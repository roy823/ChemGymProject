from __future__ import annotations

import argparse
import csv
import hashlib
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from chem_gym.config import EnvConfig
from chem_gym.envs.chem_env import ChemGymEnv
from main import maybe_load_oracle


def parse_case(spec: str) -> Tuple[float, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid --case format: {spec}. Use mu:path")
    mu_s, path_s = spec.split(":", 1)
    return float(mu_s), path_s


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_hybrid_oracle(args) -> object:
    oracle_args = Namespace(
        oracle_ckpt=None,
        oracle_mode="hybrid",
        ads_task=args.ads_task,
        disable_ads_ensemble=False,
        ads_sm_ckpt=args.ads_sm_ckpt,
        ads_md_ckpt=args.ads_md_ckpt,
        eq2_ckpt=args.eq2_ckpt,
        uma_ckpt=args.uma_ckpt,
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=False,
    )
    oracle = maybe_load_oracle(oracle_args)
    if oracle is None:
        raise RuntimeError("Hybrid oracle unavailable.")
    return oracle


def resolve_model_paths(run_dir: Path) -> Tuple[Path, Path]:
    cand_models = [run_dir / "model.zip", run_dir / "latest_model.zip"]
    cand_stats = [run_dir / "vec_normalize.pkl", run_dir / "latest_vec_normalize.pkl"]
    model = next((p for p in cand_models if p.exists()), None)
    stats = next((p for p in cand_stats if p.exists()), None)
    if model is None:
        raise FileNotFoundError(f"Model file not found in {run_dir}")
    if stats is None:
        raise FileNotFoundError(f"VecNormalize stats not found in {run_dir}")
    return model, stats


def hash_atoms(atoms) -> str:
    symbols = ",".join(atoms.get_chemical_symbols())
    positions = np.round(atoms.get_positions(), 3)
    payload = f"{symbols}|{positions.tobytes()}"
    return hashlib.md5(payload.encode("latin1", errors="ignore")).hexdigest()


def uma_total_energy(oracle, atoms) -> float:
    slab_oracle = getattr(oracle, "slab_oracle", None)
    if slab_oracle is not None:
        if hasattr(slab_oracle, "compute_total_energy"):
            return float(slab_oracle.compute_total_energy(atoms, relax=False))
        e = float(slab_oracle.compute_energy(atoms, relax=False))
        if abs(e) < 20.0:
            e *= len(atoms)
        return float(e)
    return float(oracle.compute_slab_energy(atoms, relax=False))


def oc_slab_energy(oracle, atoms) -> float:
    ads_oracle = getattr(oracle, "ads_oracle", None)
    if ads_oracle is not None and hasattr(ads_oracle, "compute_energy"):
        return float(ads_oracle.compute_energy(atoms, relax=False))
    return float(oracle.compute_adsorbate_system_energy(atoms, relax=False))


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def verdict_from_metrics(std_delta: float, ratio: float, corr_theta: float, corr_dnco: float) -> str:
    def abs_or_zero(v: float) -> float:
        return 0.0 if not np.isfinite(v) else abs(float(v))

    c_theta = abs_or_zero(corr_theta)
    c_dnco = abs_or_zero(corr_dnco)

    if std_delta >= 0.10 or ratio >= 0.35 or c_theta >= 0.25 or c_dnco >= 0.25:
        return "fail"
    if std_delta >= 0.05 or ratio >= 0.20 or c_theta >= 0.10 or c_dnco >= 0.10:
        return "warning"
    return "pass"


def main():
    parser = argparse.ArgumentParser(description="Delta drift diagnostic for hybrid objective consistency")
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case spec in format mu:path_to_run_dir. Repeat for multiple mu points.",
    )
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--e-cu-co", type=float, default=-0.55)
    parser.add_argument("--e-pd-co", type=float, default=-1.35)
    parser.add_argument("--save-dir", type=str, default="ProjectMain/checkpoints/week3_delta_diag_v1")
    parser.add_argument("--ads-task", type=str, default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="ProjectMain/checkpoints/esen_md_direct.pt")
    parser.add_argument("--eq2-ckpt", type=str, default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="ProjectMain/checkpoints/uma-m-1p1.pt")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    oracle = build_hybrid_oracle(args)
    delta_cache: Dict[str, float] = {}

    raw_rows: List[Dict] = []
    case_rows: List[Dict] = []

    for case_spec in args.case:
        mu_co, run_path = parse_case(case_spec)
        run_dir = Path(run_path)
        model_path, stats_path = resolve_model_paths(run_dir)
        case_id = f"mu_{mu_co:+.2f}".replace(".", "p").replace("+", "pos").replace("-", "neg")

        pooled_delta: List[float] = []
        pooled_d_delta: List[float] = []
        pooled_d_omega: List[float] = []
        pooled_theta: List[float] = []
        pooled_d_nco: List[float] = []

        for seed in seeds:
            env_cfg = EnvConfig(
                mode="graph",
                init_seed=int(seed),
                mu_co=float(mu_co),
                bulk_pd_fraction=0.08,
                e_cu_co=float(args.e_cu_co),
                e_pd_co=float(args.e_pd_co),
                delta_omega_scale=20.0,
                reward_shift=0.1,
                max_steps=max(args.steps + 32, 128),
            )

            def _make():
                return ChemGymEnv(env_cfg, oracle=oracle)

            base_venv = DummyVecEnv([_make])
            venv = VecNormalize.load(str(stats_path), base_venv)
            venv.training = False
            venv.norm_reward = False
            model = MaskablePPO.load(str(model_path.with_suffix("")), env=venv)

            obs = venv.reset()
            deltas: List[float] = []
            omegas: List[float] = []
            thetas: List[float] = []
            ncos: List[int] = []

            for step in range(args.steps):
                masks = get_action_masks(venv)
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                obs, _, dones, infos = venv.step(action)
                info = infos[0]

                omega = float(info.get("omega", np.nan))
                theta = float(info.get("pd_surface_coverage", np.nan))
                nco = int(info.get("n_co", -1))
                metal_atoms = info.get("metal_atoms", None)
                if metal_atoms is None:
                    continue

                key = hash_atoms(metal_atoms)
                if key in delta_cache:
                    delta = delta_cache[key]
                else:
                    e_uma = uma_total_energy(oracle, metal_atoms)
                    e_oc = oc_slab_energy(oracle, metal_atoms)
                    delta = float(e_uma - e_oc)
                    delta_cache[key] = delta

                deltas.append(delta)
                omegas.append(omega)
                thetas.append(theta)
                ncos.append(nco)
                raw_rows.append(
                    {
                        "case_id": case_id,
                        "mu_co": mu_co,
                        "seed": seed,
                        "step": step + 1,
                        "omega": omega,
                        "theta_pd": theta,
                        "n_co": nco,
                        "delta": delta,
                    }
                )
                if dones[0]:
                    obs = venv.reset()

            if len(deltas) < 5:
                continue

            d_delta = np.diff(np.asarray(deltas, dtype=float))
            d_omega = np.diff(np.asarray(omegas, dtype=float))
            d_nco = np.diff(np.asarray(ncos, dtype=float))
            delta_arr = np.asarray(deltas, dtype=float)
            theta_arr = np.asarray(thetas, dtype=float)

            std_delta = float(np.std(delta_arr))
            std_d_delta = float(np.std(d_delta))
            std_d_omega = float(np.std(d_omega))
            ratio = float(std_d_delta / max(1e-12, std_d_omega))
            corr_theta = corr_safe(delta_arr, theta_arr)
            corr_dnco = corr_safe(d_delta, d_nco)
            seed_verdict = verdict_from_metrics(std_delta, ratio, corr_theta, corr_dnco)

            case_rows.append(
                {
                    "case_id": case_id,
                    "mu_co": mu_co,
                    "seed": seed,
                    "std_delta": std_delta,
                    "std_d_delta": std_d_delta,
                    "std_d_omega": std_d_omega,
                    "ratio_d_delta_over_d_omega": ratio,
                    "corr_delta_theta": corr_theta,
                    "corr_d_delta_d_nco": corr_dnco,
                    "verdict": seed_verdict,
                }
            )

            pooled_delta.extend(delta_arr.tolist())
            pooled_d_delta.extend(d_delta.tolist())
            pooled_d_omega.extend(d_omega.tolist())
            pooled_theta.extend(theta_arr.tolist())
            pooled_d_nco.extend(d_nco.tolist())

        if pooled_delta and pooled_d_delta and pooled_d_omega:
            pooled_delta_arr = np.asarray(pooled_delta, dtype=float)
            pooled_d_delta_arr = np.asarray(pooled_d_delta, dtype=float)
            pooled_d_omega_arr = np.asarray(pooled_d_omega, dtype=float)
            pooled_theta_arr = np.asarray(pooled_theta, dtype=float)
            pooled_d_nco_arr = np.asarray(pooled_d_nco, dtype=float)

            std_delta = float(np.std(pooled_delta_arr))
            ratio = float(np.std(pooled_d_delta_arr) / max(1e-12, np.std(pooled_d_omega_arr)))
            corr_theta = corr_safe(pooled_delta_arr, pooled_theta_arr)
            corr_dnco = corr_safe(pooled_d_delta_arr, pooled_d_nco_arr)
            v = verdict_from_metrics(std_delta, ratio, corr_theta, corr_dnco)
            case_rows.append(
                {
                    "case_id": case_id,
                    "mu_co": mu_co,
                    "seed": "pooled",
                    "std_delta": std_delta,
                    "std_d_delta": float(np.std(pooled_d_delta_arr)),
                    "std_d_omega": float(np.std(pooled_d_omega_arr)),
                    "ratio_d_delta_over_d_omega": ratio,
                    "corr_delta_theta": corr_theta,
                    "corr_d_delta_d_nco": corr_dnco,
                    "verdict": v,
                }
            )

    raw_csv = save_dir / "delta_trace.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "mu_co", "seed", "step", "omega", "theta_pd", "n_co", "delta"],
        )
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

    summary_csv = save_dir / "delta_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "mu_co",
                "seed",
                "std_delta",
                "std_d_delta",
                "std_d_omega",
                "ratio_d_delta_over_d_omega",
                "corr_delta_theta",
                "corr_d_delta_d_nco",
                "verdict",
            ],
        )
        writer.writeheader()
        for row in case_rows:
            writer.writerow(row)

    pooled_rows = [r for r in case_rows if str(r["seed"]) == "pooled"]
    overall = "pass"
    if any(r["verdict"] == "fail" for r in pooled_rows):
        overall = "fail"
    elif any(r["verdict"] == "warning" for r in pooled_rows):
        overall = "warning"

    print("DELTA_DIAG_SUMMARY")
    for r in pooled_rows:
        print(r)
    print("DELTA_DIAG_OVERALL", overall)
    print("RAW_CSV", raw_csv)
    print("SUMMARY_CSV", summary_csv)


if __name__ == "__main__":
    main()
