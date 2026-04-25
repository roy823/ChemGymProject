"""Week 7 — best-structure physical audit.

For every trained run in a list of input directories, replay one greedy
episode using the saved MaskablePPO checkpoint, snapshot the metal slab at
the step with minimum *feasible* Omega, and compute physical descriptors
that map back to the literature on adsorbate-driven Cu-Pd segregation
(`Nat Commun 2021`, https://www.nature.com/articles/s41467-021-21555-z):

- Warren-Cowley short-range order parameters via
  ``chem_gym.analysis.sro_analysis.calculate_wcp``.
- Per-layer Pd fraction.
- Surface-vs-bulk Pd enrichment ratio.

Outputs:
- ``<save-root>/structure_audit_per_run.csv``
- ``<save-root>/structure_audit_summary.csv``
- ``<save-root>/REPORT.md``
- A `.cif` and `.xyz` of the best feasible structure under each run dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from chem_gym.analysis.sro_analysis import calculate_wcp
from chem_gym.config import (
    COAdsorptionConfig,
    ConstraintConfig,
    EnvConfig,
    RewardConfig,
    UMAPBRSConfig,
)
from chem_gym.envs.chem_env import ChemGymEnv
from main import maybe_load_oracle


# Directory-name conventions used by week5/6 queue scripts.
RUN_NAME_RE = re.compile(
    r"(?P<prefix>week[56]_(?:masksweep|maskedopen|boundedopen|longrun|swapctrl|open))_"
    r"m(?P<mu>\d{2})"
    r"(?:_(?:strictstop|md(?P<md>\d+)))*"
    r"(?:_(?P<budget>\d+k))?",
    re.IGNORECASE,
)


def parse_run_metadata(run_dir: Path) -> Dict[str, object]:
    """Best-effort metadata extraction from the run directory name."""
    name = run_dir.name
    meta: Dict[str, object] = {"run_dir": str(run_dir), "run_name": name}
    m = RUN_NAME_RE.search(name)
    if m:
        mu_raw = m.group("mu")
        if mu_raw:
            meta["mu_co"] = -float(mu_raw) / 10.0
        md_raw = m.group("md")
        if md_raw:
            meta["max_deviation"] = int(md_raw)
        budget_raw = m.group("budget")
        if budget_raw:
            meta["budget_steps"] = int(budget_raw.lower().replace("k", "")) * 1024
    return meta


def find_seed_dirs(run_dir: Path) -> List[Path]:
    """Return seed directories under run_dir.

    week4_action_reward_ablation organizes runs as:
        run_dir / <profile> / seed_<seed> / latest_model.zip
    """
    seed_dirs: List[Path] = []
    for profile_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in profile_dir.iterdir() if p.is_dir()):
            if (seed_dir / "latest_model.zip").exists():
                seed_dirs.append(seed_dir)
    return seed_dirs


def parse_seed_from_dir(seed_dir: Path) -> int:
    name = seed_dir.name
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse seed from {seed_dir}")
    return int(m.group(1))


def detect_profile(seed_dir: Path) -> str:
    return seed_dir.parent.name


def build_audit_env_config(
    profile: str,
    mu_co: float,
    seed: int,
    eval_steps: int,
    max_deviation: Optional[int],
) -> EnvConfig:
    reward_cfg = RewardConfig(
        mu_co=float(mu_co),
        delta_omega_scale=1.0,
        reward_profile="pure_delta_omega",
    )
    constraint_cfg = ConstraintConfig(
        constraint_update_mode="frozen",
        constraint_weight=0.0,
        constraint_lambda_init=0.0,
        constraint_lambda_min=0.0,
        constraint_lambda_max=0.0,
    )
    co_cfg = COAdsorptionConfig(co_max_coverage=1.0)
    uma_cfg = UMAPBRSConfig(use_uma_pbrs=False)
    cfg = EnvConfig(
        mode="graph",
        init_seed=int(seed),
        max_steps=int(eval_steps + 32),
        bulk_pd_fraction=0.08,
        n_active_layers=4,
        action_mode="swap" if profile.startswith("swap_") else "mutation",
        enable_noop_action=False,
        stop_terminates=True,
        min_stop_steps=8,
        use_deviation_mask=bool(max_deviation is not None),
        reward=reward_cfg,
        constraint=constraint_cfg,
        co_adsorption=co_cfg,
        uma_pbrs=uma_cfg,
    )
    if max_deviation is not None:
        cfg.max_deviation = int(max_deviation)
    return cfg


def replay_best_feasible(
    seed_dir: Path,
    profile: str,
    mu_co: float,
    eval_steps: int,
    max_deviation: Optional[int],
    oracle,
) -> Optional[Dict]:
    model_path = seed_dir / "latest_model"
    stats_path = seed_dir / "latest_vec_normalize.pkl"
    if not model_path.with_suffix(".zip").exists() or not stats_path.exists():
        return None

    seed = parse_seed_from_dir(seed_dir)
    env_cfg = build_audit_env_config(
        profile=profile,
        mu_co=mu_co,
        seed=seed,
        eval_steps=int(eval_steps),
        max_deviation=max_deviation,
    )

    def _make_single():
        return ChemGymEnv(env_cfg, oracle=oracle)

    base_venv = DummyVecEnv([_make_single])
    venv = VecNormalize.load(str(stats_path), base_venv)
    venv.training = False
    venv.norm_reward = False
    model = MaskablePPO.load(str(model_path), env=venv)

    obs = venv.reset()
    best_omega = float("inf")
    best_atoms = None
    best_metal = None
    best_info: Optional[Dict] = None
    feasible_omega_log: List[float] = []

    for _ in range(int(eval_steps)):
        masks = get_action_masks(venv)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        omega = float(info.get("omega", float("nan")))
        violation = float(info.get("constraint_violation", float("nan")))
        is_feasible = bool(np.isfinite(violation) and violation <= 1e-12)
        if is_feasible and np.isfinite(omega):
            feasible_omega_log.append(omega)
            if omega < best_omega:
                best_omega = omega
                best_atoms = info.get("atoms")
                best_metal = info.get("metal_atoms")
                best_info = dict(info)
        if dones[0]:
            obs = venv.reset()

    if best_atoms is None or best_metal is None:
        return None

    return {
        "seed_dir": seed_dir,
        "best_omega": float(best_omega),
        "best_atoms": best_atoms,
        "best_metal": best_metal,
        "best_info": best_info,
        "feasible_n_steps": int(len(feasible_omega_log)),
    }


def per_layer_pd_fraction(metal_atoms, n_layers: int = 4) -> Dict[str, float]:
    """Bin metal atoms by z and compute per-layer Pd fraction.

    The slab's bottom layer is fixed substrate Cu, so the active layers are
    typically the top `n_active = n_layers` layers. This helper returns
    fractions for layers 0 (bottom) through n_layers-1 (top surface).
    """
    if metal_atoms is None or len(metal_atoms) == 0:
        return {}
    z = np.asarray(metal_atoms.get_positions()[:, 2], dtype=float)
    symbols = np.asarray(metal_atoms.get_chemical_symbols())
    z_min, z_max = float(np.min(z)), float(np.max(z))
    if z_max - z_min < 1e-6:
        return {f"layer{i}_pd_frac": float("nan") for i in range(n_layers)}

    bins = np.linspace(z_min - 1e-3, z_max + 1e-3, num=n_layers + 1)
    out: Dict[str, float] = {}
    for i in range(n_layers):
        mask = (z >= bins[i]) & (z < bins[i + 1])
        n_in = int(np.sum(mask))
        if n_in == 0:
            out[f"layer{i}_pd_frac"] = float("nan")
            out[f"layer{i}_n_atoms"] = 0
            continue
        n_pd = int(np.sum(symbols[mask] == "Pd"))
        out[f"layer{i}_pd_frac"] = float(n_pd / n_in)
        out[f"layer{i}_n_atoms"] = n_in
    # Surface-vs-bulk enrichment: top vs (mean of remaining)
    top_key = f"layer{n_layers - 1}_pd_frac"
    rest = [out.get(f"layer{i}_pd_frac", float("nan")) for i in range(n_layers - 1)]
    rest_arr = np.asarray([v for v in rest if np.isfinite(v)], dtype=float)
    bulk_avg = float(np.mean(rest_arr)) if rest_arr.size else float("nan")
    out["bulk_avg_pd_frac"] = bulk_avg
    out["surface_pd_frac"] = out.get(top_key, float("nan"))
    if bulk_avg and bulk_avg > 1e-6 and np.isfinite(out["surface_pd_frac"]):
        out["surface_to_bulk_ratio"] = float(out["surface_pd_frac"] / bulk_avg)
    else:
        out["surface_to_bulk_ratio"] = float("nan")
    return out


def warren_cowley(metal_atoms) -> Dict[str, float]:
    if metal_atoms is None or len(metal_atoms) == 0:
        return {"wcp_cu_cu": float("nan"), "wcp_cu_pd": float("nan"), "wcp_pd_pd": float("nan")}
    try:
        wcp = calculate_wcp(metal_atoms, ["Cu", "Pd"])
    except Exception as exc:  # pragma: no cover - structural pathologies
        sys.stderr.write(f"[Week7-Audit] WCP calc failed: {exc}\n")
        return {"wcp_cu_cu": float("nan"), "wcp_cu_pd": float("nan"), "wcp_pd_pd": float("nan")}
    return {
        "wcp_cu_cu": float(wcp.get("Cu-Cu", float("nan"))),
        "wcp_cu_pd": float(wcp.get("Cu-Pd", float("nan"))),
        "wcp_pd_pd": float(wcp.get("Pd-Pd", float("nan"))),
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_save_structure(out_dir: Path, atoms_with_co, metal_atoms, omega: float) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    if metal_atoms is not None:
        try:
            metal_atoms.write(out_dir / "best_metal.cif")
            paths["best_metal_cif"] = str(out_dir / "best_metal.cif")
        except Exception:
            pass
    if atoms_with_co is not None:
        try:
            atoms_with_co.write(out_dir / "best_with_co.xyz")
            paths["best_with_co_xyz"] = str(out_dir / "best_with_co.xyz")
        except Exception:
            pass
    paths["best_omega"] = f"{omega:.6f}"
    return paths


def aggregate_summary(rows: Sequence[Dict]) -> List[Dict]:
    if not rows:
        return []
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = (
            float(row.get("mu_co", float("nan"))),
            row.get("max_deviation", "open/fixed"),
            row.get("budget_steps", -1),
            row.get("profile", "unknown"),
        )
        grouped.setdefault(key, []).append(row)

    out: List[Dict] = []
    for (mu_co, md, budget, profile), sub in sorted(
        grouped.items(),
        key=lambda kv: (
            kv[0][0],
            999 if kv[0][1] == "open/fixed" else kv[0][1],
            kv[0][2],
            kv[0][3],
        ),
    ):
        agg = {
            "mu_co": mu_co,
            "max_deviation": md,
            "budget_steps": budget,
            "profile": profile,
            "n_seeds": len(sub),
            "mean_best_omega": float(np.nanmean([r["best_omega"] for r in sub])),
            "mean_surface_pd_frac": float(np.nanmean([r.get("surface_pd_frac", float("nan")) for r in sub])),
            "mean_bulk_avg_pd_frac": float(np.nanmean([r.get("bulk_avg_pd_frac", float("nan")) for r in sub])),
            "mean_surface_to_bulk_ratio": float(np.nanmean([r.get("surface_to_bulk_ratio", float("nan")) for r in sub])),
            "mean_wcp_pd_pd": float(np.nanmean([r.get("wcp_pd_pd", float("nan")) for r in sub])),
            "mean_wcp_cu_pd": float(np.nanmean([r.get("wcp_cu_pd", float("nan")) for r in sub])),
            "mean_wcp_cu_cu": float(np.nanmean([r.get("wcp_cu_cu", float("nan")) for r in sub])),
            "mean_n_co": float(np.nanmean([r.get("n_co", float("nan")) for r in sub])),
            "mean_pd_surface_coverage": float(np.nanmean([r.get("pd_surface_coverage", float("nan")) for r in sub])),
        }
        out.append(agg)
    return out


def write_markdown(path: Path, per_run_rows: Sequence[Dict], summary_rows: Sequence[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Week 7 Best-Structure Audit")
    lines.append("")
    lines.append(
        "Each row replays the best `latest_model.zip` for one greedy episode, "
        "snapshots the metal slab at the step with minimum feasible Omega, and "
        "computes Warren-Cowley short-range order plus per-layer Pd fraction."
    )
    lines.append("")

    if summary_rows:
        lines.append("## Summary by (mu_CO, max_deviation, budget)")
        lines.append("")
        lines.append(
            "| mu_CO | max_dev | budget | profile | n_seeds | mean(best_omega) | "
            "mean(surface_Pd) | mean(bulk_Pd) | surf/bulk | mean(WCP_PdPd) | mean(WCP_CuPd) | mean(N_CO) |"
        )
        lines.append("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in summary_rows:
            md = row["max_deviation"]
            md_str = "open/fixed" if md == "open/fixed" else f"{int(md)}"
            budget = int(row["budget_steps"]) if isinstance(row["budget_steps"], (int, float)) else "?"
            lines.append(
                "| "
                f"{row['mu_co']:.2f} | "
                f"{md_str} | "
                f"{budget} | "
                f"{row['profile']} | "
                f"{int(row['n_seeds'])} | "
                f"{row['mean_best_omega']:.4f} | "
                f"{row['mean_surface_pd_frac']:.3f} | "
                f"{row['mean_bulk_avg_pd_frac']:.3f} | "
                f"{row['mean_surface_to_bulk_ratio']:.3f} | "
                f"{row['mean_wcp_pd_pd']:.3f} | "
                f"{row['mean_wcp_cu_pd']:.3f} | "
                f"{row['mean_n_co']:.2f} |"
            )
        lines.append("")

    if per_run_rows:
        lines.append("## Per Run")
        lines.append("")
        lines.append(
            "| run | seed | mu_CO | max_dev | best_omega | surface_Pd | bulk_Pd | WCP_PdPd | WCP_CuPd | N_CO |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in per_run_rows:
            md = row.get("max_deviation", "open/fixed")
            md_str = "open/fixed" if md == "open/fixed" else f"{int(md)}"
            lines.append(
                "| "
                f"{Path(row['run_dir']).name} | "
                f"{int(row['seed'])} | "
                f"{row['mu_co']:.2f} | "
                f"{md_str} | "
                f"{row['best_omega']:.4f} | "
                f"{row.get('surface_pd_frac', float('nan')):.3f} | "
                f"{row.get('bulk_avg_pd_frac', float('nan')):.3f} | "
                f"{row.get('wcp_pd_pd', float('nan')):.3f} | "
                f"{row.get('wcp_cu_pd', float('nan')):.3f} | "
                f"{row.get('n_co', float('nan')):.2f} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-7 best-structure physical audit")
    parser.add_argument(
        "--input-runs",
        type=str,
        nargs="+",
        required=True,
        help="One or more run directories produced by week4_action_reward_ablation.",
    )
    parser.add_argument("--save-root", type=str, required=True)
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument(
        "--max-deviation-override",
        type=int,
        default=None,
        help="Force max_deviation regardless of the value parsed from run-name.",
    )
    parser.add_argument("--mu-co-override", type=float, default=None)
    parser.add_argument("--oracle-mode", choices=["hybrid", "uma", "none"], default="hybrid")
    parser.add_argument("--ads-task", type=str, default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="ProjectMain/checkpoints/esen_md_direct.pt")
    parser.add_argument("--eq2-ckpt", type=str, default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="ProjectMain/checkpoints/uma-s-1p1.pt")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.eval_steps = min(int(args.eval_steps), 16)

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    if str(args.oracle_mode).lower() == "none":
        oracle = None
    else:
        oracle_args = Namespace(
            oracle_ckpt=None,
            oracle_mode=args.oracle_mode,
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
        if oracle is None and not args.smoke:
            raise RuntimeError("Audit requires an oracle. Use --oracle-mode hybrid with valid checkpoints.")

    per_run_rows: List[Dict] = []

    for run_str in args.input_runs:
        run_dir = Path(run_str)
        if not run_dir.exists():
            sys.stderr.write(f"[Week7-Audit] Skip missing dir: {run_dir}\n")
            continue
        meta = parse_run_metadata(run_dir)
        mu_co = float(args.mu_co_override) if args.mu_co_override is not None else float(meta.get("mu_co", -0.6))
        max_dev = (
            int(args.max_deviation_override)
            if args.max_deviation_override is not None
            else (int(meta["max_deviation"]) if "max_deviation" in meta else None)
        )
        budget = int(meta.get("budget_steps", 0))

        seed_dirs = find_seed_dirs(run_dir)
        if not seed_dirs:
            sys.stderr.write(f"[Week7-Audit] No seed runs under {run_dir}\n")
            continue

        for seed_dir in seed_dirs:
            profile = detect_profile(seed_dir)
            seed = parse_seed_from_dir(seed_dir)
            print(f"[Week7-Audit] {run_dir.name} :: {profile} :: seed={seed}")
            replay = replay_best_feasible(
                seed_dir=seed_dir,
                profile=profile,
                mu_co=mu_co,
                eval_steps=int(args.eval_steps),
                max_deviation=max_dev,
                oracle=oracle,
            )
            if replay is None:
                sys.stderr.write(f"[Week7-Audit] No feasible-best snapshot in {seed_dir}\n")
                continue
            structure_paths = maybe_save_structure(
                out_dir=seed_dir / "week7_audit",
                atoms_with_co=replay["best_atoms"],
                metal_atoms=replay["best_metal"],
                omega=replay["best_omega"],
            )
            wcp = warren_cowley(replay["best_metal"])
            layers = per_layer_pd_fraction(replay["best_metal"], n_layers=4)
            info = replay["best_info"] or {}
            row = {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "profile": profile,
                "seed": seed,
                "mu_co": mu_co,
                "max_deviation": max_dev if max_dev is not None else "open/fixed",
                "budget_steps": budget,
                "best_omega": float(replay["best_omega"]),
                "n_co": int(info.get("n_co", -1)),
                "pd_surface_coverage": float(info.get("pd_surface_coverage", float("nan"))),
                "feasible_n_steps": int(replay["feasible_n_steps"]),
                **wcp,
                **layers,
                **structure_paths,
            }
            per_run_rows.append(row)

    summary_rows = aggregate_summary(per_run_rows)
    write_csv(save_root / "structure_audit_per_run.csv", per_run_rows)
    write_csv(save_root / "structure_audit_summary.csv", summary_rows)
    write_markdown(save_root / "REPORT.md", per_run_rows, summary_rows)
    (save_root / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[Week7-Audit] wrote {len(per_run_rows)} per-run rows to {save_root}")


if __name__ == "__main__":
    main()
