"""Week 7 — paper-grade aggregate Pareto report.

Pulls together CSVs from week 6 (existing envelope study) and week 7 (new
baselines, phase scan, structure audit) into one Markdown report ready to
drop into the manuscript SI.

Inputs (paths relative to ProjectMain/):
- ``checkpoints/week6_envelope_report/envelope_summary.csv``        (week 6)
- ``checkpoints/week6_envelope_report/per_train_seed_metrics.csv``  (week 6)
- ``checkpoints/week7_baselines_*/standard_eval_by_train_seed.csv`` (week 7 phase 2)
- ``checkpoints/week7_phase_md4_m*/standard_eval_profile_summary.csv`` (week 7 phase 3, via week6_envelope_report.py)
- ``checkpoints/week7_structure_audit/structure_audit_summary.csv`` (week 7 phase 4)

Output: ``checkpoints/week7_pareto_report/REPORT.md`` plus a consolidated
``summary.csv`` for paper SI.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
DEFAULT_SAVE_ROOT = CHECKPOINTS / "week7_pareto_report"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def safe_min(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.min(arr))


def ci95_halfwidth(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(arr.size))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_envelope_summary(path: Path) -> List[Dict]:
    rows = read_csv_rows(path)
    out: List[Dict] = []
    for row in rows:
        max_dev_raw = row.get("max_deviation", "")
        try:
            max_dev = int(float(max_dev_raw))
        except (TypeError, ValueError):
            max_dev = None
        out.append(
            {
                "method": "ppo_mask" if "bounded" in row.get("label", "") else
                          ("ppo_open" if "open" in row.get("label", "") else
                           ("fixed_swap" if "fixed" in row.get("label", "") else "unknown")),
                "label": row.get("label", ""),
                "mu_co": to_float(row.get("mu_co")),
                "budget_steps": int(float(row.get("budget_steps", 0)) or 0),
                "max_deviation": max_dev,
                "n_seeds": int(float(row.get("n_train_seeds", 0)) or 0),
                "mean_best_omega": to_float(row.get("mean_best_omega")),
                "mean_feasible_best_omega": to_float(row.get("mean_feasible_best_omega")),
                "ci95_feasible_best_omega": to_float(row.get("ci95_feasible_best_omega")),
                "mean_constraint_valid_frac": to_float(row.get("mean_constraint_valid_frac")),
                "mean_constraint_d_frac": to_float(row.get("mean_constraint_d_frac")),
                "best_omega_global": to_float(row.get("best_omega_global")),
                "feasible_best_omega_global": to_float(row.get("feasible_best_omega_global")),
            }
        )
    return out


def load_baseline_dir(path: Path, method_label: str) -> List[Dict]:
    """Load `<dir>/standard_eval_by_train_seed.csv` and aggregate to one row."""
    rows = read_csv_rows(path / "standard_eval_by_train_seed.csv")
    if not rows:
        return []

    feasible = [to_float(r.get("mean_feasible_best_omega")) for r in rows]
    best = [to_float(r.get("mean_best_omega")) for r in rows]
    valid = [to_float(r.get("mean_constraint_valid_frac")) for r in rows]
    d_frac = [to_float(r.get("mean_constraint_d_frac")) for r in rows]
    n = len(rows)
    mu_co = to_float(rows[0].get("mu_co")) if rows else float("nan")

    return [
        {
            "method": method_label,
            "label": path.name,
            "mu_co": mu_co,
            "budget_steps": 0,
            "max_deviation": None,
            "n_seeds": n,
            "mean_best_omega": safe_mean(best),
            "mean_feasible_best_omega": safe_mean(feasible),
            "ci95_feasible_best_omega": ci95_halfwidth(feasible),
            "mean_constraint_valid_frac": safe_mean(valid),
            "mean_constraint_d_frac": safe_mean(d_frac),
            "best_omega_global": safe_min(best),
            "feasible_best_omega_global": safe_min(feasible),
        }
    ]


def discover_baseline_dirs(checkpoints_root: Path) -> List[Path]:
    if not checkpoints_root.exists():
        return []
    candidates = sorted(checkpoints_root.glob("week7_baselines_*"))
    return [c for c in candidates if c.is_dir()]


def headline_table(
    envelope_rows: List[Dict],
    baseline_rows: List[Dict],
    target_mus: List[float],
    target_budgets: List[int],
) -> List[Dict]:
    """For each (mu_co, budget), pick the strongest bounded mask result and
    line it up against fixed_swap, ppo_open, random_mutation, sa_mutation."""
    out: List[Dict] = []
    for mu in target_mus:
        for budget in target_budgets:
            best_bounded: Optional[Dict] = None
            for row in envelope_rows:
                if row["method"] != "ppo_mask":
                    continue
                if abs(row["mu_co"] - mu) > 1e-3:
                    continue
                if row["budget_steps"] != budget:
                    continue
                if best_bounded is None or (
                    np.isfinite(row["mean_feasible_best_omega"])
                    and row["mean_feasible_best_omega"] < best_bounded["mean_feasible_best_omega"]
                ):
                    best_bounded = row
            row_dict = {"mu_co": mu, "budget_steps": budget}
            row_dict["ppo_mask_best_label"] = best_bounded["label"] if best_bounded else ""
            row_dict["ppo_mask_max_deviation"] = best_bounded["max_deviation"] if best_bounded else None
            row_dict["ppo_mask_n_seeds"] = best_bounded["n_seeds"] if best_bounded else 0
            row_dict["ppo_mask_mean_feasible_best_omega"] = best_bounded["mean_feasible_best_omega"] if best_bounded else float("nan")
            row_dict["ppo_mask_ci95"] = best_bounded["ci95_feasible_best_omega"] if best_bounded else float("nan")

            # Match remaining methods at the same mu and budget.
            for method in ("ppo_open", "fixed_swap", "random_mutation", "sa_mutation"):
                rows = (envelope_rows + baseline_rows)
                rows = [r for r in rows if r["method"] == method and abs(r["mu_co"] - mu) < 1e-3 and (r["budget_steps"] == 0 or r["budget_steps"] == budget)]
                if not rows:
                    row_dict[f"{method}_mean_feasible_best_omega"] = float("nan")
                    row_dict[f"{method}_ci95"] = float("nan")
                    row_dict[f"{method}_n_seeds"] = 0
                    continue
                # Prefer matching budget; fall back to budget=0 (baseline rows).
                rows.sort(key=lambda r: (abs((r["budget_steps"] or budget) - budget), -(r["n_seeds"] or 0)))
                m = rows[0]
                row_dict[f"{method}_mean_feasible_best_omega"] = m["mean_feasible_best_omega"]
                row_dict[f"{method}_ci95"] = m["ci95_feasible_best_omega"]
                row_dict[f"{method}_n_seeds"] = m["n_seeds"]
            out.append(row_dict)
    return out


def write_report(
    save_root: Path,
    envelope_rows: List[Dict],
    baseline_rows: List[Dict],
    headline_rows: List[Dict],
    structure_rows: List[Dict],
) -> None:
    save_root.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Week 7 Paper-Grade Pareto Report")
    lines.append("")
    lines.append(
        "Aggregates the bounded-mask Pareto evidence from week 6 with the new "
        "feasibility-aware random / SA baselines, the mu_CO phase scan, and the "
        "best-structure physical audit."
    )
    lines.append("")

    if headline_rows:
        lines.append("## Headline: feasible best Omega at matched budget")
        lines.append("")
        lines.append(
            "| mu_CO | budget | ppo_mask (best md) | ppo_mask Omega +/- CI | ppo_open | fixed_swap | random | SA |"
        )
        lines.append(
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in headline_rows:
            md = row.get("ppo_mask_max_deviation")
            md_str = "?" if md is None else str(int(md))
            lines.append(
                "| "
                f"{row['mu_co']:.2f} | "
                f"{int(row['budget_steps'])} | "
                f"{row['ppo_mask_best_label']} (md={md_str}, n={int(row['ppo_mask_n_seeds'])}) | "
                f"{row['ppo_mask_mean_feasible_best_omega']:.3f} +/- {row['ppo_mask_ci95']:.3f} | "
                f"{row['ppo_open_mean_feasible_best_omega']:.3f} (n={int(row['ppo_open_n_seeds'])}) | "
                f"{row['fixed_swap_mean_feasible_best_omega']:.3f} (n={int(row['fixed_swap_n_seeds'])}) | "
                f"{row['random_mutation_mean_feasible_best_omega']:.3f} (n={int(row['random_mutation_n_seeds'])}) | "
                f"{row['sa_mutation_mean_feasible_best_omega']:.3f} (n={int(row['sa_mutation_n_seeds'])}) |"
            )
        lines.append("")

    if envelope_rows:
        lines.append("## Envelope Pareto: feasible best Omega vs (mu_CO, max_deviation)")
        lines.append("")
        lines.append(
            "| mu_CO | budget | label | max_dev | n_seeds | mean(feas Omega) | CI95 | valid_frac | mean(d_frac) |"
        )
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(envelope_rows, key=lambda r: (r["mu_co"], r["budget_steps"], r.get("max_deviation") or 999)):
            md = row.get("max_deviation")
            md_str = "open/fixed" if md is None else str(int(md))
            lines.append(
                "| "
                f"{row['mu_co']:.2f} | "
                f"{int(row['budget_steps'])} | "
                f"{row['label']} | "
                f"{md_str} | "
                f"{int(row['n_seeds'])} | "
                f"{row['mean_feasible_best_omega']:.3f} | "
                f"{row['ci95_feasible_best_omega']:.3f} | "
                f"{row['mean_constraint_valid_frac']:.3f} | "
                f"{row['mean_constraint_d_frac']:.3f} |"
            )
        lines.append("")

    if baseline_rows:
        lines.append("## Budget-matched random / SA baselines (Phase 2)")
        lines.append("")
        lines.append(
            "| method | label | mu_CO | n_seeds | mean(feas Omega) | CI95 | valid_frac |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in baseline_rows:
            lines.append(
                "| "
                f"{row['method']} | "
                f"{row['label']} | "
                f"{row['mu_co']:.2f} | "
                f"{int(row['n_seeds'])} | "
                f"{row['mean_feasible_best_omega']:.3f} | "
                f"{row['ci95_feasible_best_omega']:.3f} | "
                f"{row['mean_constraint_valid_frac']:.3f} |"
            )
        lines.append("")

    if structure_rows:
        lines.append("## Structural validation: Pd segregation and Warren-Cowley SRO")
        lines.append("")
        lines.append(
            "| mu_CO | max_dev | budget | profile | n_seeds | mean(best_omega) | "
            "surface_Pd | bulk_Pd | surf/bulk | WCP_PdPd | WCP_CuPd | mean(N_CO) |"
        )
        lines.append("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in structure_rows:
            md = row.get("max_deviation")
            md_str = "open/fixed" if md == "open/fixed" or md is None else str(int(md))
            budget_v = row.get("budget_steps", 0) or 0
            try:
                budget_int = int(budget_v)
            except (TypeError, ValueError):
                budget_int = 0
            lines.append(
                "| "
                f"{to_float(row.get('mu_co')):.2f} | "
                f"{md_str} | "
                f"{budget_int} | "
                f"{row.get('profile', '')} | "
                f"{int(to_float(row.get('n_seeds')))} | "
                f"{to_float(row.get('mean_best_omega')):.3f} | "
                f"{to_float(row.get('mean_surface_pd_frac')):.3f} | "
                f"{to_float(row.get('mean_bulk_avg_pd_frac')):.3f} | "
                f"{to_float(row.get('mean_surface_to_bulk_ratio')):.3f} | "
                f"{to_float(row.get('mean_wcp_pd_pd')):.3f} | "
                f"{to_float(row.get('mean_wcp_cu_pd')):.3f} | "
                f"{to_float(row.get('mean_n_co')):.2f} |"
            )
        lines.append("")

    (save_root / "REPORT.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-7 paper-grade aggregate")
    parser.add_argument(
        "--envelope-summary",
        type=str,
        default=str(CHECKPOINTS / "week6_envelope_report" / "envelope_summary.csv"),
    )
    parser.add_argument(
        "--baselines-glob-root",
        type=str,
        default=str(CHECKPOINTS),
        help="Directory containing one or more `week7_baselines_*` subdirectories.",
    )
    parser.add_argument(
        "--structure-summary",
        type=str,
        default=str(CHECKPOINTS / "week7_structure_audit" / "structure_audit_summary.csv"),
    )
    parser.add_argument(
        "--target-mus",
        type=str,
        default="-0.2,-0.6",
        help="Comma-separated mu_CO values to include in the headline panel.",
    )
    parser.add_argument(
        "--target-budgets",
        type=str,
        default="2048,4096",
        help="Comma-separated training budgets to include in the headline panel.",
    )
    parser.add_argument("--save-root", type=str, default=str(DEFAULT_SAVE_ROOT))
    args = parser.parse_args()

    save_root = Path(args.save_root)
    target_mus = [float(x) for x in args.target_mus.split(",") if x.strip()]
    target_budgets = [int(x) for x in args.target_budgets.split(",") if x.strip()]

    envelope_rows = load_envelope_summary(Path(args.envelope_summary))

    baseline_rows: List[Dict] = []
    baseline_root = Path(args.baselines_glob_root)
    for sub in discover_baseline_dirs(baseline_root):
        method_label = "random_mutation" if "_random_" in sub.name else (
            "sa_mutation" if "_sa_" in sub.name else "baseline"
        )
        baseline_rows.extend(load_baseline_dir(sub, method_label=method_label))

    headline_rows = headline_table(envelope_rows, baseline_rows, target_mus, target_budgets)
    structure_rows = read_csv_rows(Path(args.structure_summary))

    write_csv(save_root / "envelope_pareto.csv", envelope_rows)
    write_csv(save_root / "baselines_aggregated.csv", baseline_rows)
    write_csv(save_root / "headline_table.csv", headline_rows)
    write_report(save_root, envelope_rows, baseline_rows, headline_rows, structure_rows)
    print(f"[Week7-Pareto] wrote report to {save_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
