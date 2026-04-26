"""Week 7 — statistical significance analysis on the existing envelope CSV.

Pulls ``checkpoints/week6_envelope_report/per_train_seed_metrics.csv`` and
emits two-sided Welch's t-tests + Cohen's d for the headline contrasts:

- bounded_md{best} vs fixed_swap
- bounded_md{best} vs open_mutation (feasible filter)
- open_mutation feasible best vs fixed_swap
- max_deviation in {2, 4, 6} vs each other within same (mu_CO, budget)

Outputs:
- ``checkpoints/week7_pareto_report/significance.csv``
- Markdown table appended to ``REPORT.md``.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"


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


def welch_t(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Welch's t. Returns (t_stat, df, p_two_sided) using a Normal approximation."""
    aa = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    bb = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if aa.size < 2 or bb.size < 2:
        return float("nan"), float("nan"), float("nan")
    ma, mb = float(np.mean(aa)), float(np.mean(bb))
    va, vb = float(np.var(aa, ddof=1)), float(np.var(bb, ddof=1))
    se = math.sqrt(va / aa.size + vb / bb.size)
    if se <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    t = (ma - mb) / se
    df_num = (va / aa.size + vb / bb.size) ** 2
    df_den = (va / aa.size) ** 2 / max(1, aa.size - 1) + (vb / bb.size) ** 2 / max(1, bb.size - 1)
    df = df_num / max(df_den, 1e-12)
    # Use erfc on |t|/sqrt(2) for the two-sided p (Normal approx; df is reported separately).
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(df), float(p)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    bb = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    pooled_var = ((aa.size - 1) * np.var(aa, ddof=1) + (bb.size - 1) * np.var(bb, ddof=1)) / max(1, (aa.size + bb.size - 2))
    if pooled_var <= 1e-12:
        return float("nan")
    return float((np.mean(aa) - np.mean(bb)) / math.sqrt(pooled_var))


def ci95(a: Sequence[float]) -> float:
    arr = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


def gather(detail_rows: Sequence[Dict], mu_co: float, budget_steps: int, label_match: str) -> List[float]:
    out: List[float] = []
    for row in detail_rows:
        if row["label"] != label_match:
            continue
        if abs(to_float(row["mu_co"]) - mu_co) > 1e-3:
            continue
        if int(float(row["budget_steps"]) or 0) != int(budget_steps):
            continue
        v = to_float(row.get("mean_feasible_best_omega"))
        if np.isfinite(v):
            out.append(v)
    return out


def gather_baseline(baseline_dirs: Sequence[Path], mu_co: float, method: str) -> Tuple[List[float], int, str]:
    """Pick the largest-budget baseline run for the given (mu_co, method)."""
    best_budget = -1
    best_vals: List[float] = []
    best_name = ""
    for path in baseline_dirs:
        rows = read_csv_rows(path / "standard_eval_by_train_seed.csv")
        if not rows:
            continue
        if abs(to_float(rows[0].get("mu_co")) - mu_co) > 1e-3:
            continue
        if not rows[0].get("profile", "").startswith(method):
            continue
        # Parse budget from dir name.
        budget = 0
        for token in path.name.split("_"):
            if token.lower().endswith("k") and token.lower()[:-1].isdigit():
                budget = int(token.lower()[:-1]) * 1024
                break
            if token.isdigit():
                budget = int(token)
                break
        if budget > best_budget:
            best_budget = budget
            best_vals = [to_float(r.get("mean_feasible_best_omega")) for r in rows]
            best_vals = [v for v in best_vals if np.isfinite(v)]
            best_name = path.name
    return best_vals, best_budget, best_name


def best_bounded_label(detail_rows: Sequence[Dict], mu_co: float, budget_steps: int) -> Optional[str]:
    bounded_means: Dict[str, float] = {}
    for row in detail_rows:
        if row["method"] != "bounded_mutation":
            continue
        if abs(to_float(row["mu_co"]) - mu_co) > 1e-3:
            continue
        if int(float(row["budget_steps"]) or 0) != int(budget_steps):
            continue
        bounded_means.setdefault(row["label"], []).append(to_float(row["mean_feasible_best_omega"]))
    if not bounded_means:
        return None
    means = {label: float(np.nanmean(vs)) for label, vs in bounded_means.items()}
    return min(means, key=means.get)


def build_significance_rows(
    detail_rows: Sequence[Dict],
    mu_budgets: Sequence[Tuple[float, int]],
    baseline_dirs: Sequence[Path] = (),
) -> List[Dict]:
    rows: List[Dict] = []
    for mu_co, budget in mu_budgets:
        best_bounded = best_bounded_label(detail_rows, mu_co, budget)
        if best_bounded is None:
            continue
        bounded_vals = gather(detail_rows, mu_co, budget, best_bounded)
        fixed_label = f"fixed_m{abs(int(mu_co * 10)):02d}_{int(budget // 1024)}k"
        open_label = f"open_m{abs(int(mu_co * 10)):02d}_{int(budget // 1024)}k"
        fixed_vals = gather(detail_rows, mu_co, budget, fixed_label)
        open_vals = gather(detail_rows, mu_co, budget, open_label)

        def _emit(name: str, ref_label: str, ref_vals: List[float], ref_n_label: int = -1) -> None:
            t, df, p = welch_t(bounded_vals, ref_vals)
            d = cohens_d(bounded_vals, ref_vals)
            rows.append(
                {
                    "mu_co": mu_co,
                    "budget_steps": budget,
                    "test": f"{best_bounded} vs {ref_label}",
                    "n_bounded": len(bounded_vals),
                    "n_ref": len(ref_vals) if ref_n_label < 0 else ref_n_label,
                    "mean_bounded": float(np.nanmean(bounded_vals)) if bounded_vals else float("nan"),
                    "mean_ref": float(np.nanmean(ref_vals)) if ref_vals else float("nan"),
                    "delta_mean": (
                        float(np.nanmean(bounded_vals) - np.nanmean(ref_vals))
                        if bounded_vals and ref_vals
                        else float("nan")
                    ),
                    "ci95_bounded": ci95(bounded_vals),
                    "ci95_ref": ci95(ref_vals),
                    "welch_t": t,
                    "welch_df_eff": df,
                    "welch_p_two_sided_normal_approx": p,
                    "cohens_d": d,
                }
            )

        _emit("vs_fixed", fixed_label, fixed_vals)
        _emit("vs_open_feasible", open_label, open_vals)

        # Also compare to random / SA baselines (not budget-matched in general,
        # but we still report the gap so the manuscript can frame sample efficiency).
        for method in ("random_mutation", "sa_mutation"):
            ref_vals, ref_budget, ref_name = gather_baseline(baseline_dirs, mu_co, method)
            if not ref_vals:
                continue
            ref_label = f"{ref_name} (budget={ref_budget})"
            _emit(f"vs_{method}", ref_label, ref_vals)
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_markdown(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if path.exists():
        body = path.read_text(encoding="utf-8")
        if "## Significance tests" in body:
            # Replace existing block.
            head, _, _ = body.partition("## Significance tests")
            path.write_text(head.rstrip() + "\n", encoding="utf-8")
        lines.append("")
    lines.append("## Significance tests (Welch t, Normal-approx p, Cohen's d)")
    lines.append("")
    lines.append(
        "| mu_CO | budget | comparison | n_b | n_r | mean_b | mean_r | dmean | CI95_b | CI95_r | Welch t | df_eff | p~ | Cohen's d |"
    )
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row['mu_co']:.2f} | "
            f"{int(row['budget_steps'])} | "
            f"{row['test']} | "
            f"{int(row['n_bounded'])} | "
            f"{int(row['n_ref'])} | "
            f"{row['mean_bounded']:.3f} | "
            f"{row['mean_ref']:.3f} | "
            f"{row['delta_mean']:.3f} | "
            f"{row['ci95_bounded']:.3f} | "
            f"{row['ci95_ref']:.3f} | "
            f"{row['welch_t']:.3f} | "
            f"{row['welch_df_eff']:.2f} | "
            f"{row['welch_p_two_sided_normal_approx']:.2e} | "
            f"{row['cohens_d']:.3f} |"
        )
    lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-7 significance analysis for envelope study")
    parser.add_argument(
        "--detail-csv",
        type=str,
        default=str(CHECKPOINTS / "week6_envelope_report" / "per_train_seed_metrics.csv"),
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=str(CHECKPOINTS / "week7_pareto_report"),
    )
    parser.add_argument("--mus", type=str, default="-0.2,-0.6")
    parser.add_argument("--budgets", type=str, default="2048,4096")
    parser.add_argument(
        "--baselines-glob-root",
        type=str,
        default=str(CHECKPOINTS),
        help="Directory containing one or more `week7_baselines_*` subdirectories.",
    )
    args = parser.parse_args()

    detail_rows = read_csv_rows(Path(args.detail_csv))
    mus = [float(x) for x in args.mus.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    mu_budgets = [(mu, b) for mu in mus for b in budgets]
    baseline_dirs = sorted(Path(args.baselines_glob_root).glob("week7_baselines_*"))
    sig_rows = build_significance_rows(detail_rows, mu_budgets, baseline_dirs=baseline_dirs)

    save_root = Path(args.save_root)
    write_csv(save_root / "significance.csv", sig_rows)
    append_markdown(save_root / "REPORT.md", sig_rows)
    print(f"[Week7-Significance] wrote {len(sig_rows)} rows to {save_root}")


if __name__ == "__main__":
    main()
