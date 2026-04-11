from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"


@dataclass(frozen=True)
class ExperimentGroup:
    label: str
    mu_co: float
    method: str
    sources: tuple[str, ...]


GROUPS: tuple[ExperimentGroup, ...] = (
    ExperimentGroup(
        label="mutation_open_m02",
        mu_co=-0.2,
        method="open_composition_mutation",
        sources=(
            "week5_longrun_m02_strictstop_2k_s2",
            "week5_longrun_m02_strictstop_2k_seed33",
            "week5_longrun_m02_strictstop_2k_seed4455",
        ),
    ),
    ExperimentGroup(
        label="swap_fixed_m02",
        mu_co=-0.2,
        method="fixed_composition_swap",
        sources=(
            "week5_swapctrl_m02_strictstop_2k_s3",
            "week5_swapctrl_m02_strictstop_2k_seed4455",
        ),
    ),
    ExperimentGroup(
        label="mutation_open_m06",
        mu_co=-0.6,
        method="open_composition_mutation",
        sources=(
            "week5_longrun_m06_strictstop_2k_s2",
            "week5_longrun_m06_strictstop_2k_seed33",
            "week5_longrun_m06_strictstop_2k_seed4455",
        ),
    ),
    ExperimentGroup(
        label="swap_fixed_m06",
        mu_co=-0.6,
        method="fixed_composition_swap",
        sources=(
            "week5_swapctrl_m06_strictstop_2k_s3",
            "week5_swapctrl_m06_strictstop_2k_seed4455",
        ),
    ),
)


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


def to_int(value) -> int:
    return int(round(float(value)))


def collect_train_rows(group: ExperimentGroup) -> List[Dict]:
    dedup: Dict[int, Dict] = {}
    for source_name in group.sources:
        source_dir = CHECKPOINTS / source_name
        if not source_dir.exists():
            continue

        root_csv = source_dir / "standard_eval_by_train_seed.csv"
        rows = read_csv_rows(root_csv)
        if rows:
            for row in rows:
                train_seed = to_int(row["train_seed"])
                row_copy = dict(row)
                row_copy["source_dir"] = str(source_dir.relative_to(ROOT))
                row_copy["source_mtime"] = source_dir.stat().st_mtime
                prev = dedup.get(train_seed)
                if prev is None or float(row_copy["source_mtime"]) >= float(prev["source_mtime"]):
                    dedup[train_seed] = row_copy
            continue

        for seed_dir in sorted(p for p in source_dir.glob("*/*") if p.is_dir()):
            summary_csv = seed_dir / "standard_eval_summary.csv"
            for row in read_csv_rows(summary_csv):
                train_seed = to_int(row["train_seed"])
                row_copy = dict(row)
                row_copy["source_dir"] = str(source_dir.relative_to(ROOT))
                row_copy["source_mtime"] = source_dir.stat().st_mtime
                prev = dedup.get(train_seed)
                if prev is None or float(row_copy["source_mtime"]) >= float(prev["source_mtime"]):
                    dedup[train_seed] = row_copy

    out: List[Dict] = []
    for train_seed, row in sorted(dedup.items()):
        out.append(
            {
                "label": group.label,
                "mu_co": group.mu_co,
                "method": group.method,
                "train_seed": train_seed,
                "mean_best_omega": to_float(row.get("mean_best_omega")),
                "best_omega_global": to_float(row.get("best_omega_global")),
                "mean_best_theta_pd": to_float(row.get("mean_best_theta_pd")),
                "mean_best_n_co": to_float(row.get("mean_best_n_co")),
                "mean_noop_ratio": to_float(row.get("mean_noop_ratio")),
                "mean_mutation_ratio": to_float(row.get("mean_mutation_ratio")),
                "mean_constraint_valid_frac": to_float(row.get("mean_constraint_valid_frac")),
                "mean_constraint_violation": to_float(row.get("mean_constraint_violation")),
                "max_constraint_violation": to_float(row.get("max_constraint_violation")),
                "mean_constraint_d_frac": to_float(row.get("mean_constraint_d_frac")),
                "mean_uncertainty": to_float(row.get("mean_uncertainty")),
                "source_dir": row["source_dir"],
            }
        )
    return out


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1))


def safe_nanmax(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def ci95_halfwidth(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    std = float(np.std(arr, ddof=1))
    return float(1.96 * std / np.sqrt(arr.size))


def summarize_group(rows: List[Dict]) -> Dict:
    return {
        "label": rows[0]["label"],
        "mu_co": rows[0]["mu_co"],
        "method": rows[0]["method"],
        "n_train_seeds": len(rows),
        "mean_best_omega": safe_mean(r["mean_best_omega"] for r in rows),
        "std_best_omega": safe_std(r["mean_best_omega"] for r in rows),
        "ci95_best_omega": ci95_halfwidth(r["mean_best_omega"] for r in rows),
        "best_omega_global": float(np.nanmin([r["best_omega_global"] for r in rows])),
        "mean_best_theta_pd": safe_mean(r["mean_best_theta_pd"] for r in rows),
        "mean_best_n_co": safe_mean(r["mean_best_n_co"] for r in rows),
        "mean_noop_ratio": safe_mean(r["mean_noop_ratio"] for r in rows),
        "mean_mutation_ratio": safe_mean(r["mean_mutation_ratio"] for r in rows),
        "mean_constraint_valid_frac": safe_mean(r["mean_constraint_valid_frac"] for r in rows),
        "mean_constraint_violation": safe_mean(r["mean_constraint_violation"] for r in rows),
        "max_constraint_violation": safe_nanmax(r["max_constraint_violation"] for r in rows),
        "mean_constraint_d_frac": safe_mean(r["mean_constraint_d_frac"] for r in rows),
        "mean_uncertainty": safe_mean(r["mean_uncertainty"] for r in rows),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, detail_rows: List[Dict], summary_rows: List[Dict], gap_rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Week 5 Operando Value Report")
    lines.append("")
    lines.append("This report compares the open-composition mutation mainline against the fixed-composition swap control under matched 2048-step hybrid-oracle budgets.")
    lines.append("")

    if summary_rows:
        lines.append("## Method Summary")
        lines.append("")
        lines.append("| mu_CO (eV) | Method | n_train_seeds | mean(best_omega) | std | 95% CI | best_omega_global | mean(theta_Pd) | mean(N_CO) | valid_frac | noop_ratio |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(summary_rows, key=lambda r: (r["mu_co"], r["method"])):
            lines.append(
                "| "
                f"{row['mu_co']:.1f} | "
                f"{row['method']} | "
                f"{int(row['n_train_seeds'])} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['std_best_omega']:.6f} | "
                f"{row['ci95_best_omega']:.6f} | "
                f"{row['best_omega_global']:.6f} | "
                f"{row['mean_best_theta_pd']:.6f} | "
                f"{row['mean_best_n_co']:.6f} | "
                f"{row['mean_constraint_valid_frac']:.6f} | "
                f"{row['mean_noop_ratio']:.6f} |"
            )
        lines.append("")

    if gap_rows:
        lines.append("## Open-vs-Fixed Gap")
        lines.append("")
        lines.append("| mu_CO (eV) | open_mean(best_omega) | fixed_mean(best_omega) | gap(open-fixed) | open_n | fixed_n |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(gap_rows, key=lambda r: r["mu_co"]):
            lines.append(
                "| "
                f"{row['mu_co']:.1f} | "
                f"{row['open_mean_best_omega']:.6f} | "
                f"{row['fixed_mean_best_omega']:.6f} | "
                f"{row['gap_open_minus_fixed']:.6f} | "
                f"{int(row['open_n_train_seeds'])} | "
                f"{int(row['fixed_n_train_seeds'])} |"
            )
        lines.append("")

    if detail_rows:
        lines.append("## Per Train-Seed Rows")
        lines.append("")
        lines.append("| label | mu_CO (eV) | method | train_seed | mean(best_omega) | best_omega_global | theta_Pd | N_CO | valid_frac | source_dir |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in sorted(detail_rows, key=lambda r: (r["mu_co"], r["method"], r["train_seed"])):
            lines.append(
                "| "
                f"{row['label']} | "
                f"{row['mu_co']:.1f} | "
                f"{row['method']} | "
                f"{int(row['train_seed'])} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['best_omega_global']:.6f} | "
                f"{row['mean_best_theta_pd']:.6f} | "
                f"{row['mean_best_n_co']:.6f} | "
                f"{row['mean_constraint_valid_frac']:.6f} | "
                f"{row['source_dir']} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_gap_rows(summary_rows: List[Dict]) -> List[Dict]:
    by_mu: Dict[float, Dict[str, Dict]] = {}
    for row in summary_rows:
        by_mu.setdefault(float(row["mu_co"]), {})[str(row["method"])] = row

    gaps: List[Dict] = []
    for mu_co, sub in sorted(by_mu.items()):
        open_row = sub.get("open_composition_mutation")
        fixed_row = sub.get("fixed_composition_swap")
        if open_row is None or fixed_row is None:
            continue
        gaps.append(
            {
                "mu_co": float(mu_co),
                "open_mean_best_omega": float(open_row["mean_best_omega"]),
                "fixed_mean_best_omega": float(fixed_row["mean_best_omega"]),
                "gap_open_minus_fixed": float(open_row["mean_best_omega"] - fixed_row["mean_best_omega"]),
                "open_n_train_seeds": int(open_row["n_train_seeds"]),
                "fixed_n_train_seeds": int(fixed_row["n_train_seeds"]),
            }
        )
    return gaps


def main() -> None:
    detail_rows: List[Dict] = []
    for group in GROUPS:
        detail_rows.extend(collect_train_rows(group))

    summary_rows: List[Dict] = []
    grouped: Dict[str, List[Dict]] = {}
    for row in detail_rows:
        grouped.setdefault(row["label"], []).append(row)
    for label in sorted(grouped):
        summary_rows.append(summarize_group(grouped[label]))

    gap_rows = build_gap_rows(summary_rows)

    save_dir = CHECKPOINTS / "week5_operando_value_report"
    write_csv(save_dir / "per_train_seed_metrics.csv", detail_rows)
    write_csv(save_dir / "method_summary.csv", summary_rows)
    write_csv(save_dir / "open_vs_fixed_gap.csv", gap_rows)
    write_markdown(
        path=save_dir / "REPORT.md",
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        gap_rows=gap_rows,
    )

    print(save_dir / "per_train_seed_metrics.csv")
    print(save_dir / "method_summary.csv")
    print(save_dir / "open_vs_fixed_gap.csv")
    print(save_dir / "REPORT.md")


if __name__ == "__main__":
    main()
