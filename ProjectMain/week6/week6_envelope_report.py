from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
SAVE_DIR = CHECKPOINTS / "week6_envelope_report"


@dataclass(frozen=True)
class EnvelopeGroup:
    label: str
    mu_co: float
    budget_steps: int
    max_deviation: Optional[int]
    method: str
    sources: tuple[str, ...]


GROUPS: tuple[EnvelopeGroup, ...] = (
    EnvelopeGroup(
        label="fixed_m02_2k",
        mu_co=-0.2,
        budget_steps=2048,
        max_deviation=None,
        method="fixed_swap",
        sources=(
            "week5_swapctrl_m02_strictstop_2k_s3",
            "week5_swapctrl_m02_strictstop_2k_seed4455",
        ),
    ),
    EnvelopeGroup(
        label="open_m02_2k",
        mu_co=-0.2,
        budget_steps=2048,
        max_deviation=None,
        method="open_mutation",
        sources=(
            "week5_longrun_m02_strictstop_2k_s2",
            "week5_longrun_m02_strictstop_2k_seed33",
            "week5_longrun_m02_strictstop_2k_seed4455",
        ),
    ),
    EnvelopeGroup(
        label="open_m02_4k",
        mu_co=-0.2,
        budget_steps=4096,
        max_deviation=None,
        method="open_mutation",
        sources=("week6_open_m02_strictstop_4k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md4_m02_2k",
        mu_co=-0.2,
        budget_steps=2048,
        max_deviation=4,
        method="bounded_mutation",
        sources=(
            "week5_maskedopen_m02_strictstop_2k_s3",
            "week5_maskedopen_m02_strictstop_2k_seed4455",
        ),
    ),
    EnvelopeGroup(
        label="bounded_md2_m02_2k",
        mu_co=-0.2,
        budget_steps=2048,
        max_deviation=2,
        method="bounded_mutation",
        sources=("week6_masksweep_m02_md2_2k_s3",),
    ),
    EnvelopeGroup(
        label="fixed_m06_2k",
        mu_co=-0.6,
        budget_steps=2048,
        max_deviation=None,
        method="fixed_swap",
        sources=(
            "week5_swapctrl_m06_strictstop_2k_s3",
            "week5_swapctrl_m06_strictstop_2k_seed4455",
        ),
    ),
    EnvelopeGroup(
        label="open_m06_2k",
        mu_co=-0.6,
        budget_steps=2048,
        max_deviation=None,
        method="open_mutation",
        sources=(
            "week5_longrun_m06_strictstop_2k_s2",
            "week5_longrun_m06_strictstop_2k_seed33",
            "week5_longrun_m06_strictstop_2k_seed4455",
        ),
    ),
    EnvelopeGroup(
        label="open_m06_4k",
        mu_co=-0.6,
        budget_steps=4096,
        max_deviation=None,
        method="open_mutation",
        sources=("week6_open_m06_strictstop_4k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md2_m06_2k",
        mu_co=-0.6,
        budget_steps=2048,
        max_deviation=2,
        method="bounded_mutation",
        sources=("week6_masksweep_m06_md2_2k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md4_m06_2k",
        mu_co=-0.6,
        budget_steps=2048,
        max_deviation=4,
        method="bounded_mutation",
        sources=("week5_maskedopen_m06_strictstop_2k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md6_m06_2k",
        mu_co=-0.6,
        budget_steps=2048,
        max_deviation=6,
        method="bounded_mutation",
        sources=("week6_masksweep_m06_md6_2k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md4_m06_4k",
        mu_co=-0.6,
        budget_steps=4096,
        max_deviation=4,
        method="bounded_mutation",
        sources=("week6_boundedopen_m06_md4_4k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md2_m06_4k",
        mu_co=-0.6,
        budget_steps=4096,
        max_deviation=2,
        method="bounded_mutation",
        sources=("week6_masksweep_m06_md2_4k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md4_m02_4k",
        mu_co=-0.2,
        budget_steps=4096,
        max_deviation=4,
        method="bounded_mutation",
        sources=("week6_boundedopen_m02_md4_4k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md6_m02_2k",
        mu_co=-0.2,
        budget_steps=2048,
        max_deviation=6,
        method="bounded_mutation",
        sources=("week6_masksweep_m02_md6_2k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md6_m02_4k",
        mu_co=-0.2,
        budget_steps=4096,
        max_deviation=6,
        method="bounded_mutation",
        sources=("week6_masksweep_m02_md6_4k_s3",),
    ),
    EnvelopeGroup(
        label="bounded_md6_m06_4k",
        mu_co=-0.6,
        budget_steps=4096,
        max_deviation=6,
        method="bounded_mutation",
        sources=("week6_masksweep_m06_md6_4k_s3",),
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


def row_value_none(value) -> bool:
    return value is None or (isinstance(value, float) and not np.isfinite(value))


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


def safe_nanmin(values: Iterable[float]) -> float:
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


def collect_train_rows(group: EnvelopeGroup) -> List[Dict]:
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
                "budget_steps": group.budget_steps,
                "max_deviation": group.max_deviation,
                "method": group.method,
                "train_seed": train_seed,
                "mean_best_omega": to_float(row.get("mean_best_omega")),
                "best_omega_global": to_float(row.get("best_omega_global")),
                "mean_best_omega_is_feasible": to_float(row.get("mean_best_omega_is_feasible")),
                "mean_feasible_best_omega": to_float(row.get("mean_feasible_best_omega")),
                "feasible_best_omega_global": to_float(row.get("feasible_best_omega_global")),
                "mean_best_omega_feasibility_gap": to_float(row.get("mean_best_omega_feasibility_gap")),
                "mean_best_theta_pd": to_float(row.get("mean_best_theta_pd")),
                "mean_best_n_co": to_float(row.get("mean_best_n_co")),
                "mean_constraint_valid_frac": to_float(row.get("mean_constraint_valid_frac")),
                "mean_constraint_violation": to_float(row.get("mean_constraint_violation")),
                "mean_constraint_d_frac": to_float(row.get("mean_constraint_d_frac")),
                "mean_noop_ratio": to_float(row.get("mean_noop_ratio")),
                "source_dir": row["source_dir"],
            }
        )
    return out


def summarize(rows: List[Dict]) -> Dict:
    return {
        "label": rows[0]["label"],
        "mu_co": rows[0]["mu_co"],
        "budget_steps": rows[0]["budget_steps"],
        "max_deviation": rows[0]["max_deviation"],
        "method": rows[0]["method"],
        "n_train_seeds": len(rows),
        "mean_best_omega": safe_mean(r["mean_best_omega"] for r in rows),
        "std_best_omega": safe_std(r["mean_best_omega"] for r in rows),
        "ci95_best_omega": ci95_halfwidth(r["mean_best_omega"] for r in rows),
        "best_omega_global": safe_nanmin(r["best_omega_global"] for r in rows),
        "mean_feasible_best_omega": safe_mean(r["mean_feasible_best_omega"] for r in rows),
        "std_feasible_best_omega": safe_std(r["mean_feasible_best_omega"] for r in rows),
        "ci95_feasible_best_omega": ci95_halfwidth(r["mean_feasible_best_omega"] for r in rows),
        "feasible_best_omega_global": safe_nanmin(r["feasible_best_omega_global"] for r in rows),
        "mean_best_omega_is_feasible": safe_mean(r["mean_best_omega_is_feasible"] for r in rows),
        "mean_best_omega_feasibility_gap": safe_mean(r["mean_best_omega_feasibility_gap"] for r in rows),
        "mean_best_theta_pd": safe_mean(r["mean_best_theta_pd"] for r in rows),
        "mean_best_n_co": safe_mean(r["mean_best_n_co"] for r in rows),
        "mean_constraint_valid_frac": safe_mean(r["mean_constraint_valid_frac"] for r in rows),
        "mean_constraint_violation": safe_mean(r["mean_constraint_violation"] for r in rows),
        "mean_constraint_d_frac": safe_mean(r["mean_constraint_d_frac"] for r in rows),
        "mean_noop_ratio": safe_mean(r["mean_noop_ratio"] for r in rows),
    }


def build_comparison_rows(summary_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple[float, int], List[Dict]] = {}
    for row in summary_rows:
        grouped.setdefault((row["mu_co"], row["budget_steps"]), []).append(row)

    out: List[Dict] = []
    for (mu_co, budget_steps), rows in sorted(grouped.items()):
        fixed = next((r for r in rows if r["method"] == "fixed_swap"), None)
        open_row = next((r for r in rows if r["method"] == "open_mutation"), None)
        bounded_rows = sorted(
            (r for r in rows if r["method"] == "bounded_mutation"),
            key=lambda r: r["mean_feasible_best_omega"],
        )
        if not bounded_rows:
            continue

        best_bounded = bounded_rows[0]
        out.append(
            {
                "mu_co": mu_co,
                "budget_steps": budget_steps,
                "best_bounded_label": best_bounded["label"],
                "best_bounded_max_deviation": best_bounded["max_deviation"],
                "best_bounded_n_train_seeds": best_bounded["n_train_seeds"],
                "best_bounded_mean_feasible_best_omega": best_bounded["mean_feasible_best_omega"],
                "fixed_mean_feasible_best_omega": float("nan") if fixed is None else fixed["mean_feasible_best_omega"],
                "open_mean_feasible_best_omega": float("nan") if open_row is None else open_row["mean_feasible_best_omega"],
                "gap_vs_fixed_feasible": float("nan")
                if fixed is None
                else best_bounded["mean_feasible_best_omega"] - fixed["mean_feasible_best_omega"],
                "gap_vs_open_feasible": float("nan")
                if open_row is None
                else best_bounded["mean_feasible_best_omega"] - open_row["mean_feasible_best_omega"],
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, detail_rows: List[Dict], summary_rows: List[Dict], comparison_rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Week 6 Envelope Report")
    lines.append("")
    lines.append("This report tracks feasible-performance as a function of composition-envelope width across chemical potentials.")
    lines.append("")

    if summary_rows:
        lines.append("## Summary")
        lines.append("")
        lines.append("| mu_CO | label | budget | max_deviation | n_seeds | mean(best_omega) | mean(feasible_best_omega) | best_feasible_frac | feasible_gap | valid_frac | mean(d_frac) |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(
            summary_rows,
            key=lambda r: (
                r["mu_co"],
                r["budget_steps"],
                999 if row_value_none(r["max_deviation"]) else r["max_deviation"],
                r["label"],
            ),
        ):
            md = "open/fixed" if row_value_none(row["max_deviation"]) else str(int(row["max_deviation"]))
            lines.append(
                "| "
                f"{row['mu_co']:.1f} | "
                f"{row['label']} | "
                f"{int(row['budget_steps'])} | "
                f"{md} | "
                f"{int(row['n_train_seeds'])} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['mean_feasible_best_omega']:.6f} | "
                f"{row['mean_best_omega_is_feasible']:.6f} | "
                f"{row['mean_best_omega_feasibility_gap']:.6f} | "
                f"{row['mean_constraint_valid_frac']:.6f} | "
                f"{row['mean_constraint_d_frac']:.6f} |"
            )
        lines.append("")

    if comparison_rows:
        lines.append("## Best Bounded Envelope vs Controls")
        lines.append("")
        lines.append("| mu_CO | budget | best_bounded | max_deviation | n_seeds | feasible_best | gap_vs_fixed | gap_vs_open_feasible |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(comparison_rows, key=lambda r: (r["mu_co"], r["budget_steps"])):
            lines.append(
                "| "
                f"{row['mu_co']:.1f} | "
                f"{int(row['budget_steps'])} | "
                f"{row['best_bounded_label']} | "
                f"{int(row['best_bounded_max_deviation'])} | "
                f"{int(row['best_bounded_n_train_seeds'])} | "
                f"{row['best_bounded_mean_feasible_best_omega']:.6f} | "
                f"{row['gap_vs_fixed_feasible']:.6f} | "
                f"{row['gap_vs_open_feasible']:.6f} |"
            )
        lines.append("")

    if detail_rows:
        lines.append("## Per Seed")
        lines.append("")
        lines.append("| label | train_seed | mu_CO | budget | max_deviation | mean(best_omega) | mean(feasible_best_omega) | gap | valid_frac | source_dir |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in sorted(
            detail_rows,
            key=lambda r: (
                r["mu_co"],
                r["budget_steps"],
                999 if row_value_none(r["max_deviation"]) else r["max_deviation"],
                r["train_seed"],
                r["label"],
            ),
        ):
            md = "open/fixed" if row_value_none(row["max_deviation"]) else str(int(row["max_deviation"]))
            lines.append(
                "| "
                f"{row['label']} | "
                f"{int(row['train_seed'])} | "
                f"{row['mu_co']:.1f} | "
                f"{int(row['budget_steps'])} | "
                f"{md} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['mean_feasible_best_omega']:.6f} | "
                f"{row['mean_best_omega_feasibility_gap']:.6f} | "
                f"{row['mean_constraint_valid_frac']:.6f} | "
                f"{row['source_dir']} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    detail_rows: List[Dict] = []
    for group in GROUPS:
        detail_rows.extend(collect_train_rows(group))

    grouped: Dict[str, List[Dict]] = {}
    for row in detail_rows:
        grouped.setdefault(row["label"], []).append(row)

    summary_rows = [summarize(rows) for _, rows in sorted(grouped.items())]
    comparison_rows = build_comparison_rows(summary_rows)
    write_csv(SAVE_DIR / "per_train_seed_metrics.csv", detail_rows)
    write_csv(SAVE_DIR / "envelope_summary.csv", summary_rows)
    write_csv(SAVE_DIR / "bounded_vs_controls.csv", comparison_rows)
    write_markdown(
        SAVE_DIR / "REPORT.md",
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
    )

    print(SAVE_DIR / "per_train_seed_metrics.csv")
    print(SAVE_DIR / "envelope_summary.csv")
    print(SAVE_DIR / "bounded_vs_controls.csv")
    print(SAVE_DIR / "REPORT.md")


if __name__ == "__main__":
    main()
