from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..common.results import collect_completed_part2_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Part 2 experiment metrics into a single CSV.")
    parser.add_argument("--output-root", default="outputs/part2")
    parser.add_argument("--summary-name", default="comparison_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = collect_completed_part2_runs(output_root)

    summary_path = output_root / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        summary_path.write_text("", encoding="utf-8")
        print(f"No completed runs found under {output_root}")
        return

    summary_rows = [
        {
            "dataset_version": row["dataset_version"],
            "model_name": row["model_name"],
            "strategy": row["strategy"],
            "pretrained": row["pretrained"],
            "val_loss": row["val_loss"],
            "val_top1_accuracy": row["val_top1_accuracy"],
            "val_macro_f1": row["val_macro_f1"],
            "test_loss": row["test_loss"],
            "test_top1_accuracy": row["test_top1_accuracy"],
            "test_macro_f1": row["test_macro_f1"],
            "output_dir": row["output_dir"],
        }
        for row in rows
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} rows to {summary_path}")



if __name__ == "__main__":
    main()
