from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Part 2 experiment metrics into a single CSV.")
    parser.add_argument("--output-root", default="outputs/part2")
    parser.add_argument("--summary-name", default="comparison_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows: list[dict[str, object]] = []

    if not output_root.exists():
        print(f"No output directory found at {output_root}")
        return

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_config_path = run_dir / "run_config.json"
        test_metrics_path = run_dir / "test_metrics.json"
        history_path = run_dir / "history.json"
        if not (run_config_path.exists() and test_metrics_path.exists() and history_path.exists()):
            continue

        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        last_epoch = history[-1]
        rows.append(
            {
                "dataset_version": run_config.get("dataset_version"),
                "model_name": run_config.get("model_name"),
                "strategy": run_config.get("strategy"),
                "pretrained": run_config.get("pretrained"),
                "val_loss": last_epoch.get("val_loss"),
                "val_top1_accuracy": last_epoch.get("val_accuracy"),
                "val_macro_f1": last_epoch.get("val_macro_f1"),
                "test_loss": test_metrics.get("test_loss"),
                "test_top1_accuracy": test_metrics.get("test_accuracy"),
                "test_macro_f1": test_metrics.get("test_macro_f1"),
                "output_dir": str(run_dir),
            }
        )

    summary_path = output_root / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        summary_path.write_text("", encoding="utf-8")
        print(f"No completed runs found under {output_root}")
        return

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {summary_path}")


if __name__ == "__main__":
    main()
