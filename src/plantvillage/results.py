from __future__ import annotations

import csv
import json
from pathlib import Path


def write_experiment_summary(summary_path: str | Path, row: dict[str, object]) -> None:
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not summary_path.exists()

    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_completed_part2_runs(output_root: str | Path) -> list[dict[str, object]]:
    output_root = Path(output_root)
    rows: list[dict[str, object]] = []

    if not output_root.exists():
        return rows

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_config_path = run_dir / "run_config.json"
        test_metrics_path = run_dir / "test_metrics.json"
        history_path = run_dir / "history.json"
        if not (run_config_path.exists() and history_path.exists()):
            continue

        run_config = load_json(run_config_path)
        test_metrics = load_json(test_metrics_path) if test_metrics_path.exists() else {}
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            continue

        training_config = run_config.get("training", {})
        data_config = run_config.get("data", {})
        last_epoch = history[-1]
        rows.append(
            {
                "dataset_version": run_config.get("dataset_version"),
                "model_name": run_config.get("model_name"),
                "strategy": run_config.get("strategy"),
                "pretrained": run_config.get("pretrained"),
                "epochs": training_config.get("epochs"),
                "epochs_recorded": len(history),
                "batch_size": data_config.get("batch_size"),
                "num_workers": data_config.get("num_workers"),
                "val_loss": last_epoch.get("val_loss"),
                "val_top1_accuracy": last_epoch.get("val_accuracy"),
                "val_macro_f1": last_epoch.get("val_macro_f1"),
                "test_loss": test_metrics.get("test_loss"),
                "test_top1_accuracy": test_metrics.get("test_accuracy"),
                "test_macro_f1": test_metrics.get("test_macro_f1"),
                "output_dir": str(run_dir),
            }
        )

    return rows
