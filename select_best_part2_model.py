from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.plantvillage.results import collect_completed_part2_runs

DEFAULT_CANDIDATES = [
    ("resnet50", "from_scratch"),
    ("efficientnet_b3", "linear_probing"),
    ("efficientnet_b3", "full_finetune"),
    ("vit_small", "full_finetune"),
]
ABLATION_DATASETS = ["color", "grayscale", "background-segmented"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best Part 2 base model."
    )
    parser.add_argument("--output-root", default="outputs/part2")
    parser.add_argument("--dataset-version", default="color")
    parser.add_argument("--ranking-name", default="best_model_ranking.csv")
    parser.add_argument("--summary-name", default="best_model_summary.json")
    return parser.parse_args()


def sort_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    test_macro_f1 = row["test_macro_f1"]
    test_top1_accuracy = row["test_top1_accuracy"]
    return (
        -float(row["val_macro_f1"]),
        -float(row["val_top1_accuracy"]),
        float(row["val_loss"]),
        -float(test_macro_f1) if test_macro_f1 is not None else float("inf"),
        -float(test_top1_accuracy) if test_top1_accuracy is not None else float("inf"),
    )


def build_ablation_command(best_row: dict[str, object], dataset_version: str) -> str:
    pretrained_flag = "--pretrained" if best_row["pretrained"] else "--no-pretrained"
    return (
        "python .\\train_part2.py "
        f"--dataset-version {dataset_version} "
        f"--model {best_row['model_name']} "
        f"--strategy {best_row['strategy']} "
        f"--epochs {best_row['epochs']} "
        f"--batch-size {best_row['batch_size']} "
        f"--num-workers {best_row['num_workers']} "
        f"{pretrained_flag}"
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = collect_completed_part2_runs(output_root)

    candidate_rows = [
        row
        for row in rows
        if row["dataset_version"] == args.dataset_version
        and (row["model_name"], row["strategy"]) in DEFAULT_CANDIDATES
    ]

    if not candidate_rows:
        raise SystemExit(
            f"No completed base-model runs found for dataset_version={args.dataset_version!r} under {output_root}"
        )

    ranked_rows = sorted(candidate_rows, key=sort_key)
    best_row = ranked_rows[0]

    ranking_path = output_root / args.ranking_name
    with ranking_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked_rows)

    ablation_commands = {
        dataset_version: build_ablation_command(best_row, dataset_version)
        for dataset_version in ABLATION_DATASETS
    }

    summary_payload = {
        "selection_rule": {
            "primary_metric": "val_macro_f1",
            "tie_breakers": [
                "val_top1_accuracy",
                "val_loss (lower is better)",
                "test_macro_f1",
                "test_top1_accuracy",
            ],
        },
        "base_dataset_version": args.dataset_version,
        "best_model": best_row,
        "ranked_candidates": ranked_rows,
        "ablation_commands": ablation_commands,
        "artifacts": {
            "ranking_csv": str(ranking_path),
            "summary_json": str(output_root / args.summary_name),
        },
    }

    summary_path = output_root / args.summary_name
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Best base model:")
    print(
        f"  {best_row['model_name']} + {best_row['strategy']} "
        f"(val_macro_f1={float(best_row['val_macro_f1']):.4f}, "
        f"val_top1_accuracy={float(best_row['val_top1_accuracy']):.4f})"
    )
    print(f"Wrote ranking CSV to {ranking_path}")
    print(f"Wrote summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
