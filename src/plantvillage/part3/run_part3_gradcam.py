from __future__ import annotations

import argparse
from pathlib import Path

from .explainability import (
    analyze_run,
    generate_comparison_figures,
    load_run_context,
    write_analysis_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 GradCAM analysis for the selected ViT-Small model.")
    parser.add_argument("--color-run", default="outputs/part2/color_vit_small_full_finetune")
    parser.add_argument("--grayscale-run", default="outputs/part2/grayscale_vit_small_full_finetune")
    parser.add_argument("--segmented-run", default="outputs/part2/background_segmented_vit_small_full_finetune")
    parser.add_argument("--output-root", default="outputs/part3/vit_small_full_finetune")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for Part 3 inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 by default for Windows DataLoader stability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    color_context = load_run_context(args.color_run, batch_size=args.batch_size, num_workers=args.num_workers)
    reference_keys = [sample.canonical_key for sample in color_context.test_dataset.samples]
    grayscale_context = load_run_context(
        args.grayscale_run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        reference_keys=reference_keys,
    )
    segmented_context = load_run_context(
        args.segmented_run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        reference_keys=reference_keys,
    )

    color_analysis = analyze_run(color_context, output_root=output_root)
    grayscale_analysis = analyze_run(grayscale_context, output_root=output_root)
    segmented_analysis = analyze_run(segmented_context, output_root=output_root)

    comparison_summary = generate_comparison_figures(
        color_context=color_context,
        color_analysis=color_analysis,
        segmented_context=segmented_context,
        segmented_analysis=segmented_analysis,
        output_dir=output_root / "comparisons",
    )

    analyses = [color_analysis, grayscale_analysis, segmented_analysis]
    contexts = [color_context, grayscale_context, segmented_context]
    write_analysis_markdown(output_root, analyses, contexts, comparison_summary)

    print(f"Wrote Part 3 analysis to {output_root}")
    print(f"Generated analysis markdown at {output_root / 'analysis.md'}")
    print(f"Aligned grayscale test samples to color split: {grayscale_context.matched_sample_count}/{grayscale_context.reference_key_count}")
    print(f"Aligned background-segmented test samples to color split: {segmented_context.matched_sample_count}/{segmented_context.reference_key_count}")
    print(f"Generated color vs background-segmented comparisons: {comparison_summary['created']}")


if __name__ == "__main__":
    main()
