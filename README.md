# COMP6341_Project

Reusable PlantVillage project scaffold for:

- Part 1: data preparation and reusable training pipeline
- Part 2: model training, transfer learning comparison, and ablation studies

## Project layout

```text
COMP6341_Project/
  Input/
    color/
    grayscale/
    segmented/
  outputs/
    part1/
      baseline_resnet18/
      smoke_test/
    part2/
      <dataset>_<model>_<strategy>/
      experiment_runs.csv
      comparison_results.csv
  src/plantvillage/
    data.py
    training.py
    models.py
    results.py
  train_baseline.py
  train_part2.py
  summarize_results.py
```

## Dataset layout

The project expects PlantVillage image folders under `Input/`:

```text
Input/
  color/
  grayscale/
  segmented/
```

This matches the local repository structure already present in this workspace.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Part 1

### Scope

- `src/plantvillage/data.py`
  - `ImageFolder` loading
  - fixed-seed `80/10/10` split
  - `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomResizedCrop`, `ColorJitter`, `MixUp`
- `src/plantvillage/training.py`
  - unified training loop
  - logging, checkpoints, loss curve
- `train_baseline.py`
  - Part 1 baseline runner

### Run the baseline

```bash
python3 train_baseline.py \
  --data-dir Input/color \
  --output-dir outputs/part1/baseline_resnet18 \
  --model resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4
```

### Run the local smoke test

```bash
python3 train_baseline.py \
  --data-dir Input/color \
  --output-dir outputs/part1/smoke_test \
  --smoke-test
```

### Smoke test defaults

- `epochs=1`
- `samples_per_class=8`
- `batch_size=8`
- `max_train_steps=5`
- `max_val_steps=2`

### Part 1 outputs

```bash
outputs/part1/
  baseline_resnet18/
  smoke_test/
```

Each Part 1 run writes:

- `train.log`: training log file
- `run_config.json`: run configuration snapshot
- `split_manifest.json`: saved split indices for reproducibility across teammates
- `history.json`: epoch-wise loss, Top-1 accuracy, and Macro F1 history
- `loss_curve.png`: training/validation loss curve
- `best.pt`: checkpoint with the lowest validation loss
- `last.pt`: latest checkpoint
- `checkpoint_epoch_<n>.pt`: periodic epoch checkpoints
- `test_metrics.json`: final test set metrics

## Part 2

### Scope

- `train_part2.py`
  - unified runner for Part 2 experiments
- `src/plantvillage/models.py`
  - model and strategy factory
- `summarize_results.py`
  - experiment table aggregation

### Supported experiments

- `resnet50` + `from_scratch`
- `efficientnet_b3` + `linear_probing`
- `efficientnet_b3` + `full_finetune`
- `vit_small` + `full_finetune`
- dataset versions: `color`, `grayscale`, `background-segmented`

### Example commands

```bash
# ResNet-50 from scratch baseline
python3 train_part2.py \
  --dataset-version color \
  --model resnet50 \
  --strategy from_scratch

# EfficientNet-B3 linear probing
python3 train_part2.py \
  --dataset-version color \
  --model efficientnet_b3 \
  --strategy linear_probing

# EfficientNet-B3 full fine-tuning
python3 train_part2.py \
  --dataset-version color \
  --model efficientnet_b3 \
  --strategy full_finetune

# ViT-Small full fine-tuning
python3 train_part2.py \
  --dataset-version color \
  --model vit_small \
  --strategy full_finetune

# Ablation on grayscale
python3 train_part2.py \
  --dataset-version grayscale \
  --model vit_small \
  --strategy full_finetune

# Ablation on background-segmented images
python3 train_part2.py \
  --dataset-version background-segmented \
  --model vit_small \
  --strategy full_finetune
```

### Part 2 smoke test

```bash
python3 train_part2.py \
  --dataset-version color \
  --model resnet50 \
  --strategy from_scratch \
  --smoke-test \
  --num-workers 0
```

### Part 2 outputs

```bash
outputs/part2/
  color_resnet50_from_scratch/
  color_efficientnet_b3_linear_probing/
  color_efficientnet_b3_full_finetune/
  color_vit_small_full_finetune/
  grayscale_vit_small_full_finetune/
  background_segmented_vit_small_full_finetune/
  experiment_runs.csv
  comparison_results.csv
```

Each Part 2 run writes the same per-run artifacts as Part 1 plus:

- `experiment_runs.csv`: row-wise run registry appended after each training run
- `comparison_results.csv`: clean comparison table generated after aggregation

### Aggregate experiment results

```bash
python3 summarize_results.py --output-root outputs/part2
```

This writes `outputs/part2/comparison_results.csv`.

## Notes

- Training code supports both standard hard labels and MixUp soft labels.
- Training history and test metrics record both `Top-1 Accuracy` and `Macro F1`.
- The saved `split_manifest.json` lets all teammates train on the exact same partition.
- On Apple Silicon / MPS, `pin_memory` is automatically disabled to avoid PyTorch laptop warnings.
- The assignment text mentions `54,306` images; the local `Input/color` copy in this workspace contains `54,305` files.
