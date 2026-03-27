# COMP6341_Project

Part 1 for the course project is now implemented as a reusable PlantVillage data and training scaffold.

## What is included

- `src/plantvillage/data.py`
  - Loads PlantVillage with `torchvision.datasets.ImageFolder`
  - Uses a fixed `torch.manual_seed(42)` compatible split strategy
  - Creates reproducible `80/10/10` train/validation/test partitions
  - Applies all requested augmentations: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomResizedCrop`, `ColorJitter`, and `MixUp`
- `src/plantvillage/training.py`
  - Unified training loop
  - Console and file logging
  - Checkpoint saving (`best.pt`, `last.pt`, and epoch checkpoints)
  - Loss history persistence and loss curve plotting
- `train_baseline.py`
  - Example entrypoint other teammates can reuse by swapping the model backbone

## Dataset layout

The code expects the PlantVillage color dataset to be stored like this:

```text
Input/
  color/
    Apple___Apple_scab/
    Apple___Black_rot/
    ...
```

This matches the local repository structure already present in this workspace. In the current copy, `Input/color` contains `38` classes and `54,305` images.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run the baseline

```bash
python3 train_baseline.py \
  --data-dir Input/color \
  --output-dir outputs/resnet18_baseline \
  --model resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4
```

## Run a local smoke test

This mode is meant to prove the pipeline works end to end on a laptop. It samples a small, class-balanced subset, runs only a few steps, and still writes the same reusable outputs.

```bash
python3 train_baseline.py \
  --data-dir Input/color \
  --output-dir outputs/smoke_test \
  --smoke-test
```

Smoke test defaults:

- `epochs=1`
- `samples_per_class=8`
- `batch_size=8`
- `max_train_steps=5`
- `max_val_steps=2`

You can override them if needed:

```bash
python3 train_baseline.py \
  --data-dir Input/color \
  --output-dir outputs/smoke_test_custom \
  --smoke-test \
  --samples-per-class 12 \
  --max-train-steps 8 \
  --max-val-steps 3
```

## Outputs

Each run writes the following into `--output-dir`:

- `train.log`: training log file
- `run_config.json`: run configuration snapshot
- `split_manifest.json`: saved split indices for reproducibility across teammates
- `history.json`: epoch-wise loss and accuracy history
- `loss_curve.png`: training/validation loss curve
- `best.pt`: checkpoint with the lowest validation loss
- `last.pt`: latest checkpoint
- `checkpoint_epoch_<n>.pt`: periodic epoch checkpoints
- `test_metrics.json`: final test set metrics

For smoke-test acceptance, the run is considered successful if it:

- enters epoch 1
- completes a few train and validation batches
- writes `train.log`, `history.json`, `loss_curve.png`, `last.pt`, and `test_metrics.json`

## Reuse guide

- For a new model, keep `create_dataloaders(...)` unchanged and only replace the backbone in `train_baseline.py`.
- Training code supports both standard hard labels and MixUp soft labels.
- The saved `split_manifest.json` lets all teammates train on the exact same partition.
- On Apple Silicon / MPS, `pin_memory` is automatically disabled to avoid the PyTorch warning seen in laptop smoke tests.

## Important note

The assignment text mentions `54,306` images. The dataset currently present under `Input/color` in this repository contains `54,305` files, so results in this workspace will reflect that exact local copy.
