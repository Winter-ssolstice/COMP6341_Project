# COMP6341_Project

Reusable PlantVillage project scaffold for:

- Part 1: data preparation and reusable training pipeline
- Part 2: model training, transfer learning comparison, and ablation studies
- Part 3: GradCAM-based explainability analysis for the selected ViT-Small model

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
        best.pt
        checkpoint_epoch_<n>.pt
        history.json
        last.pt
        loss_curve.png
        run_config.json
        split_manifest.json
        test_metrics.json
        train.log
    part2/
      <dataset>_<model>_<strategy>/
        best.pt
        checkpoint_epoch_<n>.pt
        history.json
        last.pt
        loss_curve.png
        run_config.json
        split_manifest.json
        test_metrics.json
        train.log
      best_model_ranking.csv
      best_model_summary.json
      comparison_results.csv
      experiment_runs.csv
    part3/
      vit_small_full_finetune/
        <dataset_version>/
          correct_samples/
          incorrect_samples/
          metrics.json
          predictions.csv
          representative_samples.csv
        comparisons/
      analysis.md
  src/plantvillage/
    data.py
    training.py
    models.py
    results.py
    explainability.py
  train_baseline.py
  train_part2.py
  summarize_results.py
  select_best_part2_model.py
  run_part3_gradcam.py
```

## Dataset layout

The project expects PlantVillage image folders under `Input/`:

```text
Input/
  color/
  grayscale/
  segmented/
```

This matches the local repository structure already present in this workspace. This dataset can be downloaded from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset.

## Install

The `requirements.txt` file is pinned to the official PyTorch CUDA 12.8 wheels for Windows + NVIDIA GPU environments.

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
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

Recommended for stronger Part 1 results on a typical Windows workstation with an NVIDIA GPU: `epochs=20`, `batch_size=32`, `num_workers=4`.
If you hit out-of-memory errors, reduce `batch-size` first.

```powershell
python .\train_baseline.py --data-dir .\Input\color --output-dir .\outputs\part1\baseline_resnet18 --model resnet18 --epochs 10 --batch-size 32 --num-workers 4
```

### Part 1 outputs

```text
outputs/part1/
  baseline_resnet18/
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

### Recommended workflow

Recommended starting points for better Part 2 results on a typical Windows workstation with an NVIDIA GPU:

- ResNet-50 from scratch: `epochs=30`, `batch_size=32`, `num_workers=4`
- EfficientNet-B3 linear probing: `epochs=15`, `batch_size=16`, `num_workers=4`
- EfficientNet-B3 full fine-tuning: `epochs=20`, `batch_size=16`, `num_workers=4`
- ViT-Small full fine-tuning: `epochs=20`, `batch_size=16`, `num_workers=4`
- Ablation runs: reuse the selected best base model and its hyperparameters across `color`, `grayscale`, and `background-segmented` for a fair comparison

If you hit out-of-memory errors, reduce `batch-size` before reducing `epochs`.

### Step 1: Run the four base `color` experiments

```powershell
# ResNet-50 from scratch baseline
python .\train_part2.py --dataset-version color --model resnet50 --strategy from_scratch --epochs 15 --batch-size 32 --num-workers 4

# EfficientNet-B3 linear probing
python .\train_part2.py --dataset-version color --model efficientnet_b3 --strategy linear_probing --epochs 15 --batch-size 32 --num-workers 4

# EfficientNet-B3 full fine-tuning
python .\train_part2.py --dataset-version color --model efficientnet_b3 --strategy full_finetune --epochs 15 --batch-size 32 --num-workers 4

# ViT-Small full fine-tuning
python .\train_part2.py --dataset-version color --model vit_small --strategy full_finetune --epochs 20 --batch-size 16 --num-workers 4
```

### Step 2: Select the best base model for ablation

After the four base `color` runs finish, select the best model by validation Macro F1:

```powershell
python .\select_best_part2_model.py --output-root .\outputs\part2 --dataset-version color
```

This writes:

- `outputs/part2/best_model_ranking.csv`: ranked comparison of the four base models
- `outputs/part2/best_model_summary.json`: the selected best model and the selection rule

### Step 3: Run ablation with the selected best model

Open `outputs/part2/best_model_summary.json`, copy the `ablation_commands` entries, and run them one by one:

```powershell
python .\train_part2.py --dataset-version grayscale --model vit_small --strategy full_finetune --epochs 20 --batch-size 16 --num-workers 4 --pretrained
python .\train_part2.py --dataset-version background-segmented --model vit_small --strategy full_finetune --epochs 20 --batch-size 16 --num-workers 4 --pretrained
```

### Step 4: Aggregate all Part 2 experiment results

After the four base runs and the ablation runs are complete, aggregate everything into one comparison table:

```powershell
python .\summarize_results.py --output-root .\outputs\part2
```

This writes `outputs/part2/comparison_results.csv`.

### Part 2 outputs

```text
outputs/part2/
  color_resnet50_from_scratch/
  color_efficientnet_b3_linear_probing/
  color_efficientnet_b3_full_finetune/
  color_vit_small_full_finetune/
  color_<best_model>_<strategy>/
  grayscale_<best_model>_<strategy>/
  background_segmented_<best_model>_<strategy>/
  best_model_ranking.csv
  best_model_summary.json
  comparison_results.csv
```

Each Part 2 run writes the same per-run artifacts as Part 1 plus:

- `best_model_ranking.csv`: ranked comparison of the four base `color` experiments
- `best_model_summary.json`: selected best base model plus ablation commands and selection metadata
- `comparison_results.csv`: final combined comparison table generated after all base-model and ablation runs are complete

## Part 3

### Scope

- `run_part3_gradcam.py`
  - fixed Part 3 entry point for `vit_small + full_finetune`
- `src/plantvillage/explainability.py`
  - reloads the three Part 2 ViT runs
  - re-runs test set inference
  - aligns `grayscale` and `background_segmented` to the `color` test split for fair cross-version comparison
  - generates GradCAM and GradCAM++ representative figures
  - writes prediction tables and Markdown analysis

### Assumptions

Part 3 is fixed to the selected best model:

- `color_vit_small_full_finetune`
- `grayscale_vit_small_full_finetune`
- `background_segmented_vit_small_full_finetune`

Each run directory is expected to contain:

- `best.pt`
- `run_config.json`
- `split_manifest.json`

### Run Part 3

Use the default command to analyze the three ViT-Small runs and write all outputs under `outputs/part3/vit_small_full_finetune/`.

```powershell
python .\run_part3_gradcam.py
```

If you want a smaller-memory inference setup, you can override both batch size and workers explicitly:

```powershell
python .\run_part3_gradcam.py --batch-size 8 --num-workers 0
```

### Part 3 outputs

```text
outputs/part3/
  vit_small_full_finetune/
    analysis.md
    comparisons/
      correct_<class>.png
      incorrect_<class>.png
    color/
      metrics.json
      predictions.csv
      representative_samples.csv
      correct_samples/
      incorrect_samples/
    grayscale/
      metrics.json
      predictions.csv
      representative_samples.csv
      correct_samples/
      incorrect_samples/
    background_segmented/
      metrics.json
      predictions.csv
      representative_samples.csv
      correct_samples/
      incorrect_samples/
```

Part 3 writes:

- `analysis.md`: report-ready qualitative summary with metrics, hardest classes, confusion patterns, and background-segmentation comparison notes
- `predictions.csv`: per-image test predictions for each dataset version
- `representative_samples.csv`: one correct and one incorrect representative sample per class when available
- `correct_samples/*.png` and `incorrect_samples/*.png`: original image plus GradCAM and GradCAM++ overlays
- `comparisons/*.png`: matched `color` vs `background_segmented` visual comparisons for representative samples

## Notes

- Training code supports both standard hard labels and MixUp soft labels.
- Training history and test metrics record both `Top-1 Accuracy` and `Macro F1`.
- The saved `split_manifest.json` lets all teammates train on the exact same partition.
- On Apple Silicon / MPS, `pin_memory` is automatically disabled to avoid PyTorch laptop warnings.
- The assignment text mentions `54,306` images; the local `Input/color` copy in this workspace contains `54,305` files.
