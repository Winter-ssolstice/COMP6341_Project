# COMP6341 Project

PlantVillage image classification project for COMP6341, covering:

- Part 1: baseline training
- Part 2: transfer-learning comparison and modality ablation
- Part 3: GradCAM-based explainability

## Structure

```text
COMP6341_Project/
  Input/
    color/
    grayscale/
    segmented/
  outputs/
    part1/
    part2/
    part3/
  report/
    report.md
    cvpr/
      PaperForReview.tex
      egbib.bib
      cvpr.sty
      ieee_fullname.bst
  src/plantvillage/
    common/
      data.py
      training.py
      models.py
      results.py
    part1/
      train_baseline.py
    part2/
      train_part2.py
      select_best_part2_model.py
      summarize_results.py
    part3/
      explainability.py
      run_part3_gradcam.py
```

## Dataset

Expected local dataset layout:

```text
Input/
  color/
  grayscale/
  segmented/
```

Dataset source: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

## Install

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

## Run

Run commands from the repository root.

### Part 1

```powershell
python -m src.plantvillage.part1.train_baseline --data-dir .\Input\color --output-dir .\outputs\part1\baseline_resnet18 --model resnet18 --epochs 10 --batch-size 32 --num-workers 4
```

### Part 2

Base experiments:

```powershell
python -m src.plantvillage.part2.train_part2 --dataset-version color --model resnet50 --strategy from_scratch --epochs 15 --batch-size 32 --num-workers 4
python -m src.plantvillage.part2.train_part2 --dataset-version color --model efficientnet_b3 --strategy linear_probing --epochs 15 --batch-size 32 --num-workers 4
python -m src.plantvillage.part2.train_part2 --dataset-version color --model efficientnet_b3 --strategy full_finetune --epochs 15 --batch-size 32 --num-workers 4
python -m src.plantvillage.part2.train_part2 --dataset-version color --model vit_small --strategy full_finetune --epochs 20 --batch-size 16 --num-workers 4
```

Model selection:

```powershell
python -m src.plantvillage.part2.select_best_part2_model --output-root .\outputs\part2 --dataset-version color
```

Ablation runs:

```powershell
python -m src.plantvillage.part2.train_part2 --dataset-version grayscale --model vit_small --strategy full_finetune --epochs 20 --batch-size 16 --num-workers 4 --pretrained
python -m src.plantvillage.part2.train_part2 --dataset-version background-segmented --model vit_small --strategy full_finetune --epochs 20 --batch-size 16 --num-workers 4 --pretrained
```

Result aggregation:

```powershell
python -m src.plantvillage.part2.summarize_results --output-root .\outputs\part2
```

### Part 3

```powershell
python -m src.plantvillage.part3.run_part3_gradcam
```

This expects the ViT-Small runs under:

- `outputs/part2/color_vit_small_full_finetune`
- `outputs/part2/grayscale_vit_small_full_finetune`
- `outputs/part2/background_segmented_vit_small_full_finetune`

## Reports

- [report/report.md](/Users/yang/Study/Code/PycharmProjects/COMP6341_Project/report/report.md): readable project report
- [report/cvpr/PaperForReview.tex](/Users/yang/Study/Code/PycharmProjects/COMP6341_Project/report/cvpr/PaperForReview.tex): CVPR-style submission source

Generated artifacts are written to `outputs/`.
