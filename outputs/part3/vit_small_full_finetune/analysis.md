# Part 3 GradCAM Analysis

Best model fixed for Part 3: `vit_small + full_finetune`.

## Test Metrics Overview

| Dataset | Test Loss | Top-1 Accuracy | Macro F1 | Correct Samples | Incorrect Samples |
| --- | --- | --- | --- | --- | --- |
| color | 0.0391 | 0.9934 | 0.9909 | 38 | 17 |
| grayscale | 0.1461 | 0.9586 | 0.9446 | 38 | 25 |
| background_segmented | 0.0301 | 0.9925 | 0.9909 | 37 | 14 |

## Representative Sample Coverage

### color

- Predictions: `color/predictions.csv`
- Metrics: `color/metrics.json`
- Representative samples: `color/representative_samples.csv`
- Test samples aligned to color split: 5431/5431
- Generated figures: 55
- Missing incorrect sample(s): Apple / Apple scab, Apple / Black rot, Apple / Cedar apple rust, Apple / healthy, Blueberry / healthy, Cherry (including sour) / Powdery mildew, Cherry (including sour) / healthy, Corn (maize) / Common rust , Corn (maize) / healthy, Grape / Esca (Black Measles), Grape / Leaf blight (Isariopsis Leaf Spot), Grape / healthy, Peach / healthy, Pepper, bell / healthy, Potato / Late blight, Raspberry / healthy, Squash / Powdery mildew, Strawberry / Leaf scorch, Strawberry / healthy, Tomato / Tomato mosaic virus, Tomato / healthy

#### Hardest Classes in color

| Rank | Class | Support | Recall |
| --- | --- | --- | --- |
| 1 | Potato / healthy | 14 | 0.7857 |
| 2 | Tomato / Bacterial spot | 213 | 0.9765 |
| 3 | Peach / Bacterial spot | 264 | 0.9773 |
| 4 | Corn (maize) / Cercospora leaf spot Gray leaf spot | 45 | 0.9778 |
| 5 | Corn (maize) / Northern Leaf Blight | 94 | 0.9787 |

#### Top Confusions in color

| Rank | True Label | Predicted Label | Count |
| --- | --- | --- | --- |
| 1 | Peach / Bacterial spot | Tomato / Septoria leaf spot | 5 |
| 2 | Corn (maize) / Northern Leaf Blight | Corn (maize) / Cercospora leaf spot Gray leaf spot | 2 |
| 3 | Orange / Haunglongbing (Citrus greening) | Pepper, bell / healthy | 2 |
| 4 | Pepper, bell / Bacterial spot | Pepper, bell / healthy | 2 |
| 5 | Potato / healthy | Soybean / healthy | 2 |

### grayscale

- Predictions: `grayscale/predictions.csv`
- Metrics: `grayscale/metrics.json`
- Representative samples: `grayscale/representative_samples.csv`
- Test samples aligned to color split: 5431/5431
- Generated figures: 63
- Missing incorrect sample(s): Apple / Black rot, Blueberry / healthy, Cherry (including sour) / healthy, Corn (maize) / Common rust , Corn (maize) / healthy, Grape / Leaf blight (Isariopsis Leaf Spot), Grape / healthy, Orange / Haunglongbing (Citrus greening), Raspberry / healthy, Soybean / healthy, Squash / Powdery mildew, Strawberry / healthy, Tomato / healthy

#### Hardest Classes in grayscale

| Rank | Class | Support | Recall |
| --- | --- | --- | --- |
| 1 | Potato / healthy | 14 | 0.5714 |
| 2 | Tomato / Early blight | 114 | 0.6754 |
| 3 | Tomato / Spider mites Two-spotted spider mite | 149 | 0.7785 |
| 4 | Potato / Late blight | 95 | 0.8000 |
| 5 | Tomato / Target Spot | 131 | 0.8550 |

#### Top Confusions in grayscale

| Rank | True Label | Predicted Label | Count |
| --- | --- | --- | --- |
| 1 | Tomato / Spider mites Two-spotted spider mite | Tomato / healthy | 21 |
| 2 | Tomato / Early blight | Tomato / Septoria leaf spot | 15 |
| 3 | Grape / Black rot | Grape / Esca (Black Measles) | 14 |
| 4 | Tomato / Target Spot | Tomato / healthy | 14 |
| 5 | Tomato / Spider mites Two-spotted spider mite | Tomato / Target Spot | 10 |

### background_segmented

- Predictions: `background_segmented/predictions.csv`
- Metrics: `background_segmented/metrics.json`
- Representative samples: `background_segmented/representative_samples.csv`
- Test samples aligned to color split: 5318/5431
- Generated figures: 51
- Missing aligned samples in this dataset version: 113
- Missing incorrect sample(s): Apple / Apple scab, Apple / Black rot, Apple / Cedar apple rust, Apple / healthy, Blueberry / healthy, Cherry (including sour) / healthy, Corn (maize) / Common rust , Corn (maize) / healthy, Grape / Black rot, Grape / Esca (Black Measles), Grape / Leaf blight (Isariopsis Leaf Spot), Grape / healthy, Orange / Haunglongbing (Citrus greening), Peach / Bacterial spot, Pepper, bell / Bacterial spot, Potato / Early blight, Potato / Late blight, Potato / healthy, Raspberry / healthy, Soybean / healthy, Squash / Powdery mildew, Strawberry / Leaf scorch, Strawberry / healthy, Tomato / healthy

#### Hardest Classes in background_segmented

| Rank | Class | Support | Recall |
| --- | --- | --- | --- |
| 1 | Tomato / Early blight | 114 | 0.9123 |
| 2 | Peach / healthy | 41 | 0.9268 |
| 3 | Corn (maize) / Cercospora leaf spot Gray leaf spot | 45 | 0.9556 |
| 4 | Corn (maize) / Northern Leaf Blight | 94 | 0.9574 |
| 5 | Tomato / Tomato mosaic virus | 31 | 0.9677 |

#### Top Confusions in background_segmented

| Rank | True Label | Predicted Label | Count |
| --- | --- | --- | --- |
| 1 | Tomato / Early blight | Tomato / Bacterial spot | 7 |
| 2 | Corn (maize) / Northern Leaf Blight | Corn (maize) / Cercospora leaf spot Gray leaf spot | 3 |
| 3 | Tomato / Spider mites Two-spotted spider mite | Tomato / Target Spot | 3 |
| 4 | Tomato / Target Spot | Tomato / Bacterial spot | 3 |
| 5 | Peach / healthy | Peach / Bacterial spot | 2 |

## Background Segmentation Comparison

- Comparison figures created: 54
- Color representative samples without segmented match: 1
- Accuracy delta (`background_segmented - color`): -0.0009
- Macro F1 delta (`background_segmented - color`): -0.0000
- Background segmentation reduced overall classification performance on this split, so any localization gain should be weighed against the accuracy drop.

Inspect the comparison figures under `comparisons/` to judge whether attention becomes more concentrated on lesion regions after background removal.

## Failure Mode Summary

### color

- Hardest classes: Potato / healthy (recall=0.786), Tomato / Bacterial spot (recall=0.977), Peach / Bacterial spot (recall=0.977)
- Most common misclassifications: Peach / Bacterial spot -> Tomato / Septoria leaf spot (5), Corn (maize) / Northern Leaf Blight -> Corn (maize) / Cercospora leaf spot Gray leaf spot (2), Orange / Haunglongbing (Citrus greening) -> Pepper, bell / healthy (2)
- Most error-prone true class: Peach / Bacterial spot (6 test errors).

### grayscale

- Hardest classes: Potato / healthy (recall=0.571), Tomato / Early blight (recall=0.675), Tomato / Spider mites Two-spotted spider mite (recall=0.779)
- Most common misclassifications: Tomato / Spider mites Two-spotted spider mite -> Tomato / healthy (21), Tomato / Early blight -> Tomato / Septoria leaf spot (15), Grape / Black rot -> Grape / Esca (Black Measles) (14)
- Most error-prone true class: Tomato / Early blight (37 test errors).

### background_segmented

- Hardest classes: Tomato / Early blight (recall=0.912), Peach / healthy (recall=0.927), Corn (maize) / Cercospora leaf spot Gray leaf spot (recall=0.956)
- Most common misclassifications: Tomato / Early blight -> Tomato / Bacterial spot (7), Corn (maize) / Northern Leaf Blight -> Corn (maize) / Cercospora leaf spot Gray leaf spot (3), Tomato / Spider mites Two-spotted spider mite -> Tomato / Target Spot (3)
- Most error-prone true class: Tomato / Early blight (10 test errors).
