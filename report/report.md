# Plant Disease Recognition via Transfer Learning and GradCAM Explainability on PlantVillage

**COMP6341 Project Report**

---

## Abstract

Accurate automated detection of plant diseases is critical for precision agriculture. We present a systematic study on the PlantVillage dataset (54,305 images, 38 classes across 14 plant species) comparing training strategies, model architectures, and input modalities for multi-class leaf disease classification. Starting from a ResNet-18 baseline (97.53% accuracy), we show that full fine-tuning of pretrained models yields the strongest overall results, with **EfficientNet-B3 reaching 99.83% Top-1 accuracy and 99.75% Macro F1** on color images. A ViT-Small pipeline is then used for the modality ablation and GradCAM analysis, where experiments over three dataset versions (color, grayscale, background-segmented) reveal that color is an essential diagnostic feature, while background content has negligible influence on classification. GradCAM and GradCAM++ visualizations confirm that the model focuses on semantically meaningful lesion regions, with identified failure modes concentrated in visually similar disease pairs.

---

## 1. Introduction

Plant pathogens annually destroy an estimated 20–40% of global crop yields, making early and accurate disease diagnosis a pressing challenge. While human experts can identify diseases from leaf imagery, scaling such inspection is infeasible. Convolutional and transformer-based neural networks trained on curated datasets offer a path to scalable, automated diagnosis.

The PlantVillage dataset [Hughes & Salathé, 2015] provides a controlled benchmark with standardized leaf photographs covering healthy and diseased conditions across 14 crop species. Despite high inter-class variety, many disease categories share subtle visual cues (e.g., *Tomato Early Blight* vs. *Tomato Bacterial Spot*), making high-confidence multi-class classification non-trivial.

This work addresses three research questions: (1) How does training strategy (from-scratch vs. linear probing vs. full fine-tuning) affect performance? (2) Does removing background information via segmentation improve accuracy? (3) Which image regions drive model decisions, and where do failures occur?

---

## 2. Dataset and Preprocessing

**PlantVillage** contains 54,305 images organized into 38 disease/health classes. Three input modalities are provided: *color* (RGB), *grayscale*, and *background-segmented* (green background removed). All images are resized to 224×224. The dataset is split deterministically with a fixed seed (80% train / 10% val / 10% test), yielding 43,444 / 5,430 / 5,431 samples for the original color version. The grayscale version reuses the same image identities, while the background-segmented evaluation is reported on the matched subset where segmented counterparts are available.

**Data augmentation** (training only): `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomResizedCrop(224)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`, and MixUp (α = 0.4).

---

## 3. Methodology

### 3.1 Models and Training Strategies

We evaluate four model–strategy combinations:

| Model | Strategy | Pretrained | Notes |
|---|---|---|---|
| ResNet-50 | From Scratch | No | No-transfer control |
| EfficientNet-B3 | Linear Probing | Yes (ImageNet) | Frozen backbone, only head trained |
| EfficientNet-B3 | Full Fine-tune | Yes (ImageNet) | All layers updated |
| ViT-Small | Full Fine-tune | Yes (ImageNet) | Patch size 16, full unfreeze |

All models use **AdamW** (weight decay = 1×10⁻⁴) and cross-entropy loss. The baseline uses lr = 1×10⁻³; all other experiments use lr = 3×10⁻⁴. Checkpoints are saved at minimum validation loss; the best checkpoint is used for test evaluation.

### 3.2 GradCAM Explainability

The ViT-Small full fine-tuning model is selected as the analysis target. **GradCAM** [Selvaraju et al., 2017] and **GradCAM++** [Chattopadhay et al., 2018] are applied to the final attention block. For each of the 38 classes, one representative *correct* and one *incorrect* prediction are identified from the test set. The three dataset versions are aligned to the color reference split where corresponding images are available, enabling a near-matched cross-modality comparison.

---

## 4. Experiments

### 4.1 ResNet-18 Baseline

A standard ResNet-18 model is trained for 10 epochs (batch size 32) as the project baseline.

<div align="center">

![Part 1 Loss Curve](../outputs/part1/baseline_resnet18/loss_curve.png)

*Figure 1: Training and validation loss for the ResNet-18 baseline (10 epochs). Validation loss converges to ~0.22, consistent with the 97.53% test accuracy.*

</div>

**Result:** Test Accuracy = **97.53%**, Test Macro F1 = **96.34%**. The baseline establishes a strong starting point but leaves room for improvement on hard classes.

### 4.2 Model Comparison

Table 1 reports validation-set model selection on the color dataset. The strongest results come from full fine-tuning of pretrained models, while linear probing remains far weaker and the from-scratch ResNet-50 control, although competitive, ranks below the best pretrained runs. Note that MixUp augmentation is applied uniformly to all training runs; while beneficial for models with learnable feature extractors, it may interact unfavorably with the frozen backbone in the linear probing setting, contributing to the elevated training loss observed for that configuration.

**Table 1: Model comparison on the color dataset (validation set).**

| Model | Strategy | Val Acc | Val Macro F1 |
|---|---|---|---|
| EfficientNet-B3 | Full Fine-tune | **99.72%** | **99.54%** |
| ViT-Small | Full Fine-tune | 99.30% | 98.89% |
| ResNet-50 | From Scratch | 98.08% | 97.30% |
| EfficientNet-B3 | Linear Probing | 67.31% | 58.01% |

EfficientNet-B3 (full fine-tune) ranks first on validation Macro F1 (99.54%), with ViT-Small second at 98.89% and ResNet-50 from scratch close behind at 97.30%. The main contrast is therefore between full fine-tuning and linear probing: pretrained backbones adapted end-to-end are clearly strongest, while freezing the backbone leaves substantial performance on the table.

Table 2 reports held-out test metrics across all evaluated configurations. EfficientNet-B3 is the top-performing classifier overall, but ViT-Small is retained for the ablation study and explainability analysis to keep the modality comparison and GradCAM workflow on a single explainability-ready pipeline.

**Table 2: Held-out test results for runs with exported test metrics.**

| Model | Strategy | Test Acc | Test Macro F1 |
|---|---|---|---|
| EfficientNet-B3 | Full Fine-tune | **99.83%** | **99.75%** |
| ViT-Small | Full Fine-tune | 99.34% | 99.09% |
| ResNet-50 | From Scratch | 97.72% | 97.18% |
| ResNet-18 | Baseline | 97.53% | 96.34% |
| EfficientNet-B3 | Linear Probing | 67.41% | 58.50% |

<div align="center">

| ResNet-18 Baseline | EfficientNet-B3 Full Fine-tune |
|---|---|
| ![ResNet-18 Loss Curve](../outputs/part1/baseline_resnet18/loss_curve.png) | ![EfficientNet-B3 Loss Curve](../outputs/part2/color_efficientnet_b3_full_finetune/loss_curve.png) |
| *Figure 2(a): ResNet-18 (10 epochs).* | *Figure 2(b): EfficientNet-B3 (15 epochs).* |

*Figure 2: Training and validation loss curves for the baseline and the best overall model. EfficientNet-B3 converges faster and reaches a lower validation loss than the ResNet-18 baseline.*

</div>

**Key observations:** (1) Linear probing yields only 58% Macro F1, confirming that frozen ImageNet features alone are insufficient for fine-grained disease classification. (2) Full fine-tuning still gives the best overall results, with EfficientNet-B3 outperforming both ViT-Small and the from-scratch ResNet-50 control on the color dataset. (3) The from-scratch ResNet-50 run remains competitive at 97.18% test Macro F1, so the clearest takeaway is not that training from scratch fails outright, but that pretrained end-to-end adaptation is the strongest strategy among the tested configurations.

### 4.3 Input-Modality Ablation

Using ViT-Small (full fine-tune) as the fixed architecture, we compare performance across the three PlantVillage modalities. The color and grayscale rows use the original held-out test split, while the background-segmented row is reported on the matched subset where segmented counterparts are available.

**Table 3: Ablation study — ViT-Small across dataset versions.**

| Dataset Version | Test Accuracy | Test Macro F1 | ΔAcc vs. Color |
|---|---|---|---|
| Color | **99.34%** | **99.09%** | — |
| Background Segmented | 99.25% | 99.09% | −0.09% |
| Grayscale | 95.86% | 94.46% | −3.48% |

Removing color information (grayscale) causes a **−3.48% accuracy drop**, the largest degradation observed. This confirms that chromatic texture is a primary diagnostic cue for diseases such as rust, mold, and yellowing. Background segmentation produces only a **−0.09% drop** on the matched subset used for comparison, suggesting the model is inherently robust to background clutter, but this number should be interpreted as a near-matched comparison rather than a perfectly identical test-set evaluation.

### 4.4 Explainability and Failure Analysis

**Hardest classes.** Table 4 lists the hardest class (lowest per-class recall) for each input modality. The *Potato / Healthy* class is consistently the most difficult, likely due to low test-set support (14 samples) and visual similarity to *Soybean / Healthy*. The most frequent confusion pairs are: *Peach / Bacterial Spot* → *Tomato / Septoria Leaf Spot* (5 errors, color); *Tomato / Spider Mites* → *Tomato / Healthy* (21 errors, grayscale); and *Tomato / Early Blight* → *Tomato / Bacterial Spot* (7 errors, background-segmented).

**Correct prediction.** Figure 3 shows a representative correct classification of *Tomato Early Blight* (confidence = 0.994). GradCAM highlights the necrotic lesion spots distributed across the leaf surface, indicating the model has learned disease-relevant texture patterns rather than spurious correlations.

<div align="center">

![GradCAM Correct](../outputs/part3/vit_small_full_finetune/color/correct_samples/correct_tomato___early_blight.png)

*Figure 3: GradCAM visualization for a correctly classified Tomato Early Blight sample (color, conf = 0.994). Left: original image; center: GradCAM overlay; right: GradCAM++ overlay. Activation concentrates on lesion regions.*

</div>

**Failure mode.** Figure 4 illustrates a misclassification of *Corn Cercospora Leaf Spot* as *Northern Leaf Blight* (confidence = 0.754). The two diseases produce visually similar elongated lesions, and GradCAM reveals diffuse attention across the entire blade rather than localized discriminative regions — a pattern correlated with model uncertainty.

<div align="center">

![GradCAM Incorrect](../outputs/part3/vit_small_full_finetune/color/incorrect_samples/incorrect_corn__maize____cercospora_leaf_spot_gray_leaf_spot.png)

*Figure 4: GradCAM for a misclassified Corn Cercospora Leaf Spot sample (color, conf = 0.754, predicted: Northern Leaf Blight). Attention is diffuse, indicating the model is uncertain about the discriminative region.*

</div>

**Table 4: Hardest classes per dataset version (lowest per-class recall).**

| Dataset | Class | Support | Recall |
|---|---|---|---|
| Color | Potato / Healthy | 14 | 78.6% |
| Background Segmented | Tomato / Early Blight | 114 | 91.2% |
| Grayscale | Potato / Healthy | 14 | 57.1% |

---

## 5. Conclusion

We systematically studied plant disease classification on PlantVillage under multiple training strategies, architectures, and input modalities. Full fine-tuning of pretrained models provides the best overall performance, with **EfficientNet-B3 achieving 99.83% Top-1 accuracy and 99.75% Macro F1** on the color test set. The ViT-Small pipeline remains strong and is used for the modality ablation and GradCAM analysis. These experiments establish that **color is a critical diagnostic modality** (−3.5% accuracy when removed), while background removal yields minimal benefit. GradCAM analysis validates that the model attends to lesion-bearing regions on correct predictions; failure modes are concentrated in low-support classes and visually ambiguous disease pairs. Future work should explore class-balanced sampling for tail classes and contrastive pre-training on domain-specific leaf imagery.

---

## References

- Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV.
- Chattopadhay, A., et al. (2018). *Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks.* WACV.
- Dosovitskiy, A., et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR.
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML.
