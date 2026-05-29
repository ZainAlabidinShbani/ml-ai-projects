# Chest X-Ray Pneumonia Classification

Professional summary and visual results for the chest X-ray pneumonia classification project.

## Project Overview

This project compares multiple CNN architectures (VGG16, VGG19, EfficientNetB0) for binary classification of chest X-ray images (NORMAL vs PNEUMONIA). The notebook evaluates performance, model complexity, and explainability artifacts to support model selection.

## Figures

All figures below are stored in the `figures/` folder.

### Sample Images

![Sample X-Ray Images](figures/sample_images.png)

### Dataset Class Distribution

![Dataset Class Distribution](figures/class_distribution.png)

### Confusion Matrices (Test Set)

![Confusion Matrices](figures/confusion_matrices.png)

### Performance Metrics Comparison

![Performance Metrics Comparison](figures/metrics_comparison.png)

### ROC Curves

![ROC Curves](figures/roc_curves.png)

### Model Complexity

![Model Complexity](figures/model_complexity.png)

### Training Curves

![Training Curves](figures/training_curves.png)

### Grad-CAM Heatmaps

![Grad-CAM Heatmaps](figures/gradcam.png)

## Notes

- The `notebook/` directory contains the experiment notebook and reproducible analysis.
- The `csv/` directory includes the summary table of results.
- Figures are intended for quick comparison across models.
