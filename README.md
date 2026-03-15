# CellSight 🔬
### Explainable Malaria Parasite Localization and Classification

Automated malaria detection from blood smear cell images using deep learning and gradient-based visual explanation methods.

---

## Overview

CellSight evaluates three CNN architectures on the NIH Malaria Cell Image dataset and explains their predictions using Grad-CAM and a Modified Grad-CAM++ with true second-order derivatives.

| Model | Accuracy | Input Size |
|---|---|---|
| ResNet-50 | 90%+ | 128×128 |
| VGG-19 | 90%+ | 224×224 |
| MobileNetV2 | 90%+ | 224×224 |

---

## Dataset

- **NIH Malaria Cell Images** — 27,558 cell images
- **Classes:** Parasitized, Uninfected
- **Source:** [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

---

## Methods

**Grad-CAM** — Computes channel importance weights by globally averaging gradients of the class score with respect to the final convolutional layer activations.

**Modified Grad-CAM++** — Uses true second-order derivatives via nested TensorFlow GradientTapes instead of the algebraic approximation used in standard Grad-CAM++, producing tighter and more precise saliency maps.

---

## Project Structure

```
CellSight/
├── training/
│   ├── training_resnet50.ipynb
│   ├── training_vgg19.ipynb
│   └── training_mobilenetv2.ipynb
├── implementation/
│   └── full_implementation.ipynb
├── app/
│   └── app_malaria.py
└── README.md
```

---

## Setup

```bash
pip install tensorflow streamlit opencv-python numpy matplotlib seaborn
```

---

## Training

Open the relevant notebook in `training/` and update the dataset path:

```python
base_path = "path/to/cell_images"
```

Run all cells. The best model is saved automatically via `ModelCheckpoint`.

---

## Running the App

Place your trained model files in the same directory as `app_malaria.py`, then:

```bash
streamlit run app_malaria.py
```

Upload a blood smear cell image to get:
- Predicted class (Parasitized / Uninfected) with confidence
- Grad-CAM heatmap overlay
- Modified Grad-CAM++ heatmap overlay
- Difference map (ILCAN − Grad-CAM)

---

## Results

All three models achieve above 98% classification accuracy. VGG-19 produces the most clinically meaningful heatmaps, consistently localizing the Plasmodium parasite regions within infected cells.

---

## Requirements

```
tensorflow == 2.19
streamlit==1.45.1
opencv_python==4.12.0.88
numpy==2.4.3
matplotlib==3.10.8
```
