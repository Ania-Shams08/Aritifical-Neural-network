# Cats vs Dogs Classification — ANN vs SVM

This repository implements and compares an **Artificial Neural Network (ANN)** built with TensorFlow/Keras against a **Support Vector Machine (SVM)** baseline for binary image classification (cats vs. dogs), using the same preprocessed dataset for both models.

## 📌 Project Overview

- **Task:** Binary image classification — cat (0) vs dog (1)
- **Dataset:** Cats and Dogs image dataset, split into `train/` and `test/` folders, each containing `cats/` and `dogs/` subfolders
- **Models compared:**
  - Artificial Neural Network (Keras Sequential)
  - Support Vector Machine (RBF kernel, with PCA dimensionality reduction)

## 🗂️ Dataset Structure

```
data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

## ⚙️ Preprocessing

- Images resized to **64×64**
- Converted from BGR (OpenCV default) to **RGB**
- Pixel values normalized to **[0, 1]**
- Labels: `cats = 0`, `dogs = 1`

## 🧠 ANN Architecture

| Layer | Output Shape | Activation |
|---|---|---|
| Flatten (Input) | 12,288 | — |
| Dense (Hidden 1) | 256 | ReLU |
| Dense (Hidden 2) | 128 | ReLU |
| Dense (Output) | 1 | Sigmoid |

**Training configuration:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 20
- Batch size: 32
- Validation split: 20%

## 🔵 SVM Configuration

- Kernel: RBF
- `C = 1.0`, `gamma = 'scale'`
- Dimensionality reduced via **PCA (100 components)** before training, to make training time tractable on raw pixel data

## 📊 Results

| Metric | ANN | SVM |
|---|---|---|
| Accuracy | 63.10% | 65.51% |
| Precision (cats / dogs) | 0.62 / 0.65 | 0.64 / 0.68 |
| Recall (cats / dogs) | 0.66 / 0.61 | 0.69 / 0.62 |
| F1-Score (cats / dogs) | 0.64 / 0.63 | 0.66 / 0.65 |

### Confusion Matrices

**ANN**

|              | Predicted Cat | Predicted Dog |
|--------------|:---:|:---:|
| **Actual Cat** | 1656 | 844 |
| **Actual Dog** | 964 | 1536 |

**SVM**

|              | Predicted Cat | Predicted Dog |
|--------------|:---:|:---:|
| **Actual Cat** | 1731 | 769 |
| **Actual Dog** | 938 | 1562 |

## 📝 Discussion

Both models achieved moderate, comparable performance (~63–66% accuracy). The SVM slightly outperformed the ANN across every metric, though the gap is small. Both models misclassify dogs as cats more often than the reverse, suggesting genuine pixel-level overlap between the two classes in this dataset rather than a flaw specific to either algorithm.

**ANN — strengths:** Learns a non-linear decision boundary directly from raw pixels without manual feature engineering; trains end-to-end via backpropagation; can scale to more complex architectures (e.g., CNNs).

**ANN — limitations:** A plain (non-convolutional) ANN has no built-in notion of spatial structure — it treats each pixel as an independent feature. This is the most likely cause of its limited accuracy; a CNN would typically perform substantially better on this task.

**SVM — strengths:** With PCA-reduced features, found a marginally more effective decision boundary; less prone to overfitting on moderate-sized datasets.

**SVM — limitations:** Training scales poorly with feature/sample count, requiring PCA dimensionality reduction (losing some raw information); lacks hierarchical feature learning.

**Conclusion:** For this dataset and feature representation (raw/PCA pixels), ANN and SVM performed similarly, with SVM marginally ahead. The modest accuracy of both highlights that image classification benefits significantly from spatial feature extraction (as in CNNs), which neither model exploits when given flat pixel vectors.

## 🛠️ Requirements

```
tensorflow
opencv-python
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

## 🚀 How to Run

1. Place the dataset under `data/train/` and `data/test/` as shown above
2. Open the notebook in Jupyter
3. Run all cells sequentially (image loading → ANN training/evaluation → SVM training/evaluation)

## 📁 Repository Contents

| File | Description |
|---|---|
| `ANN_vs_SVM_CatsDogs.ipynb` | Full notebook: preprocessing, ANN, SVM, evaluation, confusion matrices |
| `README.md` | This file |

## 📚 Acknowledgements

Dataset: Cats and Dogs image classification dataset (lab-provided).
