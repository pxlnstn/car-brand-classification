# Car Brand Classification with CNN

A computer vision project to classify **33 car brands** from images using Convolutional Neural Networks (CNNs) and Transfer Learning. The work is delivered as a **Kaggle notebook** (code + outputs) and a **GitHub repository** (documentation and structure).

> **Goal**: Build a robust multiclass classifier, document each step clearly, evaluate with standard metrics, and explain predictions via Grad‑CAM.

---

## 1) Dataset

* **Source**: *Car Brand Classification* (Kaggle), ~16k images across **33 classes** (Audi, BMW, Ford, Mercedes, Tesla, …).
* **Task**: Multiclass image classification.
* **Split**: Train / Validation / Test ≈ **80% / 10% / 10%**.
* **Augmentation**: rotation, horizontal flip, zoom, brightness/color jitter, shear.

> Note: The dataset is consumed directly on Kaggle. **Raw images are not committed** to GitHub.

---

## 2) Project Structure

```
.
├─ car-brand-classification.ipynb
├─ utils/
│  └─ gradcam.py                    # Reusable Grad‑CAM helper
├─ models/                          # Saved weights/checkpoints (git‑ignored)
├─ README.md
├─ LICENSE
└─ .gitignore
```

**.gitignore (excerpt)**

```
# notebooks & data
*.ipynb_checkpoints/
datasets/
models/

# python
__pycache__/
*.pyc

# OS/editor
.DS_Store
```

---

## 3) Methods

* **Baseline CNN**: 2–4 Conv2D blocks → GlobalAveragePooling/Flatten → Dropout → Dense(Softmax).
* **Transfer Learning**: EfficientNet or ResNet with ImageNet weights. Freeze → warm‑up → unfreeze top blocks for fine‑tuning.
* **Regularization**: Data augmentation, Dropout, **L2 weight decay**, EarlyStopping, ReduceLROnPlateau.
* **Optimization**: Adam (optional cosine/one‑cycle schedulers). Mixed precision if available.

> **Framework**: TensorFlow/Keras (GPU enabled on Kaggle).

---

## 4) How to Run (Kaggle)

1. Open the notebook and **Enable GPU** in *Notebook Settings*.
2. Attach the dataset from Kaggle Datasets.
3. Run in order: `01_data_eda` → `02_baseline_cnn` → `03_transfer_learning`.
4. Use `04_eval_gradcam` to produce the **classification report**, **confusion matrix**, and **Grad‑CAM** figures.


---

## 5) Evaluation

* **Metrics**: Accuracy (top‑1), per‑class precision/recall/F1.
* **Visuals**: Training/validation curves, confusion matrix heatmap.
* **Robustness checks**: Class imbalance review, failure case analysis.

> Provide short, focused commentary for each figure: what improved, where it still fails, why.

---

## 6) Explainability (Grad‑CAM)

We visualize salient regions for both **correct** and **incorrect** predictions. This helps validate whether the model attends to the vehicle—not the background.

---

## 7) Hyperparameter Tuning

We vary:

* Learning rate & scheduler
* Batch size
* Augmentation strength (flip/rotation/zoom/color)
* Dropout rate and **L2**
* Optimizer variants

Record the best configuration and compare against the baseline.

---

## 8) Limitations & Future Work

* Visually similar models and badges cause confusions.
* Consider higher‑resolution inputs, **label smoothing**, mixup/cutmix, TTA.
* Explore stronger backbones or ViT‑based encoders.

---

## 9) Links

* **Kaggle Notebook**: https://www.kaggle.com/code/pelinsuustun/car-brand-classification
* **Dataset**: https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset

---

## 10) License

Released under the terms specified in `LICENSE`.


