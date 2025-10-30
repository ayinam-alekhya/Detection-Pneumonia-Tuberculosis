# 🩺 Detection of Pneumonia & Tuberculosis from Chest X-rays (ResNet50)

Transfer-learning pipeline built with **TensorFlow/Keras** to classify chest X-ray images into three classes:  
**NORMAL**, **PNEUMONIA**, and **TUBERCULOSIS** using a **ResNet50** backbone.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayinam-alekhya/Detection-Pneumonia-Tuberculosis/blob/main/MajorPro.ipynb)

---

## 📁 Repository Structure

```
Detection-Pneumonia-Tuberculosis/
├── MajorPro.ipynb               # Clean notebook for GitHub rendering
├── MajorPro_Full.ipynb          # Full notebook (large outputs, may not render)
├── ICCES804_FinalPaper.pdf      # Project paper / write-up
├── data/                        # Dataset folder (see layout below)
├── README.md
└── (.gitattributes, .gitignore) # Optional LFS & ignore rules
```

### Expected Dataset Layout

```
data/
├── NORMAL/
│   ├── img_1.png
│   └── ...
├── PNEUMONIA/
│   ├── img_2.png
│   └── ...
└── TUBERCULOSIS/
    ├── img_3.png
    └── ...
```

> **Note:** If your dataset uses `TB/` for tuberculosis, rename it to `TUBERCULOSIS/`:
> ```bash
> mv data/TB data/TUBERCULOSIS
> ```

---

## 🚀 Quick Start (Google Colab)

1. Click the **"Open in Colab"** button above.
2. *(Optional)* Set runtime type → **GPU** for faster training.
3. If dataset is on Google Drive, mount it and update the path variable in the notebook.
4. Run all cells in `MajorPro.ipynb`.  
   The trained model will be saved as **`ResNet50.h5`**.

---

## 🧠 Model Architecture (High-Level Overview)

| Component | Description |
|------------|-------------|
| **Backbone** | ResNet50 (`weights="imagenet"`, `include_top=False`, `pooling="avg"`) |
| **Head** | Dense(128, relu) → Dense(128, relu) → Dense(3, softmax) |
| **Input Size** | 227×227×3 |
| **Data Pipeline** | `ImageDataGenerator` with `preprocess_input` |
| **Split** | `train_test_split(..., test_size=0.25, random_state=42)` |
| **Regularization** | `EarlyStopping(monitor="val_accuracy", patience=3)` |

---

## 🛠️ Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install tensorflow==2.* opencv-python matplotlib seaborn pandas scikit-learn imutils
```

### ▶️ Training via Notebook

Open and run `MajorPro.ipynb`. It will:

- Discover image files and build a DataFrame (`Filepath`, `Label`)
- Create train/val/test generators (target size **227×227**)
- Build and train the **ResNet50-based classifier**
- Save the model to `ResNet50.h5`
- Plot accuracy/loss curves
- Print a **classification report** on the test set

---

## 🔍 Inference (Single Image Example)

```python
import cv2, numpy as np, tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = tf.keras.models.load_model("ResNet50.h5")

# Read & preprocess
img_path = "data/PNEUMONIA/some_image.jpeg"
img = cv2.imread(img_path)[:, :, ::-1]  # BGR → RGB
img = cv2.resize(img, (227, 227))
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Predict
probs = model.predict(x)[0]
classes = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
print(dict(zip(classes, (probs * 100).round(2))))
print("Prediction:", classes[int(np.argmax(probs))])
```

---

## 📊 Results

The notebook outputs:

- **Test Accuracy** and **Test Loss**
- **Classification report** (Precision / Recall / F1-score)
- **Visual grids** comparing true vs predicted labels

> Results may vary with dataset size, augmentations, and random seed.

---

## 🧭 Tips & Recommendations

- ✅ Folder names **must match labels exactly** (`NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`)
- 📘 Keep two notebooks:
  - `MajorPro.ipynb` → Clean version (no output, GitHub-friendly)
  - `MajorPro_Full.ipynb` → Full version (download to view)
- ⚡ Use GPU (e.g., Google Colab) for efficient training

---

## 📄 Reference Paper

Read the full project paper:  
📄 [`ICCES804_FinalPaper.pdf`](./ICCES804_FinalPaper.pdf)
