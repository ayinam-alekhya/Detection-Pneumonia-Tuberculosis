# Detection of Pneumonia & Tuberculosis from Chest X-rays (ResNet50)

Transfer-learning pipeline (TensorFlow/Keras) to classify **NORMAL**, **PNEUMONIA**, and **TUBERCULOSIS** chest X-ray images using **ResNet50**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayinam-alekhya/Detection-Pneumonia-Tuberculosis/blob/main/MajorPro.ipynb)

---

## 📦 Repository Structure

Detection-Pneumonia-Tuberculosis/
├─ MajorPro.ipynb # Clean, GitHub-renderable notebook
├─ MajorPro_Full.ipynb # Full notebook (may be large; might not render in GitHub UI)
├─ ICCES804_FinalPaper.pdf # Conference paper / write-up
├─ data/ # Dataset folder (see layout below)
├─ README.md
└─ (.gitattributes, .gitignore) # optional: LFS & ignore rules

### Expected dataset layout

Place images in class-named directories (folder names become labels):

data/
├─ NORMAL/
│ ├─ img_1.png
│ └─ ...
├─ PNEUMONIA/
│ ├─ img_2.png
│ └─ ...
└─ TUBERCULOSIS/
├─ img_3.png
└─ ...

> **Note:** If your dataset currently uses `TB/` for tuberculosis, rename it to `TUBERCULOSIS/` so labels match:
> ```bash
> mv data/TB data/TUBERCULOSIS
> ```

---

## 🚀 Quick Start (Colab)

1. Click **Open in Colab** above.
2. (Optional) **Runtime → Change runtime type → GPU** for faster training.
3. If your dataset is on Drive, mount it in the notebook and point `b = "/content/drive/MyDrive/..."` to your dataset.
4. Run all cells in `MajorPro.ipynb`. The trained model will be saved as `ResNet50.h5`.

---

## 🧠 Method (High Level)

- **Backbone:** `ResNet50` (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
- **Head:** `Dense(128, relu) → Dense(128, relu) → Dense(3, softmax)`
- **Input size:** `227×227×3` (kept consistent throughout the pipeline)
- **Data pipeline:** `ImageDataGenerator` with `preprocess_input`
- **Split:** `train_test_split(..., test_size=0.25, random_state=42)`
- **Regularization:** `EarlyStopping(monitor="val_accuracy", patience=3)`

---

## 🛠️ Local Setup

```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install tensorflow==2.* opencv-python matplotlib seaborn pandas scikit-learn imutils
## ▶️ Training (via Notebook)

Open and run `MajorPro.ipynb`. The notebook will:

- Discover image files and build a DataFrame (`Filepath`, `Label`)
- Create `train_gen` / `val_gen` / `test_gen` with target size **227×227**
- Build the **ResNet50**-based classifier
- Train with **early stopping**
- Save the model to **`ResNet50.h5`**
- Plot **Accuracy/Loss** curves
- Evaluate on the test split and print a **classification report**

---

## 🔍 Inference (Single Image Example)

```python
import cv2, numpy as np, tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load trained model
model = tf.keras.models.load_model("ResNet50.h5")

# Read and preprocess image
img_path = "data/PNEUMONIA/some_image.jpeg"
img = cv2.imread(img_path)[:, :, ::-1]     # BGR -> RGB
img = cv2.resize(img, (227, 227))
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Predict
probs = model.predict(x)[0]                 # [p(NORMAL), p(PNEUMONIA), p(TUBERCULOSIS)]
classes = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

print(dict(zip(classes, (probs*100).round(2))))
print("Prediction:", classes[int(np.argmax(probs))])


## 📊 Results

The notebook reports:

- **Test Loss** and **Test Accuracy** on your split
- A detailed **classification report** (precision / recall / F1)
- Example grids showing **True vs. Predicted** labels

> Results depend on dataset balance/size, augmentations, and random seed.

---

## 🧭 Tips

- **Folder names = labels.** Ensure class folders are exactly:
  - `NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`
- **Large notebooks on GitHub:** Big outputs may not render in GitHub’s viewer. Keep:
  - `MajorPro.ipynb` → cleaned (no outputs) for easy viewing
  - `MajorPro_Full.ipynb` → full version (download to view if GitHub can’t render)
- **GPU recommended** for training (Colab GPU is sufficient for this project scale).

---

## 📄 Paper

See the project write-up in [`ICCES804_FinalPaper.pdf`](./ICCES804_FinalPaper.pdf).




