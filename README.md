# 🩺 Detection of Pneumonia & Tuberculosis from Chest X-rays (ResNet50 + Flask)

This project provides a complete pipeline — from **training a ResNet50 deep learning model** to classify **Chest X-rays** into *Normal*, *Pneumonia*, or *Tuberculosis*, to **deploying** the trained model as an **interactive Flask web application**.

---

## 📁 Repository Overview

```
Detection-Pneumonia-Tuberculosis/
│
├── data/                           # Dataset folder
│
├── ResNet50_Model/                 # Model training files
│   ├── MajorPro.ipynb              # Clean notebook (for GitHub)
│   ├── MajorPro_Full.ipynb         # Full notebook (with outputs)
│   ├── README.md                   # Model-only readme (training details)
│   └── ResNet50.h5                 # Trained model (tracked via Git LFS)
│
├── Website/                        # Flask deployment folder
│   ├── app.py                      # Flask backend
│   ├── static/                     # Static assets (images, favicon)
│   ├── templates/                  # HTML templates
│   ├── .gitignore
│   ├── README.md                   # Web app documentation
│   └── .venv310/                   # Virtual environment (ignored)
│
├── ICCES804_FinalPaper.pdf         # Project paper / write-up
└── .gitignore
```

---

## 🧠 Part 1 – Model Training (ResNet50)

The `ResNet50_Model` folder contains all training files used to build the model.

### ⚙️ Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.* opencv-python matplotlib seaborn pandas scikit-learn imutils
```

### ▶️ Training Steps
1. Open `MajorPro.ipynb` (or `MajorPro_Full.ipynb`).
2. Mount dataset folder if using Google Drive.
3. Run all cells to:
   - preprocess X-ray images (227×227)
   - train ResNet50 (transfer learning)
   - evaluate and save the model as `ResNet50.h5`.

**Output Metrics:**
- Test Accuracy & Loss  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix & Visualization of Predictions  

### 🧩 Model Architecture
| Component | Description |
|------------|-------------|
| **Backbone** | ResNet50 (`weights="imagenet"`, `include_top=False`, `pooling="avg"`) |
| **Head** | Dense(128, relu) → Dense(128, relu) → Dense(3, softmax) |
| **Input Size** | 227×227×3 |
| **Classes** | NORMAL, PNEUMONIA, TUBERCULOSIS |

---

## 🌐 Part 2 – Flask Web Application

The `Website` folder contains the Flask application that uses the trained model for live image predictions.

### 🚀 Features
- Upload a chest X-ray and receive instant AI prediction.
- Displays class-specific HTML pages:
  - 🟢 `Normal.html`
  - 🟡 `Pneumonia.html`
  - 🔴 `Tuberculosis.html`
- Uses TensorFlow/Keras backend and NumPy for preprocessing.

### ⚙️ How to Run
```bash
cd Website
python3.10 -m venv .venv310
source .venv310/bin/activate
pip install flask tensorflow numpy
flask --app app run
```
Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧭 Combined Workflow

1. **Train Model:**  
   Run `MajorPro.ipynb` to generate `ResNet50.h5`.

2. **Deploy:**  
   Move the `.h5` file into `Website/`, and run `app.py`.

3. **Predict:**  
   Upload chest X-ray → app returns predicted class and opens result page.

---

## 📊 Tech Stack
| Layer | Technology |
|--------|-------------|
| Model | TensorFlow / Keras (ResNet50) |
| Frontend | HTML5, CSS3 (Jinja Templates) |
| Backend | Flask |
| Language | Python 3.10 |
| IDE | VS Code |
| Deployment | Local / Render (future) |

---

## 📄 Reference
📘 **Published Paper:** [IEEE Xplore – Detection of Pneumonia & Tuberculosis from Chest X-rays (ResNet50)](https://ieeexplore.ieee.org/document/10192888)  

📘 **Paper:** [ICCES804_FinalPaper.pdf](./ICCES804_FinalPaper.pdf)

---

## 👩‍💻 Author
**Alekhya Ayinam**  
🎓 M.S. Computer Science – University of South Florida  
🔗 [GitHub](https://github.com/ayinam-alekhya)
