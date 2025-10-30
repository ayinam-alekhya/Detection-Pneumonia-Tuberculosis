# 🩺 AI-Powered Chest X-Ray Disease Detection (Flask + ResNet50)

This web application provides an intuitive interface for classifying **Chest X-ray** images into **Normal**, **Pneumonia**, or **Tuberculosis** categories using a **ResNet50 deep learning model** built with TensorFlow/Keras.

---

## 🚀 Features
- Upload a chest X-ray image directly through the web interface.  
- Predicts and classifies the image into one of three classes:  
  - 🟢 Normal  
  - 🟡 Pneumonia  
  - 🔴 Tuberculosis  
- Built with **Flask** as the backend and **TensorFlow ResNet50** for deep learning inference.  
- Includes dedicated HTML result pages for each prediction class.  

---

## 🧠 Model Overview
- **Model Used:** ResNet50 (Transfer Learning)
- **Framework:** TensorFlow/Keras
- **Trained Classes:** `Normal`, `Pneumonia`, `Tuberculosis`
- **File:** `ResNet50.h5`

The model was trained using transfer learning on a curated dataset of chest X-ray images and saved for real-time inference in this Flask web app.

---

## 🗂️ Project Structure

```
Website/
│
├── app.py                        # Flask app logic
├── ResNet50.h5                   # Trained model
│
├── static/                       # Images, styles, etc.
│   ├── favicon.png
│   ├── normal.jpg
│   ├── tuberculosis.jpg
│   └── user uploaded/
│
├── templates/                    # HTML templates
│   ├── index.html                # Homepage (upload page)
│   ├── homepg.html               # Landing page
│   ├── Normal.html               # Output page for Normal result
│   ├── Pneumonia.html            # Output page for Pneumonia result
│   └── Tuberculosis.html         # Output page for Tuberculosis result
│
└── README.md
```

---

## ⚙️ How to Run the App

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayinam-alekhya/Detection-Pneumonia-Tuberculosis-Website.git
   cd Detection-Pneumonia-Tuberculosis-Website
   ```

2. **Create & activate a virtual environment**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(or manually: `pip install flask tensorflow numpy`)*

4. **Run the Flask server**
   ```bash
   flask --app app run
   ```

5. **Open the app**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📊 Tech Stack
| Component | Technology |
|------------|-------------|
| Frontend | HTML5, CSS3 (Jinja Templates) |
| Backend | Flask |
| Model | TensorFlow / Keras (ResNet50) |
| Language | Python 3.10 |
| Environment | macOS |

---

## 📸 Sample Output Pages
- **Normal Prediction Page** → `Normal.html`  
- **Pneumonia Prediction Page** → `Pneumonia.html`  
- **Tuberculosis Prediction Page** → `Tuberculosis.html`

---

## 👩‍💻 Author
**Alekhya Ayinam**  
🎓 M.S. Computer Science, University of South Florida  
🔗 [GitHub](https://github.com/ayinam-alekhya)
