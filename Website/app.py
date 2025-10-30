from flask import Flask, render_template, request,url_for,redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import os


currentdirectory = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
model_path = "ResNet50.h5"
model = load_model(model_path)
category = {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"}


def model_predict(image_path, model):
    print(image_path)
    image = load_img(image_path, target_size=(227, 227))
    image = img_to_array(image)
    image=preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(image))
    print(result)
    if result == 0:
        return "Normal", "Normal.html"
    elif result == 1:
        return "Pneumonia", "Pneumonia.html"
    elif result == 2:
        return "Tuberculosis", "Tuberculosis.html"

@app.route("/")
def home():
    return render_template("homepg.html")


@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        pred,output_page = model_predict(file_path, model)
        return render_template(output_page, pred_output=pred, user_image=file_path)
if __name__ == "__main__":
    app.run(debug=True)
