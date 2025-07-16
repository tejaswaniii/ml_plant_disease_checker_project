import os
import pickle
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pymongo

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and encoder
with open('model/disease_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["plant_disease_db"]
collection = db["predictions"]

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, -1)
    return img_array

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)

    filename = secure_filename(image.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)

    processed_img = preprocess_image(file_path)
    prediction = model.predict(processed_img)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Save to MongoDB
    collection.insert_one({"filename": filename, "prediction": predicted_label})

    return render_template("index.html", prediction=predicted_label, image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
