import os
import pickle
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pymongo

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and label encoder
with open('model/disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["plant_disease_db"]
collection = db["predictions"]

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, -1)

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
    pred_encoded = model.predict(processed_img)[0]
    prediction = le.inverse_transform([pred_encoded])[0]

    collection.insert_one({"filename": filename, "prediction": prediction})

    return render_template("index.html", prediction=prediction, image_path=file_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
