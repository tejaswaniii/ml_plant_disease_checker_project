from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import joblib
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load("model/disease_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["plant_disease"]
collection = db["predictions"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            img = Image.open(file_path).resize((64, 64)).convert("RGB")
            img_arr = np.array(img).flatten().reshape(1, -1)
            pred = model.predict(img_arr)
            label = label_encoder.inverse_transform(pred)[0]
            prediction = label

            # Save to MongoDB
            collection.insert_one({
                "filename": file.filename,
                "prediction": label,
                "timestamp": datetime.now()
            })

            return render_template("index.html", prediction=prediction, image_path=file_path)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
