import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

dataset_path = "dataset/PlantVillage"
image_size = (64, 64)

def load_data():
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                try:
                    img_path = os.path.join(label_path, file)
                    img = Image.open(img_path).resize(image_size).convert('RGB')
                    X.append(np.array(img).flatten())
                    y.append(label)
                except Exception:
                    continue
    return np.array(X), np.array(y)

def train():
    X, y = load_data()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/disease_model.pkl")
    joblib.dump(encoder, "model/label_encoder.pkl")
    print("✅ Model training complete.")

if __name__ == "__main__":
    train()
