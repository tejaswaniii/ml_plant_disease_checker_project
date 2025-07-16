import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Path to dataset
dataset_path = "dataset/PlantVillage"
img_size = (128, 128)

X = []
y = []

# Load and preprocess images
for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                X.append(img_array.flatten())
                y.append(label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and label encoder
os.makedirs("model", exist_ok=True)
with open("model/disease_model.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model and label encoder saved in 'model/' folder.")
