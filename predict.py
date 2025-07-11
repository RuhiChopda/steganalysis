import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

IMG_SIZE = 128
MODEL_PATH = "steganalysis_model.h5"
FOLDER_TO_SCAN = "data/clean"  # Change to 'data/stego' to test stego images

def predict_image(model, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not read {path}")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    prediction = model.predict(img)[0]
    label = "Clean" if np.argmax(prediction) == 0 else "Stego"
    confidence = prediction[np.argmax(prediction)] * 100
    print(f"🖼️ {os.path.basename(path)} → Prediction: {label} ({confidence:.2f}%)")

def batch_predict(folder):
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Run model_train.py first.")
        return

    model = load_model(MODEL_PATH)

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"❗ No image files found in {folder}")
        return

    print(f"🔍 Predicting {len(files)} images in '{folder}'...\n")
    for file in files:
        predict_image(model, os.path.join(folder, file))

# Run prediction on all images in folder
batch_predict(FOLDER_TO_SCAN)
