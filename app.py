from flask import Flask, render_template, request, send_file, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import csv
import zipfile
import shutil
import uuid
from io import BytesIO
from functools import wraps

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'steganalysis_model.h5'
IMG_SIZE = 128

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model(MODEL_PATH)

# Authentication credentials
USERNAME = "admin"
PASSWORD = "1234"

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response("Login Required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Image prediction function
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Invalid", 0, 0, 0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    prediction = model.predict(img)[0]

    clean_prob = round(prediction[0] * 100, 2)
    stego_prob = round(prediction[1] * 100, 2)

    if abs(prediction[0] - prediction[1]) < 0.1:
        label = "Uncertain"
    else:
        label = "Clean" if np.argmax(prediction) == 0 else "Stego"

    confidence = round(max(prediction) * 100, 2)
    return label, confidence, clean_prob, stego_prob

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            label, confidence, clean_prob, stego_prob = predict_image(filepath)
            return render_template("index.html",
                                   filename=file.filename,
                                   label=label,
                                   confidence=confidence,
                                   clean_prob=clean_prob,
                                   stego_prob=stego_prob)
    return render_template("index.html")

@app.route("/static/uploads/<filename>")
def send_file_to_static(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

@app.route("/batch")
@requires_auth
def batch_analyze():
    image_folder = 'data/stego'
    results = []
    clean_count = stego_count = uncertain_count = 0

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(image_folder, filename)
            label, confidence, clean_prob, stego_prob = predict_image(path)
            results.append({
                "filename": filename,
                "label": label,
                "confidence": confidence,
                "clean_prob": clean_prob,
                "stego_prob": stego_prob
            })

            if label == "Clean":
                clean_count += 1
            elif label == "Stego":
                stego_count += 1
            else:
                uncertain_count += 1

    return render_template("batch.html", results=results,
                           clean_count=clean_count,
                           stego_count=stego_count,
                           uncertain_count=uncertain_count)

@app.route("/upload_zip", methods=["POST"])
@requires_auth
def upload_zip():
    if "zipfile" not in request.files:
        return "No file part", 400

    zip_file = request.files["zipfile"]
    if zip_file.filename == "":
        return "No selected file", 400

    temp_folder = f"temp_zip_{uuid.uuid4().hex}"
    os.makedirs(temp_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)

    results = []
    clean_count = stego_count = uncertain_count = 0

    for filename in os.listdir(temp_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(temp_folder, filename)
            label, confidence, clean_prob, stego_prob = predict_image(path)
            results.append({
                "filename": filename,
                "label": label,
                "confidence": confidence,
                "clean_prob": clean_prob,
                "stego_prob": stego_prob
            })

            if label == "Clean":
                clean_count += 1
            elif label == "Stego":
                stego_count += 1
            else:
                uncertain_count += 1

    shutil.rmtree(temp_folder)

    return render_template("batch.html", results=results,
                           clean_count=clean_count,
                           stego_count=stego_count,
                           uncertain_count=uncertain_count)

@app.route("/download_csv")
@requires_auth
def download_csv():
    image_folder = 'data/stego'
    rows = [["Filename", "Clean %", "Stego %", "Prediction", "Confidence"]]

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(image_folder, filename)
            label, confidence, clean_prob, stego_prob = predict_image(path)
            rows.append([filename, clean_prob, stego_prob, label, confidence])

    csv_string = "\n".join([",".join(map(str, row)) for row in rows])
    output = BytesIO(csv_string.encode("utf-8"))
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='batch_results.csv'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
run(debug=True)
