from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__, template_folder='template')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lung_disease_model.h5")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels
LABELS = ["Pulmonary Fibrosis", "Normal", "Viral Pneumonia"]

# Function to detect which lung is affected
def detect_affected_lung(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0

    h, w = img.shape
    left_img = img[:, :w//2]
    right_img = img[:, w//2:]

    left_img = left_img.reshape(1, 128, 64, 1)
    right_img = right_img.reshape(1, 128, 64, 1)

    # Resize to (128,128) to fit model input shape
    left_img = tf.image.resize(left_img, [128, 128])
    right_img = tf.image.resize(right_img, [128, 128])

    left_pred = model.predict(left_img)
    right_pred = model.predict(right_img)

    left_class = np.argmax(left_pred)
    right_class = np.argmax(right_pred)

    affected_parts = []
    if LABELS[left_class] != "Normal":
        affected_parts.append("Left lung")
    if LABELS[right_class] != "Normal":
        affected_parts.append("Right lung")

    return affected_parts if affected_parts else ["None"]

# Function to predict disease from an image
def predict_disease(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 128, 128, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)  # Convert to percentage and ensure float type

    if LABELS[predicted_class] == "Normal":
        # No need to check affected areas if the overall prediction is "Normal"
        affected_areas = ["None"]
    else:
        affected_areas = detect_affected_lung(image)

    return LABELS[predicted_class], confidence, affected_areas

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_data = file.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Error decoding image'}), 400
        
        prediction, confidence, affected_areas = predict_disease(img)
        return jsonify({'prediction': prediction, 'confidence': confidence, 'affected_areas': affected_areas})
    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({'error': 'Error processing file'}), 500

if __name__ == '__main__':
    app.run(debug=True)