import os
import numpy as np
import tensorflow as tf
import cv2

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lung_disease_model.h5")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels
LABELS = ["Pulmonary Fibrosis", "Normal", "Viral Pneumonia"]

# 🔍 Function to detect which lung is affected
def detect_affected_lung(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ["Image not found"]
    
    img = cv2.resize(img, (128, 128))
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

# 🧠 Main function to predict disease
def run_prediction_pipeline(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return {
            "error": "Image could not be loaded. Check the file path or format."
        }

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result = {
        "disease": LABELS[predicted_class],
        "confidence": f"{confidence:.2f}%",
        "description": f"The detected condition is {LABELS[predicted_class]}.",
        "status": "disease" if LABELS[predicted_class] != "Normal" else "healthy",
        "affected_areas": ", ".join(detect_affected_lung(image_path))
    }
    return result

# Test the prediction
test_image = "static/sample_image.jpg"    # Update to your actual test image
result = run_prediction_pipeline(test_image)
print(result)
