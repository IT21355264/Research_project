import numpy as np
import cv2 as cv
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and label encoder
MODEL_PATH = "recognition_model_2.h5"
LABEL_ENCODER_PATH = "label_encoder_2.pkl"

model = load_model(MODEL_PATH)
encoder = joblib.load(LABEL_ENCODER_PATH)

def preprocess_image(image_bytes):
    """Preprocesses image for model prediction."""
    image = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_GRAYSCALE)

    if image is None:
        return None  # Return None if image is invalid

    image = cv.resize(image, (64, 64))
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Reshape to (1, 64, 64, 1)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["image"]
        image_bytes = file.read()  # Read image as bytes

        # Preprocess image
        image = preprocess_image(image_bytes)

        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Predict
        prediction = model.predict(image)
        label_index = np.argmax(prediction)
        label = encoder.categories_[0][label_index]

        return jsonify({"predicted_character": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error response

if __name__ == "__main__":
    app.run(debug=True)
