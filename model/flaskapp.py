import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and label encoder
model = load_model('recognition_model_3.h5')
le = joblib.load('label_encoder_3.pkl')

def load_and_preprocess_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE).astype('float32')
    img = cv.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_path = 'temp_image.png'
    image_file.save(image_path)

    # Preprocess and predict
    test_image = load_and_preprocess_image(image_path)
    prediction = model.predict(test_image)
    predicted_label_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_label_index])[0]

    # Remove the temporary image file
    os.remove(image_path)

    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
