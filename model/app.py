from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
from fpdf import FPDF
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved model
def load_saved_model(model_path):
    return load_model(model_path, compile=False)

# Preprocess the image for word segmentation
def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh

def word_segmentation(image):
    h, w, _ = image.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    thresh_img = thresholding(image)
    kernel_line = np.ones((3, 85), np.uint8)
    dilated_line = cv2.dilate(thresh_img, kernel_line, iterations=1)
    contours, _ = cv2.findContours(dilated_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    kernel_word = np.ones((3, 15), np.uint8)
    dilated_word = cv2.dilate(thresh_img, kernel_word, iterations=1)
    words_list = []

    for line in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_word[y:y+h, x:x+w]
        cnt, _ = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])

    return image, words_list

def preprocess_image(image, img_size=(256, 64)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    final_img = np.ones((img_size[1], img_size[0])) * 255
    if w > img_size[0]:
        gray = gray[:, :img_size[0]]
    if h > img_size[1]:
        gray = gray[:img_size[1], :]
    final_img[:gray.shape[0], :gray.shape[1]] = gray
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE) / 255.0
    return np.expand_dims(final_img, axis=(0, -1))

def decode_prediction(prediction, alphabet):
    decoded_text = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1])[0][0]
    decoded_text = K.get_value(decoded_text)
    return ''.join([alphabet[i] for i in decoded_text[0] if i != -1])

@app.route('/process', methods=['POST'])
def process_image():
    model_path = "handwriting_model.h5"
    model = load_saved_model(model_path)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = np.array(Image.open(file).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    processed_img, word_boxes = word_segmentation(image)
    predictions = []

    for word in word_boxes:
        x1, y1, x2, y2 = word
        word_img = processed_img[y1:y2, x1:x2]
        processed_word_img = preprocess_image(word_img)
        prediction = model.predict(processed_word_img)
        predicted_text = decode_prediction(prediction, alphabet)
        predictions.append(predicted_text)

    # Create a temporary PDF to store results
    temp_pdf_path = os.path.join(tempfile.gettempdir(), "output.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for i, text in enumerate(predictions):
        pdf.cell(0, 10, f"{i + 1}. {text}", ln=True)

    pdf.output(temp_pdf_path)

    # Return the PDF file
    return send_file(temp_pdf_path, as_attachment=True, download_name="output.pdf")

if __name__ == "__main__":
    app.run(debug=True)
