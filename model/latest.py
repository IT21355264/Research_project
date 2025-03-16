import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
WORD_FOLDER = './segmented_words'
CHAR_FOLDER = './segmented_chars'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORD_FOLDER, exist_ok=True)
os.makedirs(CHAR_FOLDER, exist_ok=True)

model = load_model('recognition_model_3.h5')
le = joblib.load('label_encoder_3.pkl')

def segment_words(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    kernel_line = np.ones((3, 85), np.uint8)
    dilated_line = cv2.dilate(thresh, kernel_line, iterations=1)
    contours_line, _ = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_lines = sorted(contours_line, key=lambda ctr: cv2.boundingRect(ctr)[1])

    kernel_word = np.ones((3, 15), np.uint8)
    dilated_word = cv2.dilate(thresh, kernel_word, iterations=1)

    word_paths = []
    word_index = 1

    for line in sorted_lines:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_word[y:y + h, x:x + w]

        contours_word, _ = cv2.findContours(roi_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_words = sorted(contours_word, key=lambda c: cv2.boundingRect(c)[0])

        for word in sorted_words:
            if cv2.contourArea(word) < 400:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            word_img = img[y + y2:y + y2 + h2, x + x2:x + x2 + w2]
            word_path = os.path.join(WORD_FOLDER, f'word_{word_index}.png')
            cv2.imwrite(word_path, cv2.cvtColor(word_img, cv2.COLOR_RGB2BGR))
            word_paths.append(word_path)
            word_index += 1

    return word_paths

def segment_characters(image_path, word_index):
    """Segments individual characters from a word image with larger white background padding."""
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda x: x[0])

    char_paths = []
    for idx, (x, y, w, h) in enumerate(char_bboxes):
        if w > 5 and h > 10:  # Filter out small noise
            char_img = thresh[y:y + h, x:x + w]
            char_img = cv2.resize(char_img, (64, 64))

            # Ensure text is black and background is white
            char_img = 255 - char_img  # Invert colors

            # Create a larger white background canvas
            padded_size = 140  # Increased padding
            padded_img = np.ones((padded_size, padded_size), dtype=np.uint8) * 255  # Larger white background
            x_offset = (padded_size - 64) // 2
            y_offset = (padded_size - 64) // 2
            padded_img[y_offset:y_offset + 64, x_offset:x_offset + 64] = char_img
            
            char_path = os.path.join(CHAR_FOLDER, f'word_{word_index}_char_{idx}.png')
            cv2.imwrite(char_path, padded_img)
            char_paths.append(char_path)

    return char_paths

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype('float32')
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def predict_character(image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_label_index])[0]
    return predicted_label

def create_pdf(text, pdf_path='recognized_text.pdf'):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    style = getSampleStyleSheet()["BodyText"]
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 12))
    doc.build(story)

@app.route('/segment_and_recognize', methods=['POST'])
def segment_and_recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    word_paths = segment_words(image_path)
    recognized_text = ""

    for idx, word_path in enumerate(word_paths, 1):
        char_paths = segment_characters(word_path, idx)
        word = "".join(predict_character(char_path) for char_path in char_paths)
        recognized_text += word + " "

    create_pdf(recognized_text.strip())

    return jsonify({'recognized_text': recognized_text, 'pdf_path': 'recognized_text.pdf'})

if __name__ == '__main__':
    app.run(debug=True)
