import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from google.cloud import vision
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
import base64  # For encoding images as Base64
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
WORD_FOLDER = './segmented_words'
CHAR_FOLDER = './segmented_chars'
SKETCH_FOLDER = './sketches'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORD_FOLDER, exist_ok=True)
os.makedirs(CHAR_FOLDER, exist_ok=True)
os.makedirs(SKETCH_FOLDER, exist_ok=True)

# Load the trained model and label encoder
MODEL_PATH = "recognition_model_2.h5"
LABEL_ENCODER_PATH = "label_encoder_2.pkl"

model = load_model(MODEL_PATH)
encoder = joblib.load(LABEL_ENCODER_PATH)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "llmtesting-242512-55d0854ea838.json"
client = vision.ImageAnnotatorClient()

def clear_folder(folder_path):
    """Deletes all files inside a given folder."""
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def segment_words(image_path):
    """Segments words from a handwritten text image and saves them."""
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image if it's too large
    h, w, _ = img.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to grayscale and apply threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Detect lines
    kernel_line = np.ones((3, 85), np.uint8)
    dilated_line = cv2.dilate(thresh, kernel_line, iterations=1)
    contours_line, _ = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_lines = sorted(contours_line, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Detect words
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

def ocr_vision(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def segment_characters(image_path, word_index):
    """Segments characters from a word image and saves them."""
    
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
            padded_img = np.ones((200, 200), dtype=np.uint8) * 255  # White background
            x_offset = (200 - 64) // 2
            y_offset = (200 - 64) // 2
            padded_img[y_offset:y_offset + 64, x_offset:x_offset + 64] = char_img
            
            char_path = os.path.join(CHAR_FOLDER, f'word_{word_index}_char_{idx}.png')
            cv2.imwrite(char_path, padded_img)
            char_paths.append(char_path)

    return char_paths

def preprocess_image(image_path):
    """Prepares a character image for model prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32")
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=[0, -1])  # Reshape to (1, 64, 64, 1)
    return img

def predict_character(image_path):
    """Predicts a character from an image."""
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    label_index = np.argmax(prediction)
    predicted_label = encoder.categories_[0][label_index]
    return predicted_label

def extract_drawings_and_sketches(image):
    # Resize the image to 1920x1080 (if needed)
    resized_image = cv2.resize(image, (1920, 1080))

    # Crop the image from vertical coordinate 350 to the bottom (full width)
    cropped_image = resized_image[350:, :]  # This takes rows from 350 to the end and all columns

    # Convert to grayscale and apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Apply morphological operations to separate overlapping sketches
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours in the processed image
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sketches = []
    sketch_index = 1

    for contour in contours:
        # Filter based on contour area (to remove small noise)
        area = cv2.contourArea(contour)
        if area < 500:  # Adjust this threshold as needed
            continue

        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the sketch using the bounding box
        sketch = cropped_image[y:y+h, x:x+w]
        sketch_path = os.path.join(SKETCH_FOLDER, f'sketch_{sketch_index}.png')
        cv2.imwrite(sketch_path, sketch)
        sketches.append(sketch_path)
        sketch_index += 1

    return sketches

def create_pdf(text, sketch_paths, pdf_path="recognized_text.pdf"):
    """Generates a PDF from recognized text and sketches."""
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = [Paragraph(text, getSampleStyleSheet()["BodyText"]), Spacer(1, 12)]
    
    for sketch_path in sketch_paths:
        story.append(Image(sketch_path, width=200, height=200))
        story.append(Spacer(1, 12))
    
    doc.build(story)

def encode_image_as_base64(image_path):
    """Encodes an image as a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.route("/segment_and_recognize", methods=["POST"])
def segment_and_recognize():
    """API Endpoint: Accepts an image, performs segmentation & recognition."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Load the image
        img = cv2.imread(image_path)
        
        # Extract sketches
        sketch_paths = extract_drawings_and_sketches(img)
        
        # Perform word segmentation
        word_paths = segment_words(image_path)
        recognized_text = ""

        # Perform character segmentation & recognition for each word
        for idx, word_path in enumerate(word_paths, 1):
            char_paths = segment_characters(word_path, idx)
            word = "".join(predict_character(char_path) for char_path in char_paths)
        
        recognized_text = ocr_vision(image_path)
        
        # Generate the PDF
        pdf_buffer = io.BytesIO()
        create_pdf(recognized_text, sketch_paths, pdf_buffer)
        pdf_buffer.seek(0)

        # Encode sketches as Base64
        sketch_base64 = [encode_image_as_base64(sketch_path) for sketch_path in sketch_paths]

        return jsonify({
            "recognized_text": recognized_text,
            "pdf_base64": base64.b64encode(pdf_buffer.read()).decode("utf-8"),
            "sketches_base64": sketch_base64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)