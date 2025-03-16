import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
from fpdf import FPDF
import tempfile
import os

# Load the saved model
@st.cache_resource
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

# Preprocess the image for prediction
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

# Decode the prediction to text
def decode_prediction(prediction, alphabet):
    decoded_text = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1])[0][0]
    decoded_text = K.get_value(decoded_text)
    return ''.join([alphabet[i] for i in decoded_text[0] if i != -1])

# Detect diagrams and save them
def extract_diagrams_and_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            cropped_box = image[y:y+h, x:x+w]
            cropped_boxes.append(cropped_box)

    return cropped_boxes


def save_to_pdf(predictions, drawings, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add recognized text
    line_text = " ".join(predictions)
    pdf.multi_cell(0, 10, txt=line_text, align='L')

    # Add drawings in a grid
    temp_image_paths = []  # List to store temporary image paths
    if drawings:
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Detected Drawings:", ln=True, align='L')

        img_x, img_y = 10, 30  # Starting position
        img_width, img_height = 60, 60  # Resize dimensions for drawings
        max_per_row = 3  # Number of images per row

        for idx, drawing in enumerate(drawings):
            # Resize drawing for the PDF
            drawing_resized = cv2.resize(drawing, (img_width, img_height))
            temp_image_path = f"temp_image_{idx}.jpg"
            temp_image_paths.append(temp_image_path)
            cv2.imwrite(temp_image_path, cv2.cvtColor(drawing_resized, cv2.COLOR_RGB2BGR))
            pdf.image(temp_image_path, x=img_x, y=img_y, w=img_width, h=img_height)

            # Adjust position for the next image
            img_x += img_width + 10
            if (idx + 1) % max_per_row == 0:
                img_x = 10
                img_y += img_height + 10

            # If there's overflow, reset coordinates for a new page
            if img_y > 250:
                pdf.add_page()
                img_x, img_y = 10, 30

    # Save the PDF
    pdf.output(output_path)

    # Cleanup: Delete temporary images
    for temp_path in temp_image_paths:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Main Streamlit app
def main():
    st.title("Handwritten Text Recognition with Diagram Detection")
    st.write("Upload a handwritten image, segment it into words and diagrams, predict the text, and save the results to a PDF.")

    model_path = "handwriting_model_2.h5"
    model = load_saved_model(model_path)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Segmenting words..."):
            processed_img, word_boxes = word_segmentation(image)

        # Detect diagrams
        with st.spinner("Detecting diagrams..."):
            drawings = extract_diagrams_and_boxes(image)
            st.write("### Detected Diagrams")
            for i, box in enumerate(drawings):
                st.image(box, caption=f"Diagram {i + 1}", use_container_width=False)

        img_with_boxes = processed_img.copy()
        predictions = []

        with st.spinner("Predicting words..."):
            for word in word_boxes:
                x1, y1, x2, y2 = word
                word_img = processed_img[y1:y2, x1:x2]
                processed_word_img = preprocess_image(word_img)
                prediction = model.predict(processed_word_img)
                predicted_text = decode_prediction(prediction, alphabet)
                predictions.append(predicted_text)

                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_with_boxes, predicted_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        st.image(img_with_boxes, caption="Segmented and Labeled Image", use_container_width=True)

        # Save predictions and diagrams to a PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            save_to_pdf(predictions, drawings, tmp_pdf.name)
            st.success("Prediction Completed successfully!")
            with open(tmp_pdf.name, "rb") as pdf_file:
                st.download_button("Download PDF", pdf_file, file_name="predictions_with_diagrams.pdf")

        st.write("Predicted Words:")
        for i, word in enumerate(predictions):
            st.write(f"Word {i + 1}: **{word}**")

if __name__ == "__main__":
    main()
