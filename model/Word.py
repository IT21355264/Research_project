import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh

def word_segmentation(image):
    # Resize if necessary
    h, w, _ = image.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Thresholding
    thresh_img = thresholding(image)

    # Line segmentation
    kernel_line = np.ones((3, 85), np.uint8)
    dilated_line = cv2.dilate(thresh_img, kernel_line, iterations=1)
    contours, _ = cv2.findContours(dilated_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # Sort by Y-coordinate

    # Word segmentation
    kernel_word = np.ones((3, 15), np.uint8)
    dilated_word = cv2.dilate(thresh_img, kernel_word, iterations=1)
    words_list = []

    for line in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_word[y:y+h, x:x+w]
        cnt, _ = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])  # Sort by X-coordinate

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:  # Filter small regions
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])

    return image, words_list

# Streamlit Interface
st.title("Word Segmentation App")
st.write("Upload an image, and the app will segment the words and display them.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Perform word segmentation
    processed_img, word_boxes = word_segmentation(image)

    # Draw rectangles around words and display segmented words
    img_with_boxes = processed_img.copy()
    for word in word_boxes:
        x1, y1, x2, y2 = word
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)

    st.image(img_with_boxes, caption="Image with Word Boxes", use_container_width=True)

    st.write("Segmented Words:")
    for i, word in enumerate(word_boxes):
        x1, y1, x2, y2 = word
        word_img = processed_img[y1:y2, x1:x2]
        st.image(word_img, caption=f"Word {i + 1}")
