import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Load the saved model
@st.cache_resource
def load_saved_model(model_path):
    return load_model(model_path, compile=False)

# Preprocess the uploaded image
def preprocess_image(image, img_size=(256, 64)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize and pad the image
    (h, w) = gray.shape
    final_img = np.ones((img_size[1], img_size[0])) * 255  # White background
    if w > img_size[0]:
        gray = gray[:, :img_size[0]]  # Crop width
    if h > img_size[1]:
        gray = gray[:img_size[1], :]  # Crop height
    final_img[:gray.shape[0], :gray.shape[1]] = gray  # Place the image on the white canvas
    
    # Rotate and normalize
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE) / 255.0
    return np.expand_dims(final_img, axis=(0, -1))

# Decode the prediction to text
def decode_prediction(prediction, alphabet):
    decoded_text = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1])[0][0]
    decoded_text = K.get_value(decoded_text)
    return ''.join([alphabet[i] for i in decoded_text[0] if i != -1])  # Exclude CTC blank index (-1)

# Streamlit app
def main():
    st.title("Handwritten Word Recognition")
    st.write("Upload an image of a handwritten word, and the app will predict the text.")

    # Load the saved model
    model_path = "handwriting_model_2.h5"
    model = load_saved_model(model_path)

    # Define the alphabet used during training
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image and make predictions
        with st.spinner("Predicting..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_text = decode_prediction(prediction, alphabet)
        
        # Display the predicted text
        st.success(f"Predicted Text: **{predicted_text}**")

if __name__ == "__main__":
    main()
