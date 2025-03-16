import os
from google.cloud import vision
from PIL import Image

# Set Google Application Credentials (Update the path to your JSON key file)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "llmtesting-242512-55d0854ea838.json"

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

# Function to perform OCR
def ocr_google_vision(image_path):
    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    # Prepare image for OCR
    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        extracted_text = texts[0].description
    else:
        extracted_text = "No text detected."

    return extracted_text

# Upload an image (replace 'your_image.jpg' with actual image path)
image_path = "handwritten.jpg"  # Change this to your image file
recognized_text = ocr_google_vision(image_path)

print("Recognized Text:", recognized_text)
