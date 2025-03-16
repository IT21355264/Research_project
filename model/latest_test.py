import requests
import os

# Flask API endpoint
API_URL = "http://127.0.0.1:5000/segment_and_recognize"

# Test image path
TEST_IMAGE_PATH = "handwritten.jpg"  # Replace with your actual test image

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Error: Test image '{TEST_IMAGE_PATH}' not found.")
    exit()

# Open the image file in binary mode
with open(TEST_IMAGE_PATH, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(API_URL, files=files)

# Process the response
if response.status_code == 200:
    result = response.json()
    print("Recognized Text:", result.get("recognized_text", "No text recognized"))
    print("PDF Path:", result.get("pdf_path", "No PDF generated"))
else:
    print(f"Error: {response.status_code}")
    print(response.text)