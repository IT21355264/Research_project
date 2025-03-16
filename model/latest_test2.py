import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"  # Change the URL if your Flask app is hosted elsewhere

# Path to the test image
image_path = "img056-010.png"  # Change this to the path of your test image

# Open the image and send it as a file
with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

# Print the response
if response.status_code == 200:
    print("Prediction:", response.json()["predicted_character"])
else:
    print("Error:", response.text)
