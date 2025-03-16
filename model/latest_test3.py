import requests

url = "http://127.0.0.1:5000/segment_and_recognize"
image_path = "handwritten.jpg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())  # Output recognized text & PDF path
