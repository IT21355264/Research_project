import requests

# Define the Flask app URL (change if hosted remotely)
URL = "http://127.0.0.1:5002/segment_and_recognize"

# Path to the test image
IMAGE_PATH = "Notes1.jpg"  # Change this to your test image file

# Send a request with the image file
def test_segmentation_and_recognition():
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(URL, files=files)
        
        # Print the response from the server
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Error:", response.status_code, response.json())
    except Exception as e:
        print("Request failed:", str(e))

if __name__ == "__main__":
    test_segmentation_and_recognition()
