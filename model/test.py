import requests

url = 'http://127.0.0.1:5000/predict'
image_path = 'char_1.png'

with open(image_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print('Predicted Label:', response.json().get('predicted_label'))
else:
    print('Error:', response.json().get('error'))
