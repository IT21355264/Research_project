from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Define the directory where the model is saved
save_directory = "./trocr_model"

# Load the processor and model from local storage
processor = TrOCRProcessor.from_pretrained(save_directory)
model = VisionEncoderDecoderModel.from_pretrained(save_directory)

# Load an image (Replace with your own image path)
image_path = "handwritten.jpg"  # Change this to your image file
image = Image.open(image_path).convert("RGB")

# Preprocess the image
pixel_values = processor(image, return_tensors="pt").pixel_values

# Ensure model is in evaluation mode
model.eval()

# Generate text prediction
with torch.no_grad():
    generated_ids = model.generate(pixel_values)

# Decode the predicted text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Recognized Text:", generated_text)
