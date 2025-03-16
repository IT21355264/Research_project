from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Function to upload an image
def ocr_from_image(image_path):
    # Load the uploaded image
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

    return generated_text

# Upload an image (replace 'your_image.jpg' with the actual image path)
image_path = "handwritten.jpg"  # Change this to your image file
recognized_text = ocr_from_image(image_path)

print("Recognized Text:", recognized_text)
