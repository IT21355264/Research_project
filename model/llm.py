from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the pre-trained model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Define the directory where you want to save the model
save_directory = "./trocr_model"

# Save the model and processor locally
processor.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and processor saved to {save_directory}")
