import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """Convert image to grayscale, apply Gaussian blur, and adaptive thresholding."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return image, gray, thresh

def detect_drawings(thresh):
    """Detect edges using Canny and find large contours that likely represent drawings."""
    edges = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    drawing_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Filter out small noise
    return drawing_contours

def extract_individual_sketches(image, contours, output_folder):
    """Extract and save each detected drawing as an individual image."""
    sketches_folder = os.path.join(output_folder, "sketches")
    os.makedirs(sketches_folder, exist_ok=True)  # Create folder if not exists

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        sketch = image[y:y+h, x:x+w]  # Crop detected sketch
        sketch_path = os.path.join(sketches_folder, f"sketch_{i+1}.jpg")
        cv2.imwrite(sketch_path, sketch)

def save_images(image, thresh, output, sketches_only, output_folder):
    """Save processed images into the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "thresholded_image.jpg"), thresh)
    cv2.imwrite(os.path.join(output_folder, "detected_drawings.jpg"), output)
    cv2.imwrite(os.path.join(output_folder, "extracted_sketches.jpg"), sketches_only)

def main(image_path, output_folder="output"):
    """Main function to process image, detect sketches, and save results."""
    # Step 1: Preprocess Image
    image, gray, thresh = preprocess_image(image_path)

    # Step 2: Detect Drawings
    drawing_contours = detect_drawings(thresh)

    # Step 3: Draw detected sketches on a copy of the image
    output = image.copy()
    cv2.drawContours(output, drawing_contours, -1, (0, 255, 0), 2)  # Draw contours in green

    # Step 4: Remove text, keep only sketches
    mask = np.ones_like(gray) * 255
    for cnt in drawing_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Remove text areas
    sketches_only = cv2.bitwise_and(image, image, mask=mask)

    # Step 5: Save Processed Images
    save_images(image, thresh, output, sketches_only, output_folder)

    # Step 6: Extract & Save Individual Sketches
    extract_individual_sketches(image, drawing_contours, output_folder)

    print(f"Processed images and individual sketches saved in '{output_folder}'.")

# Run the script
image_path = 'Notes1.jpg'  # Change this to your image path
main(image_path)
