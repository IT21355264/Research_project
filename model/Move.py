import os
import random
import shutil

# Function to copy a random subset of image files
def copy_random_images(source_folder, destination_folder, num_images_to_copy):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get a list of all image files in the source folder
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")  # Add more extensions as needed
    all_files = [file for file in os.listdir(source_folder) 
                 if file.lower().endswith(valid_extensions) 
                 and os.path.isfile(os.path.join(source_folder, file))]

    # Check if the number of requested files exceeds the available files
    if num_images_to_copy > len(all_files):
        raise ValueError(f"Requested {num_images_to_copy} files, but only {len(all_files)} are available.")

    # Randomly select the specified number of files
    selected_files = random.sample(all_files, num_images_to_copy)

    # Copy each selected file to the destination folder
    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        try:
            shutil.copy(source_path, destination_path)
            print(f"Copied: {file}")
        except Exception as e:
            print(f"Failed to copy {file}: {e}")

    print(f"Successfully copied {len(selected_files)} images to {destination_folder}")

# Parameters
source_folder = r"G:\GitHub\Handwritten_Image_Recognition\model\images\train_v2\train"  # Replace with the absolute path to your source folder
destination_folder = r"G:\GitHub\Handwritten_Image_Recognition\model\images\train"  # Replace with the absolute path to your destination folder
num_images_to_copy = 100000  # Number of images to copy

# Execute the function
try:
    copy_random_images(source_folder, destination_folder, num_images_to_copy)
except Exception as e:
    print(f"Error: {e}")
