# made by ChatGPT
import os
from PIL import Image

# Input directory containing the JPG files
input_directory = 'faces'  # Change to your directory path
output_directory = 'faces2'  # Change to your output directory path

# Make sure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_directory):
    # Check if the file is a JPG image
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        # Load the image
        image_path = os.path.join(input_directory, filename)
        image = Image.open(image_path)
        
        # Resize to 32x32
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Save as PGM file with the same name but different extension
        new_filename = os.path.splitext(filename)[0] + ".pgm"
        output_path = os.path.join(output_directory, new_filename)
        
        # Save in PGM format
        image.save(output_path, "PPM")