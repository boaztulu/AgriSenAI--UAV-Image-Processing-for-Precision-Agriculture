import os
import numpy as np
import rasterio
from scipy.ndimage import median_filter, binary_fill_holes
from skimage import morphology

class NoiseRemove:
    def __init__(self, path):
        self.filePath = path

    def process_image(self, image_path, output_path):
        with rasterio.open(image_path) as src:
            thermal_image = src.read(1)  # Read the first band
            profile = src.profile

        # Apply a median filter to reduce noise
        noise_reduced_image = median_filter(thermal_image, size=3)

        # Use a threshold to create a binary image if necessary
        # Here you could use a specific threshold value that you know is suitable for your images
        # For now, I'll just create a binary image with all non-zero values
        binary_image = noise_reduced_image > 0

        # Optionally remove small objects from the binary image
        cleaned_binary = morphology.remove_small_objects(binary_image, min_size=450)

        # Optionally fill holes in the binary image
        filled_image = binary_fill_holes(cleaned_binary)

        # Create a masked thermal image using the binary image as a mask
        masked_thermal_image = np.where(filled_image, noise_reduced_image, 0)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(masked_thermal_image, 1)

    def removeNoise(self):
        try:
            # Directory containing the thermal images
            input_dir = self.filePath
            output_dir = os.path.dirname(input_dir)
            output_dir = os.path.join(output_dir, 'Noise_Removal Result')
            os.makedirs(output_dir)

            # Process each file in the directory
            for filename in os.listdir(input_dir):
                if filename.endswith('.tif'):
                    image_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, f'noise_reduced_{filename}')
                    self.process_image(image_path, output_path)
            return 'Process Finished Successfully!', os.path.dirname(output_path)
        except Exception as e:
            return f"{e}\nPlease, Use proper files and steps", ''
