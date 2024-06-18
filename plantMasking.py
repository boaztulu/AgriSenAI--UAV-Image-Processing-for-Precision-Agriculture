'''
create three folders as: thermal, rgb and output
thermal is for thermal images, rgb is for rgb images and ouput is for the output
Save the same the rgb image and thermal image with the same name and move it into rgb and thermal folder
'''
import os
import rasterio
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.pyplot as plt

# Function to read a single band from the image
class PlantDetection:
    def __init__(self,rgbpath,thermalpath):
        self.rgbPath = rgbpath
        self.thermalPath = thermalpath

    def read_band(self, image_path, band_index):
        with rasterio.open(image_path) as src:
            band = src.read(band_index)
        return band

    def plantDetection_function(self):
        # Define paths to the folders containing images
        rgb_folder = self.rgbPath  # Rgb path
        thermal_folder = self.thermalPath  # thermal path
        output_folder = os.path.join(os.path.dirname(rgb_folder), "Plant_Detection_Result")
        output_folder.replace('\\', '/')
        os.makedirs(output_folder)
        ##previues or data one outp.r'C:\Users\analm\Documents\ther rgb\comparation\first data\align\step 3\Jan 31' #output path

        # Get a list of files in both folders
        rgb_images = os.listdir(rgb_folder)
        thermal_images = os.listdir(thermal_folder)

        # Iterate through images in both folders
        for i in range(min(len(rgb_images), len(thermal_images))):
            # Construct full paths for each pair of corresponding images
            rgb_image_path = os.path.join(rgb_folder, rgb_images[i])
            print(f'Image {i} : {rgb_image_path}')
            thermal_image_path = os.path.join(thermal_folder, thermal_images[i])

            if rgb_image_path.lower().endswith('.tif') and thermal_image_path.lower().endswith('.tif'):
                # Read red and green bands from RGB image
                red_band = self.read_band(rgb_image_path, 1)  # Adjust the band index if needed
                green_band = self.read_band(rgb_image_path, 2)  # Adjust the band index if needed

                # Calculate RGRI
                epsilon = 1e-8
                RGRI = (red_band.astype(float) + epsilon) / (green_band.astype(float) + epsilon)

                # Normalize RGRI to [0, 1] range for display purposes
                RGRI_normalized = (RGRI - np.min(RGRI)) / (np.max(RGRI) - np.min(RGRI))

                # Apply a clipping threshold to RGRI to remove extreme values
                clip_percentile = 99
                clip_value = np.percentile(RGRI, clip_percentile)
                RGRI_clipped = np.clip(RGRI, a_min=None, a_max=clip_value)

                # Normalize RGRI after clipping
                RGRI_normalized = cv2.normalize(RGRI_clipped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_32F)

                # Apply Otsu's thresholding to the normalized RGRI image
                otsu_thresh = threshold_otsu(RGRI_normalized)
                binary_mask = RGRI_normalized > otsu_thresh

                # Invert the mask: vegetation becomes white (True), and non-vegetation becomes black (False)
                inverted_binary_mask = np.logical_not(binary_mask)

                # Load the thermal image to get its dimensions and transform
                with rasterio.open(thermal_image_path) as thermal_src:
                    thermal_width = thermal_src.width
                    thermal_height = thermal_src.height

                # Perform nearest-neighbor resampling on the inverted binary mask
                inverted_binary_mask_2d = inverted_binary_mask.squeeze()

                # Resample the binary mask to match the thermal image's resolution
                resampled_fvc = resize(inverted_binary_mask_2d,
                                       (thermal_height, thermal_width),
                                       order=0,  # Nearest-neighbor interpolation
                                       preserve_range=True,
                                       anti_aliasing=False).astype(bool)

                # Load the thermal image data
                with rasterio.open(thermal_image_path) as thermal_src:
                    thermal_data = thermal_src.read(1)  # Assuming there is only one band
                    nodata = thermal_src.nodata  # Get the nodata value from the thermal image

                # Replace the nodata value with NaN before the multiplication
                thermal_data = thermal_data.astype('float')
                thermal_data[thermal_data == nodata] = np.nan

                # Ensure that both the resampled FVC and thermal data have the same shape
                if resampled_fvc.shape != thermal_data.shape:
                    raise ValueError("The shapes of the resampled FVC and thermal data do not match.")

                # Perform the multiplication
                tc_map = np.where(np.isnan(thermal_data), np.nan, thermal_data * resampled_fvc)

                # Save the resulting Tc map, handling NaN values correctly
                with rasterio.open(thermal_image_path) as src:
                    profile = src.profile
                    profile.update(nodata=np.nan, dtype=rasterio.float32)

                tc_map_output_path = os.path.join(output_folder, f"tc_{os.path.splitext(rgb_images[i])[0]}.tif")
                with rasterio.open(tc_map_output_path, 'w', **profile) as dst:
                    dst.write(tc_map, 1)

        return 'All Process Finished Successfully',output_folder
