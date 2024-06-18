
import csv
import os
import rasterio

class ExtractData:
    def __init__(self, path):
        self.inputPath =  path

    def extract_csv(self):
        try:
            # Path to the folder containing input thermal image files
            input_folder = self.inputPath

            # Create the output folder if it doesn't exist
            output_folder = os.path.join(os.path.dirname(input_folder), 'CSV output')
            os.makedirs(output_folder, exist_ok=True)

            # Loop through all files in the input folder
            for filename in os.listdir(input_folder):
                if filename.endswith('.tif') or filename.endswith('.tiff'):
                    # Construct the paths for input and output files
                    input_image_path = os.path.join(input_folder, filename)
                    image_name = os.path.splitext(os.path.basename(input_image_path))[0]
                    output_csv_path = os.path.join(output_folder, f'{image_name}_csv.csv')

                    # Process each thermal image file
                    with rasterio.open(input_image_path) as src:
                        # Read pixel values (assuming a single-band thermal image)
                        thermal_data = src.read(1)

                        # Open a CSV file to write temperature values
                        with open(output_csv_path, 'w', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)

                            # Write header
                            csv_writer.writerow(['Row', 'Column', 'Temperature'])

                            # Loop through all pixels and write their temperature values to the CSV file
                            for row_idx in range(thermal_data.shape[0]):
                                for col_idx in range(thermal_data.shape[1]):
                                    temperature_value = thermal_data[row_idx, col_idx]
                                    csv_writer.writerow([row_idx, col_idx, temperature_value])

            return "Extracted Successfully!",f'Saved in : { output_folder} file path'
        except Exception as e:
            return f'{e} \n Please! Use the correct files and steps', 'file not saved'