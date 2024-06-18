import sys

import arcpy
import os





# Define coordinates for each plot

def fildExtract(inpFolder,x1,y1,x2,y2):
    try:
        # Set the workspace (where your raster images are stored)
        arcpy.env.workspace = inpFolder
        base_output_workspace = os.path.dirname(inpFolder)  # saving in the parent directory of the input
        # Define coordinates for each plot
        plot_coordinates = {
            'Field': (x1, y1, x2, y2),
            # Add the coordinates for all 24 plots
        }
        '''
        plot_coordinates = {
            'Field': (-80.501035, 25.513385, -80.500408, 25.512777),
            # Add the coordinates for all 24 plots
        }'''

        # Loop through each raster in the workspace
        for raster in arcpy.ListRasters():
            # Extract the date from the raster filename
            # Adjust the following line according to your filename format
            # Example: for a filename like 'image_01-05-22.tif', use raster.split('_')[1]
            date_part = raster.split('_')[1]  # Adjust this as per your filename format

            # Create a new directory for each date
            date_output_workspace = os.path.join(base_output_workspace, date_part)  # the save directory
            if not os.path.exists(date_output_workspace):
                os.makedirs(date_output_workspace)

            # Loop through each plot and crop
            for plot_name, coords in plot_coordinates.items():
                xmin, ymin, xmax, ymax = coords
                # print("Plot Cropping for  " + str(plot_name))

                # Define output path with date and plot number
                output_raster = os.path.join(date_output_workspace, "{}_{}_{}".format(date_part, plot_name, raster))

                # Perform the cropping
                arcpy.Clip_management(raster, "{} {} {} {}".format(xmin, ymin, xmax, ymax), output_raster, "#", "#",
                                      "NONE")

        outputdirectory = date_output_workspace.replace('\\', '/')
        return outputdirectory
    except Exception as e:
        return 'Not Processed. Please Select proper file and step'


if __name__ == "__main__":
    # Extract the parameters passed from main.py
    inpFolder = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Call the function with the parameters

    outputPath = fildExtract(inpFolder,x1,y1,x2,y2)
    print(outputPath)



