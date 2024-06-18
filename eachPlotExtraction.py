import json
import sys

import arcpy
import os

# Set the workspace (where your raster images are stored)


# Define coordinates for each plot
#this is for 2020/2021 Data
def plotextrting(iputPath, cordinatePath):
    arcpy.env.workspace = iputPath
    base_output_workspace = os.path.dirname(iputPath)
    base_output_workspace = os.path.join(base_output_workspace, 'ExtactedPlot')
    os.makedirs(base_output_workspace)

    #loading the coordinate points from json file
    with open(cordinatePath, 'r') as json_file:
        plot_coordinates = json.load(json_file)
    '''
    plot_coordinates =  {
        'Plot1': (-80.500547, 25.513289, -80.500441, 25.513240),
        'Plot2': (-80.500700, 25.513286, -80.500594, 25.513236),
        'Plot3': (-80.500853, 25.513284, -80.500747, 25.513234),
        'Plot4': (-80.501004, 25.513283, -80.500898, 25.513233),
        'Plot5': (-80.501003, 25.513210, -80.500898, 25.513160),
        'Plot6': (-80.500852, 25.513211, -80.500747, 25.513161),
        'Plot7': (-80.500700, 25.513212, -80.500595, 25.513162),
        'Plot8': (-80.500546, 25.513213, -80.500440, 25.513163),
        'Plot9': (-80.500546, 25.513140, -80.500440, 25.513090),
        'Plot10': (-80.500699, 25.513140, -80.500593, 25.513090),
        'Plot11': (-80.500850, 25.513137, -80.500744, 25.513087),
        'Plot12': (-80.501003, 25.513135, -80.500898, 25.513085),
        'Plot13': (-80.501003, 25.513062, -80.500898, 25.513011),
        'Plot14': (-80.500849, 25.513063, -80.500744, 25.513012),
        'Plot15': (-80.500699, 25.513064, -80.500594, 25.513014),
        'Plot16': (-80.500549, 25.513065, -80.500443, 25.513015),
        'Plot17': (-80.500545, 25.512993, -80.500440, 25.512942),
        'Plot18': (-80.500699, 25.512992, -80.500594, 25.512941),
        'Plot19': (-80.500849, 25.512988, -80.500743, 25.512939),
        'Plot20': (-80.501002, 25.512988, -80.500896, 25.512937),
        'Plot21': (-80.501002, 25.512913, -80.500897, 25.512863),
        'Plot22': (-80.500845, 25.512915, -80.500740, 25.512864),
        'Plot23': (-80.500700, 25.512917, -80.500595, 25.512867),
        'Plot24': (-80.500543, 25.512918, -80.500438, 25.512868),
        # Add the coordinates for all 24 plots
    }'''

    # Loop through each raster in the workspace
    for raster in arcpy.ListRasters():
        # Extract the date from the raster filename
        date_part = raster.split('_')[1]  # Adjust this as per your filename format

        # Create a new directory for each date if it doesn't exist
        date_output_workspace = os.path.join(base_output_workspace, date_part)
        if not os.path.exists(date_output_workspace):
            os.makedirs(date_output_workspace)

        # Loop through each plot and crop
        for plot_name, coords in plot_coordinates.items():
            xmin, ymin, xmax, ymax = coords
            #print("Plot Cropping for  " + str(plot_name))

            # Define output path with date and plot number
            output_raster = os.path.join(date_output_workspace, "{}_{}_{}".format(date_part, plot_name, raster))

            # Perform the cropping
            arcpy.Clip_management(raster, "{} {} {} {}".format(xmin, ymin, xmax, ymax), output_raster, "#", "#", "NONE")

    ouputFolder = date_output_workspace.replace('\\','/')

    return ouputFolder


if __name__=="__main__":
    inputpath = sys.argv[1]
    cordinatePath = sys.argv[2]
    outputpath = plotextrting(inputpath, cordinatePath)
    print(outputpath)

