"""
Transform data from the Aconity CMOS high speed camera and assign machine
coordinates by utilizing date from the pyrometers.
"""
import imageio
import numpy as np
import os
import pylab


# Edit this line to select the job to be processed
job_name = "A103"
# Specify the CMOS camera settings.
Offset_X = 832
Offset_Y = 808
Width = 288
Height = 300

def crop_CMOS_image(image):
    return None

def create_file_list(job):
    """
    Creates a list of paths to individual part files, based on the CMOS data.
    """
    file_list = []
    filename_prefix = "Jobs/{0}/hsCamera/".format(job)
    part_number_list = os.listdir(filename_prefix)
    for part_number in part_number_list:
        layer_list = os.listdir(filename_prefix + part_number)
        for layer in layer_list:
            filename = filename_prefix + part_number + "/" + layer
            file_list.append(filename)
    return(file_list)

def create_image_array_from_mkv(filename):
    """
    Process a .mkv file into an array of images. Also reduce the size of the
    image to reduce storage requirements.
    """
    CMOS_video = imageio.get_reader(filename, 'ffmpeg')
    for frame_number, image in enumerate(CMOS_video, start=1):
        # Convert image to black and white
        image_bw = image[:,:,0].reshape(image.shape[0],image.shape[1])
        #display_image(image_bw)

        # Crop image to relevant area

def display_image(image):
    """
    Display an image. This is a function meant for debugging.
    """
    pylab.imshow(image, cmap="Greys_r", vmin=0, vmax=255)
    pylab.show()

if __name__ == "__main__":
    
    file_list = create_file_list(job_name)
    # todo: Load all CMOS images into an array
    for file in file_list:
        # Load CMOS images into array
        create_image_array_from_mkv(file)

    # todo: Transform and process CMOS images

    # todo: (optional) store processed CMOS images

    # todo: load pyro data into array

    # todo: process pyro data

    # todo: fit pyro & CMOS data

    # todo: store CMOS image table