"""
Transform data from the Aconity CMOS high speed camera and assign machine
coordinates by utilizing date from the pyrometers.
"""
import imageio
import numpy as np
import os


# Edit this line to select the job to be processed
job_name = "A103"


if __name__ == "__main__":
    
    # todo: Load all CMOS images into an array
    filename_prefix = "Jobs/{0}/hsCamera/".format(job_name)
    part_number_list = os.listdir(filename_prefix)
    for part_number in part_number_list:
        layer_list = os.listdir(filename_prefix + part_number)
        for layer in layer_list:
            filename = filename_prefix + part_number + "/" + layer
            CMOS_video = imageio.get_reader(filename, 'ffmpeg')
            for frame_number, image in enumerate(CMOS_video, start=1):
                pass

    # todo: Transform and process CMOS images

    # todo: (optional) store processed CMOS images

    # todo: load pyro data into array

    # todo: process pyro data

    # todo: fit pyro & CMOS data

    # todo: store CMOS image table
    pass