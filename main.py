"""
Transform data from the Aconity CMOS high speed camera and assign machine
coordinates by utilizing date from the pyrometers.
"""
import statistics
import imageio
import numpy as np
import os
import pylab
import time
import pandas
import matplotlib.pyplot as plt

# Edit this line to select the job to be processed
job_name = "B002"
# Specify the CMOS camera settings.
Offset_X = 832
Offset_Y = 808
Width = 256
Height = 300

# Specify settings for evaluating the main script
# The script prints what it's doing
verbal = True
# Set limit to reducing computing time for image processing. Default = None
image_number_limit =None

def create_file_list(job):
    """
    Creates a list of dictionaries with data of and paths to individual part 
    files, based on the CMOS data.
    """
    file_list = []
    filename_camera_prefix = "Jobs/{0}/hsCamera/".format(job)
    filename_pyro1_prefix = "Jobs/{0}/2Pyrometer/pyrometer1/".format(job)
    filename_pyro2_prefix = "Jobs/{0}/2Pyrometer/pyrometer2/".format(job)
    part_number_list = os.listdir(filename_camera_prefix)
    for part_number in part_number_list:
        layer_list = os.listdir(filename_camera_prefix + part_number)
        for layer in layer_list:
            filename_camera = filename_camera_prefix + part_number + "/" + layer
            l_pyro = layer.split('.')[0] + '.pcd'
            filename_pyro1 = filename_pyro1_prefix + part_number + "/" + l_pyro
            filename_pyro2 = filename_pyro2_prefix + part_number + "/" + l_pyro
            file_dict ={'filename_camera':filename_camera,
                'filename_pyro1': filename_pyro1,
                'filename_pyro2': filename_pyro2,
                'part_number':part_number,
                'layer':layer.split('.')[0]}
            file_list.append(file_dict)
    return(file_list)

def process_mkv(file):
    """
    Process a .mkv file into a dataframe.
    """
    image_array = []
    image_index_array = []
    intensity_array = []


    CMOS_video = imageio.get_reader(file['filename_camera'], 'ffmpeg')
    # Image borders where the melt pool is situated:
    x_min = 850
    x_max = 950
    y_min = 810
    y_max = 910
    # Set a threshhold value to prevent melt pools from being classified as
    # noise.
    noise_threshold_value = 23
    # Create an empty noise mask which is subtracted from all pictures.
    noise_mask = np.full([y_max-y_min,x_max-x_min],0)
    noise_picture_total = 0

    for frame_number, image in enumerate(CMOS_video, start=1):
        if image_number_limit is not None and frame_number > image_number_limit:
            break
        # Convert image to black and white
        image_bw = image[:,:,0].reshape(image.shape[0],image.shape[1])
        #display_image(image_bw)

        # Crop image to relevant area
        image_cropped = image_bw[
            (y_min - Offset_Y):(y_max - Offset_Y),
            (x_min - Offset_X):(x_max - Offset_X)
        ]
        # Update noise_mask
        if np.amax(image_cropped) < noise_threshold_value:
            noise_mask += image_cropped
            noise_picture_total +=1

        image_array.append(image_cropped)
        image_index_array.append(frame_number)

    # average noise mask
    noise_mask = np.divide(noise_mask,
        np.full(noise_mask.shape,noise_picture_total))

    for index, image in enumerate(image_array):
        # denoise images
        image_min_noise = (image - noise_mask)
        # convert negative pixels to 0
        image_min_noise[image_min_noise < 0] = 0
        image_array[index] = image_min_noise.astype(np.uint8)
        # calculate total intensity
        intensity_array.append(np.sum(image_array[index]))
    intensity_array = np.array(intensity_array, dtype=np.int64)


    # Calculate upper threshold value for Laser OFF intensity level
    # Note: the first value usually is zero
    OFF_level_array = np.array(intensity_array[1:21])
    mean = statistics.mean(OFF_level_array)
    OFF_threshold = mean + 6 * statistics.pstdev(OFF_level_array,mean)
    if verbal:
        print("Threshold value = " + str(OFF_threshold))
    
    # Determine if below or above threshold
    signs_array = np.sign(intensity_array - OFF_threshold)
    pass
    
    fig, ax = plt.subplots()
    ax.plot(intensity_array,color="blue")
    plt.axhline(OFF_threshold, color="green")
    ax2 = ax.twinx()
    ax2.plot(signs_array,color="red")
    plt.show()

    df = pandas.DataFrame(index=image_index_array)
    df['image'] = image_array
    df['intensity'] = intensity_array
    df['threshold'] = signs_array
    df['part'] = file['part_number']
    df['layer'] = file['layer']
#    a = df["intensity"].rolling(5).mean()
#    plt.plot(a)
#    plt.show()
    return df

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
        # Fetch and process images form the .mkv file
        if verbal == True:
            print("Started processing: " + file['filename_camera'])
            tstart = time.time()
        image_df = process_mkv(file)
        if verbal:
            print("Process finished in {0} seconds".format(time.time()-tstart))
        pass

        # todo: (optional) store processed CMOS images

        # todo: process pyro data

        # todo: fit pyro & CMOS data

    # todo: store CMOS image table