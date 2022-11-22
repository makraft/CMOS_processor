"""
Transform data from the Aconity CMOS high speed camera and assign machine
coordinates by utilizing date from the pyrometers.
"""
import csv
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
            l_pyro = layer.split('.')[0].replace("-",".") + '.pcd'
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
    
#    fig, ax = plt.subplots()
#    ax.plot(intensity_array,color="blue")
#    plt.axhline(OFF_threshold, color="green")
#    ax2 = ax.twinx()
#    ax2.plot(signs_array,color="red")
#    plt.show()

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


def process_pcd(file):
    df = pandas.DataFrame()
    with open(file['filename_pyro1'], "r", newline='\n') as pyro_file:
        reader = csv.reader(pyro_file, delimiter=' ')
        scanner_id = next(reader)
        scanner_protocol = next(reader)
        scanner_x_field_size = next(reader)
        scanner_y_field_size = next(reader)
        scanner_x_offset = next(reader)
        scanner_y_offset = next(reader)
        scanner_rotation = next(reader)
        scanner_field_correction_file = next(reader)
        pyro_data = np.array(list(reader)).astype(np.int64)
        assert np.sum(pyro_data[:,2]-pyro_data[:,3]) == 0, "The data in the two pyro value columns does not match"
#        window_width = 10000
#        intensity = np.convolve(pyro_data[:,2], np.ones(window_width), 'valid') / window_width
#        plt.plot(intensity)
#        plt.show()
        # Compute velocity profile
        dt = 1     # time interval for dx/dt, dy/dt
        Dx = np.diff(pyro_data[:,0], dt).astype(np.float)
        Dy = np.diff(pyro_data[:,1], dt).astype(np.float)
        Dx_mm, Dy_mm = bit2mm(Dx, Dy)
        velocity_array_mm = np.linalg.norm(np.stack((Dx_mm,Dy_mm), axis=1), axis=1) / dt
        window_width = 50
        velocity_array_smoothed_mm = np.convolve(velocity_array_mm, np.ones(window_width),mode='same') / window_width
        # Convert velocity from mm*100kHz to mm/s
        velocity_array_smoothed_mmps = velocity_array_smoothed_mm * 1e5
        # Determine vectors based upon hatch speed
        scan_velocity_hatch_mmps = 900 #mm/s
        scan_velocity_contour_mmps = 1200 #mm/s
        # check lower boundary
        signs_array_lower = (np.sign(velocity_array_smoothed_mmps - (scan_velocity_hatch_mmps -200)) + 1)/2
        # check upper boundary
        signs_array_upper = (np.sign(scan_velocity_hatch_mmps + 200 - velocity_array_smoothed_mmps) +1)/2
        initial_high_speed = np.where(signs_array_upper < 1)
        lower = initial_high_speed[0][0]
        upper = initial_high_speed[0][1]
        signs_array_upper[np.arange(lower-50,upper+21,1)] = 0
        signs_array = np.multiply(signs_array_lower,  signs_array_upper)
#        signs_array = (np.sign(velocity_array_smoothed_mmps - (scan_velocity_hatch_mmps -200)) + 
#            np.sign(scan_velocity_hatch_mmps + 200 - velocity_array_smoothed_mmps) - 1)


        fig, ax = plt.subplots()
        ax.plot(velocity_array_smoothed_mmps,color="blue")
        plt.axhline(scan_velocity_hatch_mmps, color="green")
        ax2 = ax.twinx()
        ax2.plot(signs_array,color="red")
        plt.show()

        # Pack data into dataframe and return
        df['x'] = pyro_data[dt:,0]
        df['y'] = pyro_data[dt:,1]
        df['intensity'] = pyro_data[dt:,2]
        df['velocity'] = velocity_array_smoothed_mmps
        df['threshold'] = signs_array
    return df

def display_image(image):
    """
    Display an image. This is a function meant for debugging.
    """
    pylab.imshow(image, cmap="Greys_r", vmin=0, vmax=255)
    pylab.show()

def bit2mm(x, y, fieldsize=600, sl2=True):
    """
    Transform two arrays x,y of distance data from bit to mm
    """
    x_mm = x
    y_mm = y

    if sl2:

        scaling = (float(fieldsize) / 2.0) / 524287.0  # The scaling according to the protocol (20 bits, signed).

        x_mm = - x * scaling
        y_mm = y * scaling

    else:

        scaling = float(fieldsize) / 32768.0  # The scaling according to the protocol (16 bits, signed).

        x_mm = - (x + 16384) * scaling
        y_mm = (y + 16384) * scaling

    return (x_mm, y_mm)

if __name__ == "__main__":
    
    file_list = create_file_list(job_name)
    # Load all CMOS images into an array
    for file in file_list:
        # Fetch and process images form the .mkv file
        if verbal == True:
            print("Started processing: " + file['filename_camera'])
            tstart = time.time()
#        image_df = process_mkv(file)
        if verbal:
            print("Process finished in {0} seconds".format(time.time()-tstart))
        pass

        # todo: (optional) store processed CMOS images

        # Fetch and process pyro data from the .pcd file
        if verbal == True:
            print("Started processing: " + file['filename_pyro1'])
            tstart = time.time()
        pyro1_df = process_pcd(file)
        if verbal:
            print("Process finished in {0} seconds".format(time.time()-tstart))


#        fig, ax = plt.subplots()
#        ax.plot(pyro1_df['threshold'],color="blue")
#        ax2 = ax.twinx()
#        ax2.plot(image_df['threshold'],color="red")
#        plt.show()
        # todo: fit pyro & CMOS data

    # todo: store CMOS image table