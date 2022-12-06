"""
Transform data from the Aconity CMOS high speed camera and assign machine
coordinates by utilizing data from the pyrometers.
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
import pickle

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
# Plots are generated and shown for intermediate results
visual = [True,True,True,True,True]
# 0: threshold plot camera
# 1: threshold plot pyrometer1
# 2: combined threshold plot
# 3: scatter plot of intensity @x,y position
# 4: scatter plot of ON/OFF @x,y position

# Tell program if it should only process one selected part/layer combination
# Set True or False
cherrypick = True
# If set to true, specify which one
cherry = {
    "part" : "17",
    "layer": "0-06"
}

# Set limit to reduce computing time for image processing. Default = None
image_number_limit =None


def create_file_list(job, **kwargs):
    """
    Creates a list of dictionaries with data of and paths to individual part 
    files, based on the CMOS data.
    """
    file_list = []
    filename_camera_prefix = "Jobs/{0}/hsCamera/".format(job)
    filename_pyro1_prefix = "Jobs/{0}/2Pyrometer/pyrometer1/".format(job)
    filename_pyro2_prefix = "Jobs/{0}/2Pyrometer/pyrometer2/".format(job)

    # Check if only cherry picked files are requested or all job files
    if kwargs.get('cherry', None) is not None:
        part_number = kwargs.get("cherry")["part"]
        layer = kwargs.get("cherry")["layer"]
        filename_camera = filename_camera_prefix + part_number + "/" + layer + ".mkv"
        l_pyro = layer.split('.')[0].replace("-",".") + '.pcd'
        filename_pyro1 = filename_pyro1_prefix + part_number + "/" + l_pyro
        filename_pyro2 = filename_pyro2_prefix + part_number + "/" + l_pyro
        file_dict ={'filename_camera':filename_camera,
            'filename_pyro1': filename_pyro1,
            'filename_pyro2': filename_pyro2,
            'part_number':part_number,
            'layer':layer.split('.')[0]}
        file_list.append(file_dict)
    else:
        part_number_list = os.listdir(filename_camera_prefix)
        # Create a dictionary with file information for every single file
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
    meltpool_area_array = []
    image_area_array = []


    CMOS_video = imageio.get_reader(file['filename_camera'], 'ffmpeg')
    # Image borders where the melt pool is situated:
    x_min = 850
    x_max = 950
    y_min = 810
    y_max = 910
    # Set a threshhold value to prevent melt pools from being classified as
    # noise.
    noise_threshold_value = 23
    meltpool_threshold_value = 23
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

        # calculate melt pool area
        image_area = image_min_noise
        image_area[image_area< meltpool_threshold_value] = 0
        image_area[image_area>= meltpool_threshold_value] = 1
        meltpool_area = np.sum(image_area)
        meltpool_area_array.append(meltpool_area)
        # compute a melt pool image based on the area calculation
        meltpool_image = image_area * 255
        image_area_array.append(meltpool_image)

    intensity_array = np.array(intensity_array, dtype=np.int64)


    # Calculate upper threshold value for Laser OFF intensity level
    # Note: the first value usually is zero
    OFF_level_array = np.array(intensity_array[1:21])
    mean = statistics.mean(OFF_level_array)
    OFF_threshold = mean + 6 * statistics.pstdev(OFF_level_array,mean)
    if verbal:
        print("Threshold value = " + str(OFF_threshold))
    
    # Determine if below or above threshold
    signs_array = (np.sign(intensity_array - OFF_threshold) +1)/2
    
    # store processed images as a python pickle file
    df_images = pandas.DataFrame(index=image_index_array)
    df_images['image'] = image_array
    df_images['image_area'] = image_area_array
    filename_pkl = file['filename_camera'].replace(".mkv",".pkl")
    df_images.to_pickle(filename_pkl)
    
    # create dataframe that is returned
    df = pandas.DataFrame(index=image_index_array)
    df['intensity'] = intensity_array
    df['meltpool_area'] = meltpool_area_array
    df['ON_OFF'] = signs_array
    df['threshold'] = OFF_threshold
    df['part'] = file['part_number']
    df['layer'] = file['layer']
    df['index'] = image_index_array
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
        window_width = 20
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
        try:
            lower = initial_high_speed[0][0]
            upper = initial_high_speed[0][1]
            signs_array_upper[np.arange(lower-50,upper+21,1)] = 0
            signs_array = np.multiply(signs_array_lower,  signs_array_upper)
        except:
            print("No initial high velocity in pyrometer data detected.")
            # todo: properly handle this case
#        signs_array = (np.sign(velocity_array_smoothed_mmps - (scan_velocity_hatch_mmps -200)) + 
#            np.sign(scan_velocity_hatch_mmps + 200 - velocity_array_smoothed_mmps) - 1)

        # Pack data into dataframe and return
        df['x'] = pyro_data[dt:,0]
        df['y'] = pyro_data[dt:,1]
        df['intensity'] = pyro_data[dt:,2]
        df['velocity'] = velocity_array_smoothed_mmps
        df['ON_OFF'] = signs_array
        df['threshold'] = scan_velocity_hatch_mmps
    return df


def extend_CMOS_data(df_camera, df_pyro):
    """
    Extend the CMOS camera dataframe with coordinates for each image, derived
    from the pyrometer data.
    """
    camera_ON = np.where(df_camera['ON_OFF'] == 1)
    camera_ON_start = camera_ON[0][0]
    camera_ON_end = camera_ON[0][-1]
    pyro_ON = np.where(df_pyro['ON_OFF'] == 1)
    pyro_ON_start = pyro_ON[0][0]
    pyro_ON_end = pyro_ON[0][-1]
    # compute linear scaling factors
    slope = (pyro_ON_end - pyro_ON_start)/(camera_ON_end - camera_ON_start)
    offset = pyro_ON_end - slope * camera_ON_end
    df_camera['index_pyro'] = df_camera['index'] * slope + offset
    df_camera['index_pyro'] = df_camera['index_pyro'].round()
    x_array = []
    y_array = []
    for index_pyro in df_camera['index_pyro']:
        x_array.append(df_pyro.at[int(index_pyro),'x'],)
        y_array.append(df_pyro.at[int(index_pyro),'y'],)
    df_camera['x'] = x_array
    df_camera['y'] = y_array
    
    #todo: improve matching by:
    #todo: 1. counting number of vectors for both sensors
    #todo: 2. delete/unify short pyro vectors
    #todo: 3. compare vectors of both sensors in length and number
    #todo: 4. scale pyro time on a vector basis

    #todo: give measure for quality from:
    #todo: 1 number of vectors
    #todo: 2 plot distribution of scaling factors compared to vector length
    #todo: plot melt pool area vs. pyrometer value
    #todo: plot CMOS 2D and pyro-value 2D

    return


def plot_data(df_camera, df_pyro, selection):
    """
    Generate plots from the data generated during the sensor data processing.
    """
    part = df_camera["part"][1]
    layer = df_camera["layer"][1]
    if selection[0]:
        # Plot the results of the CMOS ON/OFF detection
        fig, ax = plt.subplots()
        line1 = ax.plot(df_camera["intensity"],color="cornflowerblue",label="intensity")
        line2 = ax.axhline(df_camera["threshold"][1], color="navy",label="intensity threshold")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("image number")
        ax2 = ax.twinx()
        line3 = ax2.plot(df_camera["ON_OFF"],color="orangered",label="ON / OFF")
        ax2.set_ylabel("OFF / ON")
        # These lines are required to get one combined legend
        line_sum = line1 + [line2] + line3
        labels = [line.get_label() for line in line_sum]
        ax.legend(line_sum, labels, loc=0)
        ax.set_title("ON/OFF detection of CMOS image: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()
    if selection[1]:
        # Plot the results of the pyro velocity ON/OFF detection
        fig, ax = plt.subplots()
        line1 = ax.plot(df_pyro["velocity"],color="cornflowerblue",label="velocity")
        line2 = ax.axhline(df_pyro["threshold"][1], color="navy",label="velocity_threshold")
        ax.set_ylabel("Velocity")
        ax.set_xlabel("measurement number")
        ax2 = ax.twinx()
        line3 = ax2.plot(df_pyro["ON_OFF"],color="orangered",label="ON / OFF")
        ax2.set_ylabel("OFF / ON")
        # These lines are required to get one combined legend
        line_sum = line1 + [line2] + line3
        labels = [line.get_label() for line in line_sum]
        ax.legend(line_sum, labels, loc=0)
        ax.set_title("ON/OFF detection of pyro velocity: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()
    if selection[2]:
        # Compare the results from the two ON/OFF detections
        fig, ax = plt.subplots()
        line1 = ax.plot(df_camera['index_pyro'],df_camera['ON_OFF'],color="navy",label="CMOS camera")
        ax.set_ylabel("OFF / ON")
        ax.set_xlabel("pyrometer measurement index")
        ax2 = ax.twinx()
        line2 = ax2.plot(df_pyro['ON_OFF'],color="orangered",label="pyrometer 1")
        ax2.set_ylabel("OFF / ON")
        # These lines are required to get one combined legend
        line_sum = line1 + line2
        labels = [line.get_label() for line in line_sum]
        ax.legend(line_sum, labels, loc=0)
        ax.set_title("ON/OFF detection comparison: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()
    if selection[3]:
        # show images at their x & y positions with their intensity
        # todo: x,y scaling, units
        fig, ax = plt.subplots()
        ax.scatter(df_camera["x"], df_camera["y"], c=df_camera["intensity"],
            cmap="inferno")
        ax.set_title("CMOS image position with image intensity: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()
    if selection[4]:
        # show images at their x & y positions with ON / OFF detection
        # todo: x,y scaling, units
        # todo: include uncertainty in alignment
        fig, ax = plt.subplots()
        df_camera_ON = df_camera.loc[df_camera['ON_OFF'] == 1.0]
        df_camera_OFF = df_camera.loc[df_camera['ON_OFF'] == 0.0]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(df_camera_ON["x"], df_camera_ON["y"], c="darkorange", 
            alpha=0.5, label="ON")
        ax.scatter(df_camera_OFF["x"], df_camera_OFF["y"], c="slategray",
            alpha=0.5, label="OFF")
        ax.legend()
        ax.set_title("CMOS image position with ON/OFF detection: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()


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

def main():
    if cherrypick:
        file_list = create_file_list(job_name, cherry=cherry)
    else:
        file_list = create_file_list(job_name)
    for file in file_list:
        # todo: check if processed files are available as .pkl files
        # Fetch and process images form the .mkv file
        if verbal == True:
            print("Started processing: " + file['filename_camera'])
            tstart = time.time()
        image_df = process_mkv(file)
        if verbal:
            print("Process finished in {0} seconds".format(time.time()-tstart))
        pass

        # Fetch and process pyro data from the .pcd file
        if verbal == True:
            print("Started processing: " + file['filename_pyro1'])
            tstart = time.time()
        pyro1_df = process_pcd(file)
        if verbal:
            print("Process finished in {0} seconds".format(time.time()-tstart))

        # fit pyro & CMOS data
        extend_CMOS_data(image_df, pyro1_df)
        # plot results
        plot_data(image_df,pyro1_df,visual)

if __name__ == "__main__":
    main()