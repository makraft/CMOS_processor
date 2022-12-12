"""
Transform data from the Aconity CMOS high speed camera and assign machine
coordinates by utilizing data from the pyrometers.
"""
import math
import csv
import statistics
import imageio
import numpy as np
import os
import pylab
import time
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
visual = [False,False,False,True,True,True,False,True]
# 0: ON/OFF plot camera
# 1: ON/OFF plot pyrometer1
# 2: combined ON/OFF plot
# 3: scatter plot of intensity @x,y position
# 4: scatter plot of ON/OFF @x,y position
# 5: scatter plot of vector length vs. slope
# 6: scatter plot of melt pool area vs. pyrometer value
# 7: two adjacent scatter plots with image position and pyrometer position

# Tell program if it should only process one selected part/layer combination
# Set True or False
cherrypick = True
# If set to true, specify which one
cherry = {
    "part" : "13",
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
        signs_array_upper = (np.sign(scan_velocity_hatch_mmps + 300 - velocity_array_smoothed_mmps) +1)/2
        initial_high_speed = np.where(signs_array_upper < 1)[0]
        try:
            lower = initial_high_speed[0]
            upper = initial_high_speed[-1]
            signs_array_upper[np.arange(lower-50,upper+71,1)] = 0
            signs_array = np.multiply(signs_array_lower,  signs_array_upper)
        except:
            print("Warning: No initial high velocity in pyrometer data detected.")
            signs_array = np.multiply(signs_array_lower,  signs_array_upper)

#        window_width =20
#        intensity = np.convolve(pyro_data[:,2], np.ones(window_width), 'valid') / window_width
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
    # Create dictionary for results which do not fit in the df_camera or df_pyro
    results = {}

    # note: interval value comes from experimenting, is dependent on skywriting 
    # strategy.
    camera_off_interval = 4
    camera_ON = np.where(df_camera['ON_OFF'] == 1)[0]
    # calculate distance between ON images
    camera_dist_on_on = np.diff(camera_ON)
    # check where distance is corresponds to a minimal OFF-interval expected 
    # between 2 scan vectors. The indice corresponds to the start of the
    # interval.
    camera_off_interval_start = np.where(camera_dist_on_on >= camera_off_interval)[0]
    camera_off_interval_length = camera_dist_on_on[camera_off_interval_start]
    # find midpoints of intervals within the CMOS indices
    camera_off_midpoints = np.round(camera_ON[camera_off_interval_start] + 1 +
        camera_off_interval_length/2)
    camera_num_scan_vectors = len(camera_off_interval_start) + 1
    results.update({
        "camera_midpoints": camera_off_midpoints,
        "camera_interval_length": camera_off_interval_length,
        "camera_num_vectors": camera_num_scan_vectors})

    # do the same thing for the pyrometer data
    # note: interval value comes from experimenting, is dependent on skywriting 
    # strategy.
    pyro_off_interval = 40
    pyro_ON = np.where(df_pyro['ON_OFF'] == 1)[0]
    # calculate distance between ON data points
    pyro_dist_on_on = np.diff(pyro_ON)
    # check where distance is corresponds to a minimal OFF-interval expected 
    # between 2 scan vectors. The indice corresponds to the start of the
    # interval.
    pyro_off_interval_start = np.where(pyro_dist_on_on >= pyro_off_interval)[0]
    pyro_off_interval_length = pyro_dist_on_on[pyro_off_interval_start]
    # find midpoints of intervals within the pyrometer indices
    pyro_off_midpoints = np.round(pyro_ON[pyro_off_interval_start] + 1 +
        pyro_off_interval_length/2)
    pyro_num_scan_vectors = len(pyro_off_interval_start) + 1
    # todo (optional): remove very short scan vectors from pyro vectors if the 
    # camera is unable to recognize them.
    results.update({
        "pyro_midpoints": pyro_off_midpoints,
        "pyro_interval_length": pyro_off_interval_length,
        "pyro_num_vectors": pyro_num_scan_vectors})
    results.update({"slopes":[]})
    results.update({"slope_errors":[]})
    results.update({"camera_delta":[]})
    results.update({"pyro_delta":[]})

    print("DETECTED MIDPOINTS: CAMERA={}  PYRO={}".format(camera_num_scan_vectors,pyro_num_scan_vectors))
    df_camera['index_pyro'] = np.nan
    if camera_num_scan_vectors == pyro_num_scan_vectors:
        print("Midpoint numbers match, do pointwise scaling")
        # linearly scale the CMOS timescales between each midpoint interval
        for index, camera_midpoint in enumerate(camera_off_midpoints[:-1]):
            camera_start = camera_midpoint
            camera_end = camera_off_midpoints[index+1]
            pyro_start = pyro_off_midpoints[index]
            pyro_end = pyro_off_midpoints[index+1]
            # compute linear scaling factors
            camera_delta = camera_end-camera_start
            pyro_delta = pyro_end-pyro_start
            slope = (pyro_delta)/(camera_delta)
            results["slopes"].append(slope)
            slope_err = slope_error(camera_delta,pyro_delta)
            results["slope_errors"].append(slope_err)
            results["camera_delta"].append(camera_delta)
            results["pyro_delta"].append(pyro_delta)
            # get offset from end points since they tend to be more accurate 
            # than start points
            offset = pyro_end - slope*camera_end
            df_camera.loc[ int(camera_start):int(camera_end),'index_pyro' ] = df_camera.loc[
                int(camera_start):int(camera_end),'index'] * slope + offset
        # linearly scale all images outside the midpoints
        camera_start = camera_off_midpoints[0]
        camera_end = camera_off_midpoints[-1]
        pyro_start = pyro_off_midpoints[0]
        pyro_end = pyro_off_midpoints[-1]
        # compute linear scaling factors
        slope = (pyro_end - pyro_start)/(camera_end - camera_start)
        offset = pyro_end - slope*camera_end
        df_camera.loc[:int(camera_start),'index_pyro'] = df_camera.loc[
            :int(camera_start),'index'] * slope + offset
        df_camera.loc[int(camera_end):,'index_pyro'] = df_camera.loc[
            int(camera_end):,'index'] * slope + offset
        # round values to the closest index
        df_camera['index_pyro'] = df_camera['index_pyro'].round()
        results.update({"method": "linear pointwise"})
    else:
        print("Midpoint numbers do not match, do simple linear scaling")
        # linearly scale the entire CMOS timescale
        camera_ON = np.where(df_camera['ON_OFF'] == 1)
        camera_ON_start = camera_ON[0][0]
        camera_ON_end = camera_ON[0][-1]
        pyro_ON = np.where(df_pyro['ON_OFF'] == 1)
        pyro_ON_start = pyro_ON[0][0]
        pyro_ON_end = pyro_ON[0][-1]
        # compute linear scaling factors
        camera_delta=camera_ON_end-camera_ON_start
        pyro_delta=pyro_ON_end-pyro_ON_start
        slope = (pyro_delta)/(camera_delta)
        results["slopes"].append(slope)
        slope_err = slope_error(camera_delta,pyro_delta)
        results["slope_errors"].append(slope_err)
        results["camera_delta"].append(camera_delta)
        results["pyro_delta"].append(pyro_delta)
        offset = pyro_ON_end - slope * camera_ON_end
        df_camera['index_pyro'] = df_camera['index'] * slope + offset
        df_camera['index_pyro'] = df_camera['index_pyro'].round()
        results.update({"method": "linear single"})

    # compute machine coordinates of each camera image
    x_array = []
    y_array = []
    pyro_value_array = []
    for index_pyro in df_camera['index_pyro']:
        x_array.append(df_pyro.at[int(index_pyro),'x'],)
        y_array.append(df_pyro.at[int(index_pyro),'y'],)
        pyro_value_array.append(df_pyro.at[int(index_pyro),'intensity'],)
    df_camera['x'] = x_array
    df_camera['y'] = y_array
    df_camera['pyro_value'] = pyro_value_array
    

    #todo: (prio 3)store dataframe
    #todo: (prio 3)manual on how to access stored images from coordinates

    return(df_camera, df_pyro, results)


def slope_error(dc, dp):
    """
    Compute the error of the slope function. c = camera, p = pyrometer
    """
    delta_dc = 2
    delta_dp = 1
    dp_err = (1/dc)*delta_dp
    dc_err = (dp/(dc**2))*(-1)*delta_dc
    slope_err = math.sqrt(dp_err**2 + dc_err**2)
    return(slope_err)

def plot_data(df_camera, df_pyro, results, selection):
    """
    Generate plots from the data generated during the sensor data processing.
    """
    part = df_camera["part"][1]
    layer = df_camera["layer"][1]
    if selection[0]:
        # Plot the results of the CMOS ON/OFF detection
        fig, ax = plt.subplots()
        line1, = ax.plot(df_camera["intensity"],color="cornflowerblue",label="intensity")
        line2 = ax.axhline(df_camera["threshold"][1], color="navy",label="intensity threshold")
        x = results["camera_midpoints"]
        y = np.ones(len(x)) * df_camera['threshold'][1]
        line3 = ax.scatter(x,y,c="green",label="midpoints")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("image number")
        ax2 = ax.twinx()
        line4, = ax2.plot(df_camera["ON_OFF"],color="orangered",label="ON / OFF")
        ax2.set_ylabel("OFF / ON")
        ax.legend(handles=[line1,line2,line3,line4],loc=0)
        ax.set_title("ON/OFF detection of CMOS image: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()
    if selection[1]:
        # Plot the results of the pyro velocity ON/OFF detection
        fig, ax = plt.subplots()
        line1, = ax.plot(df_pyro["velocity"],color="cornflowerblue",label="velocity")
        line2 = ax.axhline(df_pyro["threshold"][1], color="navy",label="velocity_threshold")
        x = results["pyro_midpoints"]
        y = np.ones(len(x)) * df_pyro['threshold'][1]
        line3 = ax.scatter(x,y,c="darkgreen",label="midpoints")
        ax.set_ylabel("Velocity")
        ax.set_xlabel("measurement number")
        ax2 = ax.twinx()
        line4, = ax2.plot(df_pyro["ON_OFF"],color="orangered",label="ON / OFF")
        ax2.set_ylabel("OFF / ON")
        ax.legend(handles=[line1,line2,line3,line4],loc=0)
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
        # todo: (prio 1)x,y scaling, units
        fig, ax = plt.subplots()
        x,y = bit2mm(df_camera["x"], df_camera["y"])
        ax.scatter(x, y, c=df_camera["intensity"],
            cmap="inferno")
        ax.set_title("CMOS image position with image intensity: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        ax.set_xlabel("x position [mm] in machine coordinates")
        ax.set_ylabel("y position [mm] in machine coordinates")
        spacing=1
        x_grid_locator = MultipleLocator(spacing)
        y_grid_locator = MultipleLocator(spacing)
        ax.xaxis.set_minor_locator(x_grid_locator)
        ax.yaxis.set_minor_locator(y_grid_locator)
        ax.grid(True,which='minor')
        plt.show()
    if selection[4]:
        # show images at their x & y positions with ON / OFF detection
        # todo: (prio 1)x,y scaling, units
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

    if selection[5]:
        # plot vector length(camera delta) vs. slope as scatterplot
        fig,ax = plt.subplots()
#        ax.errorbar(results["camera_delta"],results["slopes"],
#            yerr=results["slope_errors"],c="navy",label="scan vectors",fmt='o')
        
        # calculate the average slope
        slope_total=0
        weight_total=0
        for index, slope in enumerate(results["slopes"]):
            weight = results["camera_delta"][index]
            slope_weighted = slope * weight
            slope_total+=slope_weighted
            weight_total+=weight
        slope_average=slope_total/weight_total
        # compute theoretical errors for each camera delta
        camera_delta_all = range(int(min(results["camera_delta"])), 
            int(max(results["camera_delta"]))+1, 1)
        vector_error_all = []
        for camera_delta in camera_delta_all:
            pyro_delta=slope*camera_delta
            vector_error_all.append(slope_error(camera_delta,pyro_delta))
        # add errorbars to plot
        ax.errorbar(camera_delta_all,
            np.full(len(camera_delta_all), slope_average),
            yerr=vector_error_all,c="orange",label="theoretical errors",
            fmt='none',capsize=4)
        # add data points to the plot
        ax.scatter(results["camera_delta"],results["slopes"], c="navy",
            label="scan vectors")
        # show average slope value
        ax.axhline(slope_average,color="orange",label="average scaling factor",
            linestyle="--")
        ax.set_xlabel("vector length [pyrometer data points]")
        ax.set_ylabel("Scaling factor [pyrometer data points / CMOS image]")
        ax.legend()
        plt.show()

    if selection[6]:
        # plot melt pool area vs. pyrometer value
        fig,ax = plt.subplots()
        x = df_camera['meltpool_area']
        y = df_camera['pyro_value']
        ax.scatter(x,y)
        ax.set_xlabel("meltpool area")
        ax.set_ylabel("pyrometer value")
        ax.legend()
        plt.show()

    if selection[7]:
        # plot CMOS 2D and pyro-value 2D in adjacent plots
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.scatter(df_camera["x"], df_camera["y"], c=df_camera["intensity"],
            cmap="inferno")
        ax1.set_title("CMOS image position with image intensity: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        ax2.scatter(df_pyro["x"], df_pyro["y"], c = df_pyro["intensity"],
            cmap="inferno")
        ax2.set_title("Pyrometer positions with measurement value: " +
            "PART {} | LAYER NUMBER {}".format(part, layer))
        plt.show()

def display_image(image):
    """
    Display an image. This is a helper function meant for debugging.
    """
    # todo: (prio 1)normalize pixels in image to max intensity of a series of images
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
        # todo: (prio 3)check if processed files are available as .pkl files
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
        image_df,pyro1_df,results = extend_CMOS_data(image_df, pyro1_df)
        # plot results
        plot_data(image_df,pyro1_df,results, visual)

if __name__ == "__main__":
    main()