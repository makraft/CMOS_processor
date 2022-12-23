# CMOS_processor

## Description
This is a tool to provide a custom functionality for the Aconity Midi+ machine.
The main goal is to provide coordinates to extracted images from the .mkv file that gets generated for the CMOS camera when exposing a part with process monitoring turned on.
The tool utilizes the .pcd files which are generated for the pyrometers when exposing a part.

The tool was developed in the scope of a semester thesis at the AM|Z laboratory of ETH Zürich.

## How to use this program
Clone the repository:
```sh
git clone https://github.com/makraft/CMOS_processor.git
```

install python 3. This program was developed with Python 3.7.3

Install the required libraries:
```sh
pip install requirements.txt
```

On the Aconity Midi+ main computer:
Copy all folders in the \Aconity_jobname\sensors folder into a folder inside the \Jobs folder in this repository.
The folder structure as it is generated on the Aconity PC:
    \Aconity_jobname
        \sensors
            \2Pyrometer
                \pyrometer1
                \pyrometer2
            \hsCamera
        \topics

The folder structure as it should look like in your clone of the repository:
    \CMOS_processor
        \Jobs
            \your_jobname
                \2Pyrometer
                    \pyrometer1
                    \pyrometer2
                \hsCamera

The folders \pyrometer1 and \pyrometer2 already contain the part folders with the .pcd files
The CMOS camera data needs to be manually copied from the camera PC into the \hsCamera folder.
Afterwards, each one of the folders \pyrometer1, \pyrometer2 and \hsCamera should contain one numbered folder per part.
In each one of these folders, for every layer either a .pcd file from the pyrometers or a .mkv file from the camera should be present.

In main.py, change the job_name variable to the name of the job you want to process.

In main.py, adjust the settings variables at the top according to what you need.

Run the program to process your data:
```sh
python main.py
```

The program generates .pkl files of pandas Dataframes that contain the generated data.
They can be loaded like this:
```sh
python
```

```sh
import pandas
```

```sh
data = pandas.read_pickle("path to .pkl file")
```

The files ending in "_images.pkl" contain a large dataframe with all images.
The files ending in just ".pkl" contain the dataframe with the image indices, the computed features and coordinates for each image.
If you want the index of an image at a certain position, search for it in the X and Y columns of the latter Dataframe.
The same index in the '_images.pkl' Dataframe contains the denoised image.
To display an image, use the display_image() function in main.py.
To make the image brighter, process the image with the normalized_image() function in main.py.
To convert all pixels above a threshold to white, process the image with the area_image() function in main.py
Note: These processes can get quite slow because the image Dataframes tend to become large.

Note: These functions can be easily called while in the debug mode of your editor.
In VS Code, a breakpoint can be set on the line that calls plot_data() in the main() function.

Note for making plots:
At the top, set the desired entries in the 'visual' array to True. Then run main.py.
Unless you set 'cherrypick' to True, plots are generated for all parts and all layers.

Note for processing:
Once the .pkl files have been generated, the corresponding data files will be skipped during repeaded calls of main.py with the same job_name.
This makes it possible to rapidly generate plots without having to wait through the processing.