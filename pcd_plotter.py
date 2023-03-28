import os
import matplotlib.pyplot as plt

# Set the root directory where the .pcd files are located
root_directory = 'Jobs/Pyro_example_data'

# Loop over all subfolders and find all .pcd files
file_list = []
for subdir, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.pcd'):
            file_list.append(os.path.join(subdir, file))

# Loop over each file and create a plot of the last entry in each line
for filename in file_list:
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [int(line.split()[-1]) for line in lines]
        plt.plot(data)
        plt.title(filename)
        plt.show()
