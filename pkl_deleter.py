import os

# Set the root directory where the .pkl files are located
root_directory = 'Jobs/Pyro_example_data'

# Loop over all subfolders and delete all .pkl files
for subdir, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.pkl'):
            os.remove(os.path.join(subdir, file))
