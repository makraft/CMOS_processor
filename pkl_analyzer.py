import pickle

# Set the path to the .pkl file you want to open
file_path = 'Jobs/Pyro_example_data/2Pyrometer/pyrometer1/100W/5.13.pkl_filtered_1.pkl'


# Open the file in binary mode and load the data using pickle.load()
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Do something with the loaded data
pass