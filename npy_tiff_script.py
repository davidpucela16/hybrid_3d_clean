import numpy as np 
from PIL import Image
import os
import sys
import pdb
script_path = os.path.abspath(sys.argv[0])
folder_path = os.path.dirname(script_path)
tif_path=os.path.join(folder_path,"output_tif")
os.makedirs(tif_path, exist_ok=True)
# Get a list of all .npy files in the folder
npy_files = [file for file in os.listdir(folder_path) if file.endswith(".npy")]

# Iterate over the .npy files and load them
for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)
    if len(data.shape)>1:
        if data.shape[0]==data.shape[1] and len(data.shape)==2:
            # Process the loaded data as needed
            image=Image.fromarray(data)
            filename= npy_file[:-4]  +".tif"
            image.save(os.path.join(tif_path, filename))
            # Example: Print the shape of the loaded array
            print(f"Loaded {npy_file}, Shape: {data.shape}")

