#==================================================================================================#
# Description: Converts .npz files to ROOT files.--------------------------------------------------
#==================================================================================================#

import os
import uproot
import numpy as np

# Define the directory containing the .npz files
input_dir = "data/train_bkg/Summer20UL18"

# Define the directory where you want to save the ROOT files
output_dir = "root_data/train_bkg/Summer20UL18"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through .npz files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".npz"):
        # Form the full path to the input .npz file
        input_file = os.path.join(input_dir, filename)

        # Load data from the .npz file
        data = np.load(input_file, allow_pickle=True)  # You can use allow_pickle=True if needed

        # Define the corresponding output ROOT file name (without extension)
        output_filename = os.path.splitext(filename)[0] + ".root"
        
        # Form the full path to the output ROOT file
        output_file = os.path.join(output_dir, output_filename)

        # Convert and save the data to a ROOT file
        with uproot.recreate(output_file) as root_file:
            # Loop through the arrays in the .npz file and make it a ttree to be saved to the root file
            for array_name in data.files:
                array_data = data[array_name]
                #root_file[array_name] = uproot.newtree({"data": array_data})
                root_file[array_name] = {"data": array_data}
                        
        print(f"Converted {filename} to {output_filename}")

print("Conversion complete.")



