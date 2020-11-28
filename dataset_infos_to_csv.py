## Imports

# Math
import numpy as np

# IO
import csv
import os
import h5py
import sys

# Paths
working_directory_path = os.getcwd()
input_directory = os.path.join(working_directory_path, "..", "data", "CHUM", "h5_v2")

# Other
count = 0
min_value = 0
max_value = 0

############################### Main ###############################
# Generate a csv for future selection with
#   ID | number of available doses | number of available masks | names of available masks 
# | Length | Width | Height | Minimun value | Maximum value | Mean value
with open('data_infos.csv', 'w', newline='') as csv_file:
#with open('validation_info.csv', 'w', newline='') as csv_file:
    fieldnames = ['ID', 'nb_masks', 'masks_names', 'height', 'ct_min_value', 'ct_max_value', 'ct_mean_value', 'ct_std_value', 'ct_sum_value']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for file in os.listdir(input_directory):
        h5_file = h5py.File(os.path.join(input_directory, file), "r")

        #print("Getting min, max, mean values..")
        # Get min, max, mean values
        min_value = np.min(h5_file["scans"])
        max_value = np.max(h5_file["scans"])
        mean_value = np.mean(h5_file["scans"])
        std_value = np.std(h5_file["scans"])
        sum_value = np.sum(h5_file["scans"])

        #print("Writing..")
        # Write the full row
        writer.writerow({'ID' : file.split(".")[0], 
                        'nb_masks' : h5_file["masks"].shape[0], 
                        'masks_names' : [mask_name for mask_name in h5_file["masks"].attrs["names"]],
                        'height' : h5_file["doseplans"][0].shape[2],
                        'ct_min_value' : min_value,
                        'ct_max_value' : max_value,
                        'ct_mean_value' : mean_value,
                        'ct_std_value' : std_value,
                        'ct_sum_value' : sum_value
                        })

####################################################################

