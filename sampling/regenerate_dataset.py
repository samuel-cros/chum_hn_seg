#################
### IMPORTS
#################
# Math
import numpy as np
from binary_morphology import *

# DeepL

# IO
import os
import h5py
import matplotlib.pyplot as plt
import sys
import time

#######################################################################################################################
### MAIN
#######################################################################################################################
# Paths
pwd = os.getcwd()
path_to_data = os.path.join(pwd, "..", "..", "data", "CHUM", "h5_v2")

# Parameters
dilation_radius = 20

# Init
IDs = np.load(os.path.join('..', 'stats', 'oars_proportion', '16_oars_IDs.npy'))
extra_volumes = ["ptv 1", "ctv 1", "gtv 1", "nerf optique g"]

## Generate h5 file:
#   - ID.h5
#       - CT: 512*512*H
#       - name_organ_1: 512*512*H
#       - name_organ_2: 512*512*H
#       - ...

h5_file = h5py.File(os.path.join('..', '..', 'data', 'CHUM', 'h5_v3', 'regenerated_dataset.h5'), 'w')

# For each patient
for ID in IDs:

    # Load data
    data = h5py.File(os.path.join(path_to_data, ID + '.h5'), "r")

    # Add CT
    h5_file.create_dataset(ID + '/ct', data = data["scans"], compression="gzip")

    # Add masks
    # For each organ
    h5_index = 0
    for channel_name in data["masks"].attrs["names"]:
        if channel_name not in extra_volumes:

            # Add mask
            h5_file.create_dataset(ID + '/mask/' + channel_name, data=data["masks"][h5_index], compression="gzip")

            # Add dilated mask
            dilated_data = binary_dilation(data["masks"][h5_index], (1,1,1), dilation_radius)
            h5_file.create_dataset(ID + '/dilated_mask/' + channel_name, data = dilated_data, compression="gzip")

        h5_index += 1
        