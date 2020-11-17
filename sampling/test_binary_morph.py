#################
### IMPORTS
#################
# Math
import numpy as np
from data_tools.binary_morphology import *

# DeepL

# IO
import os
import h5py
import matplotlib.pyplot as plt

#######################################################################################################################
### MAIN
#######################################################################################################################
# Paths
pwd = os.getcwd()
path_to_data = os.path.join(pwd, "..", "..", "data", "CHUM", "h5_v2")

# Load example patient

# Init
IDs = np.load(os.path.join('..', 'stats', 'oars_proportion', '20_plus_IDs.npy'))
tumor_volumes = ["ptv 1", "ctv 1", "gtv 1"]
data = h5py.File(os.path.join(path_to_data, IDs[0] + '.h5'), "r")

# Show binary mask slice by slice
'''
image = None
for h in range(0, data["masks"].shape[3], 4):
    if image is None:
        image = plt.imshow(data["masks"][4, :, :, h], 'gray')
        plt.title('Slice ' + str(h))
    else:
        image.set_data(data["masks"][4, :, :, h])
        image.autoscale()
        plt.title('Slice ' + str(h))
    plt.pause(0.0001)
    plt.draw()
'''

# Run dilation function
dilated_data = binary_dilation(data["masks"][4], (1,1,1), 15)
dilated_data[np.where(data["masks"][4])] = 0

print(dilated_data.shape)

# Show dilated mask slice by slice
image = None
for h in range(0, dilated_data.shape[2], 4):
    if image is None:
        image = plt.imshow(dilated_data[:,:,h], 'gray')
        plt.title('Slice ' + str(h))
    else:
        image.set_data(dilated_data[:,:,h])
        image.autoscale()
        plt.title('Slice ' + str(h))
    plt.pause(0.0001)
    plt.draw()