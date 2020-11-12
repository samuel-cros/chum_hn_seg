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

#############################################################
### MAIN
#############################################################

# Paths
pwd = os.getcwd()
chum_directory = os.path.join(pwd, "..", "..", "data", "CHUM", "h5_v2")

# Parameters
patch_dim = (256, 256, 64)
n_output_channels = 1
n_input_channels = 1

list_oars = ["mandibule"]
ID = '00779'
dilation_radius = 20

# Open input file
h5_file = h5py.File(os.path.join(chum_directory, ID + '.h5'), "r")
shape_scans = h5_file["scans"].shape

#############################################################
### PATCH SAMPLING
#############################################################
tumor_volumes = ["ptv 1", "ctv 1", "gtv 1"]

# Compute dilation map
combined_mask = np.zeros((shape_scans[0], shape_scans[1], shape_scans[2]))

h5_index = 0
for channel_name in h5_file["masks"].attrs["names"]:
    if channel_name not in tumor_volumes and channel_name in list_oars:
        combined_mask += h5_file["masks"][h5_index] # += in case of 'left' and 'right' organ, such as eyes
    h5_index += 1

dilated_mask = binary_dilation(combined_mask, (1,1,1), dilation_radius)

# Pick a nonzero value
nonzero_values = np.where(dilated_mask)
random_index = np.random.randint(0, len(nonzero_values[0]))
L_center = nonzero_values[0][random_index]
W_center = nonzero_values[1][random_index]
H_center = nonzero_values[2][random_index]

# Compute patch position
L = L_center - patch_dim[0]//2
W = W_center - patch_dim[1]//2
H = H_center - patch_dim[2]//2

#print(L,W,H)

## Compute offset
# Idea = we need to use padding when the patch lands outside the input
L_offset, W_offset, H_offset = abs(min(0, L)), abs(min(0, W)), abs(min(0, H))

L_lower, W_lower, H_lower = max(0, L), max(0, W), max(0, H)
L_upper, W_upper, H_upper = min(shape_scans[0]-1, L+patch_dim[0]), min(shape_scans[1]-1, W+patch_dim[1]), min(shape_scans[2]-1, H+patch_dim[2])

L_dist, W_dist, H_dist = L_upper - L_lower, W_upper - W_lower, H_upper - H_lower

#############################################################
### OUTPUT
#############################################################

# Init
new_output = np.zeros((patch_dim[0], patch_dim[1], patch_dim[2], n_output_channels)) #

h5_index = 0
for channel_name in h5_file["masks"].attrs["names"]:
    if channel_name not in tumor_volumes and channel_name in list_oars:
        new_output[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] += h5_file["masks"][h5_index, L_lower:L_upper, W_lower:W_upper, H_lower:H_upper]
    h5_index += 1

#############################################################
### INPUT
#############################################################
# Init
min_value = -1000.0 # -1000.0, search DONE for all 1000+ cases
max_value = 3071.0 # 3071.0, search DONE for all 1000+ cases
new_input = np.full((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), min_value)

# Fill the CT channel
new_input[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] = h5_file["scans"][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper]

# Scaling factor
new_input[:, :, :, 0] -= min_value 
new_input[:, :, :, 0] /= (max_value - min_value)

#############################################################
### DISPLAY
#############################################################

'''
# Show dilated mask
image = None
for h in range(0, dilated_mask.shape[2], 1):
    if image is None:
        image = plt.imshow(dilated_mask[:,:,h], 'gray')
        plt.title('Slice ' + str(h))
    else:
        image.set_data(dilated_mask[:,:,h])
        image.autoscale()
        plt.title('Slice ' + str(h))
    plt.pause(0.0001)
    plt.draw()
'''

# Show new input
image = None
for h in range(0, new_input.shape[2], 1):
    if image is None:
        image = plt.imshow(new_input[:,:,h,0], 'gray')
        plt.title('Slice ' + str(h))
    else:
        image.set_data(new_input[:,:,h,0])
        image.autoscale()
        plt.title('Slice ' + str(h))
    plt.pause(0.0001)
    plt.draw()

# Show new output
image = None
for h in range(0, new_output.shape[2], 1):
    if image is None:
        image = plt.imshow(new_output[:,:,h,0], 'gray')
        plt.title('Slice ' + str(h))
    else:
        image.set_data(new_output[:,:,h,0])
        image.autoscale()
        plt.title('Slice ' + str(h))
    plt.pause(0.0001)
    plt.draw()