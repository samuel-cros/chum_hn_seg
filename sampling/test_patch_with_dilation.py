#################
### IMPORTS
#################
# Math
import numpy as np
from data_tools.binary_morphology import *

# DeepL

# IO
import os
import sys
sys.path.append(os.getcwd())
import h5py
import matplotlib.pyplot as plt

# Others
from utils.data_standardization import standardize

#############################################################
### MAIN
#############################################################

# Parameters
patch_dim = (256, 256, 64)
n_output_channels = 1
n_input_channels = 1

list_oars = ["tronc"]
ID = '00726'

# Open input file
dataset = h5py.File(os.path.join('..', 'data', 'CHUM', 'h5_v3', 'regenerated_dataset.h5'), "r")
input_shape = dataset[ID + '/ct'].shape

#############################################################
### PATCH SAMPLING
#############################################################

# Compute dilation map
dilated_mask = np.zeros((input_shape[0], input_shape[1], input_shape[2]))

for oar in list_oars:
    dilated_mask += dataset[ID + '/dilated_mask/' + oar]

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
L_upper, W_upper, H_upper = min(input_shape[0]-1, L+patch_dim[0]), min(input_shape[1]-1, W+patch_dim[1]), min(input_shape[2]-1, H+patch_dim[2])

L_dist, W_dist, H_dist = L_upper - L_lower, W_upper - W_lower, H_upper - H_lower

#############################################################
### OUTPUT
#############################################################

# Init
new_output = np.zeros((patch_dim[0], patch_dim[1], patch_dim[2], n_output_channels)) #

for oar in list_oars:
        new_output[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] += dataset[ID + '/mask/' + oar][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper]

#############################################################
### INPUT
#############################################################
# Init
new_input = np.zeros((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels))

# Fill the CT channel
new_input[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] = standardize(dataset[ID + '/ct'][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper])

#############################################################
### DISPLAY
#############################################################

#'''
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
#'''

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