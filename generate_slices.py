###############################################################################
### IMPORTS
###############################################################################
# Math
import numpy as np

# DeepL
import keras
from keras import backend as K
import tensorflow as tf

# IO
import sys
import h5py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import csv
import time

###############################################
## Limit memory allocation to minimum needed
###############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

###############################################################################
### SUBFUNCTIONS
###############################################################################
# Parse a list of IDs XXXXX-XXXXX-XXXXX-... into a proper pythonic list
def parse_IDs(list_of_IDs):
    return list_of_IDs.split('-')

###############################################################################
### ARGS
###############################################################################
# Path to model
# - ex: lr/1e-3_e20, loss/dice_sum_average_e20
# Data to test
# - validation, test or manual

# Additional params
dropout_value = 0.0
n_convolutions_per_block = 2

if len(sys.argv) >=7:
    path_to_model_dir = sys.argv[1]
    model_name = sys.argv[2]
    model_depth = sys.argv[3]
    mode_for_H = sys.argv[4]
    kind_of_oars = sys.argv[5]
    train_validation_or_test_or_manual = sys.argv[6]
    if train_validation_or_test_or_manual == 'manual':
        list_IDs_if_manual = sys.argv[7]

    # Manage model depth
    if model_depth == '64':
        from unet_seg_64 import unet
    elif model_depth == '512':
        from unet_seg_512 import unet
    else:
        raise NameError('Unhandled model depth: ' + model_depth)

    # Manage OARs
    all_oars = ["canal medullaire", "canal medul pv", "oesophage", 
                "cavite orale", "mandibule", "parotide g", "parotide d", 
                "tronc", "tronc pv", "trachee", "oreille int g", 
                "oreille int d", "oeil g", "oeil d", "sous-max g", 
                "sous-max d", "nerf optique g"]
    
    if kind_of_oars == 'down':
        list_oars = ["canal medullaire", "canal medul pv", "cavite orale", 
                    "oesophage", "mandibule", "tronc", "trachee", "tronc pv"]
        oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'aquamarine', 
                        'cyan', 'magenta']
    elif kind_of_oars == 'up':
        list_oars = ["parotide g", "parotide d", "oreille int g", 
                    "oreille int d", "oeil g", "oeil d", "sous-max g", 
                    "sous-max d", "nerf optique g"]
        oar_colors = ['green', 'green', 'deepskyblue', 'deepskyblue', 'blue', 
                        'blue', 'purple', 'purple', 'deeppink']
    elif kind_of_oars == 'all':
        list_oars = all_oars
        oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'green', 
                        'green', 'aquamarine', 'cyan', 'deepskyblue', 
                        'deepskyblue', 'blue', 'blue', 'purple', 'purple', 
                        'magenta', 'deeppink']
    elif kind_of_oars == 'parotides':
        list_oars = ['parotide d', 'parotide g']
        left_right = True
    elif kind_of_oars == 'yeux':
        list_oars = ['oeil d', 'oeil g']
        left_right = True
    elif kind_of_oars == 'sous-maxs':
        list_oars = ['sous-max d', 'sous-max g']
        left_right = True
    elif kind_of_oars == 'oreilles':
        list_oars = ['oreille int d', 'oreille int g']
        left_right = True
    # HANDLES SINGLE ORGAN SEG
    else:
        if kind_of_oars in all_oars:
            list_oars = [kind_of_oars]
            oar_colors = ['red']
        else:
            raise NameError('Unknown kind of oars: ' + kind_of_oars)

    dict_oars = {}
    count = 0
    for oar in all_oars:
        dict_oars[oar] = count
        count += 1

   

else:
    print("Wrong number of arguments, see example below.")
    print("python generate_slices.py path_to_model_dir model_name " \
            "model_depth mode_for_H kind_of_oars " \
            "train_validation_or_test_or_manual list_IDs_if_manual")
    print("    -> format for model_depth: 64 or 512")
    print("    -> format for mode_for_H: centered or up")
    print("    -> format for kind_of_oars: up or down or all or OAR_NAME")
    print("    -> format for list_IDs_if_manual: XXXXX-XXXXX-XXXXX-...")
    sys.exit()

###############################################################################
### MAIN
###############################################################################
## Init
# Paths
pwd = os.getcwd()
path_to_data = os.path.join(pwd, "..", "data", "CHUM", "h5_v2")

# Net infos
patch_dim = (256, 256, 64)
n_input_channels = 1
n_output_channels = 1
L, W = 512//2 - patch_dim[1]//2, 64

# Data

# Visuals parameters
cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 
        'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 
        'PuBuGn', 'BuGn']
oars_colors_dict = {'canal medullaire': 'red',
                    'canal medul pv': 'orange',
                    'oesophage': 'green',
                    'cavite orale': 'gold',
                    'mandibule': 'yellow',
                    'parotide g': 'blue',
                    'parotide d': 'blue',
                    'tronc': 'cyan',
                    'tronc pv': 'aquamarine',
                    'trachee': 'lime',
                    'oreille int g': 'deepskyblue',
                    'oreille int d': 'deepskyblue',
                    'oeil g': 'purple',
                    'oeil d': 'purple',
                    'sous-max g': 'magenta',
                    'sous-max d': 'magenta',
                    'nerf optique g': 'deeppink'}

oar_cmap = mcolors.ListedColormap(oars_colors_dict.values())
oar_boundaries = [x for x in range(len(all_oars))]
oar_norm = mcolors.BoundaryNorm(oar_boundaries, oar_cmap.N, clip=True)
oar_patches = [mpatches.Patch(color=oars_colors_dict[oar], label= oar) \
                for oar in all_oars]

# Channels management
extra_volumes = ['ptv 1', 'gtv 1', 'ctv 1']

## Loading
# Load model
optim, lr = 'adam', '5e-4' # doesn't matter at test time
model = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
        n_output_channels, float(dropout_value), int(n_convolutions_per_block),
        optim, float(lr))

model.summary()

model.load_weights(os.path.join(path_to_model_dir, model_name))

# Load data IDs
list_IDs = parse_IDs(list_IDs_if_manual) \
            if (train_validation_or_test_or_manual == "manual") \
            else (np.load(os.path.join(path_to_model_dir, 
                    train_validation_or_test_or_manual + "_IDs.npy")))

# Saving results
path_to_results = os.path.join(path_to_model_dir, 'results_slices_' + \
                                train_validation_or_test_or_manual)
path_to_expected_volume = os.path.join(path_to_results, 'expected')
path_to_predicted_volume = os.path.join(path_to_results, 'predicted')

Path(path_to_results).mkdir(parents=True, exist_ok=True)
Path(path_to_expected_volume).mkdir(parents=True, exist_ok=True)
Path(path_to_predicted_volume).mkdir(parents=True, exist_ok=True)

###############################################################################
## RESULTS
###############################################################################
## Generate folder(s) for results
dice_coeff_all = np.zeros(n_output_channels+1)
count = 0

## Go through validation/test data
for ID in list_IDs:

    count += 1
    print('Generating results for patient: ' + ID + ' (' + str(count) + ')')

    data = h5py.File(os.path.join(path_to_data, ID + ".h5"), "r")

    ## Generate results
    # Visuals
    ct = data['scans']
    masks = data['masks']  

    if mode_for_H == 'centered':
        H = ct.shape[2]//2 - patch_dim[2]//2 
        # centered for lower organs within the volume
    elif mode_for_H == 'up':
        H = ct.shape[2] - patch_dim[2] 
        # up for upper organs within the volume
    else:
        raise NameError('Unhandled mode for H: ' + mode_for_H)

    ##########################
    # EXPTECTED VOLUME
    ##########################

    # Reshape groundtruth
    groundtruth = np.zeros((patch_dim[0], patch_dim[1], patch_dim[2]))
    n = 0
    for channel_name in masks.attrs['names']:
        if channel_name in list_oars:
            groundtruth[:, :, :] += masks[n, 
                                        L:L+patch_dim[0], 
                                        W:W+patch_dim[1], 
                                        H:H+patch_dim[2]]
        n += 1

    ##########################
    # PREDICTED VOLUME
    ##########################

    # Predict one patch
    patch_formatted = np.zeros((1, patch_dim[0], patch_dim[1], patch_dim[2], 
                                n_input_channels))
    patch_formatted[0, :, :, :, 0] = ct[L:L+patch_dim[0], 
                                        W:W+patch_dim[1], 
                                        H:H+patch_dim[2]]
    patch_formatted -= -1000.0
    patch_formatted /= 3071.0
    prediction = model.predict(patch_formatted)
    prediction = prediction[0, :, :, :]

    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0

    ##########################
    # Slices
    ##########################

    # Path for CT + GT
    path_to_expected_volume_for_ID = os.path.join(path_to_expected_volume, ID)
    Path(path_to_expected_volume_for_ID).mkdir(parents=True, exist_ok=True)

    # Path to PRED
    path_to_predicted_volume_for_ID = os.path.join(path_to_predicted_volume, 
                                                    ID)
    Path(path_to_predicted_volume_for_ID).mkdir(parents=True, exist_ok=True)

    #'''
    # CSV
    csv_file = open(os.path.join(path_to_predicted_volume_for_ID, ID + \
                                    '_dice_scores.csv'), 'w', newline='')
    fieldnames = ['Slice'] + ['Dice']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    #'''

    # For each slice
    for h in range(0, prediction.shape[2], 4):
        
        begin = time.time()
        print('Producing slice ' + str(H+h))

        #print('CT..')
        # Produce CT
        ct_fig = plt.figure(1, figsize=(8,5))
        plt.imshow(ct[L:L+patch_dim[0], W:W+patch_dim[1], H+h].T, cmap='gray')
        ct_fig.savefig(os.path.join(path_to_expected_volume_for_ID, 
                                    'ct_slice_' + str(H+h)))
        plt.close()

        #'''
        #print('GT..')
        # Produce GT
        gt_fig = plt.figure(2, figsize=(8,5))
        plt.imshow(np.ma.masked_where(groundtruth[:, :, h] == 0, 
                    groundtruth[:, :, h] == 0).T * dict_oars[list_oars[0]], 
                    norm=oar_norm, cmap=oar_cmap)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        gt_fig.savefig(os.path.join(path_to_expected_volume_for_ID, 
                                    'gt_slice_' + str(H+h)))
        plt.close()

        #print('PRED..')
        # Produce PRED
        pred_fig = plt.figure(3, figsize=(8,5))
        plt.imshow(np.ma.masked_where(prediction[:, :, h, 0] == 0, 
            prediction[:, :, h, 0]).T * dict_oars[list_oars[0]], 
            norm=oar_norm, cmap=oar_cmap)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        pred_fig.savefig(os.path.join(path_to_predicted_volume_for_ID, 
                                        'pred_slice_' + str(H+h)))
        plt.close()

        #print('Computing Dice..')
        # Produce Dice
        dice_join = np.sum(np.multiply(groundtruth[:,:,h], 
                                        prediction[:,:,h,0]), axis=(0,1))
        dice_union = np.sum(groundtruth[:,:,h], axis=(0,1)) + \
                        np.sum(prediction[:,:,h,0], axis=(0,1))
        dice_coeff = (2*dice_join + 1) / (dice_union + 1)

        #print(dice_coeff)

        row = {}
        row['Slice'] = H+h
        row['Dice'] = dice_coeff
        writer.writerow(row)

        print(time.time()-begin)
        #'''
        
    
    
    csv_file.close()





