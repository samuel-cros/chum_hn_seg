###############################################################################
### IMPORTS
###############################################################################
# Math
import numpy as np

# DeepL
import keras
from keras import backend as K
import tensorflow as tf
from utils.data_standardization import standardize
from model import unet_3D

# IO
import argparse
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

parser = argparse.ArgumentParser(description='Generate slices for a given '\
                                    'model')

# Arguments
parser.add_argument('-path', '--path_to_model_dir', type=str, required=True,
                    help='Path to the model directory')
parser.add_argument('-n', '--model_name', type=str, required=True,
                    help='Name of the model')
parser.add_argument('-depth', '--model_depth', type=int, required=True,
                    help='Depth of the model')
parser.add_argument('-oars', '--kind_of_oars', type=str, required=True,
                    help='Kind of oars predicted')
parser.add_argument('-set', '--kind_of_set', type=str, required=True,
                    help='Kind of set for the prediction : train, validation' \
                        ', test or manual')
parser.add_argument('-ids', '--ids_if_manual', type=str,
                    help='IDs for manual set with format: XXXXX-XXXXX-...')
parser.add_argument('-H', '--mode_for_H', type=str, required=True,
                    help='Mode used for the height: centered or up')

args = parser.parse_args()

# Manage OARs
all_oars = ["canal medullaire", "canal medul pv", "oesophage", 
            "cavite orale", "mandibule", "parotide g", "parotide d", 
            "tronc", "tronc pv", "trachee", "oreille int g", 
            "oreille int d", "oeil g", "oeil d", "sous-max g", 
            "sous-max d", "nerf optique g"]

if args.kind_of_oars == 'down':
    list_oars = ["canal medullaire", "canal medul pv", "cavite orale", 
                "oesophage", "mandibule", "tronc", "trachee", "tronc pv"]
    oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'aquamarine', 
                'cyan', 'magenta']
elif args.kind_of_oars == 'up':
    list_oars = ["parotide g", "parotide d", "oreille int g", 
                "oreille int d", "oeil g", "oeil d", "sous-max g", 
                "sous-max d", "nerf optique g"]
    oar_colors = ['green', 'green', 'deepskyblue', 'deepskyblue', 'blue', 
                'blue', 'purple', 'purple', 'deeppink']
elif args.kind_of_oars == 'all':
    list_oars = all_oars
    oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'green', 
                'green', 'aquamarine', 'cyan', 'deepskyblue', 
                'deepskyblue', 'blue', 'blue', 'purple', 'purple', 
                'magenta', 'deeppink']
elif args.kind_of_oars == 'parotides':
    list_oars = ['parotide d', 'parotide g']
    left_right = True
elif args.kind_of_oars == 'yeux':
    list_oars = ['oeil d', 'oeil g']
    left_right = True
elif args.kind_of_oars == 'sous-maxs':
    list_oars = ['sous-max d', 'sous-max g']
    left_right = True
elif args.kind_of_oars == 'oreilles':
    list_oars = ['oreille int d', 'oreille int g']
    left_right = True
# HANDLES SINGLE ORGAN SEG
else:
    if args.kind_of_oars in all_oars:
        list_oars = [args.kind_of_oars]
        oar_colors = ['red']
    else:
        raise ValueError('Unknown kind of oars: ' + args.kind_of_oars)

dict_oars = {}
count = 0
for oar in all_oars:
    dict_oars[oar] = count
    count += 1

###############################################################################
### MAIN
###############################################################################
## Init
h5_dataset = h5py.File(os.path.join('..', 'data', 'CHUM', 'h5_v3', 
                                    'regenerated_dataset.h5'), "r")

# Net infos
params = {'patch_dim': (256, 256, 64),
          'batch_size': 1,
          'n_input_channels': 1,
          'n_output_channels': 1,
          'dataset': h5_dataset,
          'shuffle': True}
L, W = 512//2 - params['patch_dim'][1]//2, 64

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
model = unet_3D((params['patch_dim'][0], params['patch_dim'][1],
                    params['patch_dim'][2], params['n_input_channels']), 
                    args.model_depth, 0.0, 'adam', 5e-4)

model.summary()

model.load_weights(os.path.join(args.path_to_model_dir, args.model_name))

# Load data IDs
list_IDs = parse_IDs(args.ids_if_manual) \
            if (args.kind_of_set == "manual") \
            else (np.load(os.path.join(args.path_to_model_dir, 
                    args.kind_of_set + "_IDs.npy")))

# Saving results
path_to_results = os.path.join(args.path_to_model_dir, 'results_slices_' + \
                                args.kind_of_set)
path_to_expected_volume = os.path.join(path_to_results, 'expected')
path_to_predicted_volume = os.path.join(path_to_results, 'predicted')

Path(path_to_results).mkdir(parents=True, exist_ok=True)
Path(path_to_expected_volume).mkdir(parents=True, exist_ok=True)
Path(path_to_predicted_volume).mkdir(parents=True, exist_ok=True)

###############################################################################
## RESULTS
###############################################################################
## Generate folder(s) for results
dice_coeff_all = np.zeros(params['n_output_channels']+1)
count = 0

## Go through validation/test data
for ID in list_IDs:

    count += 1
    print('Generating results for patient: ' + ID + ' (' + str(count) + ')')

    ## Generate results
    # Visuals
    ct = h5_dataset[ID + '/ct']

    if args.mode_for_H == 'centered':
        H = ct.shape[2]//2 - params['patch_dim'][2]//2 
        # centered for lower organs within the volume
    elif args.mode_for_H == 'up':
        H = ct.shape[2] - params['patch_dim'][2] 
        # up for upper organs within the volume
    else:
        raise ValueError('Unhandled mode for H: ' + mode_for_H)

    ##########################
    # EXPTECTED VOLUME
    ##########################

    # Reshape groundtruth
    groundtruth = np.zeros((params['patch_dim'][0], params['patch_dim'][1], 
                            ct.shape[2], params['n_output_channels']))

    for oar in list_oars:
        groundtruth[:, :, :, 0] += h5_dataset[ID + '/mask/' + oar]\
                [L:L+params['patch_dim'][0], W:W+params['patch_dim'][1], :]

    ##########################
    # PREDICTED VOLUME
    ##########################
    min_value = -1000.0 # -1000.0, search DONE for all 1000+ cases
    max_value = 3071.0 # 3071.0, search DONE for all 1000+ cases

    # Predict one patch
    patch_formatted = np.zeros((1, params['patch_dim'][0],
                                    params['patch_dim'][1], 
                                    params['patch_dim'][2], 
                                    params['n_input_channels']))
    patch_formatted[0, :, :, :, 0] = \
        standardize(ct[L:L+params['patch_dim'][0], 
                       W:W+params['patch_dim'][1], 
                       H:H+params['patch_dim'][2]])
    
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

    # CSV
    csv_file = open(os.path.join(path_to_predicted_volume_for_ID, ID + \
                                    '_dice_scores.csv'), 'w', newline='')
    fieldnames = ['Slice'] + ['Dice']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # For each slice
    for h in range(0, prediction.shape[2], 4):
        
        print('Producing slice ' + str(H+h))

        # Produce CT
        ct_fig = plt.figure(1, figsize=(8,5))
        plt.imshow(ct[L:L+params['patch_dim'][0], 
                        W:W+params['patch_dim'][1], H+h].T, cmap='gray')
        ct_fig.savefig(os.path.join(path_to_expected_volume_for_ID, 
                                    'ct_slice_' + str(H+h)))
        plt.close()

        # Produce GT
        gt_fig = plt.figure(2, figsize=(8,5))
        plt.imshow(np.ma.masked_where(groundtruth[:, :, h, 0] == 0, 
                    groundtruth[:, :, h, 0] == 0).T * dict_oars[list_oars[0]], 
                    norm=oar_norm, cmap=oar_cmap)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        gt_fig.savefig(os.path.join(path_to_expected_volume_for_ID, 
                                    'gt_slice_' + str(H+h)))
        plt.close()

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

        # Produce Dice
        dice_intersection = np.sum(np.multiply(groundtruth[:,:,h,0], 
                                        prediction[:,:,h,0]), axis=(0,1))
        dice_summation = np.sum(groundtruth[:,:,h,0], axis=(0,1)) + \
                        np.sum(prediction[:,:,h,0], axis=(0,1))
        dice_coeff = (2*dice_intersection + 1) / (dice_summation + 1)

        row = {}
        row['Slice'] = H+h
        row['Dice'] = dice_coeff
        writer.writerow(row)
        
    
    
    csv_file.close()