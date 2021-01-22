## Imports
# Math
import numpy as np
from skimage.measure import label 
from utils.data_standardization import standardize

# DeepL
from keras import backend as K
import tensorflow as tf
from model import unet_3D

# IO
import argparse
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import h5py

###############################################################################
### SUBROUTINES
###############################################################################

###############################################################################
# GETLARGESTCC
###############################################################################
# get the largest connected component
def getLargestCC(segmentation, paired):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    # For unpaired organs get the first CC
    if not paired:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    # For paired organs get the first two CC
    else:
        largestCC = \
            (labels == np.argsort(np.bincount(labels.flat)[1:])[-1]+1) | \
            (labels == np.argsort(np.bincount(labels.flat)[1:])[-2]+1)
    return largestCC.astype(segmentation.dtype)


###############################################################################
# PREDICT_FULL_VOLUME
###############################################################################
# basic idea: 
# - overlap X patches of 256*256*64 within the 512*512*H to get
#   a prediction over the full volume
# - to manage the overlapping, we will use a map that saves for each voxel
#   the number of overlaps so we can reduce the prediction accordingly
def predict_full_volume(model, input_ct, patch_dim):  

    # Init
    prediction_mask = np.zeros((input_ct.shape[1], 
                                input_ct.shape[2], 
                                input_ct.shape[3]))
    overlap_map = np.zeros((input_ct.shape[1], 
                            input_ct.shape[2], 
                            input_ct.shape[3]))

    # Init current height to 0
    h = 0
    ###########################################################################
    # While the patch fits in height
    while h + patch_dim[2] < input_shape[2]:
        # Go along the length, 128 by 128
        for l in range(0, 384, 128):
            # Go along the width, 128 by 128
            for w in range(0, 384, 128):

                # Predict
                prediction_patch = \
                    model.predict(input_ct[:, 
                                           l:l+patch_dim[0], 
                                           w:w+patch_dim[1], 
                                           h:h+patch_dim[2],
                                           :])
                # Store prediction
                prediction_mask[l:l+patch_dim[0],
                                w:w+patch_dim[1],
                                h:h+patch_dim[2]] += \
                                    prediction_patch[0, :, :, :, 0]

                # Compute overlap map
                overlap_map[l:l+patch_dim[0],
                            w:w+patch_dim[1],
                            h:h+patch_dim[2]] += 1

        # Increment h
        h += 64

    ###########################################################################
    ## Predict the last 64 slices
    h = input_shape[2]-patch_dim[2]
    # Go along the length, 128 by 128
    for l in range(0, 384, 128):
        # Go along the width, 128 by 128
        for w in range(0, 384, 128):

            # Predict
            prediction_patch = \
                    model.predict(input_ct[:,
                                           l:l+patch_dim[0], 
                                           w:w+patch_dim[1], 
                                           h:h+patch_dim[2],
                                           :])
            # Store prediction
            prediction_mask[l:l+patch_dim[0],
                            w:w+patch_dim[1],
                            h:h+patch_dim[2]] += \
                                prediction_patch[0, :, :, :, 0]

            # Compute overlap map
            overlap_map[l:l+patch_dim[0],
                        w:w+patch_dim[1],
                        h:h+patch_dim[2]] += 1

    ###########################################################################
    # Average prediction given the overlap map
    overlap_map[overlap_map == 0] = 1
    prediction_mask /= overlap_map

    # Thresholding
    prediction_mask[prediction_mask > 0.5] = 1
    prediction_mask[prediction_mask <= 0.5] = 0

    # Convert back to int
    prediction_mask = prediction_mask.astype(np.int32)

    # Denoise using biggest connected component method
    #prediction_mask = getLargestCC(prediction_mask, paired)

    return prediction_mask
###############################################################################

###############################################################################
### END SUBROUTINES
###############################################################################




###############################################################################
### ARGPARSE
###############################################################################

###############################################################################
# Parse
parser = argparse.ArgumentParser(description='Combine outputs from different'+\
    ' models by simple average')

parser.add_argument('-oar', '--oar_name', type=str, required=True,
                    help='Name of the organ to segment')
parser.add_argument('-paths', '--paths_to_models', nargs='+', type=str, 
                    required=True, help='Paths to the models')
parser.add_argument('-o_path', '--output_path', type=str, 
                    required=True, help='Path for the output results')
parser.add_argument('-set', '--kind_of_set', type=str, required=True)
parser.add_argument('--seed', type=str, required=True)
parser.add_argument('-ids', '--list_of_ids', nargs='+', type=str,
                    required=False, 
                    help='List of ids to test if set is manual')

# Default
args = parser.parse_args()

###############################################################################
# Manage arguments
all_oars = ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", 
            "mandibule", "parotide g", "parotide d", "tronc", "trachee", 
            "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g",
            "tronc pv", "sous-max d"]
if args.oar_name == 'parotides':
    list_oars = ['parotide d', 'parotide g']
    paired = True
elif args.oar_name == 'yeux':
    list_oars = ['oeil d', 'oeil g']
    paired = True
elif args.oar_name == 'sous-maxs':
    list_oars = ['sous-max d', 'sous-max g']
    paired = True
elif args.oar_name == 'oreilles':
    list_oars = ['oreille int d', 'oreille int g']
    paired = True
else:
    if args.oar_name in all_oars:
        list_oars = [args.oar_name]
        paired = False
    else:
        raise ValueError('Unknown kind of oars: ' + args.oar_name)

###############################################################################
### INIT
###############################################################################

###############################################################################
## Net parameters
patch_dim = (256, 256, 64)
n_input_channels = 1

###############################################################################
## Input file
dataset = h5py.File(os.path.join('..', 'data','regenerated_dataset.h5'), "r")

###############################################################################
## Input IDs
list_IDs = args.list_of_ids \
            if (args.kind_of_set == "manual") \
            else (np.load(os.path.join('sample_IDs', 'seed' + args.seed, 
                args.kind_of_set + "_IDs.npy")))

###############################################################################
## Paths
path_to_results = os.path.join(args.output_path, 
                               'results_' + args.kind_of_set)
Path(path_to_results).mkdir(parents=True, exist_ok=True)

###############################################################################
## Output csv file (to save scores)
csv_file = open(os.path.join(path_to_results, 
                             args.kind_of_set + '_dice_scores.csv'), 
                'w', 
                newline='')
fieldnames = ['ID'] + [args.oar_name]
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()
dice_coeff_acc = 0

###############################################################################
# Define the setup model
setup_model = unet_3D(input_shape=(patch_dim[0], 
                                   patch_dim[1], 
                                   patch_dim[2], 
                                   n_input_channels), 
                      number_of_pooling=2, 
                      dropout=0.0, 
                      optim='adam', 
                      lr=1e-3)

###############################################################################
# Load nets
list_models = []
for model_path in args.paths_to_models:
    setup_model.load_weights(model_path)
    list_models.append(setup_model)

###############################################################################
# For each patient
#
for ID in list_IDs:

    ###########################################################################
    ### PREP INPUT/GROUNDTRUTH
    ###########################################################################

    # Grab data shape
    input_shape = dataset[ID + '/ct'].shape
    groundtruth_mask = np.zeros((input_shape[0], 
                                input_shape[1], 
                                input_shape[2]),
                                dtype=np.int32)

    ## OAR MASK (groundtruth)
    # Sum oar masks in case of paired organs (e.g. eyes)
    for oar in list_oars:
        groundtruth_mask += dataset[ID + '/mask/' + oar]

    ## CT (input)
    input_ct = dataset[ID + '/ct']
    
    # Standardize (min-max norm)
    input_ct_standardized = np.zeros((1, 
                                    input_shape[0], 
                                    input_shape[1], 
                                    input_shape[2], 1))
    input_ct_standardized[0, :, :, :, 0] = standardize(input_ct[:,:,:])

    ###########################################################################
    ### GATHER PREDICTIONS
    ###########################################################################
    prediction_mask_acc = np.zeros((input_shape[0], 
                                    input_shape[1], 
                                    input_shape[2]))

    # for each net
    for model in list_models:
        # Predict full volume
        prediction_mask_acc += predict_full_volume(model, 
                                                   input_ct_standardized, 
                                                   patch_dim)

    # Average predictions
    prediction_mask_acc /= float(len(list_models))

    # Thresholding
    prediction_mask_acc[prediction_mask_acc > 0.5] = 1
    prediction_mask_acc[prediction_mask_acc <= 0.5] = 0

    # Convert back to int
    prediction_mask_acc = prediction_mask_acc.astype(np.int32)

    ###########################################################################
    ### COMPUTE SCORES
    ###########################################################################
    # Compute    
    intersection = np.sum(groundtruth_mask * prediction_mask_acc, axis=(0,1,2))
    summation = np.sum(groundtruth_mask + prediction_mask_acc, axis=(0,1,2))
    dice_coeff = (2 * intersection + 1) / (summation + 1)

    # Save
    row = {}
    row['ID'] = ID 
    row[args.oar_name] = round(dice_coeff, 3)
    writer.writerow(row)

    # Accumulate dice coeff across patients
    dice_coeff_acc += dice_coeff

# Compute average dice coeff across patients
row['ID'] = 'ALL'
row[args.oar_name] = dice_coeff_acc/len(list_IDs)
writer.writerow(row)

###########################################################################
### Cleanup
###########################################################################
csv_file.close()