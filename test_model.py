###############################################################################
### IMPORTS
###############################################################################
# Math
import numpy as np
from utils.data_standardization import standardize

# DeepL
import keras
import tensorflow as tf
from model import unet_3D

# IO
import argparse
import sys
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import csv

###############################################
## Limit memory allocation to minimum needed # TOTEST
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

parser = argparse.ArgumentParser(description='Test a given model')

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
parser.add_argument('-nconv', '--n_conv_per_block', type=int,
                    help='Number of convolutions per block', default=2)
parser.add_argument('-masks', '--summed_masks', dest='summed_mask', 
                    action='store_true', help='Generate summed masks')
parser.add_argument('-no-masks', '--no-summed_masks', dest='summed_mask', 
                    action='store_false', help="Don't generate summed masks")
parser.add_argument('-scores', '--dice_scores', dest='dice_scores', 
                    action='store_true', help="Generate dice scores")
parser.add_argument('-no-scores', '--no-dice_scores', dest='dice_scores', 
                    action='store_false', help="Don't generate dice scores")


# Additional defaults
parser.set_defaults(summed_masks=True, dice_scores=True)

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
oar_patches = [mpatches.Patch(color=oars_colors_dict[oar], label= oar) for
                oar in all_oars]

## Loading
# Load model
model = unet_3D(input_shape=(params['patch_dim'][0], 
                             params['patch_dim'][1],
                             params['patch_dim'][2], 
                             params['n_input_channels']), 
                model_depth=args.model_depth, 
                dropout=0.0, 
                optim='adam', 
                lr=5e-4)

model.summary()

model.load_weights(os.path.join(args.path_to_model_dir, args.model_name))

# Load data IDs
list_IDs = parse_IDs(args.ids_if_manual) \
            if (args.kind_of_set == "manual") \
            else (np.load(os.path.join(args.path_to_model_dir, 
                args.kind_of_set + "_IDs.npy")))

# Saving results
path_to_results = os.path.join(args.path_to_model_dir, 'results_' + 
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

if args.dice_scores:
    csv_file = open(os.path.join(path_to_results, 
                    args.kind_of_set + '_dice_scores.csv'), 
                    'w', newline='')
    fieldnames = ['ID'] + [oar_name for oar_name in [args.kind_of_oars]] \
                + ['average']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

## Go through validation/test data
for ID in list_IDs:

    count += 1
    print('Generating results for patient: ' + ID + ' (' + str(count) + ')')

    ## Generate results
    # Visuals
    ct = h5_dataset[ID + '/ct']

    ##########################
    # EXPTECTED VOLUME
    ##########################

    # Reshape groundtruth
    groundtruth = np.zeros((params['patch_dim'][0], params['patch_dim'][1], 
                            ct.shape[2], params['n_output_channels']))

    for oar in list_oars:
        groundtruth[:, :, :, 0] += h5_dataset[ID + '/mask/' + oar]\
                [L:L+params['patch_dim'][0], W:W+params['patch_dim'][1], :]

    # Summed masks
    if args.summed_masks:

        path_to_expected_volume_for_ID = os.path.join(path_to_expected_volume, 
                                                        ID)
        Path(path_to_expected_volume_for_ID).mkdir(parents=True, exist_ok=True)

        #######################################################################
        # Aggregating the OAR to form a single summed mask
        sum_masks = np.zeros((params['n_output_channels'], 
                                params['patch_dim'][0], 
                                params['patch_dim'][0]))
        fig2 = plt.figure(2, figsize=(8,5))

        plt.imshow(ct[L:L+params['patch_dim'][0], W:W+params['patch_dim'][1], 
                    int(2*ct.shape[2]/3)].T, cmap='gray')

        for n in range(params['n_output_channels']):
            sum_masks[n] = np.sum(groundtruth[:, :, :, n], axis=-1)
            sum_masks[n][sum_masks[n] > 0] = 1
            plt.imshow(np.ma.masked_where(sum_masks[n] == 0, sum_masks[n]).T 
                        * dict_oars[list_oars[n]], norm=oar_norm, 
                        cmap=oar_cmap, alpha=0.5)

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, 
                    borderaxespad=0.)
        plt.title('Expected segmentation for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        
        # Show
        #fig2.show()
        #plt.waitforbuttonpress()
        #plt.close()

        # Save
        fig2.savefig(os.path.join(path_to_expected_volume_for_ID, 
                    'summed_masks'))
        plt.close()
        #sys.exit()

   
    ##########################
    # PREDICTED VOLUME
    ##########################

    # Predict on the whole height
    # Prepare input
    patch_formatted = np.zeros((1, params['patch_dim'][0], 
                                params['patch_dim'][1], 
                                ct.shape[2], 
                                params['n_output_channels']))
    patch_formatted[0, :, :, :, 0] = standardize(ct[L:L+params['patch_dim'][0], 
                                        W:W+params['patch_dim'][1], :])

    # Prepare prediction_all
    prediction_all = np.zeros((params['patch_dim'][0], 
                                params['patch_dim'][1], 
                                ct.shape[2], 
                                params['n_output_channels']))

    current_h = 0
    # While current_h <= ct.shape[2]
    while (current_h + params['patch_dim'][2] <= ct.shape[2]):
        # Predict from h to h+64
        prediction = model.predict(patch_formatted[:, :, :, 
                            current_h:current_h+params['patch_dim'][2], :])

        # Store in prediction_all
        prediction_all[:, :, current_h:current_h+params['patch_dim'][2], :] = \
            prediction[0, :, :, :, :]

        # Increment h
        current_h += 64

    # Predict the last 64 slices
    prediction = model.predict(patch_formatted[:, :, :, 
                    (ct.shape[2]-params['patch_dim'][2]):ct.shape[2], :])

    # Store in prediction_all
    prediction_all[:, :, 
                    (ct.shape[2]-params['patch_dim'][2]):ct.shape[2], :] = \
                        prediction[0, :, :, :, :]

    # Summed masks
    if args.summed_masks:

        path_to_predicted_volume_for_ID = \
            os.path.join(path_to_predicted_volume, ID)
        Path(path_to_predicted_volume_for_ID).mkdir(parents=True, 
            exist_ok=True)

        #######################################################################
        # Aggregating the OAR to form a single summed mask
        sum_predictions = np.zeros((params['n_output_channels'], 
                                    prediction_all.shape[0],
                                    prediction_all.shape[1]))
        fig3 = plt.figure(3, figsize=(8,5))

        plt.imshow(ct[L:L+params['patch_dim'][0], W:W+params['patch_dim'][1], 
                    int(2*ct.shape[2]/3)].T, cmap='gray')

        # Thresholding
        threshold = 0.5
        for n in range(params['n_output_channels']):
            for h in range(ct.shape[2]):
                sum_predictions[n][prediction_all[:, :, h, n] > threshold] = 1
            plt.imshow(np.ma.masked_where(sum_predictions[n] == 0, 
                sum_predictions[n]).T * dict_oars[list_oars[n]], norm=oar_norm,
                cmap=oar_cmap, alpha=0.5)

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, 
                    borderaxespad=0.)
        plt.title('Predicted segmentation for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        
        # Show
        #fig3.show()
        #plt.waitforbuttonpress()
        #plt.close()

        # Save
        fig3.savefig(os.path.join(path_to_predicted_volume_for_ID, 
                                    'summed_masks'))
        plt.close()

    ##########################
    # Stats
    ##########################

    # Dice scores
    if args.dice_scores:
        
        # GOOD WAY KERAS
        from keras import backend as K
        import tensorflow as tf

        y_true = tf.constant(groundtruth) # (1, 256, 256, 64, _)
        y_pred = tf.constant(prediction_all) # (1, 256, 256, 64, _)

        intersection = K.sum(y_true * y_pred, axis=(0,1,2))
        summation = K.sum(y_true + y_pred, axis=(0,1,2))
        dice_coeff = (2. * intersection + 1) / (summation + 1)
        dice_coeff = K.eval(dice_coeff)
        #print(dice_coeff)
        
        print('--------------------------------------------------------------')
        for n in range(params['n_output_channels']):
            print('Dice coefficient for ' + args.kind_of_oars + ': ' + \
                    str(dice_coeff[n]))
        print('Average dice coefficient for patient ' + ID + ': ' + \
                    str(np.mean(dice_coeff)))
        print('--------------------------------------------------------------')
        #'''

        ## Save in a csv file
        # Fill a row dict
        row = {}
        row['ID'] = ID 
        for oar_name in [args.kind_of_oars]:
            row[oar_name] = dice_coeff[0]
        row['average'] = np.mean(dice_coeff)

        writer.writerow(row)

        # Save in dice_coeff_all
        dice_coeff_all += [x for x in dice_coeff] + [row['average']]

# Stats
# - Dice for each channel
# - Average Dice
if args.dice_scores:

    dice_coeff_all /= len(list_IDs)

    # Fill a row dict
    row = {}
    row['ID'] = 'ALL'
    for oar_name in [args.kind_of_oars]:
        row[oar_name] = dice_coeff_all[0]
    row['average'] = dice_coeff_all[-1]

    writer.writerow(row)

    csv_file.close()