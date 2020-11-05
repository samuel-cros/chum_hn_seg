#######################################################################################################################
### IMPORTS
#######################################################################################################################
# Math
import numpy as np

# DeepL
import keras
import tensorflow as tf

# IO
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

#######################################################################################################################
### SUBFUNCTIONS
#######################################################################################################################
# Parse a list of IDs XXXXX-XXXXX-XXXXX-... into a proper pythonic list
def parse_IDs(list_of_IDs):
    return list_of_IDs.split('-')

#######################################################################################################################
### ARGS
#######################################################################################################################
# Path to model
# - ex: lr/1e-3_e20, loss/dice_sum_average_e20
# Data to test
# - validation, test or manual

# Additional params
dropout_value = 0.0
n_convolutions_per_block = 2

if len(sys.argv) >=6:
    path_to_model_dir = sys.argv[1]
    model_name = sys.argv[2]
    model_depth = sys.argv[3]
    kind_of_oars = sys.argv[4]
    train_validation_or_test_or_manual = sys.argv[5]
    if train_validation_or_test_or_manual == 'manual':
        list_IDs_if_manual = sys.argv[6]

    # Manage model depth
    if model_depth == '64':
        from unet_seg_64 import unet
    elif model_depth == '512':
        from unet_seg_512 import unet
    else:
        raise NameError('Unhandled model depth: ' + model_depth)

    # Manage OARs
    all_oars = ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", "mandibule", "parotide g", "parotide d", "tronc", "tronc pv", "trachee", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "sous-max d", "nerf optique g"]

    if kind_of_oars == 'down':
        list_oars = ["canal medullaire", "canal medul pv", "cavite orale", "oesophage", "mandibule", "tronc", "trachee", "tronc pv"]
        oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'aquamarine', 'cyan', 'magenta']
    elif kind_of_oars == 'up':
        list_oars = ["parotide g", "parotide d", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "sous-max d", "nerf optique g"]
        oar_colors = ['green', 'green', 'deepskyblue', 'deepskyblue', 'blue', 'blue', 'purple', 'purple', 'deeppink']
    elif kind_of_oars == 'all':
        list_oars = all_oars
        oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'green', 'green', 'aquamarine', 'cyan', 'deepskyblue', 'deepskyblue', 'blue', 'blue', 'purple', 'purple', 'magenta', 'deeppink']
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
    print("python test_multi_params_for_avg.py path_to_model_dir model_name model_depth kind_of_oars train_validation_or_test_or_manual list_IDs_if_manual")
    print("    -> format for model_depth: 64 or 512")    
    print("    -> format for kind_of_oars: up or down or all or OAR_NAME")
    print("    -> format for list_IDs_if_manual: XXXXX-XXXXX-XXXXX-...")
    sys.exit()

# Get more args
'''
val = input("Generate summed masks? y/n \n") 
generate_summed_masks = True if (val == 'y') else False
'''
generate_summed_masks = True

'''
val = input("Generate dice scores? y/n \n")
generate_dice_scores = True if (val == 'y') else False
'''
generate_dice_scores = True

#######################################################################################################################
### MAIN
#######################################################################################################################
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
# ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", "mandibule", "parotide g", "parotide d", "tronc", "trachee", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "sous-max d", "tronc pv", "nerf optique g"]

# Visuals parameters
cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn']
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
#oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'green', 'green', 'aquamarine', 'cyan', 'deepskyblue', 'deepskyblue', 'blue', 'blue', 'purple', 'purple', 'magenta', 'deeppink']
#oar_colors = ['red', 'orange', 'yellow', 'gold', 'lime', 'green', 'aquamarine', 'cyan']
oar_cmap = mcolors.ListedColormap(oars_colors_dict.values())
oar_boundaries = [x for x in range(len(all_oars))]
oar_norm = mcolors.BoundaryNorm(oar_boundaries, oar_cmap.N, clip=True)
oar_patches = [mpatches.Patch(color=oars_colors_dict[oar], label= oar) for oar in all_oars]
#['red', 'orange', 'yellow', 'gold', 'lime', 'green', 'aquamarine', 'cyan', 'deepskyblue', 'blue', 'purple', 'magenta', 'deeppink']

# Channels management
extra_volumes = ['ptv 1', 'gtv 1', 'ctv 1']

## Loading
# Load model
optim, lr = 'adam', '5e-4' # doesn't matter at test time
model = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), n_output_channels, float(dropout_value), int(n_convolutions_per_block), optim, float(lr))
model.summary()

model.load_weights(os.path.join(path_to_model_dir, model_name))

# Load data IDs
list_IDs = parse_IDs(list_IDs_if_manual) if (train_validation_or_test_or_manual == "manual") else (np.load(os.path.join(path_to_model_dir, validation_or_test_or_manual + "_IDs.npy")))

# Saving results
path_to_results = os.path.join(path_to_model_dir, 'results_' + train_validation_or_test_or_manual)
path_to_expected_volume = os.path.join(path_to_results, 'expected')
path_to_predicted_volume = os.path.join(path_to_results, 'predicted')

Path(path_to_results).mkdir(parents=True, exist_ok=True)
Path(path_to_expected_volume).mkdir(parents=True, exist_ok=True)
Path(path_to_predicted_volume).mkdir(parents=True, exist_ok=True)

#######################################################################################################################
## RESULTS
#######################################################################################################################
## Generate folder(s) for results
dice_coeff_all = np.zeros(n_output_channels+1)
count = 0

if generate_dice_scores:
    csv_file = open(os.path.join(path_to_results, train_validation_or_test_or_manual + '_dice_scores.csv'), 'w', newline='')
    fieldnames = ['ID'] + [oar_name for oar_name in [kind_of_oars]] + ['average']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

## Go through validation/test data
for ID in list_IDs:

    count += 1
    print('Generating results for patient: ' + ID + ' (' + str(count) + ')')

    data = h5py.File(os.path.join(path_to_data, ID + ".h5"), "r")

    ## Generate results
    # Visuals
    ct = data['scans']
    masks = data['masks']  

    ##########################
    # EXPTECTED VOLUME
    ##########################

    # Reshape groundtruth
    groundtruth = np.zeros((patch_dim[0], patch_dim[1], ct.shape[2], n_output_channels))
    n = 0
    for channel_name in masks.attrs['names']:
        if channel_name in list_oars:
            groundtruth[:, :, :, 0] += masks[n, L:L+patch_dim[0], W:W+patch_dim[1], :]
        n += 1

    ###################################################################################################################
    # Showing CT and OAR as masks dynamically
    '''
    for h in range(0, ct.shape[2], 4):

        fig1 = plt.figure(1, figsize=(8,5))
        # Plot the ct
        plt.imshow(ct[:, :, h].T, cmap='gray')

        # Plot the OAR
        #   ---> sous max d et tronc pv are swapped within the data!
        n = 0
        for channel_name in masks.attrs['names']:
            if channel_name in list_oars:
                plt.imshow(np.ma.masked_where(masks[n, :, :, h] == 0, masks[n, :, :, h]).T * dict_oars[channel_name], norm=oar_norm, cmap=oar_cmap, alpha=0.5)
            n += 1

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Patient ID: ' + ID + ' , frame: ' + str(h))
        plt.gca().invert_xaxis()
        
        # Show
        fig1.show()
        plt.waitforbuttonpress()
        plt.pause(0.00001)
    plt.close()
    '''

    #'''
    # Summed masks
    if generate_summed_masks:

        path_to_expected_volume_for_ID = os.path.join(path_to_expected_volume, ID)
        Path(path_to_expected_volume_for_ID).mkdir(parents=True, exist_ok=True)

        ###################################################################################################################
        # Aggregating the OAR to form a single summed mask
        sum_masks = np.zeros((n_output_channels, patch_dim[0], patch_dim[0]))
        fig2 = plt.figure(2, figsize=(8,5))

        plt.imshow(ct[L:L+patch_dim[0], W:W+patch_dim[1], int(2*ct.shape[2]/3)].T, cmap='gray') # 2/3 => teeth, 3/4 => ??, 4/5 => ??

        for n in range(n_output_channels):
            sum_masks[n] = np.sum(groundtruth[:, :, :, n], axis=-1)
            sum_masks[n][sum_masks[n] > 0] = 1
            plt.imshow(np.ma.masked_where(sum_masks[n] == 0, sum_masks[n]).T * dict_oars[list_oars[n]], norm=oar_norm, cmap=oar_cmap, alpha=0.5)

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Expected segmentation for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        
        # Show
        #fig2.show()
        #plt.waitforbuttonpress()
        #plt.close()

        # Save
        fig2.savefig(os.path.join(path_to_expected_volume_for_ID, 'summed_masks'))
        plt.close()
        #sys.exit()

    #'''
   
    #'''
    ##########################
    # PREDICTED VOLUME
    ##########################

    # Predict one patch
    #H = ct.shape[2]//2 - patch_dim[2]//2
    #patch_formatted = np.zeros((1, patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels))
    #patch_formatted[0, :, :, :, 0] = ct[L:L+patch_dim[0], W:W+patch_dim[1], H:H+patch_dim[2]]
    #patch_formatted -= -1000.0
    #patch_formatted /= 3071.0
    #prediction = model.predict(patch_formatted)
    #prediction = prediction[0, :, :, :]

    # Predict on the whole height
    # Prepare input
    patch_formatted = np.zeros((1, patch_dim[0], patch_dim[1], ct.shape[2], n_input_channels))
    patch_formatted[0, :, :, :, 0] = ct[L:L+patch_dim[0], W:W+patch_dim[1], :]
    patch_formatted -= -1000.0
    patch_formatted /= 3071.0

    # Prepare prediction_all
    prediction_all = np.zeros((patch_dim[0], patch_dim[1], ct.shape[2], n_output_channels))

    current_h = 0
    # While current_h <= ct.shape[2]
    while (current_h + patch_dim[2] <= ct.shape[2]):
        # Predict from h to h+64
        prediction = model.predict(patch_formatted[:, :, :, current_h:current_h+patch_dim[2], :])

        # Store in prediction_all
        prediction_all[:, :, current_h:current_h+patch_dim[2], :] = prediction[0, :, :, :, :]

        # Increment h
        current_h += 64

    # Predict the last 64 slices
    prediction = model.predict(patch_formatted[:, :, :, (ct.shape[2]-patch_dim[2]):ct.shape[2], :])

    # Store in prediction_all
    prediction_all[:, :, (ct.shape[2]-patch_dim[2]):ct.shape[2], :] = prediction[0, :, :, :, :]

    # Summed masks
    if generate_summed_masks:

        path_to_predicted_volume_for_ID = os.path.join(path_to_predicted_volume, ID)
        Path(path_to_predicted_volume_for_ID).mkdir(parents=True, exist_ok=True)

        ###################################################################################################################
        # Aggregating the OAR to form a single summed mask
        sum_predictions = np.zeros((n_output_channels, prediction_all.shape[0], prediction_all.shape[1]))
        fig3 = plt.figure(3, figsize=(8,5))

        plt.imshow(ct[L:L+patch_dim[0], W:W+patch_dim[1], int(2*ct.shape[2]/3)].T, cmap='gray') # 2/3 => teeth, 3/4 => ??, 4/5 => ??

        # Thresholding
        threshold = 0.5
        for n in range(n_output_channels):
            for h in range(ct.shape[2]):
                #sum_predictions[n] = np.sum(prediction[:, :, :, n], axis=2)
                #sum_predictions[n][sum_predictions[n] > 0] = 1 # resp channel_number
                sum_predictions[n][prediction_all[:, :, h, n] > threshold] = 1
            plt.imshow(np.ma.masked_where(sum_predictions[n] == 0, sum_predictions[n]).T * dict_oars[list_oars[n]], norm=oar_norm, cmap=oar_cmap, alpha=0.5)
            #plt.imshow(sum_predictions[n] * n, norm=oar_norm, cmap=oar_cmap, alpha=0.5)

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Predicted segmentation for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()
        
        # Show
        #fig3.show()
        #plt.waitforbuttonpress()
        #plt.close()

        # Save
        fig3.savefig(os.path.join(path_to_predicted_volume_for_ID, 'summed_masks'))
        plt.close()

    #'''

    ##########################
    # Stats
    ##########################

    # Dice scores
    if generate_dice_scores:

        # WRONG WAY
        '''
        dice_join = np.sum(np.multiply(groundtruth, prediction_all), axis=-1)
        dice_union = np.sum(groundtruth, axis=-1) + np.sum(prediction_all, axis=-1)
        dice_coeff = np.mean((2*dice_join + 1) / (dice_union + 1))
        print(dice_coeff.shape)

        print('--------------------------------------------------------------')
        print('Average dice coefficient for patient ' + ID + ': ' + str(dice_coeff))
        print('--------------------------------------------------------------')
        '''

        # GOOD WAY NUMPY
        '''
        dice_join = np.sum(np.multiply(groundtruth, prediction_all), axis=(0,1,2))
        dice_union = np.sum(groundtruth, axis=(0,1,2)) + np.sum(prediction_all, axis=(0,1,2))
        dice_coeff = (2*dice_join + 1) / (dice_union + 1)
        '''
        
        # GOOD WAY KERAS
        from keras import backend as K
        import tensorflow as tf

        y_true = tf.constant(groundtruth) # (1, 256, 256, 64, 17)
        y_pred = tf.constant(prediction_all) # (1, 256, 256, 64, 17)

        intersection = K.sum(y_true * y_pred, axis=(0,1,2))
        summation = K.sum(y_true + y_pred, axis=(0,1,2))
        dice_coeff = (2. * intersection + 1) / (summation + 1)
        dice_coeff = K.eval(dice_coeff)
        #print(dice_coeff)
        
        print('--------------------------------------------------------------')
        for n in range(n_output_channels):
            print('Dice coefficient for ' + kind_of_oars + ': ' + str(dice_coeff[n]))
        print('Average dice coefficient for patient ' + ID + ': ' + str(np.mean(dice_coeff)))
        print('--------------------------------------------------------------')
        #'''

        ## Save in a csv file
        # Fill a row dict
        row = {}
        row['ID'] = ID 
        for oar_name in [kind_of_oars]:
            row[oar_name] = dice_coeff[0]
        row['average'] = np.mean(dice_coeff)

        writer.writerow(row)

        # Save in dice_coeff_all
        dice_coeff_all += [x for x in dice_coeff] + [row['average']]

# Stats
# - Dice for each channel
# - Average Dice
if generate_dice_scores:

    dice_coeff_all /= len(list_IDs)

    # Fill a row dict
    row = {}
    row['ID'] = 'ALL'
    for oar_name in [kind_of_oars]:
        row[oar_name] = dice_coeff_all[0]
    row['average'] = dice_coeff_all[-1]

    writer.writerow(row)

    csv_file.close()




