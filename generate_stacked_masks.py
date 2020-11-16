###############################################################################
###############################################################################
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

# Params
IDs = ['00779']
patch_dim = (256, 256, 64)
n_input_channels = 1
n_output_channels = 1
L, W = 512//2 - patch_dim[1]//2, 64

two_groups = True
group_1 = ['canal medul pv', 'canal medullaire', 'cavite orale', 'mandibule',
            'oesophage', 'trachee']
group_2 = ['tronc pv', 'tronc', 'oreilles', 'sous-maxs', 'yeux', 'parotides']

# Visual options
all_oars = ["canal medul pv", "canal medullaire", "oesophage", "cavite orale",
            "mandibule", "parotides", "tronc pv", "tronc", "trachee", 
            "oreilles", "yeux", "sous-maxs"]
dict_oars = {}
count = 0
for oar in all_oars:
    dict_oars[oar] = count
    count += 1
oars_colors_dict_save = {'canal medul pv': 'orange',
                    'canal medullaire': 'red',
                    'oesophage': 'green',
                    'cavite orale': 'gold',
                    'mandibule': 'yellow',
                    'parotides': 'blue',
                    'tronc pv': 'aquamarine',
                    'tronc': 'cyan',
                    'trachee': 'lime',
                    'oreilles': 'deepskyblue',
                    'yeux': 'purple',
                    'sous-maxs': 'magenta'}
oars_colors_dict = {'canal medul pv': 'darkorange',
                    'canal medullaire': 'orange',
                    'oesophage': 'yellowgreen',
                    'cavite orale': 'darkred',
                    'mandibule': 'red',
                    'parotides': 'deepskyblue',
                    'tronc pv': 'deeppink',
                    'tronc': 'hotpink',
                    'trachee': 'greenyellow',
                    'oreilles': 'blueviolet',
                    'yeux': 'green',
                    'sous-maxs': 'turquoise'}
oar_cmap = mcolors.ListedColormap(oars_colors_dict.values())
oar_boundaries = [x for x in range(len(all_oars)+1)]
oar_norm = mcolors.BoundaryNorm(oar_boundaries, oar_cmap.N, clip=True)
oar_patches = [mpatches.Patch(color=oars_colors_dict[oar], label= oar) \
                for oar in all_oars]

# Paths
pwd = os.getcwd()
path_to_data = os.path.join(pwd, "..", "data", "CHUM", "h5_v2")

# Define model archi
optim, lr, dropout_value, n_convolutions_per_block = 'adam', '5e-4', '0.0', '2' 
# doesn't matter at test time

# Load models
from unet_seg_64 import unet
canal_medullaire = unet((patch_dim[0], patch_dim[1], patch_dim[2], 
                    n_input_channels), n_output_channels, float(dropout_value), 
                    int(n_convolutions_per_block), optim, float(lr))
canal_medullaire.load_weights('res_baseline/canal_medullaire/' \
                                'dr_0.0_nconv_2_e_200/best_model.h5')
canal_medul_pv = unet((patch_dim[0], patch_dim[1], patch_dim[2], 
                n_input_channels), n_output_channels, float(dropout_value), 
                int(n_convolutions_per_block), optim, float(lr))
canal_medul_pv.load_weights('res_baseline/canal_medul_pv/' \
                            'dr_0.0_nconv_2_e_200/best_model.h5')
oesophage = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
            n_output_channels, float(dropout_value), 
            int(n_convolutions_per_block), optim, float(lr))
oesophage.load_weights('res_baseline/oesophage/' \
                        'dr_0.0_nconv_2_e_200/best_model.h5')
mandibule = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
            n_output_channels, float(dropout_value), 
            int(n_convolutions_per_block), optim, float(lr))
mandibule.load_weights('res_baseline/mandibule/' \
                        'dr_0.0_nconv_2_e_200/best_model.h5')
trachee = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
            n_output_channels, float(dropout_value), 
            int(n_convolutions_per_block), optim, float(lr))
trachee.load_weights('res_baseline/trachee/dr_0.0_nconv_2_e_200/best_model.h5')
tronc_pv = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
                n_output_channels, float(dropout_value), 
                int(n_convolutions_per_block), optim, float(lr))
tronc_pv.load_weights('res_baseline/tronc_pv/' \
                        'dr_0.0_nconv_2_e_200/best_model.h5')
tronc = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
        n_output_channels, float(dropout_value), int(n_convolutions_per_block),
        optim, float(lr))
tronc.load_weights('res_baseline/tronc/dr_0.0_nconv_2_e_200/best_model.h5')
cavite_orale = unet((patch_dim[0], patch_dim[1], patch_dim[2], 
                n_input_channels), n_output_channels, float(dropout_value), 
                int(n_convolutions_per_block), optim, float(lr))
cavite_orale.load_weights('res_baseline/cavite_orale/' \
                            'dr_0.0_nconv_2_e_200/best_model.h5')
parotides = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
            n_output_channels, float(dropout_value), 
            int(n_convolutions_per_block), optim, float(lr))
parotides.load_weights('res_baseline/parotides/' \
                        'dr_0.0_nconv_2_o_adam_lr_5e-4_e_200/best_model.h5')

from unet_seg_512 import unet
yeux = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
        n_output_channels, float(dropout_value), int(n_convolutions_per_block),
        optim, float(lr))
yeux.load_weights('res_baseline_512/yeux/' \
                    'dr_0.3_nconv_2_o_adam_lr_5e-4_e_50/best_model.h5')
oreilles = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
            n_output_channels, float(dropout_value), 
            int(n_convolutions_per_block), optim, float(lr))
oreilles.load_weights('res_baseline_512/oreilles/' \
                        'dr_0.3_nconv_2_o_adam_lr_5e-4_e_50/best_model.h5')
sousmaxs = unet((patch_dim[0], patch_dim[1], patch_dim[2], n_input_channels), 
            n_output_channels, float(dropout_value), 
            int(n_convolutions_per_block), optim, float(lr))
sousmaxs.load_weights('res_baseline_512/sous-maxs/' \
                        'dr_0.3_nconv_2_o_adam_lr_5e-4_e_50/best_model.h5')

oars_models_dict = {'canal medullaire': canal_medullaire,
                    'canal medul pv': canal_medul_pv,
                    'oesophage': oesophage,
                    'cavite orale': cavite_orale,
                    'mandibule': mandibule,
                    'parotides': parotides,
                    'tronc': tronc,
                    'tronc pv': tronc_pv,
                    'trachee': trachee,
                    'oreilles': oreilles,
                    'yeux': yeux,
                    'sous-maxs': sousmaxs}

# For each patient
for ID in IDs:

    # Load data
    data = h5py.File(os.path.join(path_to_data, ID + ".h5"), "r")

    # Set up data
    ct = data['scans']
    masks = data['masks']

    #
    
    # Prepare input
    patch_formatted = np.zeros((1, patch_dim[0], patch_dim[1], ct.shape[2], 
                        n_input_channels))
    patch_formatted[0, :, :, :, 0] = ct[L:L+patch_dim[0], W:W+patch_dim[1], :]
    patch_formatted -= -1000.0
    patch_formatted /= 3071.0

    ##########################
    # ONE GROUP
    ##########################
    if not two_groups:

        plt.imshow(ct[L:L+patch_dim[0], W:W+patch_dim[1], 
                    int(2*ct.shape[2]/3)].T, cmap='gray')

        for oar in all_oars:

            # Prepare prediction_all
            prediction_all = np.zeros((patch_dim[0], patch_dim[1], ct.shape[2],
                                        n_output_channels))

            current_h = 0
            # While current_h <= ct.shape[2]
            while (current_h + patch_dim[2] <= ct.shape[2]):
                # Predict from h to h+64
                prediction = oars_models_dict[oar].predict(patch_formatted[:,
                                    :, :, current_h:current_h+patch_dim[2], :])

                # Store in prediction_all
                prediction_all[:, :, current_h:current_h+patch_dim[2], :] = \
                    prediction[0, :, :, :, :]

                # Increment h
                current_h += 64

            # Predict the last 64 slices
            prediction = oars_models_dict[oar].predict(patch_formatted[:, :, :,
                                (ct.shape[2]-patch_dim[2]):ct.shape[2], :])

            # Store in prediction_all
            prediction_all[:, :, (ct.shape[2]-patch_dim[2]):ct.shape[2], :] = \
                prediction[0, :, :, :, :]

            # Summed masks

            ###################################################################
            # Aggregating the OAR to form a single summed mask
            sum_predictions = np.zeros((n_output_channels, 
                            prediction_all.shape[0], prediction_all.shape[1]))

            # Thresholding
            threshold = 0.5
            for n in range(n_output_channels):
                for h in range(ct.shape[2]):
                    sum_predictions[n][prediction_all[:, :, 
                                                        h, n] > threshold] = 1
                plt.imshow(np.ma.masked_where(sum_predictions[n] == 0, 
                    sum_predictions[n]).T * dict_oars[oar], norm=oar_norm, 
                    cmap=oar_cmap, alpha=0.5) #TODO

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, 
                    borderaxespad=0.)
        plt.title('Predicted segmentation summed mask for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()

        plt.show()
        plt.waitforbuttonpress()
        plt.close()

    ##########################
    # TWO GROUPS
    ##########################
    elif two_groups:

        # GROUP 1
        fig1 = plt.figure()
        plt.imshow(ct[L:L+patch_dim[0], W:W+patch_dim[1], 
            int(2*ct.shape[2]/3)].T, cmap='gray')

        for oar in group_1:
            

            # Prepare prediction_all
            prediction_all = np.zeros((patch_dim[0], patch_dim[1], ct.shape[2],
                                        n_output_channels))

            current_h = 0
            # While current_h <= ct.shape[2]
            while (current_h + patch_dim[2] <= ct.shape[2]):
                # Predict from h to h+64
                prediction = oars_models_dict[oar].predict(patch_formatted[:,
                                    :, :, current_h:current_h+patch_dim[2], :])

                # Store in prediction_all
                prediction_all[:, :, current_h:current_h+patch_dim[2], :] = \
                    prediction[0, :, :, :, :]

                # Increment h
                current_h += 64

            # Predict the last 64 slices
            prediction = oars_models_dict[oar].predict(patch_formatted[:, :, :,
                                    (ct.shape[2]-patch_dim[2]):ct.shape[2], :])

            # Store in prediction_all
            prediction_all[:, :, (ct.shape[2]-patch_dim[2]):ct.shape[2], :] = \
                prediction[0, :, :, :, :]

            # Summed masks

            ###################################################################
            # Aggregating the OAR to form a single summed mask
            sum_predictions = np.zeros((n_output_channels, 
                prediction_all.shape[0], prediction_all.shape[1]))

            # Thresholding
            threshold = 0.5
            for n in range(n_output_channels):
                for h in range(ct.shape[2]):
                    sum_predictions[n][prediction_all[:, :, 
                                                        h, n] > threshold] = 1
                plt.imshow(np.ma.masked_where(sum_predictions[n] == 0, 
                    sum_predictions[n]).T * dict_oars[oar], norm=oar_norm, 
                    cmap=oar_cmap, alpha=0.5) #TODO

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, 
                    borderaxespad=0.)
        plt.title('Predicted segmentation summed mask for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()

        fig1.show()
        plt.waitforbuttonpress()
        plt.close()

        # GROUP 2
        fig2 = plt.figure()
        plt.imshow(ct[L:L+patch_dim[0], W:W+patch_dim[1], 
            int(2*ct.shape[2]/3)].T, cmap='gray')

        for oar in group_2:
            
            # Prepare prediction_all
            prediction_all = np.zeros((patch_dim[0], patch_dim[1], ct.shape[2],
                                        n_output_channels))

            current_h = 0
            # While current_h <= ct.shape[2]
            while (current_h + patch_dim[2] <= ct.shape[2]):
                # Predict from h to h+64
                prediction = oars_models_dict[oar].predict(patch_formatted[:,
                                :, :, current_h:current_h+patch_dim[2], :])

                # Store in prediction_all
                prediction_all[:, :, current_h:current_h+patch_dim[2], :] = \
                    prediction[0, :, :, :, :]

                # Increment h
                current_h += 64

            # Predict the last 64 slices
            prediction = oars_models_dict[oar].predict(patch_formatted[:, :, :,
                                (ct.shape[2]-patch_dim[2]):ct.shape[2], :])

            # Store in prediction_all
            prediction_all[:, :, (ct.shape[2]-patch_dim[2]):ct.shape[2], :] = \
                prediction[0, :, :, :, :]

            # Summed masks

            ###################################################################
            # Aggregating the OAR to form a single summed mask
            sum_predictions = np.zeros((n_output_channels, 
                                        prediction_all.shape[0], 
                                        prediction_all.shape[1]))

            # Thresholding
            threshold = 0.5
            for n in range(n_output_channels):
                for h in range(ct.shape[2]):
                    sum_predictions[n][prediction_all[:, :, 
                                                        h, n] > threshold] = 1
                plt.imshow(np.ma.masked_where(sum_predictions[n] == 0, 
                    sum_predictions[n]).T * dict_oars[oar], norm=oar_norm, 
                    cmap=oar_cmap, alpha=0.5) #TODO

        # Format
        plt.legend(handles=oar_patches, bbox_to_anchor=(1.05, 1), loc=2, 
                    borderaxespad=0.)
        plt.title('Predicted segmentation summed mask for patient ' + ID)
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.invert_xaxis()

        fig2.show()
        plt.waitforbuttonpress()
        plt.close()




