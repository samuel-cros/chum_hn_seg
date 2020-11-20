## Imports

# Math
import numpy as np
import random # for random augmentation parameters
from utils.data_standardization import standardize

# DeepL
import keras
import concurrent.futures

# IO
import h5py
import csv
import os
import sys

# Parameters to test
dilation_radius = 20

############################### Subfunctions ###############################


############################### Main ###############################
# DataGenerator
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_or_validation, list_IDs, list_oars, patch_dim, batch_size, n_input_channels, n_output_channels, dataset, shuffle=True, augmentation=False):
        'Initialization'
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.list_oars = list_oars
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.dataset = dataset
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, n_input_channels, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.patch_dim, self.n_input_channels))
        y = np.empty((self.batch_size, *self.patch_dim, self.n_output_channels)) 

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        
            # Store sample
            input_patch, output_patch = self.generate_random_patch(ID)
            X[i,] = input_patch
            y[i,] = output_patch

        return X, y

    #############################################################
    ### Generate random input and output patches
    #############################################################
    def generate_random_patch(self, ID):

        input_shape = self.dataset[ID + '/ct'].shape

        #############################################################
        ### PATCH SAMPLING
        #############################################################
        # Compute dilation map
        dilated_mask = np.zeros((input_shape[0], input_shape[1], input_shape[2]))

        for oar in self.list_oars:
            dilated_mask = np.logical_or(dilated_mask, 
                                    self.dataset[ID +'/dilated_mask/'+ oar])
        dilated_mask = dilated_mask.astype(int)
        
        # Pick a nonzero value
        nonzero_values = np.where(dilated_mask)
        random_index = np.random.randint(0, len(nonzero_values[0]))
        L_center = nonzero_values[0][random_index]
        W_center = nonzero_values[1][random_index]
        H_center = nonzero_values[2][random_index]

        # Compute patch position
        L = L_center - self.patch_dim[0]//2
        W = W_center - self.patch_dim[1]//2
        H = H_center - self.patch_dim[2]//2

        ## Compute offset
        # Idea = we need to use padding when the patch lands outside the input
        L_offset, W_offset, H_offset = abs(min(0, L)), abs(min(0, W)), abs(min(0, H))

        L_lower, W_lower, H_lower = max(0, L), max(0, W), max(0, H)
        L_upper, W_upper, H_upper = min(input_shape[0]-1, L+self.patch_dim[0]), min(input_shape[1]-1, W+self.patch_dim[1]), min(input_shape[2]-1, H+self.patch_dim[2])

        L_dist, W_dist, H_dist = L_upper - L_lower, W_upper - W_lower, H_upper - H_lower

        #############################################################
        ### OUTPUT
        #############################################################

        # Init
        new_output = np.zeros((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2], self.n_output_channels)) #

        for oar in self.list_oars:
            new_output[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] = \
                np.logical_or(new_output[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0], self.dataset[ID + '/mask/' + oar][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper])
        new_output = new_output.astype(int)

        #############################################################
        ### INPUT
        #############################################################
        # Init
        new_input = np.zeros((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2], self.n_input_channels))

        # Fill the CT channel
        new_input[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] = standardize(self.dataset[ID + '/ct'][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper])

        if self.augmentation: # TOREDO

            #############################################################
            ### Augmentation
            #############################################################
            pass
        
        return new_input, new_output

##################################################################
