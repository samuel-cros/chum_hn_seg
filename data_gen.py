## Imports

# Math
import numpy as np
import random # for random augmentation parameters

# DeepL
import keras
import concurrent.futures

# IO
import h5py
import csv
import os
import sys

# Paths
chum_directory = os.path.join("..", "data", "CHUM", "h5_v2")

# Parameters to test
dilation_radius = 20

############################### Subfunctions ###############################

# Interpolate
def interpolate(input, transformation):
    interpolator = gryds.Interpolator(input)
    return interpolator.transform(transformation)

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
        dilated_mask += self.dataset[ID + '/dilated_mask/' + oar]
    
    # Pick a nonzero value
    nonzero_values = np.where(dilated_mask)
    random_index = np.random.randint(0, len(nonzero_values[0]))
    L_center = nonzero_values[0][random_index]
    W_center = nonzero_values[1][random_index]
    H_center = nonzero_values[2][random_index]

    # Compute patch position
    L = L_center - self.patch_dim[0]
    W = W_center - self.patch_dim[1]
    H = H_center - self.patch_dim[2]

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
        new_output[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] += self.dataset[ID + '/mask/' + oar][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper]

    #############################################################
    ### INPUT
    #############################################################
    # Init
    min_value = -1000.0 # -1000.0, search DONE for all 1000+ cases
    max_value = 3071.0 # 3071.0, search DONE for all 1000+ cases
    new_input = np.full((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2], self.n_input_channels), min_value)

    # Fill the CT channel
    new_input[L_offset:L_offset+L_dist, W_offset:W_offset+W_dist, H_offset:H_offset+H_dist, 0] = self.dataset[ID + '/ct'][L_lower:L_upper, W_lower:W_upper, H_lower:H_upper]

    # Scaling factor
    new_input[:, :, :, 0] -= min_value 
    new_input[:, :, :, 0] /= (max_value - min_value)

    if self.augmentation: # TOREDO

        '''
        #############################################################
        ### Augmentation
        #############################################################

        # Affine transform
        random_angle = np.pi/random.choice([neg for neg in range(-32, -15, 1)] + [pos for pos in range(16, 33, 1)])
        random_shear = np.array([[1, random.choice([0.0, 0.1, 0.2]+[-0.1,-0.2]), 0], [0, 1, 0], [0, 0, 1]])
        #random_translation = [0.01, 0, 0] # Conv = Invariant
        #random_brighting = random.randint(0, 5)/100
        
        # Apply
        #ct *= random_brighting
        affine = gryds.AffineTransformation(ndim=3, angles=[0, 0, random_angle], center=[0.5,0.5,0.5], shear_matrix=random_shear)

        # CT
        interpolator_ct = gryds.Interpolator(new_input[:, :, :, 0])
        ct_augmented = interpolator_ct.transform(affine)
        new_input[:, :, :, 0] = ct_augmented

        # OAR
        #interpolator_oar = gryds.Interpolator(new_output)
        #new_output = interpolator_oar.transform(affine)

        new_augmented_output = np.zeros((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2], self.n_output_channels))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(interpolate, new_output[:, :, :, k], affine) for k in range(self.n_output_channels)] # Out of Memory error :'(

        count = 0
        for future in concurrent.futures.as_completed(results):
            new_augmented_output[:, :, :, count] = future.result()
            count += 1

        new_output = new_augmented_output
        '''
    
    return new_input, new_output

##################################################################


############################### Main ###############################
# DataGenerator
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_or_validation, list_IDs, list_oars, patch_dim, batch_size, n_input_channels, n_output_channels, dataset, shuffle=True, augmentation=False):
        'Initialization'
        self.input_directory = chum_directory
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
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, n_input_channels, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.patch_dim, self.n_input_channels))
        y = np.empty((self.batch_size, *self.patch_dim, self.n_output_channels)) 

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
        
            # Store sample
            input_patch, output_patch = generate_random_patch(self, ID)
            X[i,] = input_patch
            y[i,] = output_patch

        return X, y

##################################################################
