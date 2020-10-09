## Imports

# Math
import numpy as np
import random # for random augmentation parameters

# DeepL
import keras
import gryds # for augmentation management
import concurrent.futures

# IO
import h5py
import csv
import os
import sys

# Paths
working_directory_path = os.getcwd()
chum_directory = os.path.join(working_directory_path, "..", "data", "CHUM", "h5_v2")

############################### Subfunctions ###############################

# Interpolate
def interpolate(input, transformation):
    interpolator = gryds.Interpolator(input)
    return interpolator.transform(transformation)

#############################################################
### Generate random input and output patches
#############################################################
def generate_random_patch(self, input_path):

    # Open input file
    h5_file = h5py.File(input_path, "r")

    #############################################################
    ### OUTPUT
    #############################################################
    shape_scans = h5_file["scans"].shape

    # PATCH MANAGEMENT
    # Cropped (limits where computed to know where the oars can be found)
    if self.cropping == 'all':
        L = random.randint(364 - self.patch_dim[0], 153) # 364 - 153 = 211, centered? low-random?
        W = random.randint(1, 395 - self.patch_dim[1])
        H = random.randint(0, shape_scans[2] - self.patch_dim[2])

    elif self.cropping == 'down': # canal med, cavité, etc
        L = random.randint(364 - self.patch_dim[0], 174) # 364 - 174 = 190, centered? low-random?
        W = random.randint(1, 395 - self.patch_dim[1])
        H = random.randint(0, shape_scans[2] - self.patch_dim[2])

    elif self.cropping == 'up': # oeil, parotide, etc
        L = random.randint(363 - self.patch_dim[0], 153) # 363 - 153 = 210, centered? low-random?
        W = random.randint(66, 376 - self.patch_dim[1])
        temp = shape_scans[2] - self.patch_dim[2]
        if temp >= 51:
            H = random.randint(51, min(shape_scans[2], 334) - self.patch_dim[2])
        else:
            H = temp

    elif self.cropping == 'parotides':
        min_3D_d, max_3D_d = self.min_locations_dict['parotide d'], self.max_locations_dict['parotide d']
        min_3D_g, max_3D_g = self.min_locations_dict['parotide g'], self.max_locations_dict['parotide g']

        min_3D = []
        for x,y in zip(min_3D_d, min_3D_g):
            min_3D.append(min(x,y))
        
        max_3D = []
        for x,y in zip(max_3D_d, max_3D_g):
            max_3D.append(max(x,y))

        # L (shape_scans[0] is constant at 512 for now)
        if (max_3D[0] - min_3D[0] >= self.patch_dim[0]):
            L = random.randint(min_3D[0], max_3D[0] - self.patch_dim[0])
        else:
            if max_3D[0] >= self.patch_dim[0]:
                L = random.randint(max_3D[0] - self.patch_dim[0], min(min_3D[0], self.patch_dim[0]))
            else:
                L = random.randint(0, min(min_3D[0], self.patch_dim[0]))
        
        # W (shape_scans[1] is constant at 512 for now)
        if (max_3D[1] - min_3D[1] >= self.patch_dim[1]):
            W = random.randint(min_3D[1], max_3D[1] - self.patch_dim[1])
        else:
            if max_3D[1] >= self.patch_dim[1]:
                W = random.randint(max_3D[1] - self.patch_dim[1], min(min_3D[1], self.patch_dim[1]))
            else:
                W = random.randint(0, min(min_3D[1], self.patch_dim[1]))

        # H (shape_scans[2] is NOT constant at around 200-400 with some particular cases at around 100)
        #print(shape_scans[2], min_3D[2], max_3D[2])

        if shape_scans[2] >= max_3D[2]:
            if (max_3D[2] - min_3D[2] >= self.patch_dim[2]):
                H = random.randint(min_3D[2], max_3D[2] - self.patch_dim[2])
            else:
                H = random.randint(max_3D[2] - self.patch_dim[2], min_3D[2])
        else:
            H = random.randint(0, shape_scans[2] - self.patch_dim[2])

    elif self.cropping == 'yeux':
        min_3D_d, max_3D_d = self.min_locations_dict['oeil d'], self.max_locations_dict['oeil d']
        min_3D_g, max_3D_g = self.min_locations_dict['oeil g'], self.max_locations_dict['oeil g']

        min_3D = []
        for x,y in zip(min_3D_d, min_3D_g):
            min_3D.append(min(x,y))
        
        max_3D = []
        for x,y in zip(max_3D_d, max_3D_g):
            max_3D.append(max(x,y))

        # L (shape_scans[0] is constant at 512 for now)
        if (max_3D[0] - min_3D[0] >= self.patch_dim[0]):
            L = random.randint(min_3D[0], max_3D[0] - self.patch_dim[0])
        else:
            if max_3D[0] >= self.patch_dim[0]:
                L = random.randint(max_3D[0] - self.patch_dim[0], min(min_3D[0], self.patch_dim[0]))
            else:
                L = random.randint(0, min(min_3D[0], self.patch_dim[0]))
        
        # W (shape_scans[1] is constant at 512 for now)
        if (max_3D[1] - min_3D[1] >= self.patch_dim[1]):
            W = random.randint(min_3D[1], max_3D[1] - self.patch_dim[1])
        else:
            if max_3D[1] >= self.patch_dim[1]:
                W = random.randint(max_3D[1] - self.patch_dim[1], min(min_3D[1], self.patch_dim[1]))
            else:
                W = random.randint(0, min(min_3D[1], self.patch_dim[1]))

        # H (shape_scans[2] is NOT constant at around 200-400 with some particular cases at around 100)
        #print(shape_scans[2], min_3D[2], max_3D[2])

        if shape_scans[2] >= max_3D[2]:
            if (max_3D[2] - min_3D[2] >= self.patch_dim[2]):
                H = random.randint(min_3D[2], max_3D[2] - self.patch_dim[2])
            else:
                H = random.randint(max_3D[2] - self.patch_dim[2], min_3D[2])
        else:
            H = random.randint(0, shape_scans[2] - self.patch_dim[2])

    
    else:
        # Random
        '''
        L = random.randint(0, shape_scans[0] - self.patch_dim[0])
        W = random.randint(0, shape_scans[1] - self.patch_dim[1])
        H = random.randint(0, shape_scans[2] - self.patch_dim[2])
        '''

        # OAR specific
        min_3D, max_3D = self.min_locations_dict[self.cropping], self.max_locations_dict[self.cropping]

        # L (shape_scans[0] is constant at 512 for now)
        if (max_3D[0] - min_3D[0] >= self.patch_dim[0]):
            L = random.randint(min_3D[0], max_3D[0] - self.patch_dim[0])
        else:
            if max_3D[0] >= self.patch_dim[0]:
                L = random.randint(max_3D[0] - self.patch_dim[0], min(min_3D[0], self.patch_dim[0]))
            else:
                L = random.randint(0, min(min_3D[0], self.patch_dim[0]))
        
        # W (shape_scans[1] is constant at 512 for now)
        if (max_3D[1] - min_3D[1] >= self.patch_dim[1]):
            W = random.randint(min_3D[1], max_3D[1] - self.patch_dim[1])
        else:
            if max_3D[1] >= self.patch_dim[1]:
                W = random.randint(max_3D[1] - self.patch_dim[1], min(min_3D[1], self.patch_dim[1]))
            else:
                W = random.randint(0, min(min_3D[1], self.patch_dim[1]))

        # H (shape_scans[2] is NOT constant at around 200-400 with some particular cases at around 100)
        #print(shape_scans[2], min_3D[2], max_3D[2])

        if shape_scans[2] >= max_3D[2]:
            if (max_3D[2] - min_3D[2] >= self.patch_dim[2]):
                H = random.randint(min_3D[2], max_3D[2] - self.patch_dim[2])
            else:
                H = random.randint(max_3D[2] - self.patch_dim[2], min_3D[2])
        else:
            H = random.randint(0, shape_scans[2] - self.patch_dim[2])       

    # Create an empty array of n_output_channels:
    new_output = np.zeros((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2], self.n_output_channels)) #

    # PATCH MANAGEMENT
    tumor_volumes = ["ptv 1", "ctv 1", "gtv 1"]

    h5_index = 0
    for channel_name in h5_file["masks"].attrs["names"]:
        if channel_name not in tumor_volumes and channel_name in self.list_oars:
            new_output[:, :, :, 0] += h5_file["masks"][h5_index, L:L+self.patch_dim[0], W:W+self.patch_dim[1], H:H+self.patch_dim[2]]
        h5_index += 1

    #############################################################
    ### INPUT
    #############################################################
    # Create an empty array of n_input_channels:
    new_input = np.zeros((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2], self.n_input_channels))

    # Fill the CT channel
    new_input[:, :, :, 0] = h5_file["scans"][L:L+self.patch_dim[0], W:W+self.patch_dim[1], H:H+self.patch_dim[2]]

    # Scaling factor
    new_input[:, :, :, 0] -= -1000.0 # -1e3, search DONE for all 1000+ cases
    new_input[:, :, :, 0] /= 3071.0 # 3071.0, search DONE for all 1000+ cases

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
    def __init__(self, train_or_validation, list_IDs, list_oars, patch_dim, batch_size, n_input_channels, n_output_channels, cropping, min_locations_dict, max_locations_dict, shuffle=True, augmentation=False):
        'Initialization'
        self.input_directory = chum_directory
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.list_oars = list_oars
        
        dict_oars = {}
        count = 0
        for oar in list_oars:
            dict_oars[oar] = count
            count += 1

        self.dict_oars = dict_oars
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.cropping = cropping
        self.min_locations_dict = min_locations_dict
        self.max_locations_dict = max_locations_dict
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
            input_path = os.path.join(self.input_directory, ID + ".h5")
            input_patch, output_patch = generate_random_patch(self, input_path)
            X[i,] = input_patch
            y[i,] = output_patch

        return X, y

##################################################################