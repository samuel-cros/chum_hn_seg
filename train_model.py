## Imports
# Math
import numpy as np
import math

# DeepL
from sklearn.model_selection import train_test_split
#from data_generator_multi import DataGenerator
from data_gen import DataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.layers import *
from model import unet_3D, load_pretrained_weights

# IO
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import h5py

###############################################
## Limit memory allocation to minimum needed
###############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

###############################################
## Sub-functions
###############################################

#

###############################################
## Input
###############################################

parser = argparse.ArgumentParser(description='Train a given model')

# Arguments
parser.add_argument('-path', '--path_to_main_folder', type=str, required=True,
                    help='Path to the output folder')
parser.add_argument('-pool', '--number_of_pooling', type=int, required=True,
                    help='Number of pooling operations')
parser.add_argument('-oars', '--kind_of_oars', type=str, required=True,
                    help='Kind of oars to predict')
parser.add_argument('-o', '--optim', type=str, required=True,
                    help='Optimizer')
parser.add_argument('-lr', type=float, required=True, help='Learning rate')
parser.add_argument('-drop', '--dropout_value', type=float, required=True,
                    help='Dropout')
parser.add_argument('-e', '--n_epochs', type=int, required=True,
                    help='Number of epochs')
parser.add_argument('-w', '--initial_weights', type=str,
                    help='Path to the initial weights')
parser.add_argument('-aug', '--augmentation', dest='augmentation',
                    action='store_true', help='Use data augmentation')
parser.add_argument('-no-aug', '--no-augmentation', dest='augmentation',
                    action='store_false', help="Don't use data augmentation")
parser.add_argument('-seed', type=int, required=True, help='Random seeding')

# Additional defaults
parser.set_defaults(augmentation=False)
args = parser.parse_args()

## Seeding
from numpy.random import seed
seed(args.seed)
from tensorflow import set_random_seed
set_random_seed(args.seed+1)

# Manage OARs
all_oars = ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", 
            "mandibule", "parotide g", "parotide d", "tronc", "trachee", 
            "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g",
            "tronc pv", "sous-max d", "nerf optique g"]

if args.kind_of_oars == 'down':
    list_oars = ["canal medullaire", "canal medul pv", "cavite orale", 
                "oesophage", "mandibule", "tronc", "trachee", "tronc pv"]
elif args.kind_of_oars == 'up':
    list_oars = ["parotide g", "parotide d", "oreille int g", "oreille int d",
                "oeil g", "oeil d", "sous-max g", "sous-max d", 
                "nerf optique g"]
elif args.kind_of_oars == 'all':
    list_oars = all_oars
elif args.kind_of_oars == 'parotides':
    list_oars = ['parotide d', 'parotide g']
elif args.kind_of_oars == 'yeux':
    list_oars = ['oeil d', 'oeil g']
elif args.kind_of_oars == 'sous-maxs':
    list_oars = ['sous-max d', 'sous-max g']
elif args.kind_of_oars == 'oreilles':
    list_oars = ['oreille int d', 'oreille int g']
# HANDLES SINGLE ORGAN SEG
else:
    if args.kind_of_oars in all_oars:
        list_oars = [args.kind_of_oars]
    else:
        raise ValueError('Unknown kind of oars: ' + args.kind_of_oars)

# Manage folder for generated files
Path(args.path_to_main_folder).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.path_to_main_folder, 
                    args.kind_of_oars.replace(' ', '_'))).mkdir(parents=True, 
                                                            exist_ok=True)
if args.initial_weights is not None:
    path_to_generated_files = os.path.join(args.path_to_main_folder, 
        args.kind_of_oars.replace(' ', '_'), 
        'dr_' + str(args.dropout_value) + '_o_' + args.optim + '_lr_' + \
                str(args.lr) + '_e_' + str(args.n_epochs) + '_transfer')
else:
    path_to_generated_files = os.path.join(args.path_to_main_folder, 
        args.kind_of_oars.replace(' ', '_'), 'dr_' + \
            str(args.dropout_value) + '_o_' + args.optim + '_lr_' + \
                    str(args.lr) + '_e_' + str(args.n_epochs))
Path(path_to_generated_files).mkdir(parents=True, exist_ok=True)

###############################################
## Splitting
###############################################
# Load IDs of patients with the required 16 oars
IDs = np.load(os.path.join('stats', 'oars_proportion', '16_oars_IDs.npy')) 
# 430 patients

# Split in train 70%, validation 15%, test 15%
train_IDs, other_IDs = train_test_split(IDs, test_size=0.3)
validation_IDs, test_IDs = train_test_split(other_IDs, test_size=0.5)

# Save for testing
np.save(os.path.join(path_to_generated_files, 'train_IDs'), train_IDs)
np.save(os.path.join(path_to_generated_files, 'validation_IDs'), 
                        validation_IDs)
np.save(os.path.join(path_to_generated_files, "test_IDs"), test_IDs)

###############################################
## Parameters
###############################################
h5_dataset = h5py.File(os.path.join('..', 
                                    'data', 
                                    'regenerated_dataset.h5'), "r")

n_input_channels= 1

training_params = {'patch_dim': (256, 256, 64),
          'batch_size': 1,
          'dataset': h5_dataset,
          'shuffle': True,
          'augmentation': args.augmentation}

validation_params = {'patch_dim': (256, 256, 64),
          'batch_size': 1,
          'dataset': h5_dataset,
          'shuffle': True,
          'augmentation': False}

# Generators
training_generator = DataGenerator("train", train_IDs, list_oars, 
                                   **training_params)
validation_generator = DataGenerator("validation", validation_IDs, list_oars,
                                     **validation_params)

# Define model
input_shape = (training_params['patch_dim'][0], 
               training_params['patch_dim'][1],
               training_params['patch_dim'][2], 
               n_input_channels)
model = unet_3D(input_shape=input_shape, 
                number_of_pooling=args.number_of_pooling, 
                dropout=args.dropout_value, 
                optim=args.optim, 
                lr=args.lr)

# Load pretrained weights
if args.initial_weights is not None:
    model = load_pretrained_weights(model, input_shape, args)

# Callbacks
mc = ModelCheckpoint(os.path.join(path_to_generated_files, 'best_model.h5'), 
                        monitor='val_dice_coefficient', mode='max', 
                        save_best_only=True, verbose=1)

callbacks = [mc]

###############################################
## Training
###############################################
history = model.fit_generator(generator=training_generator, 
                            validation_data=validation_generator,
                            epochs=args.n_epochs,
                            callbacks=callbacks,
                            max_queue_size=16,
                            workers=8) # TODO, adapt to .fit
model.save_weights(os.path.join(path_to_generated_files, "model.h5"))

###############################################
## Results
###############################################
# Get training and validation loss histories
training_loss = history.history['loss']
np.save(os.path.join(path_to_generated_files, 'training_loss'), training_loss)
validation_loss = history.history['val_loss']
np.save(os.path.join(path_to_generated_files, 'validation_loss'), 
                        validation_loss)

# Get training and validation accuracy histories
training_accuracy = history.history['dice_coefficient']
np.save(os.path.join(path_to_generated_files, 'training_dice'), 
                        training_accuracy)
validation_accuracy = history.history['val_dice_coefficient']
np.save(os.path.join(path_to_generated_files, 'validation_dice'), 
                        validation_accuracy)

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize and save loss/accuracy
plt.figure()
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(path_to_generated_files, "loss.png"))

plt.figure()
plt.plot(epoch_count, training_accuracy, 'r--')
plt.plot(epoch_count, validation_accuracy, 'b-')
plt.legend(['Training Dice', 'Validation Dice'])
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.savefig(os.path.join(path_to_generated_files, "dice.png"))
