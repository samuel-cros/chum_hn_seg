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

# IO
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import h5py

###############################################
## Limit memory allocation to minimum needed # TOTEST
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
seed_value = 38

# "canal medullaire", "canal medul pv", "oesophage", "cavite orale", "mandibule", "parotide g", "parotide d", "tronc", "trachee", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "tronc pv", "sous-max d", "nerf optique g"
#list_oars = ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", "mandibule", "parotide g", "parotide d", "tronc", "trachee", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "tronc pv", "sous-max d", "nerf optique g"]

# Manage OARs
all_oars = ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", "mandibule", "parotide g", "parotide d", "tronc", "trachee", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "tronc pv", "sous-max d", "nerf optique g"]

# Additional params
dropout_value = '0.0'
n_convolutions_per_block = '2'

if len(sys.argv) >= 7:
    path_to_main_folder = sys.argv[1]
    model_depth = sys.argv[2]
    kind_of_oars = sys.argv[3] # down, up
    optim = sys.argv[4]
    lr = sys.argv[5]
    n_epochs = sys.argv[6]
    if len(sys.argv) == 8:
        initial_weights = sys.argv[7]

    # Manage model depth
    if model_depth == '64':
        from unet_seg_64 import unet
    elif model_depth == '512':
        from unet_seg_512 import unet
    else:
        raise NameError('Unhandled model depth: ' + model_depth)

else:
    print("Wrong number of arguments, see example below.")
    print("python train_model.py output_folder model_depth kind_of_oars optim lr epochs initial_weights")
    print("    -> format for model_depth: 64 or 512")
    print("    -> format for kind_of_oars: up or down or all or one OAR_NAME")
    print("    -> format for initial_weights: path to model")
    sys.exit()

if kind_of_oars == 'down':
    list_oars = ["canal medullaire", "canal medul pv", "cavite orale", "oesophage", "mandibule", "tronc", "trachee", "tronc pv"]
elif kind_of_oars == 'up':
    list_oars = ["parotide g", "parotide d", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "sous-max d", "nerf optique g"]
elif kind_of_oars == 'all':
    list_oars = all_oars
elif kind_of_oars == 'parotides':
    list_oars = ['parotide d', 'parotide g']
elif kind_of_oars == 'yeux':
    list_oars = ['oeil d', 'oeil g']
elif kind_of_oars == 'sous-maxs':
    list_oars = ['sous-max d', 'sous-max g']
elif kind_of_oars == 'oreilles':
    list_oars = ['oreille int d', 'oreille int g']
# HANDLES SINGLE ORGAN SEG
else:
    if kind_of_oars in all_oars:
        list_oars = [kind_of_oars]
    else:
        raise NameError('Unknown kind of oars: ' + kind_of_oars)

# Manage folder for generated files
Path(path_to_main_folder).mkdir(parents=True, exist_ok=True)
Path(os.path.join(path_to_main_folder, kind_of_oars.replace(' ', '_'))).mkdir(parents=True, exist_ok=True)
if len(sys.argv) == 8:
    path_to_generated_files = os.path.join(path_to_main_folder, kind_of_oars.replace(' ', '_'), 'dr_' + dropout_value + '_nconv_' + n_convolutions_per_block + '_o_' + optim + '_lr_' + lr + '_e_' + n_epochs + '_transfer')
else:
    path_to_generated_files = os.path.join(path_to_main_folder, kind_of_oars.replace(' ', '_'), 'dr_' + dropout_value + '_nconv_' + n_convolutions_per_block + '_o_' + optim + '_lr_' + lr + '_e_' + n_epochs)
Path(path_to_generated_files).mkdir(parents=True, exist_ok=True)

###############################################
## Splitting
###############################################
# Load IDs
IDs = np.load(os.path.join('stats', 'oars_proportion', '16_oars_IDs.npy')) # 430 patients
IDs = IDs[:200] # TEST

# Split in train 70%, validation 15%, test 15%
train_IDs, other_IDs = train_test_split(IDs, test_size=0.3, random_state=seed_value)
validation_IDs, test_IDs = train_test_split(other_IDs, test_size=0.5, random_state=seed_value)

# Save for testing
np.save(os.path.join(path_to_generated_files, 'train_IDs'), train_IDs)
np.save(os.path.join(path_to_generated_files, 'validation_IDs'), validation_IDs)
np.save(os.path.join(path_to_generated_files, "test_IDs"), test_IDs)

###############################################
## Parameters
###############################################
h5_dataset = h5py.File(os.path.join('..', 'data', 'CHUM', 'h5_v3', 'regenerated_dataset.h5'), "r")

params = {'patch_dim': (256, 256, 64),
          'batch_size': 1,
          'n_input_channels': 1,
          'n_output_channels': 1,
          'dataset': h5_dataset,
          'shuffle': True}

# Generators
training_generator = DataGenerator("train", train_IDs, list_oars, **params)
validation_generator = DataGenerator("validation", validation_IDs, list_oars, **params)

# Define model
model = unet((params['patch_dim'][0], params['patch_dim'][1], params['patch_dim'][2], params['n_input_channels']), params['n_output_channels'], float(dropout_value), int(n_convolutions_per_block), optim, float(lr))

# Load pretrained weights
if len(sys.argv) == 8:
    model.load_weights(initial_weights)

# Callbacks
mc = ModelCheckpoint(os.path.join(path_to_generated_files, 'best_model.h5'), monitor='val_dice_coefficient', mode='max', save_best_only=True, verbose=1)

callbacks = [mc]

###############################################
## Training
###############################################
history = model.fit_generator(generator=training_generator, 
                            validation_data=validation_generator,
                            epochs=int(n_epochs),
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
np.save(os.path.join(path_to_generated_files, 'validation_loss'), validation_loss)

# Get training and validation accuracy histories
#training_accuracy = history.history['accuracy']
#validation_accuracy = history.history['val_accuracy']

training_accuracy = history.history['dice_coefficient']
np.save(os.path.join(path_to_generated_files, 'training_dice'), training_accuracy)
validation_accuracy = history.history['val_dice_coefficient']
np.save(os.path.join(path_to_generated_files, 'validation_dice'), validation_accuracy)

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
