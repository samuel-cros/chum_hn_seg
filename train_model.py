## Imports
# Math
import numpy as np
import math

# DeepL
from sklearn.model_selection import train_test_split
#from data_generator_multi import DataGenerator
from data_generator_multi_for_avg import DataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# IO
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

###############################################
## Limit memory allocation to minimum needed # TOTEST
###############################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

###############################################
## Sub-functions
###############################################

## get_min_max_3D
# Returns
#   - min_3D: [min_along_x, min_along_y, min_along_z] i.e minimum position in 
# the volumes where the organ appears
#   - max_3D: [max_along_x, max_along_y, max_along_z] i.e maximum position in 
# the volumes where the organ appears
# Parameters
#   - reader: a dict reader of the csv containing the oar_locations (generated 
# via /stats/compute_stats.py)
#   - oar_name: the name of the oar handled
def parse_min_max(min_max):

    #print(min_max)

    parsed = []
    min_max_splitted = ((min_max.split('('))[1].split(')'))[0].split(',')
    for i in range(len(min_max_splitted)):
        parsed.append(int(float(min_max_splitted[i])))

    #print(parsed)

    return parsed

###############################################
## Splitting
###############################################
seed_value = 38

# Load csv data
data_info_reader = csv.DictReader(open('data_infos.csv'))

# Select only rows with min_nb_masks masks
dataset_size = 200
dataset_size_check = 0
min_nb_masks = 20 # 20 ~31%, 15 ~71%, 10 ~86%, 5 ~94% of 1045 patients

parser = argparse.ArgumentParser(description='Test a given model')

# Arguments
parser.add_argument('-path', '--path_to_main_folder', type=str, required=True,
                    help='Path to the output folder')
parser.add_argument('-depth', '--model_depth', type=str, required=True,
                    help='Depth of the model')
parser.add_argument('-oars', '--kind_of_oars', type=str, required=True,
                    help='Kind of oars to predict')
parser.add_argument('-o', '--optim', type=str, required=True,
                    help='Optimizer')
parser.add_argument('--lr', type=float, required=True, help='Learning rate')
parser.add_argument('-drop', '--dropout_value', type=float, required=True,
                    help='Dropout')
parser.add_argument('-nconv', '--n_conv_per_block', type=int,
                    help='Number of convolutions per block', default=2)
parser.add_argument('-e', '--n_epochs', type=int, required=True,
                    help='Number of epochs')
parser.add_argument('-w', '--initial_weights', type=str,
                    help='Path to the initial weights')

args = parser.parse_args()

# Manage model depth
if args.model_depth == '64':
    from unet_seg_64 import unet
elif args.model_depth == '512':
    from unet_seg_512 import unet
else:
    raise NameError('Unhandled model depth: ' + args.model_depth)

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
        raise NameError('Unknown kind of oars: ' + args.kind_of_oars)


# Manage folder for generated files
Path(args.path_to_main_folder).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.path_to_main_folder, 
                    args.kind_of_oars.replace(' ', '_'))).mkdir(parents=True, 
                                                            exist_ok=True)
if args.initial_weights is not None:
    path_to_generated_files = os.path.join(args.path_to_main_folder, 
        args.kind_of_oars.replace(' ', '_'), 
        'dr_' + str(args.dropout_value) + '_nconv_' + \
            str(args.n_conv_per_block) + '_o_' + args.optim + '_lr_' + \
                str(args.lr) + '_e_' + str(args.n_epochs) + '_transfer')
else:
    path_to_generated_files = os.path.join(args.path_to_main_folder, 
        args.kind_of_oars.replace(' ', '_'), 'dr_' + \
            str(args.dropout_value) + '_nconv_' + \
                str(args.n_conv_per_block) + '_o_' + args.optim + '_lr_' + \
                    str(args.lr) + '_e_' + str(args.n_epochs))
Path(path_to_generated_files).mkdir(parents=True, exist_ok=True)

IDs = []

# OARs locations
oars_locations_reader = \
    csv.DictReader(open(os.path.join('stats', 'oars_location', 
                                        'oars_limits_20_more.csv')))

rows = []
for row in oars_locations_reader:
    rows.append(row)

#print('rows', rows)

min_locations_dict = {}
for oar in list_oars:
    min_locations_dict[oar] = parse_min_max(rows[0][oar])

#print('min_locations_dict', min_locations_dict)

max_locations_dict = {}
for oar in list_oars:
    max_locations_dict[oar] = parse_min_max(rows[1][oar])

# Select data based on nb_masks
''' 
for row in data_info_reader:
    if (int(row['nb_masks']) >= min_nb_masks and dataset_size_check < 200):
        IDs.append(row['ID'])
        dataset_size_check += 1
'''

# Select data based on masks
#''' 
for row in data_info_reader:
    if (int(row['nb_masks']) >= min_nb_masks and \
            dataset_size_check < dataset_size):
        if all(oar in row['masks_names'] for oar in list_oars):
            IDs.append(row['ID'])
            dataset_size_check += 1
#'''

# Split in train 70%, validation 15%, test 15%
train_IDs, other_IDs = train_test_split(IDs, test_size=0.3, 
                                        random_state=seed_value)
validation_IDs, test_IDs = train_test_split(other_IDs, test_size=0.5, 
                                            random_state=seed_value)

# Save for testing
np.save(os.path.join(path_to_generated_files, 'train_IDs'), train_IDs)
np.save(os.path.join(path_to_generated_files, 'validation_IDs'), 
                        validation_IDs)
np.save(os.path.join(path_to_generated_files, "test_IDs"), test_IDs)

###############################################
## Parameters
###############################################
params = {'patch_dim': (256, 256, 64),
          'batch_size': 1,
          'n_input_channels': 1,
          'n_output_channels': 1,
          'shuffle': True} # 'n_output_channels': len(list_oars),

# Generators
training_generator = DataGenerator("train", train_IDs, list_oars, **params, 
                                    cropping=args.kind_of_oars, 
                                    min_locations_dict=min_locations_dict, 
                                    max_locations_dict=max_locations_dict, 
                                    augmentation=False)
validation_generator = DataGenerator("validation", validation_IDs, list_oars, 
                                        **params, cropping=args.kind_of_oars, 
                                        min_locations_dict=min_locations_dict, 
                                        max_locations_dict=max_locations_dict)

# Define model
model = unet((params['patch_dim'][0], params['patch_dim'][1], 
                params['patch_dim'][2], params['n_input_channels']), 
                params['n_output_channels'], args.dropout_value, 
                args.n_conv_per_block, args.optim, args.lr)

# Load pretrained weights
if args.initial_weights is not None:
    model.load_weights(args.initial_weights)

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
                            epochs=int(args.n_epochs),
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
#training_accuracy = history.history['accuracy']
#validation_accuracy = history.history['val_accuracy']

training_accuracy = history.history['dice_coefficient']
np.save(os.path.join(path_to_generated_files, 'training_accuracy'), 
                        training_accuracy)
validation_accuracy = history.history['val_dice_coefficient']
np.save(os.path.join(path_to_generated_files, 'validation_accuracy'), 
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
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(path_to_generated_files, "accuracy.png"))
