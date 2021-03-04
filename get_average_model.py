## Seeding
from numpy.random import seed
seed(5)

## Imports
# Math
import numpy as np

# DeepL
from model import unet_3D

# IO
import argparse
import sys

###############################################
## Input
###############################################

parser = argparse.ArgumentParser(description='Average weights across similar'+\
    'models')

# Arguments
parser.add_argument('-p', '--paths', nargs='+', type=str, required=True,
                    help='Paths to the different models to average')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to the output model to produce')

args = parser.parse_args()

params = {'patch_dim': (256, 256, 64),
          'batch_size': 1,
          'shuffle': True}

n_input_channels= 1

# Define model
input_shape = (params['patch_dim'][0], 
               params['patch_dim'][1],
               params['patch_dim'][2], 
               n_input_channels)
setup_model = unet_3D(input_shape=input_shape, 
                number_of_pooling=2, 
                dropout=0.0, 
                optim='adam', 
                lr=1e-3)

# Load each model and extract the weights
weights = []
for model_path in args.paths:
    setup_model.load_weights(model_path)
    weights.append(setup_model.get_weights())

# Average the weights
new_weights = []
for weights_list_tuple in zip(*weights): 
    new_weights.append(np.array([np.array(w).mean(axis=0) \
        for w in zip(*weights_list_tuple)]))

# Produce the final averaged model
new_model = unet_3D(input_shape=input_shape, 
                number_of_pooling=2, 
                dropout=0.0, 
                optim='adam', 
                lr=1e-3)

new_model.set_weights(new_weights)
new_model.save_weights(args.output_path)