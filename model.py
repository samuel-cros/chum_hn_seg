## Imports
# Math
import math

# DeepL
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras import activations
from keras import regularizers
from tensorflow.keras import losses

from utils.fcn_maker.model import assemble_unet

###########################################################################
## Losses #################################################################
###########################################################################

# Dice coeff
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=(0,1,2,3))
    summation = K.sum(y_true + y_pred, axis=(0,1,2,3))
    dice_coeff = K.mean((2. * intersection + 1) / (summation + 1))
    return dice_coeff

# Dice loss
def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

###########################################################################
## Net ####################################################################
###########################################################################
def unet_3D(input_shape, model_depth, dropout, optim, lr):

    model = assemble_unet(input_shape=input_shape,
        init_num_filters=(16, 16), num_classes=1, num_pooling=int(math.log(model_depth, 2)-4), 
        short_skip=False, long_skip=True, long_skip_merge_mode='concat',
        upsample_mode='conv', dropout=dropout, normalization= BatchNormalization, weight_decay=None,
        init='he_normal', nonlinearity='relu', halve_features_on_upsample=True, ndim=3, verbose=True)

    if optim == 'adam':
            model.compile(optimizer = Adam(lr = lr), loss = dice_coefficient_loss, metrics = [dice_coefficient])
    elif optim == 'rmsprop':
        model.compile(optimizer = RMSprop(lr = lr), loss = dice_coefficient_loss, metrics = [dice_coefficient])
    else:
        raise NameError('Unknown optimizer.')

    model.summary()

    return model
