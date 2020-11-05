## Imports
# DeepL
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras import activations
from keras import regularizers
from tensorflow.keras import losses

###########################################################################
## Parameters #############################################################
###########################################################################

kernel_value = 'he_normal'
#n_convolutions_per_block = 3 # 2 ok, 3 totest # -> now a parameter ! :D
#dropout_value = 0.0 # -> now a parameter ! :D
batch_norm = True

inner_activation = 'relu'
final_activation = 'sigmoid'

#current_lr = 5e-4 # -> now a parameter ! :D

###########################################################################
## Blocks #################################################################
###########################################################################

def conv_block(output_size, previous_layer, n_convolutions, activation, kernel_initializer, batch_norm, dropout):
    # Convolve
    block = previous_layer
    for i in range(n_convolutions):
        block = Conv3D(output_size, kernel_size = 3, padding = 'same', kernel_initializer = kernel_initializer)(block)
        # BN
        if batch_norm:
            block = BatchNormalization()(block)
    # Activate
    block = Activation(activation)(block)    
    # Pool
    pool = MaxPooling3D(pool_size=(2,2,2))(block)
    # Dropout
    if dropout > 0.0:
        pool = Dropout(dropout)(pool)
    return block, pool

def up_conv_block(output_size, previous_layer, skip_connections_layer, n_convolutions, activation, kernel_initializer, batch_norm, dropout):
    block = previous_layer
    # Deconvolve
    block = Conv3DTranspose(output_size, kernel_size = 3, padding = 'same', kernel_initializer = kernel_initializer)(UpSampling3D(size = (2,2,2))(block))
    # BN
    if batch_norm:
        block = BatchNormalization()(block)
    # Activate
    block = Activation(activation)(block)
    # Concatenate
    block = concatenate([skip_connections_layer, block], axis = 4)
    # Convolve
    for i in range(n_convolutions):
        block = Conv3D(output_size, kernel_size = 3, padding = 'same', kernel_initializer = kernel_initializer)(block)
        # BN
        if batch_norm:
            block = BatchNormalization()(block)
    # Activate
    block = Activation(activation)(block)
    # Dropout
    if dropout > 0.0:
        block = Dropout(dropout)(block)

    return block

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

# Weighted BCE / DICE
def bce_dice_loss(y_true, y_pred, bce_weight=1e-3, dice_weight=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return bce_weight * losses.binary_crossentropy(y_true_f, y_pred_f) - dice_weight * dice_coefficient(y_true, y_pred)

# Current
current_loss = dice_coefficient_loss # dice_coefficient_loss (good), bce_dice_loss (similar)
current_metric = dice_coefficient

###########################################################################
## Net ####################################################################
###########################################################################

def unet(input_size, n_output_channels, dropout_value, n_convolutions_per_block, optim, lr, pretrained_weights = None):
    inputs = Input(input_size)

    ###########################################################################
    ## Architecture ###########################################################
    ###########################################################################

    # x16 layers going down
    conv16, pool16 = conv_block(16, inputs, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x32 layers going down
    conv32, pool32 = conv_block(32, pool16, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x64 layers going down
    conv64, pool64 = conv_block(64, pool32, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x128 layers going down
    conv128, pool128 = conv_block(128, pool64, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x256 layers going down
    conv256, pool256 = conv_block(256, pool128, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x512 layers
    conv512, pool512 = conv_block(512, pool256, n_convolutions_per_block*2, inner_activation, kernel_value, batch_norm, dropout_value)
    # x256 layers going up
    upconv256 = up_conv_block(256, conv512, conv256, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x128 layers going up
    upconv128 = up_conv_block(128, upconv256, conv128, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    upconv64 = up_conv_block(64, upconv128, conv64, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x32 layers going up
    upconv32 = up_conv_block(32, upconv64, conv32, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # x16 layers going up
    upconv16 = up_conv_block(16, upconv32, conv16, n_convolutions_per_block, inner_activation, kernel_value, batch_norm, dropout_value)
    # Output
    convFIN = Conv3D(n_output_channels, kernel_size = 1, activation = final_activation)(upconv16)

    ###########################################################################
    ## Model ##################################################################
    ###########################################################################
    #print(convFIN.shape)

    model = Model(inputs = inputs, outputs = convFIN)

    if optim == 'adam':
        model.compile(optimizer = Adam(lr = lr), loss = current_loss, metrics = [current_metric])
    elif optim == 'rmsprop':
        model.compile(optimizer = RMSprop(lr = lr), loss = current_loss, metrics = [current_metric])
    else:
        raise NameError('Unknown optimizer.')

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
