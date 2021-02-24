# CHUM H&N segmentation project #

This repository contains the final set of codes that allowed me to produce accurate organ segmentation for Head and Neck (H&N) cancer patients on a dataset provided by the Centre Hospitalier de l'Université de Montréal (CHUM).

### Train model ###

Launch training and generates model weights, dice values (.npy) and curves (.png), train/validation/test IDs.

#### Arguments ####

* __path__, __path_to_main_folder__ : A string specifiying the name of the folder that will contain the outputs.
* __pool__, __number_of_pooling__ : An integer specifying the number of pooling operations for the U-Net architecture.
* __oars__, __kind_of_oars__ : A string specifying the OAR or OARs to be segmented (see code for the different available modes for multi-segmentation, paired organ segmentation).
* __o__, __optim__ : A string specifiying the name optimizer to be used (supported optimizers: 'adam' or 'rmsprop').
* __lr__ : A float specifying the learning rate.
* __drop__, __dropout_value__ : A float specifying the dropout value.
* __e__, __n_epochs__ : An integer specifying the number of epochs.
* __w__, __initial_weights__ : A string specifying the path to the weights to optionally initialize the model.
* __aug__ / __no-aug__, __augmentation__ / __no-augmentation__ : A flag to either use data augmentation or not.


### Test model ###

#### Arguments ####

* __path__, __path_to_model_folder__ : A string specifiying the path to the model folder where the model file is located and where outpus will be generated.
* __mname__, __model_name__ : An string specifying the name of the model file.
* __oar__, __oar_name__ : A string specifying the name of the OAR to segment.
* __set__, __kind_of_set__ : A string specifying the set to generated for (either 'train', 'validation', 'test' or 'manual').
* __ids__, __list_of_ids__ : Multiple strings specifying the IDs of patients to be tested (required when set is 'manual').
