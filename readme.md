# CHUM H&N segmentation project #

This repository contains the final set of codes that allowed me to produce accurate organ segmentation for Head and Neck (H&N) cancer patients on a dataset provided by the Centre Hospitalier de l'Université de Montréal (CHUM).

### Train model (train_model.py) ###

Launches training and generates model weights, Dice values (.npy) and curves (.png), train/validation/test IDs.

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


### Test model (test_model.py) ###

Launches testing and generates Dice scores per patient and average across a given set of the data.

#### Arguments ####

* __path__, __path_to_model_folder__ : A string specifiying the path to the model folder where the model file is located and where outpus will be generated.
* __mname__, __model_name__ : An string specifying the name of the model file.
* __oar__, __oar_name__ : A string specifying the name of the OAR to segment.
* __set__, __kind_of_set__ : A string specifying the set to generated for (either 'train', 'validation', 'test' or 'manual').
* __ids__, __list_of_ids__ : Multiple strings specifying the IDs of patients to be tested (required when set is 'manual').

### Combine outputs (combine_outputs_learned.py) ###

Launches output combination or combined prediction respectively generating classifier weights or Dice scores per patient and average across a given set of data.

#### Arguments ####

* __paths__, __path_to_models__ : Multiple string specifying the paths to the different models to combine.
* __o_path__, __output_path__ : A string specifying the folder that will contain the outputs.
* __oar__, __oar_name__ : A string specifying the name of the OAR to segment.
* __set__, __kind_of_set__ : A string specifying the set to generated for (either 'train', 'validation', 'test' or 'manual').
* __ids__, __list_of_ids__ : Multiple strings specifying the IDs of patients to be tested (required when set is 'manual').
* __seed__ : A string specifying the seed used for training the models.
* __learn__ / __predict__ : A flag to either learn the combination of ouputs or use to it to generate predictions.

### Get average model (get_average_model.py) ###

Computes a simple averaging of input models.

#### Arguments ####

* __p__, __paths__ : Multiple string specifying the paths to the different models to average.
* __output_path__ : A string specifying the folder that will contain the outputs.

### Data gen (data_gen.py) ###

Redefinition of keras.utils.Sequence DataGenerator, the main addition being a random sample generation combined with dilated binary maps. Prior to training, we computed a binary volume per patient where the ones indicate the center of a future patch. For that we dilated (see veugene/data_tools/binary_morpholgy) the training target volumes based on our patch-size so that every patch contains a good amount of pixels to predict.

### Model (model.py) ###

Model definition of a 3D U-Net using an existing package (see veugene/fcn_maker/model) for stability and ease of modification.

### Stats ###

Folder containing scripts used to compute useful stats on the dataset such as OAR locations (bounding boxes used in early implementation) or OAR proportions (availabilty per patient, space occupied in the volume per organ, etc).

### Sampling ###

Folder containing scripts used to generate the latest iteration of the dataset that includes dilation maps (that replaced the bounding box system).


