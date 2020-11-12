#################
### IMPORTS
#################
# Math
import numpy as np

# DeepL


# IO
import h5py
import os
import csv
import time
import sys

#######################################################################################################################
### SUBFUNCTIONS
#######################################################################################################################
# The channels order is as follows : (may change, for now it's based on most occurences, decreasing order)
    # 0 : canal medullaire
    # 1 : canal medul pv
    # 2 : oesophage
    # 3 : cavite orale
    # 4 : mandibule
    # 5 : parotide g
    # 6 : parotide d
    # 7 : tronc
    # 8 : trachee
    # 9 : oreille int g
    # 10 : oreille int d
    # 11 : oeil g
    # 12 : oeil d
    # 13 : sous-max g
    # 14 : tronc pv
    # 15 : sous-max d
    # 16 : nerf optique g
#######################################################################################################################
# Returns the corresponding channel number given a channel name
def name_to_number(channel_name):
    if (channel_name == "canal medullaire"):
        return 0
    elif (channel_name == "canal medul pv"):
        return 1
    elif (channel_name == "oesophage"):
        return 2
    elif (channel_name == "cavite orale"):
        return 3
    elif (channel_name == "mandibule"):
        return 4
    elif (channel_name == "parotide g"):
        return 5
    elif (channel_name == "parotide d"):
        return 6
    elif (channel_name == "tronc"):
        return 7
    elif (channel_name == "trachee"):
        return 8
    elif (channel_name == "oreille int g"):
        return 9
    elif (channel_name == "oreille int d"):
        return 10
    elif (channel_name == "oeil g"):
        return 11
    elif (channel_name == "oeil d"):
        return 12
    elif (channel_name == "sous-max g"):
        return 13
    elif (channel_name == "tronc pv"):
        return 14
    elif (channel_name == "sous-max d"):
        return 15
    elif (channel_name == "nerf optique g"):
        return 16
    else:
        raise NameError("Unknown channel: %s" % channel_name)

# Returns the corresponding channel name given a channel number
def number_to_name(channel_number):
    if (channel_number == 0):
        return "canal medullaire"
    elif (channel_number == 1):
        return "canal medul pv"
    elif (channel_number == 2):
        return "oesophage"
    elif (channel_number == 3):
        return "cavite orale"
    elif (channel_number == 4):
        return "mandibule"
    elif (channel_number == 5):
        return "parotide g"
    elif (channel_number == 6):
        return "parotide d"
    elif (channel_number == 7):
        return "tronc"
    elif (channel_number == 8):
        return "trachee"
    elif (channel_number == 9):
        return "oreille int g"
    elif (channel_number == 10):
        return "oreille int d"
    elif (channel_number == 11):
        return "oeil g"
    elif (channel_number == 12):
        return "oeil d"
    elif (channel_number == 13):
        return "sous-max g"
    elif (channel_number == 14):
        return "tronc pv"
    elif (channel_number == 15):
        return "sous-max d"
    elif (channel_number == 16):
        return "nerf optique g"
    else:
        raise NameError("Unknown channel: %s" % channel_number)

# 

#######################################################################################################################
### MAIN
#######################################################################################################################
## Init
# Paths
pwd = os.getcwd()
path_to_data = os.path.join(pwd, "..", "data", "CHUM", "h5_v2")

# Constants
nb_oars = 17
max_dim = (512, 512, 512)

#######################################################################################################################
### OARS LOCATION
#######################################################################################################################
# get_oars_location generates a npy file containing all summed masks with shape (nb_oars, max_dim1, max_dim2, max_dim3)
def get_oars_location():

    t0 = time.time()

    # Init
    sum_masks = np.zeros((nb_oars, max_dim[0], max_dim[1], max_dim[2]))
    tumor_volumes = ["ptv 1", "ctv 1", "gtv 1"]

    # Go through the data
    for file in os.listdir(path_to_data):

        # Load data
        data = h5py.File(os.path.join(path_to_data, file), "r")

        # 
        if (data["masks"].shape[0] >= 20):

            # Go through the channels
            h5_index = 0
            for channel_name in data["masks"].attrs["names"]:
                if channel_name not in tumor_volumes:
                    sum_masks[name_to_number(channel_name), :, :, 0:data["masks"][h5_index].shape[2]] += data["masks"][h5_index, :, :, :]
                h5_index += 1

    # Save the summed masks
    try:
        os.mkdir(os.path.join("stats", "oars_location"))
    except OSError as error: 
        print(error)
    np.save(os.path.join("stats", "oars_location", "summed_masks"), sum_masks)

#######################################################################################################################
# get_oars_limits generates a csv file containing the minimum index (first row) and maximum index (second row) of nonzero values in the summed_masks (cf get_oars_location)
def get_oars_limits():

    # Init

    # Load summed_masks
    try:
        summed_masks = np.load(os.path.join("stats", "oars_location", "summed_masks.npy"))
    except FileNotFoundError:
        print("File not found, first get the oar locations!")
        sys.exit()

    # Generate csv, a column by oar, first line = min, second line = max
    with open(os.path.join("stats", "oars_location", "oars_limits.csv"), 'w', newline='') as csv_file:
        fieldnames = [number_to_name(x) for x in range(nb_oars)]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Init
        min_dim1 = np.zeros(nb_oars)
        min_dim2 = np.zeros(nb_oars)
        min_dim3 = np.zeros(nb_oars)
        max_dim1 = np.zeros(nb_oars)
        max_dim2 = np.zeros(nb_oars)
        max_dim3 = np.zeros(nb_oars)
    
        # Find dim1, dim2, dim3 limits
        for channel_index in range(summed_masks.shape[0]):
            list_of_nonzero_values = np.where(summed_masks[channel_index])
            min_dim1[channel_index], max_dim1[channel_index] = min(list_of_nonzero_values[0]), max(list_of_nonzero_values[0])
            min_dim2[channel_index], max_dim2[channel_index] = min(list_of_nonzero_values[1]), max(list_of_nonzero_values[1])
            min_dim3[channel_index], max_dim3[channel_index] = min(list_of_nonzero_values[2]), max(list_of_nonzero_values[2])

        # Write the stats: first line = min, second line = max
        first_row = {}
        second_row = {}
        count = 0
        for field in fieldnames:
            first_row[field] = (min_dim1[count], min_dim2[count], min_dim3[count])
            second_row[field] = (max_dim1[count], max_dim2[count], max_dim3[count])
            count += 1

        writer.writerow(first_row)
        writer.writerow(second_row)


#######################################################################################################################
# generate csv file containing number and percentage of nonzero pixels per organ
def get_oars_proportion():

    # Init
    IDs = np.load(os.path.join('stats', 'oars_proportion', '20_plus_IDs.npy'))
    tumor_volumes = ["ptv 1", "ctv 1", "gtv 1"]

    # Init csv
    with open(os.path.join("stats", "oars_proportion", "oars_proportion.csv"), 'w', newline='') as csv_file:

        fieldnames_number = [number_to_name(x) + ' (number)' for x in range(nb_oars)]
        fieldnames_percentage = [number_to_name(x) + ' (percentage)' for x in range(nb_oars)]

        fieldnames = ['ID'] + [number_to_name(x) + ' ' + format for x in range(nb_oars) for format in ['(number)', '(percentage)']]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # For each patient
        for ID in IDs:

            # Init
            row = {}
            number_of_nonzero = np.zeros(nb_oars)
            percentage_of_nonzero = np.zeros(nb_oars)

            # For each organ
            data = h5py.File(os.path.join(path_to_data, ID + '.h5'), "r")

            #print(data["masks"].shape[1], data["masks"].shape[2], data["masks"].shape[3])

            # Go through the channels
            h5_index = 0
            for channel_name in data["masks"].attrs["names"]:
                if channel_name not in tumor_volumes:
                    # a. Find number of nonzero values
                    number_of_nonzero[name_to_number(channel_name)] = len(np.where(data["masks"][h5_index, :, :, :])[0])
                    # b. Divide by total number of values
                    percentage_of_nonzero[name_to_number(channel_name)] = number_of_nonzero[name_to_number(channel_name)]/(data["masks"].shape[1]*data["masks"].shape[2]*data["masks"].shape[3])
                h5_index += 1

            #print(number_of_nonzero)
            #print(percentage_of_nonzero)

            # Fill the csv
            row['ID'] = ID

            count_fieldname = 0
            for fieldname in fieldnames_number:
                row[fieldname] = number_of_nonzero[count_fieldname]
                count_fieldname += 1

            count_fieldname = 0
            for fieldname in fieldnames_percentage:
                row[fieldname] = percentage_of_nonzero[count_fieldname]
                count_fieldname += 1

            writer.writerow(row)

        # Closing
        csv_file.close()

#######################################################################################################################
# generate npy file containing list of IDs of patient having 20+ OARs
def get_ids_20_plus():

    IDs = []
    # Go through the data
    for file in os.listdir(path_to_data):

        # Load data
        data = h5py.File(os.path.join(path_to_data, file), "r")

        # 
        if (data["masks"].shape[0] >= 20):
            IDs.append(file.split('.')[0])
        
    np.save(os.path.join('stats', 'oars_proportion', '20_plus_IDs'), IDs)

#######################################################################################################################
# generate npy file containing list of IDs of patient having the following OARs: 
def get_ids_16_oars():

    list_oars = ["canal medullaire", "canal medul pv", "oesophage", "cavite orale", "mandibule", "parotide g", "parotide d", "tronc", "trachee", "oreille int g", "oreille int d", "oeil g", "oeil d", "sous-max g", "tronc pv", "sous-max d"]

    IDs = []
    # Go through the data
    for file in os.listdir(path_to_data):

        # Load data
        data = h5py.File(os.path.join(path_to_data, file), "r")

        # Check that file contains all needed oars
        if (all(e in data["masks"].attrs["names"] for e in list_oars)):
            IDs.append(file.split('.')[0])

        print(len(IDs))
        
    np.save(os.path.join('stats', 'oars_proportion', '16_oars_IDs'), IDs)

#######################################################################################################################
# get average input size
def get_average_input_height():

    # Init
    IDs = np.load(os.path.join('stats', 'oars_proportion', '20_plus_IDs.npy'))

    average_height = 0
    # For each patient
    for ID in IDs:

        # Load
        data = h5py.File(os.path.join(path_to_data, ID + '.h5'), "r")

        # Accumulate heights
        average_height += data["masks"].shape[3]

    print("Average height is: ", average_height/len(IDs))





