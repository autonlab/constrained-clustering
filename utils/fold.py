import config
import os
import pickle
import numpy as np
from .utils import print_verbose
from .constraint import random_indices
from pmlb import classification_dataset_names, fetch_data

def createFold(dname, path = config.result, number_fold = 10, 
    min_points = 100, verbose = 0):
    """
        Creates fold for dataset: a file is created at the given path
        In which the train index is saved for each fold

        The training use 25% of the data and it is randomly assigned
    
        Arguments:
            dname {String} -- Dataset name
        
        Keyword Arguments:
            path {String} -- Path where to save the data
            number_fold {int} -- Number of fold (default: {10})
            min_points {int} -- Minimal points in dataset (default: {100})
            verbose {int} -- Verbose level (default: {0})
    """
    assert dname in classification_dataset_names, "Unknown dataset"
    
    # Read data and put them in good format for sklearn
    data, labelvector = fetch_data(dname, return_X_y = True, local_cache_dir = config.datadir)
    data = data.astype('float64')

    if len(labelvector) < min_points:
        print_verbose('Ignored - {}: dataset too small - {} points'.format(dname, len(labelvector)), verbose)
        return {}
    
    labels, counts = np.unique(labelvector, return_counts = True)
    classes = len(labels)
    print_verbose('{} : {} points in {} classes'.format(dname, len(labelvector), len(labels)), verbose)
    
    train_index, constraint_index = {}, {}
    for fold in range(number_fold):     
        # Split in train and test
        ## Stratified split
        train_index[fold] = []
        for label, count in zip(labels, counts):
            lentrain = int(0.25 * count)
            index_label = np.argwhere(labelvector == label).flatten()
            train_index[fold].extend(np.random.choice(index_label, size = lentrain, replace = False).tolist())

        np.random.shuffle(train_index[fold])

        # Compute constraints matrix
        ## Number constraint
        number_constraint = min(int(((len(train_index[fold])-1)*len(train_index[fold])/2.)), 10000)

        ## Indices computed only on train part
        constraint_index[fold] = random_indices(train_index[fold], number_constraint)

    # Save configuration
    info = {"Name": dname, "N_Classes": classes, "Labels": labelvector,
            "Constraint": constraint_index, "Train": train_index}
    pickle.dump(info, open(os.path.join(path, dname + "_fold.pickle"), 'wb'))

def readFold(dname, path = config.result, verbose = 0):
    """
        Iterate through all the folders at the given path

        Keyword Arguments:
            path {String} -- Path where to read the configuration

        Returns:
            Configuration
    """
    confFile = os.path.join(path, dname + "_fold.pickle")
    if not(os.path.isfile(confFile)):
        print_verbose("Dataset {} ignored".format(dname), verbose)
        return None

    return pickle.load(open(confFile, 'rb'))

def computedFold(path = config.result, verbose = 0):
    """
        Give the list of dataset for which we have fold

        Keyword Arguments:
            path {String} -- Path where to read the configuration

        Returns:
            List of fold
    """
    files = os.listdir(path)
    confFiles = [f[:f.index("_fold.pickle")] for f in files if "_fold.pickle" in f]
    return confFiles