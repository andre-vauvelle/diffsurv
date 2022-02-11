import os
import _pickle as pickle

import dill as dill


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _create_folder_if_not_exist(filename):
    """ Makes trial_n folder if the folder component of the baseline_path does not already exist. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_pickle(obj, filename, use_dill=False, protocol=4, create_folder=True):
    """ Basic pickle/dill dumping.
    Given trial_n python object and trial_n baseline_path, the method will save the object under that baseline_path.
    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        use_dill (bool): Set True to save using dill.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.
    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        if not use_dill:
            pickle.dump(obj, file, protocol=protocol)
        else:
            dill.dump(obj, file)


def load_pickle(filename, use_dill=False):
    """ Basic dill/pickle load function.
    Args:
        filename (str): Location of the object.
        use_dill (bool): Set True to load with dill.
    Returns:
        python object: The loaded object.
    """
    with open(filename, 'rb') as file:
        if not use_dill:
            obj = pickle.load(file)
        else:
            obj = dill.load(file)
    return obj
