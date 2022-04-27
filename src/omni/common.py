import importlib
import os
import _pickle as pickle

import dill as dill


def safe_string(name):
    return name.replace('/', '_').replace('.', '_')


def unsafe_string(name):
    return name.replace('_', '.').replace('_', '/')


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


def save_feather_split(df, filename, n_files=10, create_folder=True):
    """ Save a dataframe to feather format.
    Args:
        df (pandas dataframe): The dataframe to be saved.
        filename (str): Location to save the file.
        n_files (int): Number of files to split the dataframe into.
    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save each split
    for i in range(n_files):
        df_split = df.iloc[i::n_files]
        df_split.reset_index(drop=True, inplace=True)
        df_split.to_feather(filename + '_' + str(i) + '.feather')


def init_class_from(class_path, init_args):
    # Init loss class from string
    module_name = '.'.join(class_path.split('.')[:-1])
    class_name = class_path.split('.')[-1]

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    class_(**init_args)
    return class_
