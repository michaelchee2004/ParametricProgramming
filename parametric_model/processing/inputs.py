import numpy as np

from numpy.core.numeric import full
from pathlib import Path
# from typing import Union

import parametric_model
from parametric_model.config.core import config


MODEL_ROOT = Path(parametric_model.__file__).parent
DATA_DIR = MODEL_ROOT / 'data'


def read_Ab(file_name):
    data_file_path = DATA_DIR / file_name
    if data_file_path.is_file():
        Ab = np.loadtxt(data_file_path)
        A = Ab[:, :-1]
        b = Ab[:, -1].flatten()
        return A, b
    raise OSError('Did not find file at path: {data_file_path}')


def read_QmAb(file_name):
    data_file_path = DATA_DIR / file_name
    if data_file_path.is_file():
        QmAb = np.loadtxt(data_file_path)
        x_size = get_cols(QmAb) - 1
        Q = QmAb[:x_size, :-1]
        m = QmAb[x_size, :-1]
        A = QmAb[x_size+1:, :-1]
        b = QmAb[x_size+1:, -1].flatten()
        return Q, m, A, b
    raise OSError('Did not find file at path: {data_file_path}')


def read_QmA_theta_b(file_name):
    data_file_path = DATA_DIR / file_name
    if data_file_path.is_file():
        data = np.loadtxt(data_file_path, delimiter=',')
        theta_size = int(data[0, 0])
        x_size = get_cols(data) - theta_size - 1
        Q = data[1:1+x_size+theta_size, :-1]
        m = data[1+x_size+theta_size, :-1]
        A = data[1+x_size+theta_size+1:, :-1]
        b = data[1+x_size+theta_size+1:, -1].flatten()
        return Q, m, A, b, theta_size
    raise OSError('Did not find file at path: {data_file_path}')


def get_rows(M):
    """Get number of rows in a 2D array.
    """
    return np.shape(M)[0]


def get_cols(M):
    """Get number of columns in a 2D array.
    """
    return np.shape(M)[1]


def matrix_to_dict(M):
    """Convert a 1D array into a dict for pyomo input initialisation.
    """
    # makes sure input is ndarray, or else indexing fails
    if type(M) != 'numpy.ndarray':
        M = np.array(M)        

    M_dict = {}
    for i in range(get_rows(M)):
        for j in range(get_cols(M)):
            M_dict[(i+1, j+1)] = M[i, j]

    return M_dict


def vector_to_dict(V):
    """Convert a 1D vector into a dict for pyomo input initialisation.
    """
    # makes sure input is ndarray, or else indexing fails
    if type(V) != 'numpy.ndarray':
        V = np.array(V)

    V_dict = {}
    for i in range(get_rows(V)):
        V_dict[(i+1)] = V[i]

    return V_dict


def get_zeros_rows_index(matrix):
    """Return rows of a matrix where all values are close to zero.
    
    Args:
        matrix (ndarray): 2D array, i.e. a matrix

    Returns: 
        ndarray: 1D array of row indices where the rows are clsoe to zero.
    """
    element_is_zero = np.isclose(matrix, 0.0,
                                 rtol=0.0,
                                 atol=config.other_config.allclose_tol)
    row_is_zeros = np.all(element_is_zero, axis=1)
    return np.where(row_is_zeros)


def check_duplicates(checked_object, checklist=None):
    """Check if a 2D array has any duplicate rows.

    If a checklist is not provided, the function checks within the array itself
    whether there are duplicate rows. If a checklist is provided, the function
    checks the array aginst the checklist, and delete rows that also appear on the 
    checklist.

    Args:
        checked_object (ndarray): 2D array where the duplicates are to be found
        checklist (ndarray, optional): 1D array to compare against. Defaults to None

    Returns:
        ndarray: row indices where duplicates are found    
    """
    if checklist is None:
        checklist_is_target = True
        checklist = checked_object
    else:
        checklist_is_target = False

    dup_rows = []
    # Take each row in check_object,
    for row in range(checked_object.shape[0]):
        duplicate_loc = np.where(
            np.all(
                np.isclose(checklist,
                           checked_object[row],
                           rtol=0,
                           atol=config.other_config.allclose_tol),
                axis=1))

        # np where returns a tuple, so duplicate_loc needs [0]
        if checklist_is_target:
            if duplicate_loc[0].shape[0] > 1:
                dup_rows.append(duplicate_loc[0][1:])
        else:
            if duplicate_loc[0].shape[0] > 0:
                dup_rows.append(row)

    dup_rows = np.unique(dup_rows)
    return dup_rows


def remove_duplicates(checked_object, checklist=None):
    """Check if a 2D array has any duplicate rows, and also delete them. 

    See function 'check_duplicates'.

    Args:
        checked_object (ndarray): 2D array where the duplicates are to be found
        checklist (ndarray, optional): 1D array to compare against. Defaults to None

    Returns:
        ndarray: row indices where duplicates are found   
    """
    dup_rows = check_duplicates(checked_object, checklist).tolist()
    return np.delete(checked_object, dup_rows, axis=0)


