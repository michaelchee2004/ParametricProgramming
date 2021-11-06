import numpy as np
from typing import Union

def get_rows(M: Union[list, np.ndarray]) -> int:
    """
    Get number of rows in a 2d matrix.
    """
    return np.shape(M)[0]


def get_cols(M: Union[list, np.ndarray]) -> int:
    """
    Get number of columns in a 2d matrix.
    """
    return np.shape(M)[1]


def matrix_to_dict(M: Union[list, np.ndarray]) -> dict:
    """
    Convert a matrix into a dict for pyomo input initialisation.
    """
    # makes sure input is ndarray, or else indexing fails
    if type(M) != 'numpy.ndarray':
        M = np.array(M)        

    M_dict = {}
    for i in range(get_rows(M)):
        for j in range(get_cols(M)):
            M_dict[(i, j)] = M[i, j]

    return M_dict

