import numpy as np

from numpy.core.numeric import full
from pathlib import Path
from typing import Union

import parametric_model
from parametric_model.config.core import config


MODEL_ROOT = Path(parametric_model.__file__).parent
DATA_DIR = MODEL_ROOT / 'data'


def read_Ab(file_name: Path) -> np.ndarray:
    data_file_path = DATA_DIR / file_name
    if data_file_path.is_file():
        Ab = np.loadtxt(data_file_path)
        A = Ab[:, :-1]
        b = Ab[:, -1].flatten()
        return A, b
    raise OSError('Did not find file at path: {data_file_path}')


def read_QmAb(file_name: Path) -> np.ndarray:
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


def read_QmA_theta_b(file_name: Path) -> np.ndarray:
    data_file_path = DATA_DIR / file_name
    if data_file_path.is_file():
        data = np.loadtxt(data_file_path)
        theta_size = data[0, 0]
        x_size = get_cols(data) - theta_size - 1
        Q = data[1:1+x_size, :-theta_size-1]
        m = data[1+x_size, :-theta_size-1]
        A = data[1+x_size+1:, :-theta_size-1]
        theta = data[1+x_size+1:, 
                     x_size:x_size+theta_size]
        b = data[1+x_size+1:, -1].flatten()
        return Q, m, A, theta, b
    raise OSError('Did not find file at path: {data_file_path}')


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
            M_dict[(i+1, j+1)] = M[i, j]

    return M_dict


def vector_to_dict(V: Union[list, np.ndarray]) -> dict:
    """
    Convert a 1d vector into a dict for pyomo input initialisation.
    """
    # makes sure input is ndarray, or else indexing fails
    if type(V) != 'numpy.ndarray':
        V = np.array(V)

    V_dict = {}
    for i in range(get_rows(V)):
        V_dict[(i+1)] = V[i]

    return V_dict


def get_zeros_rows_index(matrix):
    element_is_zero = np.isclose(matrix, 0.0,
                               rtol=0.0,
                               atol=config.other_config.allclose_tol)
    row_is_zeros = np.all(element_is_zero, axis=1)
    return np.where(row_is_zeros)
