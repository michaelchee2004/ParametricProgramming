import collections

import pytest

from parametric_model.config.core import config
from parametric_model.processing.inputs import read_Ab, read_QmA_theta_b


@pytest.fixture()
def rc_test_data():
    A, b = read_Ab(config.app_config.rc_test_data_file)
    Ab_collection = collections.namedtuple("Ab", ["A", "b"])
    Ab = Ab_collection(A, b)
    return Ab


@pytest.fixture()
def qp_region_test_data():
    Q, m, A, W, b = read_QmA_theta_b(
        config.app_config.qp_region_test_data_file
    )
    QmAb_theta_collection = collections.namedtuple(
        "QmA_theta_b", ["Q", "m", "A", "W", "b"]
    )
    QmAb_theta = QmAb_theta_collection(Q, m, A, W, b)
    return QmAb_theta


@pytest.fixture()
def qp_mp_test_data():
    Q, m, A, W, b = read_QmA_theta_b(config.app_config.qp_mp_test_data_file)
    QmAb_theta_collection = collections.namedtuple(
        "QmA_theta_b", ["Q", "m", "A", "W", "b"]
    )
    QmAb_theta = QmAb_theta_collection(Q, m, A, W, b)
    return QmAb_theta


@pytest.fixture()
def lp_region_test_data():
    Q, m, A, b, theta_size = read_QmA_theta_b(
        config.app_config.lp_region_test_data_file
    )
    QmAb_theta_collection = collections.namedtuple(
        "QmA_theta_b", ["Q", "m", "A", "b", "theta_size"]
    )
    QmAb_theta = QmAb_theta_collection(Q, m, A, b, theta_size)
    return QmAb_theta


@pytest.fixture()
def lp_mp_test_data():
    Q, m, A, b, theta_size = read_QmA_theta_b(config.app_config.lp_mp_test_data_file)
    QmAb_theta_collection = collections.namedtuple(
        "QmA_theta_b", ["Q", "m", "A", "b", "theta_size"]
    )
    QmAb_theta = QmAb_theta_collection(Q, m, A, b, theta_size)
    return QmAb_theta
