import collections
import pytest

from parametric_model.config.core import config
from parametric_model.processing.inputs import read_Ab


@pytest.fixture()
def rc_test_data():
    A, b = read_Ab(config.app_config.rc_test_data_file)
    Ab_collection = collections.namedtuple('Ab', ['A', 'b'])
    Ab = Ab_collection(A, b)
    return Ab

