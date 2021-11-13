import numpy as np

from parametric_model.config.core import config
from parametric_model.processing.redundancy_checker import RedundancyChecker


def test_redundancy_checker(rc_test_data):
    A, b = rc_test_data
    checker = RedundancyChecker(A, b)
    reduced_A, reduced_b = checker.remove_redundancy()
    print(reduced_A)
    assert np.allclose(reduced_A, 
                       np.array([[-0.13793103,  4.20689655],
                                 [1.,           0.        ]]), 
                       rtol=0.0, 
                       atol=config.other_config.allclose_tol)

