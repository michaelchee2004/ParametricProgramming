import numpy as np

from parametric_model.solvers.region_generator import ParametricSolver


def test_region_gen(mp_test_data):
    Q, m, A, b, theta_size = mp_test_data
    mp = ParametricSolver(A, b, Q, m, theta_size)
    mp.solve()
    # test no of regions
    assert len(mp.regions) == 5
    # test the last region
    assert np.allclose(
        mp.regions[4]['added_bound_A'],
        np.array([[-3.16515, -3.7546],
                  [0.07350042,  0.05290033]]),
        rtol=.0,
        atol=1e-6)
    assert np.allclose(
        mp.regions[4]['added_bound_b'],
        np.array([-3.582576,  0.06334921]),
        rtol=.0,
        atol=1e-6)
    assert np.allclose(
        mp.regions[4]['firm_bound_A'],
        np.array([[-1., 0.],
                  [0.,  1.]]),
        rtol=.0,
        atol=1e-6)
    assert np.allclose(
        mp.regions[4]['firm_bound_b'],
        np.array([0., 1.]),
        rtol=.0,
        atol=1e-6)
