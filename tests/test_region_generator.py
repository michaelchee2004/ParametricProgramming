import numpy as np

from parametric_model.solvers.region_generator import ParametricSolver


def test_qp_region_gen(qp_mp_test_data):
    Q, m, A, b, theta_size = qp_mp_test_data
    mp = ParametricSolver(A, b, m, theta_size, Q=Q)
    mp.solve()
    # test no of regions
    assert len(mp.regions) == 5
    # test the last region
    assert np.allclose(
        mp.regions[4]["added_bound_A"],
        np.array(
            [[-3.16515, -3.7546], [2.82163241, -2.09545779], [0.07350042, 0.05290033]]
        ),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.allclose(
        mp.regions[4]["added_bound_b"],
        np.array([-3.582576, 0.04398204, 0.06335021]),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.allclose(
        mp.regions[4]["firm_bound_A"],
        np.array([[-1.0, 0.0], [0.0, 1.0]]),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.allclose(
        mp.regions[4]["firm_bound_b"], np.array([0.0, 1.0]), rtol=0.0, atol=1e-6
    )


def test_lp_region_gen(lp_mp_test_data):
    Q, m, A, b, theta_size = lp_mp_test_data
    Q = None
    mp = ParametricSolver(A, b, m, theta_size)
    mp.solve()
    # test no of regions
    assert len(mp.regions) == 2
    # test the last region
    assert np.allclose(
        mp.regions[1]["added_bound_A"],
        np.array([[0.13793103, -4.20689655]]),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.allclose(
        mp.regions[1]["added_bound_b"], np.array([-896.55172414]), rtol=0.0, atol=1e-6
    )
    assert np.allclose(
        mp.regions[1]["firm_bound_A"],
        np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.allclose(
        mp.regions[1]["firm_bound_b"],
        np.array([0.0, 6000.0, 500.0]),
        rtol=0.0,
        atol=1e-6,
    )
