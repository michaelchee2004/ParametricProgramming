import numpy as np

from parametric_model.solvers.region_solver import RegionSolver


def test_region_solver(qp_region_test_data):
    Q, m, A, b, theta_size = qp_region_test_data
    region_prob = RegionSolver(A, b, Q, m, theta_size)
    region_prob.solve()
    # test no of boundaries
    assert np.shape(region_prob.boundary_slope) == (4, 2)
    # test values in boundaries LHS
    assert np.allclose(
        region_prob.boundary_slope, 
        np.array([[-1.,          0.       ],
                  [ 0.,         1.        ],
                  [-3.16515,   -3.7546    ],
                  [ 0.07350042, 0.05290032]]),
        rtol=.0,
        atol=1e-6)
    # test values in boundaries RHS
    assert  np.allclose(
        region_prob.boundary_constant, 
        np.array([0.,
                  1.,
                  -3.582575,
                  0.06335]),
        rtol=.0,    
        atol=1e-6)
