import numpy as np

from parametric_model.solvers.generic_solver import GenericSolver


def test_quadratic_solver(qp_region_test_data):
    Q, m, A, b, theta_size = qp_region_test_data
    qp = GenericSolver(A, b, m, Q=Q)
    qp.solve()
    assert np.allclose(
        qp.soln, 
        np.array([9.87024640e-06, - 3.07055422e-05,  4.45368281e-01,  5.78734765e-01]),
        rtol=.0, 
        atol=1e-6)  