import numpy as np

from parametric_model.solvers.generic_solver import GenericSolver

# from tests.conftest import qp_region_test_data


def test_quadratic_solver(qp_region_test_data):
    Q, m, A, W, b = qp_region_test_data
    Q = np.concatenate((Q, np.zeros((2, 2))), axis=1)
    Q = np.concatenate((Q, np.zeros((2, 4))), axis=0)
    m = np.concatenate((m, np.zeros((2))))
    A = np.concatenate((A, W), axis=1)
    qp = GenericSolver(A, b, m, Q=Q)
    qp.solve()
    assert np.allclose(
        qp.soln,
        np.array([1.04190127e-05, -3.24147062e-05, 4.45367918e-01, 5.78735081e-01]),
        rtol=0.0,
        atol=1e-6,
    )


# data = qp_region_test_data()
# test_quadratic_solver(data)
