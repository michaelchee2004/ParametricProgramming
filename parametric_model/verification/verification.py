import numpy as np

from parametric_model.config.core import config
from parametric_model.generate_regions import mp_data
from parametric_model.processing.inputs import get_zeros_rows_index, load_mp
from parametric_model.solvers.generic_solver import GenericSolver

mp = load_mp()
Q, m, A, b, theta_size = mp_data()
Q_x = Q[:theta_size, :theta_size]
m_x = m[:theta_size]
A_x = A[:, :theta_size]
zeros_rows = get_zeros_rows_index(A_x)
A_x = np.delete(A_x, zeros_rows, axis=0)

np.random.seed(0)
random_inputs = np.random.uniform(
    low=0.0, high=1.0, size=(config.app_config.verif_test_size, mp.x_size)
)

for row in random_inputs:
    mp_soln = mp.get_soln(row)
    b_x = b - np.dot(A[:, -theta_size:], row.reshape(theta_size, -1)).reshape(-1)
    b_x = np.delete(b_x, zeros_rows)
    opt = GenericSolver(A_x, b_x, m_x, Q=Q_x)
    opt.solve()
    results_match = np.allclose(mp_soln, opt.soln, rtol=0.0, atol=1e-6)
    print(results_match)
