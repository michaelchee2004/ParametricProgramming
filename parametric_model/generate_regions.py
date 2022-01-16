import collections

from parametric_model.config.core import config
from parametric_model.processing.inputs import read_QmA_theta_b, save_mp
from parametric_model.solvers.region_generator import ParametricSolver


def mp_data():
    Q, m, A, W, b = read_QmA_theta_b(config.app_config.qp_mp_test_data_file)
    QmAb_theta_collection = collections.namedtuple(
        "QmA_theta_b", ["Q", "m", "A", "W", "b"]
    )
    QmAb_theta = QmAb_theta_collection(Q, m, A, W, b)
    return QmAb_theta


# A = np.array(
#     [
#         [1., 1., -1., 0.],
#         [5., -4., 0., 0.],
#         [-8., 22., 0., -1.],
#         [-4., -1., 0., 0.],
#         [0., 0., -1., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., -1.],
#         [0., 0., 0., 1.]
#     ]
# )

# b = np.array([13., 20., 121., -8., 10, 10., 100., 100.])

# Q = np.array(
#     [
#         [30. * 2., 0.,     0., 0.],
#         [0.,       1. * 2, 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]
#     ]
# )

# m = np.array([0., 0., 0., 0.])

# theta_size = 2

Q, m, A, W, b = mp_data()
mp = ParametricSolver(A, W, b, m, Q=Q)
mp.solve()
save_mp(mp)
