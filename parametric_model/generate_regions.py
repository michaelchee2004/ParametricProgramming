import collections

from parametric_model.config.core import config
from parametric_model.processing.inputs import read_QmA_theta_b, save_mp
from parametric_model.solvers.region_generator import ParametricSolver


def mp_data():
    Q, m, A, b, theta_size = read_QmA_theta_b(config.app_config.mp_test_data_file)
    QmAb_theta_collection = collections.namedtuple(
        "QmA_theta_b", ["Q", "m", "A", "b", "theta_size"]
    )
    QmAb_theta = QmAb_theta_collection(Q, m, A, b, theta_size)
    return QmAb_theta


Q, m, A, b, theta_size = mp_data()
mp = ParametricSolver(A, b, m, theta_size, Q=Q)
mp.solve()
save_mp(mp)
