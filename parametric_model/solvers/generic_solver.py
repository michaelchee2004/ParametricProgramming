import numpy as np
import pyomo.environ as pmo

from parametric_model.config.core import config
from parametric_model.processing.inputs import (
    get_cols,
    get_rows,
    matrix_to_dict,
    vector_to_dict,
)


class GenericSolver:
    """A class for solving QPs.

    The problem is posed as
    min (XT)QX + mX
    s.t.
    Ax <= b
    where
    X: a vector of optimised variables x
    XT: transposed X
    Q: coefficients for qudratic terms
    m: coefficients for linear terms
    A: LHS coefficients in constraints list
    b: RHS constants in constraints list
    If problem is LP, Q is not supplied.

    Attributes:
        A (ndarray): 1D array, LHS of constraint matrix
        b (ndarray): 1D array, RHS of constraint matrix
        Q (ndarray, optional): 2D array, coefficients of quadratic terms in objective
        m (ndarray): 1D array, coefficients of linear terms of objective
        lp_solver_path (str): full path of lp solver executable. Provided in config.yml
        lp_solver_setting (str): lp solver. See admissable solver for pyomo.
            Default provided in config.yml
        lp_activedual_tol (float): duals with values larger than this tolerance is
            considered active. See attribute 'active_const'
        qp_solver_path (str): full path of qp solver executable. Provided in config.yml
        qp_solver_setting (str): qp solver. See admissable solver for pyomo.
            Default provided in config.yml
        qp_activedual_tol (float): duals with values larger than this tolerance is
            considered active. See attribute 'active_const'
        tee (bool): set to True for pyomo to print log. Defaults to False
        x_size (int): number of optimised variables x
        c_size (int): number of rows of constraints
        soln (ndarray): optimisation solution of x
        duals (ndarray): duals of constraints
        active_const (list of bool): bool indicating whether constraints are active,
            based on attribute 'activedual_tol'
    """

    lp_solver_setting = config.solver_config.lp_solver_setting
    lp_activedual_tol = config.solver_config.lp_activedual_tol
    qp_solver_setting = config.solver_config.qp_solver_setting
    qp_activedual_tol = config.solver_config.qp_activedual_tol

    def __init__(self, A, b, m, Q=None, tee=False, find_feasible=False):
        """Initialise object by taking in inputs and creating pyomo model object."""

        self.A = A
        self.b = b
        self.Q = Q
        self.m = m
        # solver
        self.tee = tee
        self.find_feasible = find_feasible
        # get no of var and constraints
        self.x_size = get_cols(self.A)
        self.c_size = get_rows(A)
        # outputs
        self.soln = None
        self.duals = None
        # self.slacks = None
        self.active_const = None

        self._create_model()

    def _create_model(self):
        """Create pyomo model object."""

        _A_init = matrix_to_dict(self.A)
        _b_init = vector_to_dict(self.b)
        _m_init = vector_to_dict(self.m)
        if self.Q is not None:
            _Q_init = matrix_to_dict(self.Q)

        # define pyomo model
        self.model = pmo.ConcreteModel()
        self.model.n = pmo.RangeSet(1, self.x_size)
        self.model.c = pmo.RangeSet(1, self.c_size)
        self.model.A = pmo.Param(self.model.c, self.model.n, initialize=_A_init)
        self.model.b = pmo.Param(self.model.c, initialize=_b_init)
        if self.Q is not None:
            self.model.Q = pmo.Param(self.model.n, self.model.n, initialize=_Q_init)
        self.model.m = pmo.Param(self.model.n, initialize=_m_init)
        self.model.x = pmo.Var(self.model.n)
        if self.find_feasible is True:
            self.model.s = pmo.Var(self.model.c, domain=pmo.NonNegativeReals)
            self.model.z = pmo.Var()
        self.model.dual = pmo.Suffix(direction=pmo.Suffix.IMPORT)
        self.model.constraints = pmo.ConstraintList()

        # Model Ax <= b.
        # if finding feasible point, we maximise the minimum slack. Add slack var, and
        # model objective z to be smaller than all slacks
        if self.find_feasible is True:
            for c in self.model.c:
                self.model.constraints.add(
                    sum(self.model.A[c, i] * self.model.x[i] for i in self.model.n)
                    + self.model.s[c]
                    <= self.model.b[c]
                )
                self.model.constraints.add(self.model.z <= self.model.s[c])
        # if not finding feasible point, constraints are simply Ax M= b.
        else:
            for c in self.model.c:
                self.model.constraints.add(
                    sum(self.model.A[c, i] * self.model.x[i] for i in self.model.n)
                    <= self.model.b[c]
                )

        # if we are finding feasible point, we are maximising the smallest slack,
        # which is modelled as z - see constraint creation step.
        if self.find_feasible is True:
            self.model.obj = pmo.Objective(expr=-self.model.z)
        # QP objective is obj = 0.5 x^T Q x + mx
        elif self.Q is not None:
            self.model.obj = pmo.Objective(
                expr=(
                    0.5
                    * sum(
                        sum(
                            self.model.Q[i, j] * self.model.x[i] * self.model.x[j]
                            for j in self.model.n
                        )
                        for i in self.model.n
                    )
                    + sum(self.model.m[i] * self.model.x[i] for i in self.model.n)
                )
            )
        # LP objective is obj = mx
        else:
            self.model.obj = pmo.Objective(
                expr=sum(self.model.m[i] * self.model.x[i] for i in self.model.n)
            )

        # set solver depending on problem is LP or QP
        if self.Q is None or self.find_feasible is True:
            self.solver = pmo.SolverFactory(
                self.lp_solver_setting, tee=self.tee
            )
            if self.lp_solver_setting == "cplex":
                self.solver.options["lpmethod"] = 2
        else:
            self.solver = pmo.SolverFactory(
                self.lp_solver_setting, tee=self.tee
            )
            if self.qp_solver_setting == "cplex":
                self.solver.options["qpmethod"] = 2

    def solve(self):
        """Solve optimisation problem and save results to object attributes.

        Results saved include attribute 'soln', 'duals' and 'active_const'.
        """

        self.solver.solve(self.model, tee=self.tee)

        self.soln = np.empty([self.x_size])
        for i in range(self.x_size):
            self.soln[i] = self.model.x[i + 1].value

        self.duals = np.empty([self.c_size])
        for c in range(self.c_size):
            self.duals[c] = -self.model.dual[self.model.constraints[c + 1]]

        # self.slacks = np.empty([self.c_size])
        # for c in range(self.c_size):
        #     self.slacks[c] = self.model.constraints[c+1].uslack()

        if self.Q is None:
            self.active_const = self.duals >= self.lp_activedual_tol
        else:
            self.active_const = self.duals >= self.qp_activedual_tol


A = np.array(
    [
        [1., 1., -1., 0.],
        [5., -4., 0., 0.],
        [-8., 22., 0., -1.],
        [-4., -1., 0., 0.],
        [0., 0., -1., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., -1.],
        [0., 0., 0., 1.],
        [0., 0., 1., -0.],
        [0., 0., -32., 1.]
    ]
)

b = np.array([13., 20., 121., -8.,
    10, 10., 100., 100., -7.08696658, 214.99999269])

Q = np.array(
    [
        [30. * 2., 0., 0., 0.],
        [0., 1. * 2, 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ]
)

m = np.array([0., 0., 0., 0.])

theta_size = 2
opt = GenericSolver(A, b, m, Q=Q, tee=False, find_feasible=True)
opt.solve()
print(opt.soln)

# A = np.array(
#     [[1.0, .0, -3.16515, -3.7546],
#      [-1.0, .0, 3.16515, 3.7546],  # problematic
#      [-0.0609, .0, -0.17355, 0.2717],
#      [-0.0064, .0, -0.06585, -0.4714],
#      [.0, 1.0, -1.81960, 3.2841],  # problematic
#      [.0, -1.0, 1.81960, -3.2841],
#      [.0, .0, -1.0, .0],
#      [.0, .0, 1., .0],
#      [.0, .0, .0, -1.0],
#      [.0, .0, .0, 1.0],
#      # additional
#      [.0, .0, -3.16515   , -3.7546],
#      [.0, .0, 2.82163241, -2.09545779],
#      [.0, .0, 0.07350042, 0.05290032]]
# )

# b = np.array(
#     [0.417425, 3.582575, 0.413225, 0.467075, 1.090200, 2.909800, .0, 1., .0, 1.,
#      # additional
#      -3.582575, 0.04398198, 0.063350]
# )

# m = np.array(
#     [.0, .0, .0, .0]
# )

# Q = np.array(
#     [[0.0098 * 2, 0.0063, .0, .0],
#      [0.0063, 0.00995 * 2, .0, .0],
#      [.0, .0, .0, .0],
#      [.0, .0, .0, .0]]
# )

# theta_count = 2
# opt = GenericSolver(A, b, m, Q=Q, tee=False, find_feasible=True)
# opt.solve()
# print(opt.soln)
