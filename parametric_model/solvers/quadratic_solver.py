import numpy as np
import pyomo.environ as pmo

from parametric_model.config.core import config
from parametric_model.processing.inputs import get_rows, get_cols, vector_to_dict, matrix_to_dict


class GenericSolver:

    def __init__(self, A, b, Q, m, 
                 solver_path=config.solver_config.solver_path,
                 solver_setting=config.solver_config.solver_setting,
                 tee=False):
        
        self.A = A
        self.b = b
        self.Q = Q
        self.m = m
        # solver
        self.solver_path = solver_path
        self.solver_setting = solver_setting
        self.tee = tee
        # get no of var and constraints
        self.x_size = get_cols(self.A)
        self.c_size = get_rows(A)
        # outputs
        self.soln = None
        self.duals = None
        self.slacks = None
        self.active_const = None

        self._create_model()

    def _create_model(self):        
        _A_init = matrix_to_dict(self.A)
        _b_init = vector_to_dict(self.b)
        _Q_init = matrix_to_dict(self.Q)
        _m_init = vector_to_dict(self.m)

        # define pyomo model
        self.model = pmo.ConcreteModel()
        self.model.n = pmo.RangeSet(1, self.x_size)
        self.model.c = pmo.RangeSet(1, self.c_size)
        self.model.A = pmo.Param(self.model.c, self.model.n, initialize=_A_init)
        self.model.b = pmo.Param(self.model.c, initialize=_b_init)
        self.model.Q = pmo.Param(self.model.n, self.model.n, initialize=_Q_init)
        self.model.m = pmo.Param(self.model.n, initialize=_m_init)
        self.model.x = pmo.Var(self.model.n)
        self.model.dual = pmo.Suffix(direction=pmo.Suffix.IMPORT)
        self.model.constraints = pmo.ConstraintList()

        # Ax <= b
        for c in self.model.c:
            self.model.constraints.add(
                sum(self.model.A[c, i] * self.model.x[i]
                    for i in self.model.n) 
                <= self.model.b[c])

        # obj = 0.5 x^T Q x + mx
        self.model.obj = pmo.Objective(
            expr=(0.5 * sum(sum(self.model.Q[i, j] * self.model.x[i] * self.model.x[j]
                                for j in self.model.n) 
                            for i in self.model.n)
                  + sum(self.model.m[i] * self.model.x[i] for i in self.model.n)))

        # define solver
        self.solver = pmo.SolverFactory(self.solver_setting, 
                                        tee=self.tee,
                                        executable=self.solver_path)

    def solve(self):
        self.solver.solve(self.model)

        self.soln = np.empty([self.x_size])
        for i in range(self.x_size):
            self.soln[i] = self.model.x[i+1].value

        self.duals = np.empty([self.c_size])
        for c in range(self.c_size):
            self.duals[c] = -self.model.dual[self.model.constraints[c+1]]
        
        self.slacks = np.empty([self.c_size])
        for c in range(self.c_size):
            self.slacks[c] = self.model.constraints[c+1].uslack()

        self.active_const = self.slacks <= config.solver_config.feas_tol
