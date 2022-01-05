import numpy as np
import pyomo.environ as pmo

import parametric_model
from parametric_model.config.core import config
from parametric_model.processing.inputs import get_rows, get_cols, matrix_to_dict, vector_to_dict


class RedundancyChecker():
    """
    A class for removing redundant rows for a constraint system Ax <= b.

    Attributes:
        A (ndarray): a 2D which is LHS of Ax <= b
        b (ndarray): a 1D ndarray which is RHS of Sx <= b
        model (pyomo model object): pyomo model object for detecting redundancy
        solver_path (str): location of executable for solver
        solver_setting (str): which solver to use, a parameter for pyomo
        solver (pyomo solver object): pyomo solver object for solving model
        relax_tol (float, optional): a float to relax a constraint being checked for redundancy.
            Default provided in config.yml
        zero_tol (float, optional): a float, value higher than this is identified as higher than zero.
            Default provided in config.yml
        x_size (int): int, no of variables in A
        c_size (int): int, no of constraints in A
        redundancy (list of int): a boolean list indicating 1 for each redundant constraint
        reduced_A (ndarray): 2D array, A with redundant rows removed
        reduced_b (ndarray): 1D array, b with redundant rows removed   
    """

    def __init__(self, A, b, 
                 solver_path=config.solver_config.solver_path,
                 solver=config.solver_config.solver_setting,
                 relax_tol=config.redundancy_checker_config.relax_tol, 
                 zero_tol=config.redundancy_checker_config.zero_tol):
        """Inits class and create pyomo model object ready to be run.
        """

        # defining Ax <= b
        self.A = A
        self.b = b
        # opt problem
        self.model = None
        self.solver = None
        # solver
        self.solver_path = solver_path
        self.solver_setting = solver
        # tolerances
        self.relax_tol = relax_tol
        self.zero_tol = zero_tol
        # utility
        self.x_size = get_cols(self.A)
        self.c_size = get_rows(self.A)
        # output entities
        self. slack = None
        self.redundancy = None
        self.reduced_A = None
        self.reduced_b = None

        self._create_model()

    def _create_model(self):
        """Create pyomo model and solver objects.
        
        Args:
            None
        Returns:
            None
        """
        _A_init = matrix_to_dict(self.A)
        _b_init = vector_to_dict(self.b)

        # define pyomo model
        self.model = pmo.ConcreteModel()
        self.model.n = pmo.RangeSet(1, self.x_size)
        self.model.c = pmo.RangeSet(1, self.c_size)
        self.model.A = pmo.Param(self.model.c, self.model.n, initialize=_A_init)
        self.model.b = pmo.Param(self.model.c, mutable=True, initialize=_b_init)
        self.model.x = pmo.Var(self.model.n)
        self.model.dual = pmo.Suffix(direction=pmo.Suffix.IMPORT)
        self.model.constraints = pmo.ConstraintList()
        for c in self.model.c:
            self.model.constraints.add(sum(self.model.A[c, i] * self.model.x[i]
                                           for i in self.model.n) <= self.model.b[c])
        self.solver = pmo.SolverFactory(self.solver_setting, executable=self.solver_path)

    def remove_redundancy(self):
        """
        Remove redundant rows from Ax <= b.

        Args:
            None
        Returns:
            None
        """  
        # for each constraint, delete any old obj, set new obj as Ax of chosen constraint
        # and maximise it.
        # Deactivate the chosen constraint itself.
        # Then check if b-Ax to see if positive (constraint is loose).
        # If so, mark as redundant.
        self.redundancy = np.zeros([self.c_size])
        self.slack = np.zeros([self.c_size])

        for c in self.model.c:
            # delete old obj
            try:
                self.model.del_component(self.model.obj)
            except:
                pass
            # relax chosen constraint
            self.model.b[c] += self.relax_tol #* \
                # np.max(np.abs(self.A[c-1, :]))/(np.abs(self.b[c-1]) if self.b[c-1]!=0. else 1.)
            # set up chosen constraint as the new obj
            self.model.obj = pmo.Objective(expr=-sum(self.model.A[c, i] 
                                                     * self.model.x[i] for i in self.model.n) 
                                           + self.model.b[c])
            self.solver.solve(self.model, tee=False)
            # if obj is bigger than 0, mark constarint as redundant
            # self.slack[c-1] = pmo.value(self.model.obj)
            if pmo.value(self.model.obj) > self.zero_tol:
                self.redundancy[c-1] = 1
            
            # revert relaxation of constraint
            self.model.b[c] -= self.relax_tol #* \
                # np.max(np.abs(self.A[c-1, :])) / \
                # (np.abs(self.b[c-1]) if self.b[c-1] != 0. else 1.)

        # print('slack=')
        # print(self.slack)
        self.reduced_A = self.A[self.redundancy == 0]
        self.reduced_b = self.b[self.redundancy == 0]
        return self.reduced_A, self.reduced_b

# A = np.array(
#     [[ 0.0,  1.0],
#      [-3.16515, -3.7546],
#      [2.82163241, -2.09545779],
#      [-0.07350042, -0.05290033]]
# )

# b = np.array([1., -3.58257501,  0.04384359, -0.06334207])

# model = RedundancyChecker(A,b).remove_redundancy()
# print(model)

