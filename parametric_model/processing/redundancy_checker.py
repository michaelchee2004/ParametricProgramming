import numpy as np
import pyomo.environ as pmo

from parametric_model.processing.inputs import get_rows, get_cols, matrix_to_dict


class RedundancyChecker:
    """
    A class which remove redundant rows for a constraint system Ax <= b.

    Attributes:
        A: a ndarray which is LHS of Ax <= b
        b: a ndarray which is RHS of Sx <= b
        model: pyomo model object for detecting redundancy
        solver: pyomo solver object for solving model
        relax_tol: a float to relax a constraint being checked for redundancy
        zero_tol: a float, value higher than this is identified as higher than zero
        x_size: int, no of variables in A
        c_size: int, no of constraints in A
        slack: a list of how much slack is in a constraint being checked for redundancy
        redundancy: a boolean list marking a 1 for each redundant constraint
        reduced_A: A with redundant rows removed
        reduced_b: b with redundant rows removed   
    """

    def __init__(self, A, b, relax_tol=1e-6, zero_tol=1e-6):
        """
        Inits class and create opt model ready to be run.

        Args:
            A: attribute
            b: attribute
            relax_tol: attribute
            zero_tol: attribute

        Returns:
            None
        """
        # defining Ax <= b
        self.A = A
        self.b = b
        # opt problem
        self.model = None
        self.solver = None
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

        self.create_model()

    def create_model(self):
        """
        Create model object so that it can called when needed.
        
        Args:
            None
        Returns:
            None
        """
        A_init = matrix_to_dict(self.A)
        b_init = matrix_to_dict(self.b)

        # define pyomo model
        self.model = pmo.ConcreteModel()
        self.model.n = pmo.RangeSet(1, self.x_size)
        self.model.c = pmo.RangeSet(1, self.c_size)
        self.model.A = pmo.Param(self.model.c, self.model.n, initialize=A_init)
        self.model.b = pmo.Param(self.model.c, mutable=True, initialize=b_init)
        self.model.x = pmo.Var(self.model.n)
        self.model.dual = pmo.Suffix(direction=pmo.Suffix.IMPORT)
        self.model.constraints = pmo.ConstraintList()
        for c in self.model.c:
            self.model.constraints.add(
                sum(self.model.A[c, i] * self.model.x[i]
                    for i in self.model.n) <= self.model.b[c]
            )
        
        self.solver = pmo.SolverFactory('cplex')

    def remove_redundancy(self) -> None:
        """
        Remove redundant rows from Ax <= b.

        Args:
            None
        Returns:
            reduced_A: A with redundant rows removed
            reduced_b: A with redundant rows removed
        """  
        # for each constraint, delete any old obj, set new obj as Ax of chosen constraint
        # and maximise it.
        # Deactivate the chosen constraint itself.
        # Then check if b-Ax to see if positive (constraint is loose).
        # If so, mark as redundant.
        self.redundancy = np.zeros([self.c_size])

        for c in self.model.c:
            # delete old obj
            try:
                self.model.del_component(self.model.obj)
            except:
                pass
            # relax chosen constraint
            self.model.b[c] += self.relax_tol
            # set up chosen constraint as the new obj
            self.model.obj = pmo.Objective(
                expr=-sum(self.model.A[c, i] * self.model.x[i]
                          for i in self.model.n) + (self.model.b[c])
            )
            self.solver.solve(self.model, tee=False)
            # if obj is bigger than 0, mark constarint as redundant
            self.slack[c-1] = pmo.value(self.model.obj)
            if pmo.value(self.model.obj) > self.zero_tol:
                self.redundancy[c-1] = 1
            
            # revert relaxation of constraint
            self.model.b[c] -= self.relax_tol

        self.reduced_A = self.A[self.redundancy == 0]
        self.reduced_b = self.b[self.redundancy == 0]
        return self.reduced_A, self.reduced_b
