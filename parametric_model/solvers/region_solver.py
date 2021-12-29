import numpy as np
from numpy.lib.function_base import delete
import pyomo.environ as pmo

from parametric_model.config.core import config
from parametric_model.processing.inputs import get_rows, get_cols, vector_to_dict, matrix_to_dict, get_zeros_rows_index
from parametric_model.solvers.quadratic_solver import GenericSolver
from parametric_model.processing.redundancy_checker import RedundancyChecker


class RegionSolver:
    """
    Returns equations representing x, lambda and boundaries of a math problem in terms of theta.
    self.soln_slope
    self.soln_constant
    self.boundary_slope
    self.boundary_constant
    """

    def __init__(self, A, b, Q, m, theta_size):
        self.A = A
        self.b = b
        self.Q = Q
        self.m = m
        self.theta_size = theta_size
        self.x_size = get_cols(A) - theta_size
        self.var_size = get_cols(A)
        self.c_size = get_rows(A)

        # returned from _solve_theta
        self.theta = None

        # returned from _solve_x
        self.x_problem_A = None
        self.x_problem_b = None
        self.x_problem_theta_matrix = None
        self.x = None
        self.duals = None
        self.active_const = None

        # returned from _get_MN
        self.M = None
        self.N = None
        self.MN = None
        self.reduced_duals = None

        # returned from _get_soln_params
        self.soln_slope = None
        self.soln_constant = None

        # returned from _set_boundaries
        self.boundary_slope = None
        self.boundary_constant = None

    def _solve_theta(self):
        theta_only_rows = get_zeros_rows_index(self.A[:, :self.x_size])
        # problem_b = self.b.copy()
        # problem_b[theta_only_rows] = problem_b[theta_only_rows] 
        _theta_problem = GenericSolver(self.A, self.b, self.Q, self.m)
        _theta_problem.solve()
        self.theta = _theta_problem.soln[-self.theta_size:]

    def _solve_x(self):
        # define A without theta, and ignore constraints with only theta terms
        self.x_problem_A = self.A[:, :-self.theta_size]
        # define b ignoring constraints with only theta terms
        self.x_problem_b = self.b - np.dot(self.A[:, -self.theta_size:], self.theta) 
        self.x_problem_Q = self.Q[:self.x_size, :self.x_size]
        self.x_problem_m = self.m[:self.x_size]

        # delete constraints with only theta terms
        delete_rows = get_zeros_rows_index(self.x_problem_A)
        self.x_problem_A = np.delete(self.x_problem_A, delete_rows, axis=0)
        self.x_problem_b = np.delete(self.x_problem_b, delete_rows)
        self.x_problem_theta_matrix = np.delete(self.A[:, -self.theta_size:], delete_rows, axis=0)
        # this is b without the subtraction of theta
        self.x_problem_b_original = np.delete(self.b, delete_rows)

        # solve for x, duals
        x_problem = GenericSolver(self.x_problem_A, self.x_problem_b, self.x_problem_Q, self.x_problem_m)
        x_problem.solve()
        self.x = x_problem.soln
        self.duals = x_problem.duals
        # print('duals')
        # print(self.duals)
        self.active_const = x_problem.active_const
        self.duals[np.logical_not(self.active_const).tolist()] = .0

    def _get_MN(self):
        self.M_len = self.x_size + get_rows(self.x_problem_A)
        self.M = np.zeros([self.M_len, self.M_len])
        # note it is impossible for the previous row reduction to make Q unfit for top left,
        # because the first rows include the objective function so are impossible to be removed
        M_top_left_input = self.Q[:self.x_size, :self.x_size]
#         for i in range(M_top_left_input.shape[0]):
#             M_top_left_input[i, i] = M_top_left_input[i, i]/2.0
        self.M[:self.x_size, :self.x_size] = M_top_left_input
        self.M[:self.x_size, self.x_size:] = self.x_problem_A.T
        self.M[self.x_size:, :self.x_size] = np.multiply(self.x_problem_A.T, self.duals).T

        # if whole row is zero, multiplier is zero so delete row
        self.delete_rows = get_zeros_rows_index(self.M)
        self.M = np.delete(self.M, self.delete_rows, axis=0)
        self.M = np.delete(self.M, self.delete_rows, axis=1)
        self.kept_rows = np.delete(range(self.M_len), self.delete_rows)

        # M has (x_size + c_size) rows.
        # For matrices theta_matrix and duals, they only have c_size rows.
        # Here we want to delete constraints that are redundant in theta_matrix and duals, not M.
        # So go back by x_size to delete rows for theta_matrix and duals.
        delete_rows_constraints_only = self.delete_rows - \
            np.ones(len(self.delete_rows)) * self.x_size
        delete_rows_constraints_only = delete_rows_constraints_only.astype('int')
        # delete redundant rows from theta_matrix, duals and N also to avoid singular matrix
        reduced_theta_matrix = np.delete(self.x_problem_theta_matrix, delete_rows_constraints_only, axis=0)
        self.reduced_duals = np.delete(self.duals, delete_rows_constraints_only)

        self.N = np.zeros([np.shape(self.M)[0], self.theta_size])
        self.N[self.x_size:] = np.multiply(reduced_theta_matrix.T, self.reduced_duals).T

        self.MN = np.linalg.solve(self.M, self.N)

    def _get_soln_params(self):
        self.soln_slope = np.zeros((self.M_len, self.theta_size))
        self.soln_slope[self.kept_rows] = -self.MN

        self.soln_constant = np.zeros(self.M_len)
        self.soln_constant[self.kept_rows] = np.dot(-self.MN, -self.theta) \
                                             + np.concatenate((self.x, self.reduced_duals))

    def _set_boundaries(self):
        # Boundaries come from three places:
        # - subbing theta into the constraints
        # - slope calculated for duals
        # - theta-only constraints
        # 
        # For subbing theta into constraints.
        # substitute x = G * theta + H we got from previous step into Ax <= b
        # Means AG * theta + AH <= b
        # A: x_problem_A, remove active constraints
        # b: x_problem_b, remove active constraints
        # G: soln_slope, for x (so remove rows for duals)
        # H: soln_constant, for x (so remove rows for duals)
        #
        # Then need to add back the theta theta matrix into the constraints to 
        # get back the full constraint with theta. We can use x_problem_theta_matrix
        # 
        # Note for first two, both needs to remove zeros rows. They can happen in MN is 0.

        # Deal with subbing theta into constants first
        not_active_const_flag = np.logical_not(self.active_const)
        sub_A = self.x_problem_A[not_active_const_flag]
        sub_b=self.x_problem_b_original[not_active_const_flag]
        sub_theta_matrix = self.x_problem_theta_matrix[not_active_const_flag]
        sub_G = self.soln_slope[:self.x_size]
        sub_H = self.soln_constant[:self.x_size]
        AG = np.dot(sub_A, sub_G)
        AH = np.dot(sub_A, sub_H)
        AG_with_theta_matrix = AG + sub_theta_matrix
        new_rhs = sub_b - AH
        
        delete_rows = get_zeros_rows_index(AG_with_theta_matrix)
        AG_with_theta_matrix = np.delete(AG_with_theta_matrix, delete_rows, axis=0)
        new_rhs = np.delete(new_rhs, delete_rows)

        # Now deal with dual boundaries
        # We need to flip dual boundaries A. 
        # From previous step, lambda  = slope * theta + constant
        # For lambda to remain positive in the region, we need lambda = slope * theta + constant >= 0
        # Rearrange to get -slope * theta <= constant
        dual_boundaries_A = self.soln_slope[self.x_size:, :] * -1.0
        dual_boundaries_b = self.soln_constant[self.x_size:]
        delete_rows = get_zeros_rows_index(dual_boundaries_A)
        dual_boundaries_A = np.delete(dual_boundaries_A, delete_rows, axis=0)
        dual_boundaries_b = np.delete(dual_boundaries_b, delete_rows)

        # After that, deal with theta-only constraints in A.
        A_theta_only_rows_index = get_zeros_rows_index(self.A[:, :self.x_size])
        # because we use index to define x-coordinate here, the result list is put into another list
        A_theta_only_rows = self.A[A_theta_only_rows_index, -self.theta_size:][0]
        b_theta_only_rows = self.b[A_theta_only_rows_index]

        boundary_slope = np.zeros((AG_with_theta_matrix.shape[0]
                                   + A_theta_only_rows.shape[0]
                                   + dual_boundaries_A.shape[0], 
                                  self.theta_size))
        if AG_with_theta_matrix.shape[0]>=1:
            boundary_slope[0:AG_with_theta_matrix.shape[0], :] = AG_with_theta_matrix

        if A_theta_only_rows.shape[0]>=1:
            _insert_start = AG_with_theta_matrix.shape[0]
            _insert_end = AG_with_theta_matrix.shape[0] + A_theta_only_rows.shape[0]
            boundary_slope[_insert_start:_insert_end, :] = A_theta_only_rows
        
        # print('AG_with_theta_matrix')
        # print(AG_with_theta_matrix)
        # print('A_theta_only_rows')
        # print(A_theta_only_rows)
        # print('dual_boundaries_A')
        # print(dual_boundaries_A)
        if dual_boundaries_A.shape[0]>=1:
            _insert_start = (AG_with_theta_matrix.shape[0]
                             + A_theta_only_rows.shape[0])
            _insert_end = (AG_with_theta_matrix.shape[0]
                           + A_theta_only_rows.shape[0]
                           + dual_boundaries_A.shape[0])
            boundary_slope[_insert_start:_insert_end, :] = dual_boundaries_A

        boundary_constant = np.concatenate((new_rhs, 
                                            b_theta_only_rows, 
                                            dual_boundaries_b))

        reduction_problem = RedundancyChecker(boundary_slope, boundary_constant)
        self.boundary_slope, self.boundary_constant = reduction_problem.remove_redundancy()

    def solve(self):
        self._solve_theta()
        self._solve_x()
        self._get_MN()
        self._get_soln_params()
        self._set_boundaries()


########################################################################################
A = np.array(
    [[1.0, .0, -3.16515, -3.7546],
     [-1.0, .0, 3.16515, 3.7546], # problematic
     [-0.0609, .0, -0.17355, 0.2717],
     [-0.0064, .0, -0.06585, -0.4714],
     [.0, 1.0, -1.81960, 3.2841], # problematic
     [.0, -1.0, 1.81960, -3.2841],
     [.0, .0, -1.0, .0],
     [.0, .0, 1.,   .0],
     [.0, .0, .0, -1.0],
     [.0, .0, .0, 1.0],
     # additional
     [.0, .0, -3.16515   , -3.7546    ],
     [.0, .0,  2.82163241, -2.09545779],
     [.0, .0, 0.07350042,  0.05290033]]
)

b = np.array(
    [0.417425, 3.582575, 0.413225, 0.467075, 1.090200, 2.909800, .0, 1., .0, 1., 
    # additional 
     -3.582575,  0.04398198,  0.06335021]
)

m = np.array(
    [.0, .0, .0, .0]
)

Q = np.array(
    [[0.0098*2, 0.0063, .0, .0],
     [0.0063, 0.00995*2, .0, .0],
     [.0, .0, .0, .0],
     [.0, .0, .0, .0]]
)

theta_count = 2


mp = RegionSolver(A, b, Q, m, theta_count)
mp.solve()
print('mp.boundary_slope')
print(mp.boundary_slope)
print('mp.boundary_constant')
print(mp.boundary_constant)
