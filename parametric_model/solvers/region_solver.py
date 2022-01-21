import numpy as np

from parametric_model.processing.inputs import get_cols, get_rows, get_zeros_rows_index
from parametric_model.processing.redundancy_checker import RedundancyChecker
from parametric_model.solvers.generic_solver import GenericSolver


class RegionSolver:
    """Class for solving a region in multi-parametric QP/LP.

    The problem is posed as
    min(XT)QX + mX
    s.t.
    Ax + Wθ <= b
    where
    X: a vector of optimised variables
    θ: a vector of varying parameters
    XT: transposed X
    Q: coefficients for quadratic terms in objective
    m: coefficients for linear terms
    A: coefficients for X in constraints list
    W: coefficients for θ in constraints list
    b: RHS constants in constraints list
    Q is omitted if problem is LP.

    To use this class, simply create object and solve:
        problem = RegionSolver(A, W, b, m, Q=Q)
        problem.solve()

    Attributes:
        A (ndarray): 2D array, coefficients of optimised variables X in
            constraint matrix
        W (ndarray): 2D array, coefficients of varying parameters θ in
            constraint matrix
        b (ndarray): 1D array, RHS of constraint matrix
        Q (ndarray, optional): 2D array, coefficients of quadratic terms in
            objective
        m (ndarray): 1D array, coefficients of linear terms inf objective
        theta_size (int): number of varying parameters theta
        x_size (int): number of optimised variables x
        theta_size (int): number of varying parameters θ
        var_size (int): total number of optimised variables x and varying parameters
            theta
        c_size (int): number of rows of constraints
        feasible_theta (ndarray): 1D array, a starting solution of theta
        x_problem_A (ndarray): 2D array, represents LHS of constraint matrix as we pose
            problem solving x under a starting solution of theta
        x_problem_b (ndarray): 1D array, represents RHS of constraint matrix as we pose
            problem solving x under a starting solution of theta
        x_problem_W (ndarray): 2D array, the theta terms W 'left out' when
            solving for x
        x (ndarray): 1D array, solution of x under a starting solution of theta
        duals (ndarray): 1D array, duals from solving x under a starting solution of theta
        active_const (list of bool): active constraints when solving x
        M (ndarray): 2D array, derivative matrix of the lagrangian w.r.t. x and duals
        M_kept_rows (ndarray): 1D array, non-zero rows of M
        N (ndarray): 2D array, derivative of the lagrangian w.r.t. theta
        MN (ndarray): 1D array, soltuion of M^-1 * N
        reduced_duals (ndarray): 1D array, duals of active constraints only
        soln_slope (ndarray): slope of the solution of x
        soln_constant (ndarray): constant of the solution of x
        boundary_slope (ndarray): slope of the region boundaries
        boundary_constant (ndarray): constant of the region boundaries
    """

    def __init__(self, A, W, b, m, Q=None):
        self.A = A
        self.W = W
        self.b = b
        self.Q = Q
        self.m = m
        self.x_size = get_cols(A)
        self.theta_size = get_cols(W)
        self.var_size = self.x_size + self.theta_size
        self.c_size = get_rows(A)

        # returned from _solve_theta
        self.feasible_theta = None

        # returned from _solve_x
        self.x_problem_A = None
        self.x_problem_b = None
        self.x_problem_W = None
        self.x = None
        self.duals = None
        self.active_const = None

        # returned from _get_MN
        self.M = None
        self.M_kept_rows = None
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
        """Generate a starting, feasible value of theta."""

        _theta_problem_A = np.concatenate((self.A, self.W), axis=1)
        _theta_problem = GenericSolver(
            _theta_problem_A, self.b, self.m, tee=False, find_feasible=True
        )
        _theta_problem.solve()
        self.feasible_theta = _theta_problem.soln[-self.theta_size :]

    def _solve_x(self):
        """Generate a optimal solution of x based on a starting value of theta.

        The method also provides other useful outputs from solving x, such as duals.
        """
        # define A without theta, and ignore constraints with only theta terms
        self.x_problem_A = self.A
        # define b ignoring constraints with only theta terms
        self.x_problem_b = self.b - np.dot(self.W, self.feasible_theta)

        # delete constraints with only theta terms
        _delete_rows = get_zeros_rows_index(self.x_problem_A)
        self.x_problem_A = np.delete(self.x_problem_A, _delete_rows, axis=0)
        self.x_problem_b = np.delete(self.x_problem_b, _delete_rows)
        self.x_problem_W = np.delete(self.W, _delete_rows, axis=0)
        # this is b without the subtraction of theta
        self.x_problem_b_original = np.delete(self.b, _delete_rows)

        # solve for x, duals
        _x_problem = GenericSolver(self.x_problem_A, self.x_problem_b, self.m, Q=self.Q)
        _x_problem.solve()
        self.x = _x_problem.soln
        self.duals = _x_problem.duals
        # print(self.duals)
        self.active_const = _x_problem.active_const
        self.duals[np.logical_not(self.active_const).tolist()] = 0.0

    def _get_MN(self):
        """Compute attributes 'M', 'N' and 'MN'."""

        self.M_len = self.x_size + get_rows(self.x_problem_A)
        self.M = np.zeros([self.M_len, self.M_len])
        # note it is impossible for the previous row reduction to make Q unfit for top
        # left, because the first rows include the objective function so are
        # impossible to be removed
        if self.Q is not None:
            _M_top_left_input = self.Q
            self.M[: self.x_size, : self.x_size] = _M_top_left_input
        self.M[: self.x_size, self.x_size :] = self.x_problem_A.T
        self.M[self.x_size :, : self.x_size] = np.multiply(
            self.x_problem_A.T, self.duals
        ).T

        # if whole row is zero, multiplier is zero so delete row
        _delete_rows = get_zeros_rows_index(self.M)
        self.M = np.delete(self.M, _delete_rows, axis=0)
        self.M = np.delete(self.M, _delete_rows, axis=1)
        self.M_kept_rows = np.delete(range(self.M_len), _delete_rows)

        # M has (x_size + c_size) rows.
        # For matrices theta_matrix and duals, they only have c_size rows.
        # Here we want to delete constraints that are redundant in theta_matrix and
        # duals, not M. So go back by x_size to delete rows for theta_matrix and duals.
        _delete_rows_constraints_only = (
            _delete_rows - np.ones(len(_delete_rows)) * self.x_size
        )
        delete_rows_constraints_only = _delete_rows_constraints_only.astype("int")
        # delete redundant rows from theta_matrix, duals and N also to avoid singular
        # matrix
        _reduced_W = np.delete(self.x_problem_W, delete_rows_constraints_only, axis=0)
        self.reduced_duals = np.delete(self.duals, delete_rows_constraints_only)

        self.N = np.zeros([np.shape(self.M)[0], self.theta_size])
        self.N[self.x_size :] = np.multiply(_reduced_W.T, self.reduced_duals).T

        self.MN = np.linalg.solve(self.M, self.N)

    def _get_soln_params(self):
        """Compute attributes 'soln_slope' and 'soln_constant'."""

        self.soln_slope = np.zeros((self.M_len, self.theta_size))
        self.soln_slope[self.M_kept_rows] = -self.MN

        self.soln_constant = np.zeros(self.M_len)
        self.soln_constant[self.M_kept_rows] = np.dot(
            -self.MN, -self.feasible_theta
        ) + np.concatenate((self.x, self.reduced_duals))

    def _set_boundaries(self):
        """Return boundaries of the region.

        Boundaries come from three places:
        - subbing theta into the constraints
        - slope calculated for duals
        - theta-only constraints

        For subbing theta into constraints.
        substitute x = G * theta + H we got from previous step into Ax <= b
        Means AG * theta + AH <= b
        A: x_problem_A, remove active constraints
        b: x_problem_b, remove active constraints
        G: soln_slope, for x (so remove rows for duals)
        H: soln_constant, for x (so remove rows for duals)

        Then need to add back the theta theta matrix into the constraints to
        get back the full constraint with theta. We can use x_problem_theta_matrix

        Note for first two, both needs have rows of zeros removed as it can occur.
        """

        # Deal with subbing theta into constants first
        not_active_const_flag = np.logical_not(self.active_const)
        sub_A = self.x_problem_A[not_active_const_flag]
        sub_b = self.x_problem_b_original[not_active_const_flag]
        sub_W = self.x_problem_W[not_active_const_flag]
        sub_G = self.soln_slope[: self.x_size]
        sub_H = self.soln_constant[: self.x_size]
        AG = np.dot(sub_A, sub_G)
        AH = np.dot(sub_A, sub_H)
        AG_with_W = AG + sub_W
        new_rhs = sub_b - AH

        delete_rows = get_zeros_rows_index(AG_with_W)
        AG_with_W = np.delete(AG_with_W, delete_rows, axis=0)
        new_rhs = np.delete(new_rhs, delete_rows)

        # Now deal with dual boundaries
        # We need to flip dual boundaries A.
        # From previous step, dual  = slope * theta + constant
        # For duals to remain positive in the region, we need:
        #   dual = slope * theta + constant >= 0
        # Rearrange to get -slope * theta <= constant
        dual_boundaries_A = self.soln_slope[self.x_size :, :] * -1.0
        dual_boundaries_b = self.soln_constant[self.x_size :]
        delete_rows = get_zeros_rows_index(dual_boundaries_A)
        dual_boundaries_A = np.delete(dual_boundaries_A, delete_rows, axis=0)
        dual_boundaries_b = np.delete(dual_boundaries_b, delete_rows)

        # After that, deal with theta-only constraints in A.
        _theta_only_rows_index = get_zeros_rows_index(self.A)
        # because we use index to define x-coordinate here, the result list is
        # put into another list
        A_theta_only_rows = self.W[_theta_only_rows_index, :][0]
        b_theta_only_rows = self.b[_theta_only_rows_index]

        boundary_slope = np.zeros(
            (
                AG_with_W.shape[0]
                + A_theta_only_rows.shape[0]
                + dual_boundaries_A.shape[0],
                self.theta_size,
            )
        )
        if AG_with_W.shape[0] >= 1:
            boundary_slope[0 : AG_with_W.shape[0], :] = AG_with_W

        if A_theta_only_rows.shape[0] >= 1:
            _insert_start = AG_with_W.shape[0]
            _insert_end = AG_with_W.shape[0] + A_theta_only_rows.shape[0]
            boundary_slope[_insert_start:_insert_end, :] = A_theta_only_rows

        # print('AG_with_theta_matrix')
        # print(AG_with_theta_matrix)
        # print('A_theta_only_rows')
        # print(A_theta_only_rows)
        # print('dual_boundaries_A')
        # print(dual_boundaries_A)
        if dual_boundaries_A.shape[0] >= 1:
            _insert_start = AG_with_W.shape[0] + A_theta_only_rows.shape[0]
            _insert_end = (
                AG_with_W.shape[0]
                + A_theta_only_rows.shape[0]
                + dual_boundaries_A.shape[0]
            )
            boundary_slope[_insert_start:_insert_end, :] = dual_boundaries_A

        boundary_constant = np.concatenate(
            (new_rhs, b_theta_only_rows, dual_boundaries_b)
        )
        # print('AG_with_theta_matrix')
        # print(AG_with_theta_matrix)
        # print('A_theta_only_rows')
        # print(A_theta_only_rows)
        # print('dual_boundaries_A')
        # print(dual_boundaries_A)
        reduction_problem = RedundancyChecker(boundary_slope, boundary_constant)
        (
            self.boundary_slope,
            self.boundary_constant,
        ) = reduction_problem.remove_redundancy()

    def solve(self):
        """Solve the region by running methods in order."""

        self._solve_theta()
        self._solve_x()
        self._get_MN()
        self._get_soln_params()
        self._set_boundaries()


# # QP
# A = np.array(
#     [[1.0, .0, -3.16515, -3.7546],
#      [-1.0, .0, 3.16515, 3.7546], # problematic
#      [-0.0609, .0, -0.17355, 0.2717],
#      [-0.0064, .0, -0.06585, -0.4714],
#      [.0, 1.0, -1.81960, 3.2841], # problematic
#      [.0, -1.0, 1.81960, -3.2841],
#      [.0, .0, -1.0, .0],
#      [.0, .0, 1.,   .0],
#      [.0, .0, .0, -1.0],
#      [.0, .0, .0, 1.0],
#      # additional
#      [.0, .0, -3.16515   , -3.7546    ],
#      [.0, .0,  2.82163241, -2.09545779],
#      [.0, .0, 0.07350042,  0.05290032]]
# )

# b = np.array(
#     [0.417425, 3.582575, 0.413225, 0.467075, 1.090200, 2.909800, .0, 1., .0, 1.,
#     # additional
#      -3.582575,  0.04398198,  0.063350]
# )

# m = np.array(
#     [.0, .0, .0, .0]
# )

# Q = np.array(
#     [[0.0098*2, 0.0063, .0, .0],
#      [0.0063, 0.00995*2, .0, .0],
#      [.0, .0, .0, .0],
#      [.0, .0, .0, .0]]
# )

# theta_count = 2


# mp = RegionSolver(A, b, m, theta_count, Q=Q)
# mp.solve()
# print('mp.boundary_slope')
# print(mp.boundary_slope)
# print('mp.boundary_constant')
# print(mp.boundary_constant)

# # LP
# A = np.array([[0.8, 0.44, -1., 0.],
#               [0.05, 0.1, 0., -1.],
#               [0.1, 0.36, 0., 0.],
#               [-1., 0., 0., 0., ],
#               [0., -1., 0., 0.],
#               [0., 0., -1., 0.],
#               [0., 0., 1., 0.],
#               [0., 0., 0., -1.],
#               [0., 0., 0., 1.],
#               [0., 0., -0.03278688524590164,  1.]])

# b = np.array([24000, 2000., 6000., 0., 0., 0., 6000., 0., 500., 213.1147531])

# m = [-8.1, -10.8, 0., 0.]

# theta_size = 2

# mp = RegionSolver(A, b, m, theta_size)
# mp.solve()
# print(mp.boundary_slope)
# print(mp.boundary_constant)

# A = np.array(
#     [
#         [1., 1.],
#         [5., -4.],
#         [-8., 22.],
#         [-4., -1.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.]
#     ]
# )

# W = np.array(
#     [
#         [-1., 0.],
#         [ 0., 0.],
#         [ 0., -1.],
#         [ 0., 0.],
#         [ -1., 0.],
#         [ 1., 0.],
#         [ 0., -1.],
#         [ 0., 1.],
#         [ 1., -0.],
#         [ -32., 1.]
#     ]
# )

# b = np.array([13., 20., 121., -8., 10, 10., 100., 100., -7.08696658, 214.99999269])

# Q = np.array(
#     [
#         [30. * 2., 0.],
#         [0., 1. * 2]
#     ]
# )

# m = np.array([0., 0.])

# region = RegionSolver(A,W,b,m, Q=Q)
# region.solve()
# print(region.boundary_slope)
# print(region.boundary_constant)
