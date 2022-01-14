import numpy as np

from parametric_model.config.core import config
from parametric_model.processing.inputs import check_duplicates, get_cols, get_rows
from parametric_model.solvers.region_solver import RegionSolver


class ParametricSolver:
    """A solver class for multi-parametric LP/QP problems.

    The problem is posed as
    min (XT)QX + mX
    s.t.
    Ax <= b
    where
    X: a vector of optimised variables x and varying parameter theta, with theta always
        listed after x
    XT: transposed X
    Q: coefficients for qudratic terms
    m: coefficients for linear terms
    A: LHS coefficients in constraints list
    b: RHS constants in constraints list

    Attributes:
        system (dict): a dict representing the original problem (Q, b, Q, m, theta_size).
        x_size: (int): number of variables x in problem
        col_size (int): total number of x and theta
        max_iter (int): setting limiting how many times the regions list is looked through
            to find unsolved regions
        regions (dict): a dict storing results for all the regions

    Notes:
        In 'system' attribute, distinction between coefficients for x and for theta are
        made in the folllowing way:
            'theta_size' is the number of varying parameters.
            In 1D array 'm', the last theta_size elements are of theta;
            in 2D array 'Q', the last theta_size rows and columns are of theta;
            in 2D array 'A', the last theta_size columns are of theta.

        Within 'regions' dict attribute, each region is defined by a dict with the
        following keys:
            'soln_A'/'soln_b': slope and constant for the parametrised optimal solution of
                x and constraint duals
            'firm_bound_A'/'firm_bound_b: slope and constant for region boundaries which
                appear in the original problem
            'added_bound_A'/'added_bound_b': region boundaries which do not appear in
                original problem
            'flippable_bound_A'/'flippable_bound_b': same as added_bound_A/b, but for
                boundaries yet to be used to generate new regions (so can be flipped to
                    create new region)
            solve_status: 0 = region has not been solved; 1 = region has been solved; 2 =
                redundant region boundaries have been removed; 3 = region boundaries have
                been used to generate new regions
    """

    lp_newregion_tol = config.regiongen_config.lp_newregion_tol
    qp_newregion_tol = config.regiongen_config.qp_newregion_tol

    def __init__(
        self,
        A,
        b,
        m,
        theta_size,
        Q=None,
        max_iter=config.regiongen_config.max_iter_default,
    ):
        """Initialise object by taking in problem inputs and creating initial region.

        The problem is posed as
        min (XT)QX + mX
        s.t.
        Ax <= b
        where
        X: a vector of optimised variables x and varying parameter theta, with theta
            always listed after x
        XT: transposed X
        Q: coefficients for qudratic terms
        m: coefficients for linear terms
        A: LHS coefficients in constraints list
        b: RHS constants in constraints list
        Q is omitted if problem is LP.

        Args:
            A (ndarray): 2D array, LHS of constraint system
            b (ndarray): 1D array, RHS of constraint system
            m (ndarray): 1D array, coefficients for linear terms in objective
            theta_size (int): number of theta in X
            Q (ndarray, optional): 2D array, coefficients for quadratic terms in
                objective
            max_iter (int, optional): iteration limit on how many times the list of
                regions is looked through to find unsolved regions. Default is given
                in config.yml as 100

        Returns:
            None
        """

        self.system = {"A": A, "b": b, "m": m, "theta_size": theta_size, "Q": Q}
        self.x_size = get_cols(self.system["A"]) - theta_size
        self.col_size = get_rows(self.system["A"])
        self.max_iter = max_iter

        self.regions = []
        self.create_region(None, None, None, None, None, None, None, None, 0)

    def create_region(
        self,
        soln_A,
        soln_b,
        firm_bound_A,
        firm_bound_b,
        added_bound_A,
        added_bound_b,
        flippable_bound_A,
        flippable_bound_b,
        solve_status,
    ):
        """Create a new region.

        This method creates a new region element in the 'regions' dict attribute.

        Args:
            soln_A (ndarray): (see class docstring)
            soln_b (ndarray): (see class docstring)
            firm_bound_A (ndarray): (see class docstring)
            firm_bound_b (ndarray): (see class docstring)
            added_bound_A (ndarray): (see class docstring)
            added_bound_b (ndarray): (see class docstring)
            flippable_bound_A (ndarray): (see class docstring)
            flippable_bound_b (ndarray): (see class docstring)
            solve_status (int): (see class docstring)

        Returns:
            None
        """

        _region_def = {
            "soln_A": soln_A,
            "soln_b": soln_b,
            "firm_bound_A": firm_bound_A,
            "firm_bound_b": firm_bound_b,
            "added_bound_A": added_bound_A,
            "added_bound_b": added_bound_b,
            "flippable_bound_A": flippable_bound_A,
            "flippable_bound_b": flippable_bound_b,
            "solve_status": solve_status,
        }
        self.regions.append(_region_def)

    def gen_new_regions(self, region_index):
        """Generate new regions with the flippable boundaries of a region.

        This method creates as many region elements in the 'regions' dict attribute
        as there are flippable bounds in a region. After the flippable bounds are used,
        they are recategorised from 'flippable_bound' to 'added_bound' of the region.
        After this method is run, solve_status of region is set to 3.

        Args:
            region_index (int): region to generate new regions from

        Returns:
            None
        """

        if np.shape(self.regions[region_index]["flippable_bound_b"]) == ():
            return

        flippable_A = self.regions[region_index]["flippable_bound_A"]
        flippable_b = self.regions[region_index]["flippable_bound_b"]
        added_A = self.regions[region_index]["added_bound_A"]
        added_b = self.regions[region_index]["added_bound_b"]
        no_of_new_regions = np.shape(flippable_b)[0]

        next_added_A = np.empty(shape=[0, np.shape(flippable_A)[1]])
        next_added_b = np.empty(shape=[0])
        # for each region, flip and append a row (multiply by -1) from
        # flippable A/b to next_added_A/b.
        # Create new region and put next_added_A/b as flipped (i.e. these are firm)
        # Then unflip this newly appended row in next_added_A/b (which is last row)
        # to be used for the next region.
        _newregion_tol = (
            self.lp_newregion_tol if self.system["Q"] is None else self.qp_newregion_tol
        )
        for n in range(no_of_new_regions):
            next_added_A = np.append(
                next_added_A, [np.multiply(flippable_A[n], -1)], axis=0
            )
            next_added_b = np.append(next_added_b, [np.multiply(flippable_b[n], -1)])

            self.create_region(
                None,
                None,
                None,
                None,
                np.concatenate((added_A.copy(), next_added_A.copy()), axis=0),
                np.concatenate((added_b.copy(), next_added_b.copy() - _newregion_tol)),
                None,
                None,
                0,
            )
            next_added_A[-1, :] = next_added_A[-1, :] * -1.0
            next_added_b[-1] = next_added_b[-1] * -1.0

        # now label the considered constraints as flipped
        self.regions[region_index]["added_bound_A"] = np.append(
            self.regions[region_index]["added_bound_A"],
            self.regions[region_index]["flippable_bound_A"],
            axis=0,
        )
        self.regions[region_index]["added_bound_b"] = np.append(
            self.regions[region_index]["added_bound_b"],
            self.regions[region_index]["flippable_bound_b"],
        )
        self.regions[region_index]["flippable_bound_A"] = None
        self.regions[region_index]["flippable_bound_b"] = None

        self.regions[region_index]["solve_status"] = 3

    def solve_region_problem(self, region_index):
        """Solve a region, generating optimal for x and defining its boundaries.

        The method calls region_solver module, and uses the results as inputs for
        calling method 'categorise_const' to fill the region element in 'regions'
        attribute. After this method is one, 'solve_status' of the region is set
        to 1.

        Args:
            region_index (int): region to be solved

        Returns:
            None
        """

        # extended flipped bound. Bounds only have columns for theta, so need columns
        # for x.
        # A zeros matrix with no. of rows of flipped bound, and no. of cols of x, then
        # concat with theta no. of cols
        try:
            added_bound_A_with_x = np.concatenate(
                (
                    np.zeros(
                        [
                            np.shape(self.regions[region_index]["added_bound_A"])[0],
                            np.shape(self.system["A"])[1] - self.system["theta_size"],
                        ]
                    ),
                    self.regions[region_index]["added_bound_A"],
                ),
                axis=1,
            )
            region_problem_A = np.concatenate(
                (self.system["A"], added_bound_A_with_x), axis=0
            )
            region_problem_b = np.concatenate(
                (self.system["b"], self.regions[region_index]["added_bound_b"])
            )
        # index error if there is no added bound at all
        except IndexError:
            region_problem_A = self.system["A"]
            region_problem_b = self.system["b"]

        region_problem = RegionSolver(
            region_problem_A,
            region_problem_b,
            self.system["m"],
            self.system["theta_size"],
            Q=self.system["Q"],
        )
        region_problem.solve()
        self.regions[region_index]["soln_A"] = region_problem.soln_slope
        self.regions[region_index]["soln_b"] = region_problem.soln_constant

        self.categorise_const(
            region_index,
            region_problem.boundary_slope,
            region_problem.boundary_constant,
        )

        self.regions[region_index]["solve_status"] = 1

    def categorise_const(self, region_index, lhs, rhs):
        """Categorise input boundaries as either firm, added or flippable for a region.

        This method fills 'soln_A', 'soln_b', 'firm_bound_A', 'firm_bound_b',
        'flippable_bound_A', 'flippable_bound_b' of the region element specified
        by argument 'region_index'.

        Args:
            region_index (int): region to be filled with boundary information

        Returns:
            None
        """

        # if there is no boundary, then categorisation needs not happen
        if np.shape(rhs) == () or np.shape(rhs)[0] == 0:
            return
        # concatenate LHS and RHS of eqns so they can be compared together
        _boundary_concat = np.concatenate((lhs, np.array([rhs]).T), axis=1)
        _boundary_concat_normal = _boundary_concat.copy()
        # 'normalise' each set of eqns so RHS is 1 (divide all terms by RHS value)
        for row in range(np.shape(_boundary_concat)[0]):
            if _boundary_concat[row, -1] != 0.0:
                _boundary_concat_normal[row] = np.divide(
                    _boundary_concat[row], np.abs(_boundary_concat[row, -1])
                )

        # same for added_bound
        if (
            np.shape(self.regions[region_index]["added_bound_A"]) == ()
            or np.shape(self.regions[region_index]["added_bound_A"])[0] == 0
        ):
            _is_added = []
        else:
            _added_concat = np.concatenate(
                (
                    self.regions[region_index]["added_bound_A"],
                    self.regions[region_index]["added_bound_b"].reshape(-1, 1),
                ),
                axis=1,
            )
            _added_concat_normal = _added_concat.copy()
            for row in range(np.shape(_added_concat)[0]):
                if _added_concat[row, -1] != 0.0:
                    _added_concat_normal[row] = np.divide(
                        _added_concat[row], np.abs(_added_concat[row, -1])
                    )
            _is_added = check_duplicates(
                _boundary_concat_normal, checklist=_added_concat_normal
            ).tolist()

        # same for original set of eqns
        _sys_concat = np.concatenate(
            (self.system["A"], np.array([self.system["b"]]).T), axis=1
        )
        # ignore eqns with x terms
        _theta_only_rows = np.all(_sys_concat[:, : self.x_size] == 0.0, axis=1)
        if np.shape(_theta_only_rows)[0] == 0:
            _is_firm = []
        else:
            # -theta_size-1: to extract the thetas plus the rhs attached
            _sys_concat = _sys_concat[
                _theta_only_rows, -self.system["theta_size"] - 1 :
            ]
            _sys_concat_normal = _sys_concat.copy()
            for row in range(np.shape(_sys_concat)[0]):
                if _sys_concat[row, -1] != 0.0:
                    _sys_concat_normal[row] = np.divide(
                        _sys_concat[row], np.abs(_sys_concat[row, -1])
                    )
            _is_firm = check_duplicates(
                _boundary_concat_normal, checklist=_sys_concat_normal
            ).tolist()

        # return indexes from _boundary_concat
        _boundary_indices = np.arange(np.shape(_boundary_concat_normal)[0])
        _is_flippable = np.where(
            np.logical_not(
                np.in1d(_boundary_indices, np.concatenate((_is_firm, _is_added)))
            )
        )[0].tolist()

        self.regions[region_index]["firm_bound_A"] = _boundary_concat[_is_firm, :-1]
        self.regions[region_index]["firm_bound_b"] = _boundary_concat[_is_firm, -1]
        self.regions[region_index]["added_bound_A"] = _boundary_concat[_is_added, :-1]
        self.regions[region_index]["added_bound_b"] = _boundary_concat[_is_added, -1]
        self.regions[region_index]["flippable_bound_A"] = _boundary_concat[
            _is_flippable, :-1
        ]
        self.regions[region_index]["flippable_bound_b"] = _boundary_concat[
            _is_flippable, -1
        ]

    def reduce_region_bounds(self, region_index):
        """Remove flippable boundaries of a region which are duplicates in firm or added
        boundaries.

        Args:
            region_index (int): region in which flippable boundaries are checked

        Returns:
            None
        """

        firm_concat = np.concatenate(
            (
                self.regions[region_index]["firm_bound_A"],
                np.array([self.regions[region_index]["firm_bound_b"]]).T,
            ),
            axis=1,
        )

        added_concat = np.concatenate(
            (
                self.regions[region_index]["added_bound_A"],
                np.array([self.regions[region_index]["added_bound_b"]]).T,
            ),
            axis=1,
        )

        flippable_concat = np.concatenate(
            (
                self.regions[region_index]["flippable_bound_A"],
                np.array([self.regions[region_index]["flippable_bound_b"]]).T,
            ),
            axis=1,
        )

        # in flippable_concat, divide each row with b, so rhs for all eqns are 1
        flippable_concat_normal = flippable_concat.copy()
        for row in range(np.shape(flippable_concat_normal)[0]):
            if self.regions[region_index]["flippable_bound_b"][row] != 0.0:
                flippable_concat_normal[row] = np.divide(
                    flippable_concat_normal[row],
                    np.abs(flippable_concat_normal[row][-1]),
                )
        self_duplicates = check_duplicates(flippable_concat_normal)

        # check if self_reduced_flippable has duplicates in firm_bound
        firm_concat_normal = firm_concat.copy()
        for row in range(np.shape(firm_concat_normal)[0]):
            if self.regions[region_index]["firm_bound_b"][row] != 0.0:
                firm_concat_normal[row] = np.divide(
                    firm_concat_normal[row], np.abs(firm_concat_normal[row][-1])
                )
        firm_duplicates = check_duplicates(
            flippable_concat_normal, checklist=firm_concat_normal
        )

        # check if firm_reduced_flippable has duplicates in added_bound
        added_concat_normal = added_concat.copy()
        for row in range(np.shape(added_concat_normal)[0]):
            if self.regions[region_index]["added_bound_b"][row] != 0.0:
                added_concat_normal[row] = np.divide(
                    added_concat_normal[row], np.abs(added_concat_normal[row][-1])
                )
        added_duplicates = check_duplicates(
            flippable_concat_normal, checklist=added_concat_normal
        )

        index_to_delete = np.unique(
            np.concatenate((self_duplicates, firm_duplicates, added_duplicates))
        ).tolist()
        self.regions[region_index]["flippable_bound_A"] = np.delete(
            flippable_concat, index_to_delete, axis=0
        )[:, :-1]
        self.regions[region_index]["flippable_bound_b"] = np.delete(
            flippable_concat, index_to_delete, axis=0
        )[:, -1]

        self.regions[region_index]["solve_status"] = 2

    def loop_region(self, region_index):
        """Solve specified region, process its boundaries, and generate new regions.

        Args:
            region_index (int): region to process

        Returns:
            None
        """

        self.solve_region_problem(region_index)
        self.reduce_region_bounds(region_index)
        self.gen_new_regions(region_index)

    def solve(self):
        """Solve the entire MP problem.

        This method runs 'loop_region' method on any unsolved new region,
        until the problem is exhausted of unsolved regions, or iteration limit
        is reached.
        """

        for i in range(self.max_iter):
            for r in range(len(self.regions)):
                if self.regions[r]["solve_status"] == 0:
                    self.loop_region(r)

            if (
                np.all(
                    np.array(
                        [
                            self.regions[i]["solve_status"]
                            for i in range(len(self.regions))
                        ]
                    )
                    != 0
                )
                is True
            ):
                break

    def get_soln(self, theta):
        for r in range(len(self.regions)):
            _theta = np.array(theta)
            _all_boundaries_A = np.concatenate(
                (self.regions[r]["firm_bound_A"], self.regions[r]["added_bound_A"]),
                axis=0,
            )
            _all_boundaries_b = np.concatenate(
                (self.regions[r]["firm_bound_b"], self.regions[r]["added_bound_b"])
            )

            _in_boundary = (
                np.dot(_all_boundaries_A, _theta.reshape(-1, 1))
                - _all_boundaries_b.reshape(-1, 1)
                <= 0.0
            )
            _in_all_boundaries = np.all(_in_boundary)
            if _in_all_boundaries:
                _x = np.dot(
                    self.regions[r]["soln_A"], _theta.reshape(-1, 1)
                ) + self.regions[r]["soln_b"].reshape(-1, 1)
                _x = _x.reshape(-1)[: self.x_size]
                return _x

        return None


###########################################################################
# A = np.array(
#     [[1.0, .0, -3.16515, -3.7546],
#      [-1.0, .0, 3.16515, 3.7546],
#      [-0.0609, .0, -0.17355, 0.2717],
#      [-0.0064, .0, -0.06585, -0.4714],
#      [.0, 1.0, -1.81960, 3.2841],
#      [.0, -1.0, 1.81960, -3.2841],
#      [.0, .0, -1.0, .0],
#      [.0, .0, 1.0, .0],
#      [.0, .0, .0, -1.0],
#      [.0, .0, .0, 1.0]]
# )

# b = np.array(
#     [0.417425, 3.582575, 0.413225, 0.467075, 1.090200, 2.909800, .0, 1.0,
#      .0, 1.0]
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

# theta_size = 2


# mp = ParametricSolver(A, b, m, theta_size, Q=Q)
# mp.solve()
# print(mp.regions)


##########################################################
# A = np.array(
#     [
#         [0.8, 0.44, -1.0, 0.0],
#         [0.05, 0.1, 0.0, -1.0],
#         [0.1, 0.36, 0.0, 0.0],
#         [-1.0, 0.0, 0.0, 0.0],
#         [0.0, -1.0, 0.0, 0.0],
#         [0.0, 0.0, -1.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, -1.0],
#         [0.0, 0.0, 0.0, 1.0],
#     ]
# )

# b = np.array([24000, 2000.0, 6000.0, 0.0, 0.0, 0.0, 6000.0, 0.0, 500.0])

# m = [-8.1, -10.8, 0.0, 0.0]

# theta_size = 2

# mp = ParametricSolver(A, b, m, theta_size)
# mp.max_iter = 5
# mp.loop_region(0)
# mp.loop_region(1)
# print(mp.regions)
# print(mp.regions[1]["added_bound_A"][0][0])


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
#         [30. * 2., 0., 0., 0.],
#         [0., 1. * 2, 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]
#     ]
# )

# m = np.array([0., 0., 0., 0.])
# theta_size = 2
# mp = ParametricSolver(A, b, m, theta_size, Q=Q)
# mp.max_iter = 5
# mp.loop_region(0)
# mp.loop_region(1)
# mp.loop_region(2)
# # print(mp.regions)
# mp.loop_region(3)
# print(mp.regions)
