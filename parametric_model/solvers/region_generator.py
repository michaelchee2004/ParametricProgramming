import numpy as np

from parametric_model.processing.inputs import check_duplicates, remove_duplicates, get_rows, get_cols
from parametric_model.solvers.region_solver import RegionSolver



class ParametricSolver:
    """
    Solve_status:
    - 0: uns
    """

    def __init__(self, A, b, Q, m, theta_size):
        self.system = {
            'A': A,
            'b': b,
            'Q': Q,
            'm': m,
            'theta_size': theta_size
        }
        self.x_size = get_cols(self.system['A']) - theta_size
        self.col_size = get_rows(self.system['A'])

        self.regions = {}

    def create_region(self, soln_A, soln_b, firm_bound_A, firm_bound_b, added_bound_A, added_bound_b, 
                      flippable_bound_A, flippable_bound_b, solved_status):
        region_def = {
            'soln_A': soln_A,
            'soln_b': soln_b,
            'firm_bound_A': firm_bound_A,
            'firm_bound_b': firm_bound_b,
            'added_bound_A': added_bound_A,
            'added_bound_b': added_bound_b,
            'flippable_bound_A': flippable_bound_A,
            'flippable_bound_b': flippable_bound_b,
            'solved_status': solved_status
        }
        self.regions[self.new_index()] = region_def

    # def solve_original_problem(self):
    #       self.create_region(
    #         original_problem.soln_slope,
    #         original_problem.soln_constant,
    #         firm_A,
    #         firm_b,
    #         flippable_A,
    #         flippable_b,
    #         added_A,
    #         added_b,
    #         True
    #     )

    def gen_new_regions(self, region_index):
        """
        Generate new regions with the flippable constraints of a region (region specified by region_index).
        """
        if np.shape(self.regions[region_index]['flippable_bound_b'])==():
            return

        flippable_A = self.regions[region_index]['flippable_bound_A']
        flippable_b = self.regions[region_index]['flippable_bound_b']
        added_A = self.regions[region_index]['added_bound_A']
        added_b = self.regions[region_index]['added_bound_b']
        no_of_new_regions = np.shape(flippable_b)[0]

        next_added_A = np.empty(shape=[0, np.shape(flippable_A)[1]])
        next_added_b = np.empty(shape=[0])
        # for each region, flip and append a row (multiply by -1) from flippable A/b to next_added_A/b.
        # Create new region and put next_added_A/b as flipped (i.e. these are firm)
        # Then unflip this newly appended row in next_added_A/b (which is last row)
        # to be used for the next region.
        for n in range(no_of_new_regions):
            next_added_A = np.append(
                next_added_A, [np.multiply(flippable_A[n], -1)], axis=0)
            next_added_b = np.append(next_added_b, [np.multiply(flippable_b[n], -1)])
            
            self.create_region(
                None,
                None,
                None,
                None,
                np.concatenate((added_A.copy(), next_added_A.copy()), axis=0),
                np.concatenate((added_b.copy(), next_added_b.copy()-1e-6)),
                None,
                None,
                False)
            next_added_A[-1, :] = next_added_A[-1, :] * -1.0
            next_added_b[-1] = next_added_b[-1] * -1.0

        # now label the considered constraints as flipped
        self.regions[region_index]['added_bound_A'] = np.append(
            self.regions[region_index]['added_bound_A'],
            self.regions[region_index]['flippable_bound_A'],
            axis=0
        )
        self.regions[region_index]['added_bound_b'] = np.append(
            self.regions[region_index]['added_bound_b'],
            self.regions[region_index]['flippable_bound_b']
        )
        self.regions[region_index]['flippable_bound_A'] = None
        self.regions[region_index]['flippable_bound_b'] = None

    def solve_region_problem(self, region_index):
        """
        
        """
        # extended flipped bound. Bounds only have columns for theta, so need columns for x.
        # A zeros matrix with no. of rows of flipped bound, and no. of cols of x, then
        # concat with theta no. of cols
        try:
            added_bound_A_with_x = np.concatenate(
                (
                    np.zeros([np.shape(self.regions[region_index]['added_bound_A'])[0],
                            np.shape(self.system['A'])[1] - self.system['theta_size']]),
                    self.regions[region_index]['added_bound_A']),
                axis=1)
            region_problem_A = np.concatenate(
                (
                    self.system['A'],
                    added_bound_A_with_x),
                axis=0)
            region_problem_b = np.concatenate(
                (
                    self.system['b'],
                    self.regions[region_index]['added_bound_b']))
        # index error if there is no added bound at all
        except IndexError:
            region_problem_A = self.system['A']
            region_problem_b = self.system['b']

        region_problem = RegionSolver(region_problem_A,
                                      region_problem_b,
                                      self.system['Q'],
                                      self.system['m'],
                                      self.system['theta_size'])
        region_problem.solve()
        self.regions[region_index]['soln_A'] = region_problem.soln_slope
        self.regions[region_index]['soln_b'] = region_problem.soln_constant

        self.categorise_const(
            region_index,
            region_problem.boundary_slope,
            region_problem.boundary_constant)

    def new_index(self):
        try:
            last_index = list(self.regions.keys())[-1]
            return last_index + 1
        except IndexError:
            return 0

    def categorise_const(self, region_index, lhs, rhs):
        """
        Categorise boundaries provided as argument, looking at the eqn system of a region.
        Args:

        
        """
        # if there is no boundary, then categorisation needs not happen
        if np.shape(rhs) == () or np.shape(rhs)[0]==0:
            return
        # concatenate LHS and RHS of eqns so they can be compared together
        _boundary_concat = np.concatenate(
            (lhs, np.array([rhs]).T), 
            axis=1)
        _boundary_concat_normal = _boundary_concat.copy()
        # 'normalise' each set of eqns so RHS is 1 (divide all terms by RHS value)
        for row in range(np.shape(_boundary_concat)[0]):
            if _boundary_concat[row, -1] != 0.0:
                _boundary_concat_normal[row] = np.divide(_boundary_concat[row],
                                                        np.abs(_boundary_concat[row, -1]))
        
        # same for added_bound
        if (np.shape(self.regions[region_index]['added_bound_A']) == () or 
            np.shape(self.regions[region_index]['added_bound_A'])[0] == 0):
            _is_added = []
        else:
            _added_concat = np.concatenate(
                (self.regions[region_index]['added_bound_A'], self.regions[region_index]['added_bound_b'].reshape(-1, 1)),
                axis=1)
            _added_concat_normal = _added_concat.copy()
            for row in range(np.shape(_added_concat)[0]):
                if _added_concat[row, -1] != 0.0:
                    _added_concat_normal[row] = np.divide(_added_concat[row],
                                                        np.abs(_added_concat[row, -1]))
            _is_added = check_duplicates(
                _boundary_concat_normal, checklist=_added_concat_normal).tolist()          

        # same for original set of eqns
        _sys_concat = np.concatenate(
            (self.system['A'], np.array([self.system['b']]).T), 
            axis=1)
        # ignore eqns with x terms
        _theta_only_rows = np.all(_sys_concat[:, :self.x_size] == 0.0, axis=1)
        if np.shape(_theta_only_rows)[0] == 0:
            _is_firm = []
        else:
            # -theta_size-1: to extract the thetas plus the rhs attached
            _sys_concat = _sys_concat[_theta_only_rows, -self.system['theta_size']-1:]
            _sys_concat_normal = _sys_concat.copy()
            for row in range(np.shape(_sys_concat)[0]):
                if _sys_concat[row, -1] != 0.0:
                    _sys_concat_normal[row] = np.divide(_sys_concat[row],
                                                        np.abs(_sys_concat[row, -1]))
            _is_firm = check_duplicates(
                _boundary_concat_normal, checklist=_sys_concat_normal).tolist()
       
        # return indexes from _boundary_concat
        _boundary_indices = np.arange(np.shape(_boundary_concat_normal)[0])
        _is_flippable = np.where(
            np.logical_not(np.in1d(_boundary_indices,
                                   np.concatenate((_is_firm, _is_added)))))[0].tolist()

        self.regions[region_index]['firm_bound_A'] = _boundary_concat[_is_firm, :-1]
        self.regions[region_index]['firm_bound_b'] = _boundary_concat[_is_firm, -1]
        self.regions[region_index]['added_bound_A'] = _boundary_concat[_is_added, :-1]
        self.regions[region_index]['added_bound_b'] = _boundary_concat[_is_added, -1]
        self.regions[region_index]['flippable_bound_A'] = _boundary_concat[_is_flippable, :-1]
        self.regions[region_index]['flippable_bound_b'] = _boundary_concat[_is_flippable, -1]

    def reduce_region_bounds(self, region_index):
        """
        Take the flippable list of a region and modify the list so duplicates are removed.

        Check within the flippable list itself first, 
        then check against firm bounds, 
        finally check against added bounds.
        """
        firm_concat = np.concatenate(
            (
                self.regions[region_index]['firm_bound_A'],
                np.array([self.regions[region_index]['firm_bound_b']]).T),
            axis=1)

        added_concat = np.concatenate(
            (
                self.regions[region_index]['added_bound_A'],
                np.array([self.regions[region_index]['added_bound_b']]).T),
            axis=1)

        flippable_concat = np.concatenate(
            (
                self.regions[region_index]['flippable_bound_A'],
                np.array([self.regions[region_index]['flippable_bound_b']]).T),
            axis=1)

        # in flippable_concat, divide each row with b, so rhs for all eqns are 1
        flippable_concat_normal = flippable_concat.copy()
        for row in range(np.shape(flippable_concat_normal)[0]):
            if self.regions[region_index]['flippable_bound_b'][row] != 0.0:
                flippable_concat_normal[row] = np.divide(
                    flippable_concat_normal[row],
                    np.abs(flippable_concat_normal[row][-1]))
        self_duplicates = check_duplicates(flippable_concat_normal)

        # check if self_reduced_flippable has duplicates in firm_bound
        firm_concat_normal = firm_concat.copy()
        for row in range(np.shape(firm_concat_normal)[0]):
            if self.regions[region_index]['firm_bound_b'][row] != 0.0:
                firm_concat_normal[row] = np.divide(
                    firm_concat_normal[row],
                    np.abs(firm_concat_normal[row][-1]))
        firm_duplicates = check_duplicates(
            flippable_concat_normal,
            checklist=firm_concat_normal)

        # check if firm_reduced_flippable has duplicates in added_bound
        added_concat_normal = added_concat.copy()
        for row in range(np.shape(added_concat_normal)[0]):
            if self.regions[region_index]['added_bound_b'][row] != 0.0:
                added_concat_normal[row] = np.divide(
                    added_concat_normal[row],
                    np.abs(added_concat_normal[row][-1]))
        added_duplicates = check_duplicates(
            flippable_concat_normal,
            checklist=added_concat_normal)

        index_to_delete = np.unique(np.concatenate(
            (
                self_duplicates,
                firm_duplicates,
                added_duplicates))).tolist()
        self.regions[region_index]['flippable_bound_A'] = np.delete(flippable_concat, index_to_delete, axis=0)[:, :-1]
        self.regions[region_index]['flippable_bound_b'] = np.delete(flippable_concat, index_to_delete, axis=0)[:, -1]


#########################################################################################
A = np.array(
    [[1.0, .0, -3.16515, -3.7546],
     [-1.0, .0, 3.16515, 3.7546],
     [-0.0609, .0, -0.17355, 0.2717],
     [-0.0064, .0, -0.06585, -0.4714],
     [.0, 1.0, -1.81960, 3.2841],
     [.0, -1.0, 1.81960, -3.2841],
     [.0, .0, -1.0, .0],
     [.0, .0, 1.0, .0],
     [.0, .0, .0, -1.0],
     [.0, .0, .0, 1.0]]
)

b = np.array(
    [0.417425, 3.582575, 0.413225, 0.467075, 1.090200, 2.909800, .0, 1.0, .0, 1.0]
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

theta_size = 2

mp = ParametricSolver(A, b, Q, m, theta_size)
mp.create_region(None, None, None, None, None, None, None, None, False)
mp.solve_region_problem(0)
mp.reduce_region_bounds(0)
mp.gen_new_regions(0)
# print(mp.regions)
mp.solve_region_problem(1)
mp.reduce_region_bounds(1)
mp.gen_new_regions(1)
# print(mp.regions)
mp.solve_region_problem(2)
mp.reduce_region_bounds(2)
mp.gen_new_regions(2)
# print(mp.regions)
mp.solve_region_problem(3)
mp.reduce_region_bounds(3)
mp.gen_new_regions(3)
# # print(mp.regions)
mp.solve_region_problem(4)
mp.reduce_region_bounds(4)
mp.gen_new_regions(4)
# mp.solve_region_problem(5)
# mp.reduce_region_bounds(5)
# mp.gen_new_regions(5)
print(mp.regions)


