# ParametricProgramming

## Introduction
This package is made to solve simple multi-parametric (mp) LP and QP problems. <br/>
The problem is in the form: <br/>
  min (XT)QX + mX <br/>
  s.t. <br/>
  AX <= b <br/>
where <br/>
- X: an array of both optimsed variables x and varying parameters θ. Note θ are listed always after all x are declared.
- XT: transposed X
- Q: coefficients for quadratic terms in objective
- m: coefficients for linear terms in objective
- A: coefficients for constraint matrix 
- b: constants for constraint matrix

## Usage
To use, simply:

from parametric_model.solvers.regionn_generator import ParametricSolver

mp = ParametricSolver(A, b, m, theta_size, Q=Q)
mp.solve()
theta = [0.7, 0.7]
mp.get_soln(theta)

Note Q is optional. If Q is supplied, the problem is treated as mp-QP. Otherwise, it is treated as mp-LP.
theta_size is used to determine how many rows/columns in 2D array A, m and Q correspond to coefficients for theta. Note theta always occupy the last theta_size number if rows/columns.

## Notes
Unfortunately the current mp-QP solver is less robust than mp-LP. This is because it turns out that with a solver like ipopt, it is not easy to determine which constraints are active:
- whether dual is non-zero: it turns out duals for inactive constraints also have a small value;
- whether the constraint slack is zero: even worse than dual, because active constraints has a small slack, and this is bigger than the value for the duals for inactive constraints.
In config.yml a tolerance for the dual to be considered active is provided, but currently no good way to calibrate.

## Developments
Potential future developments include:
- adding logs to track/debug solver
- handling infeasible optimisation
- raise errors when incompatible/wrong sizes of Q, m, A, b are given 
- (nice to have) representing region boundaries in concatenated form [slope constant] rather than storing them separately
- solving mp-MILP
- Make a reader for LP files from Xpress/cplex, so that mp can be solved on Xpress/cplex modelled problems
