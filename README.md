# ParametricProgramming

## Introduction
This package is made to solve simple multi-parametric (mp-) LP and QP problems. <br/>
The problem is in the form: <br/>

  __min (XT)QX + mX__ <br/>
  __s.t. AX <= b__ <br/>
  
where <br/>
- X: an array of both optimsed variables x and varying parameters θ. Note θ are always listed behind x.
- XT: transposed X
- Q: coefficients for quadratic terms in objective
- m: coefficients for linear terms in objective
- A: coefficients for constraint matrix 
- b: constants for constraint matrix

## Usage
To use, simply:

```
from parametric_model.solvers.region_generator import ParametricSolver

A = ...
b = ...
m = ...
theta_size = ...
Q = ...

mp = ParametricSolver(A, b, m, theta_size, Q=Q)
mp.solve()
theta = [0.7, 0.7]
mp.get_soln(theta)
```

Note Q is optional. If Q is supplied, the problem is treated as mp-QP. Otherwise, it is treated as mp-LP.

`theta_size` is used to determine how many: 
- columns in 2D array A, 
- elements in 1D array m, and
- rows and columns in 2D array Q 

correspond to coefficients for θ. Note θ always occupy the last theta_size number of rows/columns in each case.

## Notes
Unfortunately the current mp-QP solver is less robust than mp-LP. This is because it turns out that with a solver like ipopt, it is not easy to determine which constraints are active. With no direct indicator for constraint activeness, consider the following two methods:
- __whether dual is non-zero:__ it turns out duals for inactive constraints also have a small value in ipopt; our LP solver glpk does not have this problem
- __whether the constraint slack is zero:__ even worse than using dual, because active constraints actually have a small slack, and this is bigger than the duals of inactive constraints - this means it is even harder to differentiate between slack of active vs. inactive constraints 

In config.yml a tolerance is provided on the dual where constraints with dual larger than this is considered active. However, currently no good way to calibrate the tol.

## Developments
Potential future developments include:
- adding logs to track/debug solver
- handling infeasible optimisation
- raise errors when incompatible/wrong sizes of Q, m, A, b are given 
- (nice to have) representing region boundaries in concatenated form [slope constant] rather than storing them separately
- solving mp-MILP
- Make a reader for LP files from Xpress/cplex, so that mp can be solved on Xpress/cplex modelled problems
- Tidy up commit messages on GitHub
