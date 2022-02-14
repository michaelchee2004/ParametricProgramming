# ParametricProgramming

![alt text](https://github.com/michaelchee2004/ParametricProgramming/blob/master/readme_image.png)

## Introduction
This package is made to solve simple (i.e. with no speed consideration) Multi-Parametric LP and QP problems (mp-LP, mp-QP). [Wikipedia](https://en.wikipedia.org/wiki/Parametric_programming) / [Book](https://www.wiley.com/en-ie/Multi+Parametric+Programming:+Theory,+Algorithms+and+Applications,+Volume+1-p-9783527631216)

The optimisation problem is in the form below. Notice the presence of θ that deviates from usual QP formulation: <br/>

```
min 1/2 (x_T)Qx + mx
  s.t. Ax + Wθ <= b

where
- x: optimsed variables 
- x_T: transposed x
- θ: 'unfixed' input parameters of optimisation problem
- Q: coefficients for quadratic terms in objective
- m: coefficients for linear terms in objective
- A: coefficients for x in constraint matrix 
- W: coefficients for θ in constraint matrix
- b: constants in constraint matrix
```

We would like to understand how __x*__, the optimal solution of __x__, varies depending on 
input parameters __θ__,

i.e. express __x*__ as a function of __θ__.

## Installation
Download the wheel file: [link](https://github.com/michaelchee2004/ParametricProgramming/blob/master/dist/parametric_programming-0.0.1-py3-none-any.whl)

Then, </br>
```pip install package_whl_path```

Note the package assumes cplex to be available and callable as PATH environment variable (free version is sufficient to run examples). 
Can be changed to other solvers through editing in config.yml, but a very good solver is necessary 
to ensure accurate categorisation of constraints as active/inactive. GLPK, etc. have been observed to give wrong 
results, possibly because they need tuning.

## Usage
- [mp-LP example (motivating example)](https://github.com/michaelchee2004/ParametricProgramming/blob/master/jupyter_notebooks/lp_example.ipynb)
- [mp-QP example](https://github.com/michaelchee2004/ParametricProgramming/blob/master/jupyter_notebooks/qp_example.ipynb)

## Motivation
Although packages for solving parametric-programming problems have been developed on MatLab, a painful gap exists in python. </br>
This project therefore presents a novel and interesting personal challenge to both further knowledge in optimisation and gain more experience with python. </br>

(Very recently - Sept 2021, a package for solving mp has been released by the Multi-parametric Optimization & Control group at Taxas A&M: [ppopt](https://pypi.org/project/ppopt/), although with limited functionality for now.)  

## Developments
Potential future developments include:
- adding logs to track/debug solver
- handling infeasible optimisation
- raise errors when incompatible/wrong sizes of Q, m, A, b are given 
- (nice to have) representing region boundaries in concatenated form [slope constant] rather than storing them separately
- solving mp-MILP
- Make a reader for LP files from Xpress/cplex, so that mp can be solved on Xpress/cplex modelled problems
