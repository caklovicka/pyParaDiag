# Welcome to Paralpha!
The Paralpha project is a Python based implementation of a diagonalization 
based parallel-in-time method. It is intended as a prototype and 
for educational purposes.
Problem example classes that are included here, serve also as an example 
on how to implement other discretization methods.

## Table of contents
* [Features](#features)
* [Getting started](#getting-started)
* [Minitutorial](#minitutorial)
  * [Problem classes](#problem-classes)
  * [Example of`main.py`](#example-of-`main.py`)
    * [The (&alpha;) sequence](#the-(&alpha;)-sequence)
    * [Parallelization strategy](#parallelization-strategy)
    * [Communicators](#communicators)
  * [Useful variables](#useful-variables)  
  * [Optional runtime arguments](#optional-runtime-arguments)
* [How to cite](#how-to-cite)  

## Features
- Many different problem examples already implemented
- Arbitrary choice of order for the Gauss-Radau-Right time-propagator
- Works with [mpi4py](https://pypi.org/project/mpi4py/) and 
  [petsc4py](https://bitbucket.org/petsc/petsc4py/src/master/)
- Choosing a parallelization strategy:
  - across time-steps
  - across collocation-nodes
  - across space-points
- Preset or a user-choice of a linear solver
- Manual or automatic choice of the (&alpha;) sequence

## Getting started
The implementations is fully compatiple with Python 3.6 or higher.
The following dependant libraries can be installed via `pip install`:
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [mpi4py](https://pypi.org/project/mpi4py/)
- [PySDC](https://github.com/Parallel-in-Time/pySDC)  

However, the following is recommended to be installed manually:
- [PETSc](https://www.mcs.anl.gov/petsc/) and 
  [petsc4py](https://bitbucket.org/petsc/petsc4py/src/master/) 
  compiled with scalar type complex

It is best to first install [PETSc](https://www.mcs.anl.gov/petsc/) following
installation guides on the webpage. After that, one can install 
[petsc4py](https://bitbucket.org/petsc/petsc4py/src/master/)
via `pip`, keeping in mind that the variables `PETSC_DIR` and `PETSC_ARCH` 
need to be set accordingly.

## Minitutorial
### Problem classes
Example problem classes can be found in the directory 
`problem_examples_parallel`. The user can use the existing modules or
follow comments on how to change parts of code inside the file itself. 
This class contains
- A definition of the spatial matrix `Apar` assembled in parallel 
  on the `comm_matrix` communicator
- A definition of the `bpar` function
- A definition of the initial condition function `u_initial`
- A definition of `norm`

and optional function definitions such as 
- An exact solution of the differential equation `u_exact`
- The right hand side of an equation `rhs` used for building `bpar`
- The linear solver `linear_solver`

### Example of `main.py`
After setting up a problem class, we can import it and choose the wanted setting.
A detailed explanation of parameter choices can be found in `main.py`.

#### The (&alpha;) sequence
The manual choice of the (&alpha;) sequence 
```
prob.optimal_alphas = False
prob.alphas = [1e-5, 1e-2, 0.1]
```
means that Paralpha will compute the first three iterations using the given sequence
and after that repeatedly use `prob.alphas[-1]` until it stops. To switch
on the automatic choice of the (&alpha;) sequence, the above two lines need to be
replaced with
```
prob.optimal_alphas = True
```
#### Parallelization strategy
The total number of cores needed for Paralpha is `prob.proc_col * prob.proc_row`.
Lines
```
prob.proc_col = 4
prob.proc_row = 16
```
mean that the number of cores for parallelization across time-steps is
`16` , while `4` cores will handle parallelism across collocation nodes.
Furthermore, lines 
```
prob.time_intervals = 16
prob.rolling = 3
```
mean that we want to handle the first `16` time-steps in parallel with Paralpha and repeat
this until we cover a total of `3 * 16` time-steps. For now, the number of 
`prob.time_intervals` needs to be a power of 2 to fit the radix-2 algorithm.

#### Communicators
This figure represents the communicating groups for when `prob.time_intervals = 4` 
and `prob.proc_col = 4`.
Each colored block is a part of the vector locally stored on a processor 
and the different color groups represents the subcommunicators of the `MPI_COMM_WORLD`. 
Each column of the table represents the storage for vector `prob.u_loc`.
![text](https://github.com/caklovicka/linear-petsc-fft-Paralpha/blob/master/procgrid.png)
In case when `prob.time_points = prob.proc.col`, 
groups `COMM_SUBCOL_SEQ` and `COMM_SUBCOL_ALT` would be nonexistent.
These goups can be accessed via names
- `prob.comm` which is MPI_COMM_WORLD
- `prob.comm_col`
- `prob.comm_row`
- `prob.comm_subcol_seq`
- `prob.comm_subcol_alt`
- `prob.comm_last`

### Useful variables
After the `prob.setup()` there is a list of useful variables
- `prob.dx`: spatial step
- `prob.x`: spatial points  
- `prob.t`: collocation nodes
- `prob.comm_matrix`: communicator for assembly of `prob.Apar`  
- `prob.row_beg` and `prob.row_end`: beginning and end of the chunk of rows locally stored  
- `prob.Apar`: the spatial matrix stored in parallel  
- `prob.global_size_A`: size ot the spatial matrix, equal to a product of all
  the spatial points
- `prob.Q`: the collocation matrix

After `prob.solve()` is complete, the following variables are updated
- `prob.u_loc`: the solution spread across cores as a flat `np.array`
- `prob.u_loc_last`: solution in the last time-step of Paralpha spread across cores
- `prob.err_last`: list of `norm`-differences between consecutive iterates
- `prob.iterations`: list of numbers of Paralpha iterations
- `prob.algorithm_time`: time spent in `prob.solve()`
- `prob.commnunication_time`: time spent for communication 
  (without [petsc4py]((https://bitbucket.org/petsc/petsc4py/src/master/) ) communication)
- `prob.system_time_min`
- `prob.system_time_max`

### Optional runtime arguments
Paralpha also has a set of runtime arguments, listed with `--help`:
```
  -h, --help            show this help message and exit
  --T_start T_START     Default = 0
  --T_end T_END         Default = 1
  --rolling ROLLING     Default = 1 ... number of intervals to perform one
                        paralpha and combine it sequentially for the next one.
  --time_intervals TIME_INTERVALS
                        Default = 10 ... size of the B matrix or how many
                        intervals will be treated in parallel.
  --time_points TIME_POINTS
                        Default = 3 ... number of time points for the
                        collocation problem, the size of Q.
  --proc_row PROC_ROW   Default = 1 ... number of processors for dealing with
                        paralellization of time intervals. Choose so that
                        proc_row = time_intervals or get an error.
  --proc_col PROC_COL   Default = 1 ... number fo processors dealing with
                        parallelization of the time-space collocation problem.
                        If just time parallelization, choose so that (proc_col
                        | time_points) and (proc_col >= 1). If space-time
                        parallelization, choose proc_col = k * time_point,
                        (where 0 < k < spatial_points) and (k | spatial
                        points).
  --spatial_points SPATIAL_POINTS [SPATIAL_POINTS ...]
                        Default = 24 ... number of spatial points with unknown
                        values (meaning: not including the boundary ones)
  --solver SOLVER       Default = lu ... specifying the inner linear solver:
                        lu, gmres, custom.
  --maxiter MAXITER     Default = 5 ... maximum number of iterations on one
                        rolling interval.
  --tol TOL             Default = 1e-6 ... a stopping criteria when two
                        consecutive solutions in the last time point are lower
                        than tol (in one rolling interval).
  --stol STOL           Default = 1e-6 ... an inner solver tolerance.
  --smaxiter SMAXITER   Default = 100 ... an inner solver maximum iterations.
  --document DOCUMENT   Default = None ... document to write an output.
```
These values are rewritten if they are changed in the main file. For example, 
if running with optional argument `python main.py --solver=gmres` where your
`main.py` contains a definition `prob.solver=custom`, Paralpha will use a
GMRES solver from the `scipy` library and not your custom one. This list is 
also a summary of default settings when not defined otherwise.






## How to cite