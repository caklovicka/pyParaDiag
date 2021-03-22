# Welcome to Paralpha!
The Paralpha project is a Python based implementation of a diagonalization based 
parallel-in-time method. It is intended as a prototype and for educational purposes.
Problem example classes that are included here, serve also as an example on how to
implement other discretization methods.

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





## How to cite