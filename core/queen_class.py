import argparse
import abc


class QueenClass(abc.ABC):

    alphas = []
    betas = []
    gamma = NotImplemented
    dt = NotImplemented
    t = NotImplemented
    x = NotImplemented
    dx = NotImplemented

    # control
    u0_loc = NotImplemented
    u_last_loc = NotImplemented
    u_loc = NotImplemented
    u_last_old_loc = NotImplemented

    # state
    y0_loc = NotImplemented
    y_last_loc = NotImplemented
    y_loc = NotImplemented
    y_last_old_loc = NotImplemented

    # adjoint
    pT_loc = NotImplemented
    p_first_loc = NotImplemented
    p_loc = NotImplemented
    p_first_old_loc = NotImplemented

    # for the problem class
    # Apar is state, AApar adjoint
    Apar = NotImplemented
    AApar = NotImplemented

    row_beg = NotImplemented
    row_end = NotImplemented
    global_size_A = NotImplemented  # global_size_A = global_size_AA
    spatial_points = NotImplemented

    # errors
    grad_err = NotImplemented
    residual = NotImplemented

    # other
    iterations = NotImplemented
    B = NotImplemented
    Q = NotImplemented
    P = NotImplemented
    convergence = NotImplemented
    stop = False
    document = 'None'

    # communicators
    frac = NotImplemented
    Frac = NotImplemented

    # global communicators
    comm = NotImplemented
    rank = NotImplemented
    size = NotImplemented

    # global for state and adjoint
    comm_state = NotImplemented
    comm_rank_state = NotImplemented
    comm_size_state = NotImplemented

    comm_adjoint = NotImplemented
    comm_rank_adjoint = NotImplemented
    comm_size_adjoint = NotImplemented

    # row for state and adjoint
    comm_row_state = NotImplemented
    rank_row_state = NotImplemented
    size_row_state = NotImplemented

    comm_row_adjoint = NotImplemented
    rank_row_adjoint = NotImplemented
    size_row_adjoint = NotImplemented

    # col for state and adjoint
    comm_col_state = NotImplemented
    rank_col_state = NotImplemented
    size_col_state = NotImplemented

    comm_col_adjoint = NotImplemented
    rank_col_adjoint = NotImplemented
    size_col_adjoint = NotImplemented

    # subcol_alternating for state and adjoint
    comm_subcol_alternating_state = NotImplemented
    rank_subcol_alternating_state = NotImplemented
    size_subcol_alternating_state = NotImplemented

    comm_subcol_alternating_adjoint = NotImplemented
    rank_subcol_alternating_adjoint = NotImplemented
    size_subcol_alternating_adjoint = NotImplemented

    # subcol_seq for state and adjoint
    comm_subcol_seq_state = NotImplemented
    rank_subcol_seq_state = NotImplemented
    size_subcol_seq_state = NotImplemented

    comm_subcol_seq_adjoint = NotImplemented
    rank_subcol_seq_adjoint = NotImplemented
    size_subcol_seq_adjoint = NotImplemented

    # last for state and adjoint
    comm_last_state = NotImplemented
    comm_last_adjoint = NotImplemented

    # matrix communicators for state and adjoint
    comm_matrixA = NotImplemented
    comm_matrixAA = NotImplemented

    # benchmarking
    algorithm_time = NotImplemented
    communication_time = NotImplemented
    system_time_max = NotImplemented
    system_time_min = NotImplemented

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--T_start', type=float, default=0, help='Default = 0')
        parser.add_argument('--T_end', type=float, default=1, help='Default = 1')
        parser.add_argument('--beta', type=float, default=1, help='Default = 1')
        parser.add_argument('--alpha', type=float, default=1e-6, help='Default = 1e-6')
        parser.add_argument('--time_intervals', type=int, default=1, help='Default = 10 ... size of the B matrix or how many intervals will be treated in parallel.')
        parser.add_argument('--time_points', type=int, default=3, help='Default = 3 ... number of time points for the collocation problem, the size of Q.')
        parser.add_argument('--proc_row', type=int, default=1, help='Default = 1 ... number of processors for dealing with paralellization of time intervals. Choose so that proc_row = time_intervals or get an error.')
        parser.add_argument('--proc_col', type=int, default=1, help='Default = 1 ... number fo processors dealing with parallelization of the time-space collocation problem. If just time parallelization, choose so that (proc_col | time_points) and (proc_col >= 1). If space-time parallelization, choose proc_col = k * time_point, (where 0 < k < spatial_points) and (k | spatial points).')
        parser.add_argument('--spatial_points', type=int, nargs='+', default=[24], help='Default = 24 ... number of spatial points with unknown values (meaning: not including the boundary ones)')
        parser.add_argument('--solver', type=str, default='lu', help='Default = lu ... specifying the inner linear refinement solver: lu, gmres, custom.')
        parser.add_argument('--maxiter', type=int, default=5, help='Default = 5 ... maximum number of iterations on one rolling interval.')
        parser.add_argument('--tol', type=float, default=1e-6, help='Default = 1e-6 ... a stopping criteria when two consecutive solutions in the last time point are lower than tol (in one rolling interval).')
        parser.add_argument('--stol', type=float, default=1e-6, help='Default = 1e-6 ... an inner solver tolerance.')
        parser.add_argument('--smaxiter', type=float, default=1e-6, help='Default = 100 ... an inner solver maximum iterations.')

        args = parser.parse_args().__dict__

        self.T_start = args['T_start']
        self.T_end = args['T_end']
        self.betas = [args['beta']]
        self.alphas = [args['alpha']]
        self.time_intervals = args['time_intervals']
        self.time_points = args['time_points']
        self.proc_row = args['proc_row']
        self.proc_col = args['proc_col']
        self.spatial_points = args['spatial_points']
        self.solver = args['solver']
        self.maxiter = args['maxiter']
        self.tol = args['tol']
        self.stol = args['stol']
        self.smaxiter = args['smaxiter']


    def bpar_y(self, *args):
        pass

    def bpar_p(self, *args):
        pass

    def y_initial(self, *args):
        pass

    def p_initial(self, *args):
        pass

    def gradient(self, *args):
        pass

    def objective(self, *args):
        pass

    @staticmethod
    def linear_solver(*args):
        pass

    @staticmethod
    def norm(*args):
        pass

    def linear_solver(self, *args):
        pass
