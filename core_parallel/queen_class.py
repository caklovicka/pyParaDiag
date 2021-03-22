import argparse
import abc


class QueenClass(abc.ABC):

    alphas = []
    bad_alphas = []
    m0 = 1
    dt = NotImplemented
    t = NotImplemented
    x = NotImplemented
    u0_loc = NotImplemented
    u_last_loc = NotImplemented
    u_loc = NotImplemented
    u_last_old_loc = NotImplemented
    err_last = NotImplemented
    inner_tols = NotImplemented
    iterations = NotImplemented
    B = NotImplemented
    Q = NotImplemented
    P = NotImplemented
    time_document = NotImplemented
    Apar = NotImplemented
    row_beg = NotImplemented
    row_end = NotImplemented
    spatial_points = NotImplemented
    dx = NotImplemented

    # passive
    comm = NotImplemented
    rank = NotImplemented
    size = NotImplemented
    comm_row = NotImplemented
    rank_row = NotImplemented
    rank_row = NotImplemented
    size_row = NotImplemented
    frac = NotImplemented
    Frac = NotImplemented
    comm_col = NotImplemented
    rank_col = NotImplemented
    size_col = NotImplemented
    cols_loc = NotImplemented
    rows_loc = NotImplemented
    comm_subcol_alternating = NotImplemented
    rank_subcol_alternating = NotImplemented
    size_subcol_alternating = NotImplemented
    comm_subcol_seq = NotImplemented
    rank_subcol_seq = NotImplemented
    size_subcol_seq = NotImplemented
    comm_last = NotImplemented
    comm_matrix = NotImplemented
    optimal_alphas = False
    stop = False
    global_size_A = NotImplemented

    algorithm_time = NotImplemented
    communication_time = NotImplemented
    system_time_max = NotImplemented
    system_time_min = NotImplemented

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--T_start', type=float, default=0, help='Default = 0')
        parser.add_argument('--T_end', type=float, default=1, help='Default = 1')
        parser.add_argument('--rolling', type=int, default=1, help='Default = 1 ... number of intervals to perform one paralpha and combine it sequentially for the next one.')
        parser.add_argument('--time_intervals', type=int, default=1, help='Default = 10 ... size of the B matrix or how many intervals will be treated in parallel.')
        parser.add_argument('--time_points', type=int, default=3, help='Default = 3 ... number of time points for the collocation problem, the size of Q.')
        parser.add_argument('--proc_row', type=int, default=1, help='Default = 1 ... number of processors for dealing with paralellization of time intervals. Choose so that proc_row = time_intervals or get an error.')
        parser.add_argument('--proc_col', type=int, default=1, help='Default = 1 ... number fo processors dealing with parallelization of the time-space collocation problem. If just time parallelization, choose so that (proc_col | time_points) and (proc_col >= 1). If space-time parallelization, choose proc_col = k * time_point, (where 0 < k < spatial_points) and (k | spatial points).')
        parser.add_argument('--spatial_points', type=int, nargs='+', default=[24], help='Default = 24 ... number of spatial points with unknown values (meaning: not including the boundary ones)')
        parser.add_argument('--solver', type=str, default='lu', help='Default = lu ... specifying the inner linear solver: lu, gmres, custom.')
        parser.add_argument('--maxiter', type=int, default=5, help='Default = 5 ... maximum number of iterations on one rolling interval.')
        parser.add_argument('--tol', type=float, default=1e-6, help='Default = 1e-6 ... a stopping criteria when two consecutive solutions in the last time point are lower than tol (in one rolling interval).')
        parser.add_argument('--stol', type=float, default=1e-6, help='Default = 1e-6 ... an inner solver tolerance.')
        parser.add_argument('--smaxiter', type=float, default=1e-6, help='Default = 100 ... an inner solver maximum iterations.')
        parser.add_argument('--document', type=str, default='None', help='Default = None ... document to write an output.')

        args = parser.parse_args().__dict__

        self.T_start = args['T_start']
        self.T_end = args['T_end']
        self.rolling = args['rolling']
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
        self.document = args['document']

    def bpar(self, *args):
        pass

    def u_initial(self, *args):
        pass

    @staticmethod
    def rhs(*args):
        pass

    @staticmethod
    def linear_solver(*args):
        pass

    @staticmethod
    def norm(*args):
        pass

    def linear_solver(self, *args):
        pass
