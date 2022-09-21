from mpi4py import MPI
from core_parallel.queen_class import QueenClass


class Communicators(QueenClass):

    def __init__(self):
        QueenClass.__init__(self)

    def setup(self):

        # create groups and their communicators
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.comm_row = MPI.Comm.Split(self.comm, self.rank % self.proc_col, self.rank // self.proc_col)
        self.rank_row = self.comm_row.Get_rank()
        self.size_row = self.comm_row.Get_size()

        self.comm_col = MPI.Comm.Split(self.comm, self.rank // self.proc_col, self.rank % self.proc_col)
        self.rank_col = self.comm_col.Get_rank()
        self.size_col = self.comm_col.Get_size()

        # this is where the space communicators start
        self.frac = self.proc_col // self.time_points
        self.Frac = self.time_points // self.proc_col

        self.rows_loc = (self.time_points * self.global_size_A) // self.proc_col

        # with spatial parallelization
        if self.frac > 1:

            self.comm_subcol_alternating = MPI.Comm.Split(self.comm_col, self.rank_col % self.frac, self.rank_col // self.frac)
            self.rank_subcol_alternating = self.comm_subcol_alternating.Get_rank()
            self.size_subcol_alternating = self.comm_subcol_alternating.Get_size()

            self.comm_subcol_seq = MPI.Comm.Split(self.comm_col, self.rank_col // self.frac, self.rank_col % self.frac)
            self.rank_subcol_seq = self.comm_subcol_seq.Get_rank()
            self.size_subcol_seq = self.comm_subcol_seq.Get_size()

            self.comm_matrix = self.comm_subcol_seq
            self.row_beg = self.rank_subcol_seq * self.rows_loc
            self.row_end = (self.rank_subcol_seq + 1) * self.rows_loc

            if self.rank_row == 0 or self.size - self.size_subcol_seq <= self.rank:
                if self.rank_row == 0:
                    tag = self.rank_col // self.frac + 1
                else:
                    tag = 0
                self.comm_last = MPI.Comm.Split(self.comm, self.rank_col % self.frac, tag)
            else:
                self.comm_last = MPI.Comm.Split(self.comm, self.size, self.rank)
                self.comm_last = None

        # without spatial parallelization
        else:
            self.comm_matrix = MPI.COMM_SELF
            self.row_beg = 0
            self.row_end = self.global_size_A

            if self.rank_row == 0 or self.size - 1 == self.rank:
                if self.rank_row == 0:
                    tag = self.rank_col + 1
                elif self.size - 1 == self.rank:
                    tag = 0
                self.comm_last = MPI.Comm.Split(self.comm, 1, tag)
            else:
                self.comm_last = MPI.Comm.Split(self.comm, 0, self.rank)
                self.comm_last = None

        """
                                proc_row = 5
                      -------------------------------   
                      |  0  |  4  |  8  | 12  | 16  |
                      -------------------------------
                      |  1  |  5  |  9  | 13  | 17  |
            comm =    -------------------------------   proc_col = 4
                      |  2  |  6  | 10  | 14  | 18  |
                      -------------------------------
                      |  3  |  7  | 11  | 15  | 19  |
                      -------------------------------
        """
        """
                                proc_row = 5
                      -------------------------------   
        comm_row =    |  0  |  1  |  2  |  3  |  4  |
                      -------------------------------
        comm_row =    |  0  |  1  |  2  |  3  |  4  |
                      -------------------------------   proc_col = 4
        comm_row =    |  0  |  1  |  2  |  3  |  4  |
                      -------------------------------
        comm_row =    |  0  |  1  |  2  |  3  |  4  |
                      -------------------------------
        """
        """
                                proc_row = 5
                      -------------------------------   
                      |  0  |  0  |  0  |  0  |  0  |
                      -------------------------------
                      |  1  |  1  |  1  |  1  |  1  |
                      -------------------------------   proc_col = 4
                      |  2  |  2  |  2  |  2  |  2  |
                      -------------------------------
                      |  3  |  3  |  3  |  3  |  3  |
                      -------------------------------
                         ^     ^     ^     ^     ^
                                 comm_col
        """
        """
                                proc_row = 5
                      -------------------------------   
                      |  0  |  0  |  0  |  0  |  0  |
                      -------------------------------
                      |  1  |  1  |  1  |  1  |  1  |
                      -------------------------------   proc_col = 4
                      |  0  |  0  |  0  |  0  |  0  |
                      -------------------------------
                      |  1  |  1  |  1  |  1  |  1  |
                      -------------------------------
                         ^     ^     ^     ^     ^
                              comm_subcol_seq
        """
        """
                                proc_row = 5
                      -------------------------------   
                      |  0  |  0  |  0  |  0  |  0  |
                      -------------------------------
                      |  0' |  0' |  0' |  0' |  0' |
                      -------------------------------   proc_col = 4
                      |  1  |  1  |  1  |  1  |  1  |
                      -------------------------------
                      |  1' |  1' |  1' |  1 '|  1' |
                      -------------------------------
                         ^     ^     ^     ^     ^
                          comm_subcol_alternating
        """
        """
                                proc_row = 5
                      -------------------------------   
                      |  1  |  x  |  x  |  x  |  x  |
                      -------------------------------
                      |  1' |  x  |  x  |  x  |  x  |
                      -------------------------------   proc_col = 4
                      |  2  |  x  |  x  |  x  |  0  |
                      -------------------------------
                      |  2' |  x  |  x  |  x  |  0' |
                      -------------------------------
                         ^     ^     ^     ^     ^
                             comm_last
        """