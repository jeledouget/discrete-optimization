"""
Resolve a Sudoku of size N:
- N rows
- N columns
Affect a value [1...N] to each element of the Sudoku so as to have
no duplicates on any row, any column, any of the N subsquares (sqrt(N) * sqrt(N))
==================================================================================== """


from time import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


# timing decorator
def timing(func):
    def wrapper(*args, **kwargs):
        t = time()
        name = func.__name__
        res = func(*args, **kwargs)
        self = args[0]
        if name in self.timing:
            self.timing[name].append(time() - t)
        else:
            self.timing[name] = [time() - t]
        return res
    return wrapper


class Sudoku1:

    def __init__(self, n=9, initials=(), plot=False, plot_latency=0.1, final_plot=False):
        if n not in np.array([4, 9, 16, 25, 36, 49, 64, 81, 100]):
            raise ValueError('n should be a square value in {4, 9, ..., 100}')
        self.n = n
        self.sqrt = int(np.sqrt(self.n))
        self.domain = np.ones((self.n, self.n, self.n), dtype=bool)
        self.board = np.zeros((self.n, self.n), dtype=int)
        if isinstance(initials, np.ndarray):  # unknown values : zeros
            pos = np.where(initials)
            self.initials = [(i + 1, j + 1, initials[i, j]) for i, j in zip(*pos)]
        else:
            self.initials = initials
        for (i,j,k) in self.initials:
            self.domain[i-1,j-1,:] = False
            self.domain[i-1,j-1,k-1] = True
            self.board[i-1,j-1] = k
        self.queue = []
        self.pruned_positions = []
        self.splits = 0
        self.picks = 0
        self.plot = plot
        self.final_plot = final_plot
        self.timing = dict()
        self.plot_latency = plot_latency
        self.fig = None

    @timing
    def update_plot(self):
        self.update_board()
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 7))
        self.fig.clf()
        axes = self.fig.subplots(1)
        axes.set_title('Sudoku', pad=20)
        axes.axis('off')
        for i in range(self.n + 1):
            axes.vlines(i - 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes.hlines(i - 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
        for i in range(self.sqrt, self.n, self.sqrt):
            axes.vlines(i - 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=2)
            axes.hlines(i - 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=2)
        for (i,j,_) in self.initials:
            axes.add_patch(patches.Rectangle(
                (j - 1.5, self.n - 0.5 - i), 1, 1, edgecolor=None, facecolor='y', alpha=0.3
            ))
        board = self.board
        pos = np.where(board > 0)
        for i,j in zip(*pos):
            val = self.board[i, j]
            axes.text(j, self.n - 1 - i, val, horizontalalignment='center', verticalalignment='center')
        plt.draw()
        if self.plot_latency:
            plt.pause(self.plot_latency)

    @timing
    def feasible(self):
        """
        - No 2 same values on a row
        - No 2 same values on a column
        - No 2 same values in a subsquare
        - Non-empty domain for each value in each row
        - Non-empty domain for each value in each column
        - Non-empty domain for each value in each subsquare
        """
        if not (self.domain.sum(2) >= 1).all():
            return False
        fixed_pos = np.where(self.domain.sum(2) == 1)
        for val in range(self.n):
            val_board = np.zeros((self.n, self.n), dtype=bool)
            for i,j in zip(*fixed_pos):
                if self.domain[i,j,val]:
                    val_board[i,j] = True
            if (val_board.sum(0) > 1).any():
                return False
            if (val_board.sum(1) > 1).any():
                return False
            for i in range(self.sqrt):
                for j in range(self.sqrt):
                    min_i = self.sqrt * (i // self.sqrt)
                    min_j = self.sqrt * (j // self.sqrt)
                    sub_val_board = val_board[min_i:min_i+self.sqrt,min_j:min_j+self.sqrt]
                    if sub_val_board.sum() > 1:
                        return False
        return True

    @timing
    def solved(self):
        if not (self.domain.sum(2) == 1).all():
            return False
        values = set(range(self.n))
        for i in range(self.n):
            rows = [np.where(self.domain[i, j, :])[0][0] for j in range(self.n)]
            cols = [np.where(self.domain[j, i, :])[0][0] for j in range(self.n)]
            a, b = self.sqrt * (i // self.sqrt), self.sqrt * (i % self.sqrt)
            sq_range = range(self.sqrt)
            squares = [np.where(self.domain[a+j, b+k, :])[0][0] for j in sq_range for k in sq_range]
            if (set(rows) != values) or (set(cols) != values) or (set(squares) != values):
                return False
        return True

    @timing
    def prune(self):
        """ For each value (1 to N):
        - Intersect domains (row - col - subsquare)
        """
        new_s = self.domain.sum()
        s = new_s + 1  # arbitrary initial value
        while new_s < s:
            s = new_s
            # clean all existing values
            pos = np.where(self.domain.sum(2) == 1)
            values = [(i,j,np.where(self.domain[i,j,:])[0][0])
                      for i,j in zip(*pos) if (i,j) not in self.pruned_positions]
            for i,j,val in values:
                self.domain[i,:,val] = False
                self.domain[:,j, val] = False
                min_i = self.sqrt * (i // self.sqrt)
                min_j = self.sqrt * (j // self.sqrt)
                self.domain[min_i:min_i+self.sqrt,min_j:min_j+self.sqrt, val] = False
                self.domain[i,j,val] = True
                self.pruned_positions.append((i,j))
            new_s = self.domain.sum()

    @timing
    def update_board(self):
        # remove all existing values
        pos = np.where(self.domain.sum(2) == 1)
        for i, j in zip(*pos):
            self.board[i,j] = 1 + np.where(self.domain[i,j,:])[0][0]

    @timing
    def split(self):
        """
        Remove half possible values for a random location amongst minimum size domains
        """
        self.splits += 1
        sum_domain = self.domain.sum(2)
        sum_domain[sum_domain == 1] = self.n
        choices = np.where(sum_domain == sum_domain.min())
        rand = np.random.choice(range(choices[0].shape[0]))
        i,j = choices[0][rand], choices[1][rand]
        vals = np.where(self.domain[i,j,:])[0]
        half = len(vals) // 2
        left_domain = self.domain.copy()
        left_domain[i,j,vals[:half]] = False
        right_domain = self.domain.copy()
        right_domain[i,j,vals[half:]] = False
        self.queue.append((left_domain, self.pruned_positions.copy(), self.board.copy()))
        self.queue.append((right_domain, self.pruned_positions.copy(), self.board.copy()))

    @timing
    def solve(self):
        """ If feasible, prune. If pruned, split."""
        self.queue.append((self.domain.copy(), self.pruned_positions.copy(), self.board.copy()))
        while self.queue:
            if self.plot:
                self.update_plot()
            self.picks += 1
            self.domain, self.pruned_positions, self.board = self.queue.pop()
            self.prune()
            if self.feasible():  # board is up-to-date
                if self.solved():
                    self.update_board()
                    break
                else:
                    self.split()
        if self.final_plot:
            self.update_plot()


"""
Try more efficient pruning
----------------------------- """

class Sudoku2(Sudoku1):

    @timing
    def feasible(self):
        """
        - No 2 same values on a row
        - No 2 same values on a column
        - No 2 same values in a subsquare
        - Non-empty domain for each value in each row
        - Non-empty domain for each value in each column
        - Non-empty domain for each value in each subsquare
        """
        if not (self.domain.sum(2) >= 1).all():
            return False
        fixed_pos = np.where(self.domain.sum(2) == 1)
        for val in range(self.n):
            #
            val_board = np.zeros((self.n, self.n), dtype=bool)
            for i, j in zip(*fixed_pos):
                if self.domain[i, j, val]:
                    val_board[i, j] = True
            if (val_board.sum(0) > 1).any():
                return False
            if (val_board.sum(1) > 1).any():
                return False
            for i in range(self.sqrt):
                for j in range(self.sqrt):
                    min_i = self.sqrt * (i // self.sqrt)
                    min_j = self.sqrt * (j // self.sqrt)
                    sub_val_board = val_board[min_i:min_i + self.sqrt, min_j:min_j + self.sqrt]
                    if sub_val_board.sum() > 1:
                        return False
            # check subsquares are compatible
            val_rows = set([i+x for i,j,sq in self.subsquares(val) for x in np.where(sq.sum(1))[0]])
            if val_rows != set(range(self.n)):
                return False
            val_cols = set([j+y for i,j,sq in self.subsquares(val) for y in np.where(sq.sum(0))[0]])
            if val_cols != set(range(self.n)):
                return False
        return True

    def subsquares(self, val=None):
        for i in range(0, self.n, self.sqrt):
            for j in range(0, self.n, self.sqrt):
                if val is None:
                    yield i,j,self.domain[i:i+self.sqrt, j:j+self.sqrt,:]
                else:
                    yield i,j,self.domain[i:i + self.sqrt, j:j + self.sqrt, val]

    @timing
    def prune(self):
        """ For each value (1 to N):
        - Intersect domains (row - col - subsquare)
        """
        new_s = self.domain.sum()
        s = new_s + 1  # arbitrary initial value
        while new_s < s:
            s = new_s
            # clean all existing values
            pos = np.where(self.domain.sum(2) == 1)
            values = [(i,j,np.where(self.domain[i,j,:])[0][0])
                      for i,j in zip(*pos) if (i,j) not in self.pruned_positions]
            for i,j,val in values:
                self.domain[i,:,val] = False
                self.domain[:,j, val] = False
                min_i = self.sqrt * (i // self.sqrt)
                min_j = self.sqrt * (j // self.sqrt)
                self.domain[min_i:min_i+self.sqrt,min_j:min_j+self.sqrt, val] = False
                self.domain[i,j,val] = True
                self.pruned_positions.append((i,j))
            # also: for each val, if alone in subsquare: remove all on row and col
            # also: if domain of a value in a subsquare is limited to a row or a column : propagate
            for val in range(self.n):
                if self.domain[:,:,val].sum() > self.n:
                    for sq_i,sq_j,square in self.subsquares(val):
                        if square.sum() == 1:
                            i,j = np.where(square)[0][0], np.where(square)[1][0]
                            self.domain[sq_i+i,:sq_j,val] = False
                            self.domain[sq_i + i, sq_j+self.sqrt:, val] = False
                            self.domain[:sq_i, sq_j+ j, val] = False
                            self.domain[sq_i + self.sqrt:, sq_j + j, val] = False
            for i in range(self.sqrt):
                for j in range(self.sqrt):
                    min_i = self.sqrt * i
                    min_j = self.sqrt * j
                    for val in range(self.n):
                        subsquare = self.domain[min_i:min_i+self.sqrt,min_j:min_j+self.sqrt, val]
                        if subsquare.sum() > 1:
                            pos = np.where(subsquare)
                            if (pos[0] == pos[0][0]).all():
                                row = pos[0][0] + min_i
                                self.domain[row, :min_j, val] = False
                                self.domain[row, min_j + self.sqrt:, val] = False
                            if (pos[1] == pos[1][0]).all():
                                col = pos[1][0] + min_j
                                self.domain[:min_i, col, val] = False
                                self.domain[min_i + self.sqrt:, col, val] = False
            new_s = self.domain.sum()


"""
Instantiation / benchmark
----------------------------- """

solvers = {
    1: Sudoku1,
    2: Sudoku2
}

if __name__ == '__main__':

    # difficult level
    benchmark_initials = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 3],
        [0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0],
        [4, 0, 1, 6, 0, 0, 0, 0, 0],
        [0, 0, 7, 1, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 4, 0],
        [0, 3, 0, 9, 1, 0, 0, 0, 0]
    ])

    benchmark_solution = np.array([
        [7, 4, 5, 3, 6, 8, 9, 1, 2],
        [8, 1, 9, 5, 7, 2, 4, 6, 3],
        [3, 6, 2, 4, 9, 1, 8, 5, 7],
        [6, 9, 3, 8, 2, 4, 5, 7, 1],
        [4, 2, 1, 6, 5, 7, 3, 9, 8],
        [5, 8, 7, 1, 3, 9, 6, 2, 4],
        [1, 5, 8, 7, 4, 6, 2, 3, 9],
        [9, 7, 6, 2, 8, 3, 1, 4, 5],
        [2, 3, 4, 9, 1, 5, 7, 8, 6]
    ])

    s1 = Sudoku1(initials=benchmark_initials)
    s2 = Sudoku2(initials=benchmark_initials)

    contre_christelle = np.array([
        [8, 0, 0, 1, 0, 0, 0, 7, 0],
        [0, 2, 0, 0, 4, 0, 8, 0, 0],
        [0, 6, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 7, 0, 9, 0, 8],
        [2, 4, 0, 0, 8, 0, 0, 0, 0],
        [0, 3, 8, 0, 0, 0, 0, 0, 5],
        [0, 8, 0, 6, 0, 4, 1, 0, 0],
        [9, 0, 0, 0, 0, 7, 2, 0, 4],
        [0, 0, 5, 8, 1, 0, 0, 0, 6]
    ])

    s_christelle = Sudoku1(
        initials=contre_christelle,
        plot=False,
        final_plot=True
    )
