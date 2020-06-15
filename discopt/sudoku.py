"""
Resolve a Sudoku of size N:
- N rows
- N columns
Affect a value [1...N] to each element of the Sudoku so as to have
no duplicates on any row, any column, any of the N subsquares (sqrt(N) * sqrt(N))
==================================================================================== """


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class Sudoku:

    def __init__(self, n=9, initials=(), plot=True, plot_latency=0.5, final_plot=True):
        if n not in np.array([4, 9, 16, 25, 36, 49, 64, 81, 100]):
            raise ValueError('n should be a square value in {4, 9, ..., 100}')
        self.n = n
        self.sqrt = int(np.sqrt(self.n))
        self.domain = np.ones((self.n, self.n, self.n), dtype=bool)
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.initials = initials
        self.queue = []
        self.splits = 0
        self.picks = 0
        self.plot = plot
        self.final_plot = final_plot
        self.timing = dict()
        self.plot_latency = plot_latency
        self.fig = None

    def update_plot(self):
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
                (j - 0.5, self.n - 1.5 - i), 1, 1, edgecolor=None, facecolor='y', alpha=0.3
            ))
        board = self.board
        pos = np.where(board > 0)
        for i,j in zip(*pos):
            val = self.board[i, j]
            axes.text(j, self.n - 1 - i, val)
        plt.draw()
        if self.plot_latency:
            plt.pause(self.plot_latency)

    def feasible(self):
        """
        - No 2 same values on a row
        - No 2 same values on a column
        - No 2 same values in a subsquare
        - Non-empty domain for each value in each row
        - Non-empty domain for each value in each column
        - Non-empty domain for each value in each subsquare
        """
        return (self.domain.sum(2) >= 1).all()

    def solved(self):
        if not (self.domain.sum(2) == 1).all():
            return False
        values = set(range(1, self.n + 1))
        for i in range(self.n):
            rows = [np.where(self.domain[i, j, :])[0][0] for j in range(self.n)]
            cols = [np.where(self.domain[j, i, :])[0][0] for j in range(self.n)]
            a, b = self.sqrt * (i // self.sqrt), self.sqrt * (i % self.sqrt)
            sq_range = range(self.sqrt)
            squares = [np.where(self.domain[a+j, b+k, :])[0][0] for j in sq_range for k in sq_range]
            if (set(rows) != values) or (set(cols) != values) or (set(squares) != values):
                return False
        return True

    def prune(self):
        """ For each value (1 to N):
        - Intersect domains (row - col - subsquare)
        """
        prune_again = True
        new_s = self.domain.sum()
        while prune_again:
            s = new_s
            # clean all existing values
            pos = np.where(self.domain.sum(2) == 1)
            for i,j in zip(*pos):
                val = np.where(self.domain[i,j,:])[0][0]
                self.domain[i,:,val] = False
                self.domain[:,j, val] = False
                min_i = i // self.sqrt
                min_j = j // self.sqrt
                self.domain[min_i:min_i+self.sqrt,min_j:min_j+self.sqrt, val] = False
                self.domain[i,j,val] = True
            new_s = self.domain.sum()
            prune_again = (new_s > s)
        self.update_board()

    def update_board(self):
        # remove all existing values
        pos = np.where(self.domain.sum(2) == 1)
        for i, j in zip(*pos):
            self.board[i,j] = 1 + np.where(self.domain[i,j,:])[0][0]

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
        self.queue.append(left_domain)
        self.queue.append(right_domain)

    def solve(self):
        """ If feasible, prune. If pruned, split."""
        for (i,j,k) in self.initials:
            self.domain[i,j,:] = False
            self.domain[i,j,k-1] = True
            self.board[i,j] = k
        self.queue.append(self.domain.copy())
        while self.queue:
            if self.plot:
                self.update_plot()
            self.picks += 1
            self.domain = self.queue.pop()
            self.prune()
            if self.feasible():
                if self.solved():
                    break
                else:
                    self.split()
        if self.final_plot:
            self.update_plot()


if __name__ == '__main__':
    s = Sudoku(
        n=9,
        initials=(
            (0, 7, 1),
            (1, 5, 2),
            (1, 8, 3),
            (2, 3, 4),
            (3, 6, 5),
            (4, 0, 4),
            (4, 2, 1),
            (4, 3, 6),
            (5, 2, 7),
            (5, 3, 1),
            (6, 1, 5),
            (6, 6, 2),
            (7, 4, 8),
            (7, 7, 4),
            (8, 1, 3),
            (8, 3, 9),
            (8, 4, 1)
        )
    )
