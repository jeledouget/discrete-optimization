import numpy as np
from matplotlib import pyplot as plt


class MagicSeries:

    """ Find a Magic Series of a given size. Any series is OK.
    A magic Series is a list such that : c[i] = len([x for x in c if x == i]) for all i """

    def __init__(self, n, plot=True, plot_latency=2):
        self.n = n
        self.series = -1 * np.ones(self.n, dtype=int)
        # -1 : unknown. 0: False. 1: True
        self.constraints = -1 * np.ones((self.n, self.n), dtype=int)
        self.solved = False
        self._plot = plot
        self.plot_latency = plot_latency
        self.queue = []
        self.solutions = []
        self.explorations = 0

    def is_solved(self):
        if not self.solved:
            self.solved = all([
                self.series[i] == (self.series == i).sum()
                for i in range(self.series.shape[0])
            ])
        return self.solved

    def is_feasible(self):
        """ No more that one 1 on a column; not all 0 on a column """
        c1 = ((self.constraints == 1).sum(0) <= 1).all()
        c2 = ((self.constraints == 0).sum(0) < self.n).all()
        indices = np.where(self.series >= 0)[0]
        c3 = ((self.constraints[indices, :] == 1).sum(1) <= self.series[indices]).all()
        row_indices = np.where(self.constraints == 1)[0]
        c4 = row_indices.sum() <= self.n
        return c1 and c2 and c3 and c4

    def update_series(self):
        """ assumes feasablity """
        ones = np.where(self.constraints == 1)
        self.series = -1 * np.ones(self.n, dtype=int)
        for i,j in zip(*ones):
            self.series[j] = i

    def prune(self):
        prune_again = True
        while prune_again:
            prune_again = False
            # set 0s in columns with a 1
            col_sums = (self.constraints == 1).sum(0)
            for i in np.where(col_sums == 1)[0]:
                col = self.constraints[:,i]
                if col.sum() != 1:  # -1s are present
                    row = np.where(col == 1)[0][0]
                    self.constraints[:row, i] = 0
                    self.constraints[row+1, i] = 0
                    prune_again = True
            # set 1s in columns with a all-but-1 zeros
            col_sums = (self.constraints == 0).sum(0)
            for i in np.where(col_sums == self.n - 1)[0]:
                col = self.constraints[:, i]
                row = np.where(col != 0)[0][0]
                if self.constraints[row, i] != 1:
                    self.constraints[row, i] = 1
                    prune_again = True
            # low / high bound
            for i in range(self.n):
                if (self.constraints[:,i] == 1).sum() == 0:  # self.series[i] == -1:  # unknown
                    row = self.constraints[i,:]
                    lower_bound = row[row == 1].shape[0]
                    higher_bound = row[row != 0].shape[0]
                    if (self.constraints[:lower_bound, i] == 0).sum() < self.constraints[:lower_bound, i].shape[0]:
                        self.constraints[:lower_bound, i] = 0
                        prune_again = True
                    if (self.constraints[higher_bound+1:, i] == 0).sum() < self.constraints[higher_bound+1:, i].shape[0]:
                        self.constraints[higher_bound+1:, i] = 0
                        prune_again = True
        self.update_series()

    def split(self):
        col_unknowns = (self.constraints == -1).sum(0)
        ind = np.where(col_unknowns > 0)[0]
        if len(ind) == 0:
            return
        sel_col = ind[col_unknowns[ind].argmin()]
        indices = np.where(self.constraints[:,sel_col] == -1)[0]
        half = len(indices) // 2
        # left: set first half to 0
        left_constraints = self.constraints.copy()
        left_constraints[indices[:half], sel_col] = 0
        # right:  set 2nd half to 0
        right_constraints = self.constraints.copy()
        right_constraints[indices[half:], sel_col] = 0
        # enqueue
        self.queue.append((left_constraints, self.series.copy()))
        self.queue.append((right_constraints, self.series.copy()))

    def solve(self):
        self.queue.append((self.constraints, self.series))
        while self.queue:
            self.explorations += 1
            self.solved = False
            self.constraints, self.series = self.queue.pop()
            self.prune()
            # input()
            if self.is_feasible():
                if self.is_solved():
                    print(f'found a {self.n}-magic-series : {self.series.tolist()}')
                    self.solutions.append(self.series.copy())
                else:
                    self.split()
            if self._plot:
                self.plot()
                plt.pause(self.plot_latency)

    def plot(self):
        if 'fig' not in self.__dict__:
            self.fig = plt.figure(figsize=(11, 6))
        self.fig.clf()
        self.fig.suptitle(f'Magic Series')
        axes = self.fig.subplots(1, 2)
        axes[0].text(0, 3, 'Series')
        axes[1].set_title('Domain', pad=20)
        axes[0].axis('off')
        axes[0].set_ylim(-10, 10)
        axes[1].set_xlim(-0.5, self.n - 0.5)
        axes[1].set_ylim(-0.5, self.n - 0.5)
        for i in range(self.n):
            axes[0].vlines(i + 0.5, ymin=-2, ymax=2, lw=0.5)
            axes[1].vlines(i + 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes[1].hlines(i + 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
            axes[0].text(i, 1, i)
            axes[0].text(i, -1, self.series[i])
            for j in range(self.n):
                val = self.constraints[i,j]
                axes[1].text(j, self.n - 1 - i, val if val != -1 else '?')
        axes[0].hlines(0, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
        plt.draw()
