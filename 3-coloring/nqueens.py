from time import time
import numpy as np
from matplotlib import pyplot as plt


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


class Nqueens:

    """ Based on rows as decision variables """

    def __init__(self, n=8, initials=None, plot=False, plot_latency=0.5, final_plot=False):
        self.n = n
        self.domain = np.ones((self.n, self.n), dtype=bool)
        self.initials = initials
        self.picked = []
        self.plot = plot
        self.final_plot = final_plot
        self.picks = 0
        self._solved = False
        self.timing = dict()
        if self.plot:
            self.plot_latency = plot_latency
            self.fig = plt.figure(figsize=(11, 6))
            self.update_plot()

    @property
    def board(self):
        a = self.domain.copy()
        for i in range(self.n):
            if a[i,:].sum() > 1:
                a[i,:] = False
        return a

    @timing
    def row_constraints_ok(self, row):
        nb = self.domain[row,:].sum()
        if nb == 1:
            col = np.where(self.domain[row,:])[0][0]
            return all([
                self.domain[:,col].sum() == 1,
                self.domain.diagonal(col - row).sum() == 1,
                np.fliplr(self.domain).diagonal(self.n - 1 - col - row).sum() == 1
            ])
        elif nb == 0:
            return False
        else:
            return True

    @timing
    def is_feasible(self):
        feasible = all([self.row_constraints_ok(i) for i in range(self.n)])
        return feasible

    @timing
    def is_solved(self):
        if not self._solved:
            solved = self.is_feasible() and (self.domain.sum() == self.n)
            self._solved = solved
        return self._solved

    @timing
    def update_plot(self):
        self.fig.clf()
        self.fig.suptitle(f'{self.n}-queens problem')
        axes = self.fig.subplots(1, 2)
        axes[0].set_title('Board', pad=20)
        axes[1].set_title('Domain', pad=20)
        for i in range(self.n):
            axes[0].vlines(i + 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes[1].vlines(i + 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes[0].hlines(i + 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
            axes[1].hlines(i + 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
        axes[0].matshow(self.board)
        axes[1].matshow(self.domain)
        plt.draw()
        if self.plot:
            plt.pause(self.plot_latency)

    @timing
    def prune(self):
        prune_again = True
        while prune_again:
            prune_again = False
            for i in range(self.n):
                if sum(self.domain[i,:]) == 1:
                    col = np.where(self.domain[i,:])[0][0]  # only value
                    self.domain[:i, col] = False
                    self.domain[i+1:, col] = False
                    for j in range(self.n):
                        if j != i:
                            if 0 <= col - (i - j) < self.n:
                                self.domain[j, col - (i - j)] = False
                            if 0 <= col + (i - j) < self.n:
                                self.domain[j, col + (i - j)] = False
            if self.plot:
                self.update_plot()

    @timing
    def pick(self):
        self.picks += 1
        domains = self.domain.sum(1)
        pickable = np.where(domains > 1)[0]
        i = pickable[domains[domains > 1].argmin()]
        j = min(np.where(self.domain[i, :])[0])
        picked = (self.domain.copy(), i, j)
        self.picked.append(picked)
        self.domain[i, :j] = False
        self.domain[i, j + 1:] = False

    @timing
    def rollback(self):
        # used when reaching unfeasible domain
        # come back to old domain, remove descendant choice
        if len(self.picked) == 0:
            return False  # not rollbackable
        domain, i, j = self.picked[-1]
        self.picked = self.picked[:-1]
        self.domain = domain
        self.domain[i, j] = False
        return True

    @timing
    def solve(self):
        if self.initials:
            for row, col in self.initials:
                self.domain[row,:col] = False
                self.domain[row, col+1:] = False
        while True:
            self.prune()
            if self.is_feasible():
                if self.is_solved():
                    if self.final_plot and (not self.plot):
                        self.fig = plt.figure(figsize=(11, 6))
                        self.update_plot()
                    print(f'This {self.n}-queens problem has been successfully solved!')
                    return self
                else:
                    self.pick()
            else:
                rolled_back = self.rollback()
                if not rolled_back:
                    print(f'This {self.n}-queens problem is not feasible '
                          'under given original conditions')
                    return


def place_queens(**kwargs):
    data = Nqueens(**kwargs)
    data.solve()
    if data.is_solved():
        print(f'Number of picks: {data.picks}')
    return data


def benchmark(store=False):
    N = 1000
    times = np.zeros(N)
    solved = np.zeros(N, dtype=bool)
    picks = np.zeros(N, dtype=int)
    for i in range(N):
        t = time()
        data = place_queens(
            n=i,
            initials=None,
            plot=False,
            plot_latency=0.001,
            final_plot=False
        )
        solved[i] = data.is_solved()
        picks[i] = data.picks
        times[i] = time() - t
    if store:
        import pickle
        with open('/tmp/nqueens.pkl', 'wb') as f:
            pickle.dump((solved, picks, times), f)
    else:
        return solved, times, picks
