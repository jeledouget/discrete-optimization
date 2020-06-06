import numpy as np
from matplotlib import pyplot as plt


class Nqueens:

    def __init__(self, n=8, initials=None, plot=True, plot_latency=0.5):
        self.n = n
        self.domain = np.ones((self.n, self.n), dtype=bool)
        self.initials = initials
        self.picked = []
        self.plot = plot
        if self.plot:
            self.plot_latency = plot_latency
            self.fig = plt.figure()
            self.update_plot()

    @property
    def board(self):
        a = self.domain.copy()
        for i in range(self.n):
            if a[i,:].sum() > 1:
                a[i,:] = False
        return a
        """
        x = np.zeros((self.n, self.n))
        for i, val in enumerate(self.row):
            if val != -1:
                x[i, val] = 2
        for i, (j,k,_) in enumerate(self.picked):
            x[j, k] = 1
        return x"""

    @property
    def np_domain(self):
        return self.domain
        """x = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in self.domain[i]:
                x[i, j] = 1
        return x"""

    def row_constraints_ok(self, row):
        nb = sum(self.domain[row,:])
        if nb == 1:
            col = np.where(self.domain[row,:])[0][0]
            return all(
                [self.domain[i, col] == 0 for i in range(self.n) if i != row] +
                [self.domain[i, col - (row - i)] == 0
                 for i in range(self.n)
                 if (i != row) and (0 <= col - (row - i) < self.n)
                ] +
                [self.domain[i, col + (row - i)] == 0
                 for i in range(self.n)
                 if (i != row) and (0 <= col + (row - i) < self.n)
                ]
            )
        elif nb == 0:
            return False
        else:
            return True

    def is_feasible(self):
        return all([self.row_constraints_ok(i) for i in range(self.n)])

    def is_solved(self):
        return self.is_feasible() and self.domain.sum() == self.n

    def update_plot(self):
        self.fig.clf()
        axes = self.fig.subplots(1, 2)
        for i in range(self.n):
            axes[0].vlines(i + 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes[1].vlines(i + 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes[0].hlines(i + 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
            axes[1].hlines(i + 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
        axes[0].matshow(self.board)
        axes[1].matshow(self.domain)
        plt.draw()
        plt.pause(self.plot_latency)

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

    def pick(self):
        while True:
            for i in range(self.n):
                if sum(self.domain[i,:]) > 1:
                    j = min(np.where(self.domain[i,:])[0])
                    picked = (self.domain.copy(), i, j)
                    self.picked.append(picked)
                    self.domain[i,:j] = False
                    self.domain[i, j+1:] = False
                    return True
            # otherwise rollback
            rolled_back = self.rollback()
            if not rolled_back:
                return False

    def rollback(self):
        # no possible pick: come back to old domain, remove descendant choiceru
        if len(self.picked) == 0:
            return False
        domain, i, j = self.picked[-1]
        self.picked = self.picked[:-1]
        self.domain = domain
        self.domain[i, j] = False
        return True

    def solve(self):
        if self.initials:
            for row, col in self.initials:
                self.domain[row,:col] = False
                self.domain[row, col+1:] = False
        while True:
            self.prune()
            if self.is_feasible():
                if self.is_solved():
                    print(f'This {self.n}-queens problem has been successfully solved!')
                    return self
                else:
                    picked = self.pick()
                    if not picked:
                        print(f'This {self.n}-queens problem is not feasible '
                              'under given original conditions')
                        return
            else:
                rolled_back = self.rollback()
                if not rolled_back:
                    print(f'This {self.n}-queens problem is not feasible '
                          'under given original conditions')
                    return


def place_queens(**kwargs):
    data = Nqueens(**kwargs)
    data.solve()
    return data


if __name__ == '__main__':
    data = place_queens(initials=None, plot=True, plot_latency=0.01)
