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

    def __init__(self, n=8, initials=None, plot=False, plot_latency=0.5, final_plot=False, verbose=True):
        self.n = n
        self.domain = np.ones((self.n, self.n), dtype=bool)
        self.initials = initials
        self.picked = []
        self.plot = plot
        self.final_plot = final_plot
        self.picks = 0
        self._solved = False
        self.timing = dict()
        self.verbose = verbose
        self.plot_latency = plot_latency
        if self.plot:
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
        if 'fig' not in self.__dict__:
            self.fig = plt.figure(figsize=(11, 6))
        self.fig = plt.figure(figsize=(11, 6))
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
            for i in range(self.n):  # row
                if self.domain[i,:].sum() == 1:
                    col = np.where(self.domain[i,:])[0][0]  # only value
                    if any([
                        self.domain[:, col].sum() > 1,
                        self.domain.diagonal(col - i).sum() > 1,
                        np.fliplr(self.domain).diagonal(self.n - 1 - col - i).sum() > 1
                    ]):
                        prune_again = True
                        self.domain[:, col] = False
                        diag_indices = [
                            [j, col - (i - j)]
                            for j in range(self.n)
                            if 0 <= col - (i - j) < self.n
                        ] + [
                            [j, col + (i - j)]
                            for j in range(self.n)
                            if 0 <= col + (i - j) < self.n
                        ]
                        self.domain[tuple(zip(*diag_indices))] = False
                        # put back True value
                        self.domain[i, col] = True
            if self.plot:
                self.update_plot()

    @timing
    def pick(self):
        self.picks += 1
        domains = self.domain.sum(1)
        pickable = np.where(domains > 1)[0]
        i = pickable[domains[domains > 1].argmin()]
        j = np.where(self.domain[i, :])[0].min()
        picked = (self.domain.copy(), i, j)
        self.picked.append(picked)
        self.domain[i, :j] = False
        self.domain[i, j+1:] = False

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
                    if self.verbose:
                        print(f'This {self.n}-queens problem has been successfully solved!')
                    return self
                else:
                    self.pick()
            else:
                rolled_back = self.rollback()
                if not rolled_back:
                    if self.verbose:
                        print(f'This {self.n}-queens problem is not feasible '
                              'under given original conditions')
                    return


class NqueensQueue(Nqueens):

    """ Based on rows as decision variables
     Here, no rollback: solution is abandoned if unfeasible
     2 options at each 'pick' based on domain
     """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = []

    @timing
    def split(self):
        self.picks += 1
        domains = self.domain.sum(1)
        splitable = np.where(domains > 1)[0]
        i = splitable[domains[domains > 1].argmin()]
        j = np.where(self.domain[i, :])[0]
        left_cols = j[:len(j) // 2]
        right_cols = j[len(j) // 2:]
        left_domain = self.domain.copy()
        left_domain[i, left_cols] = False
        self.queue.append(left_domain)
        right_domain = self.domain.copy()
        right_domain[i, right_cols] = False
        self.queue.append(right_domain)

    @timing
    def solve(self):
        if self.initials:
            for row, col in self.initials:
                self.domain[row,:] = False
                self.domain[row, col] = True
        self.queue.append(self.domain)
        while self.queue:
            self.domain = self.queue.pop()
            self.prune()
            if self.is_feasible():
                if self.is_solved():
                    if self.final_plot and (not self.plot):
                        self.fig = plt.figure(figsize=(11, 6))
                        self.update_plot()
                    if self.verbose:
                        print(f'This {self.n}-queens problem has been successfully solved!')
                    return self
                else:
                    self.split()
        # if we reach the end of the queue: not solvable
        if self.verbose:
            print(f'This {self.n}-queens problem is not feasible '
                  'under given original conditions')
        return


class NqueensPrime(NqueensQueue):

    """ Decompose in blocks. Solve by blocks. Fill in solution
    Can be a gain for some high values ...
    Could be super effective if we'd store a database of solutions """

    def __init__(self, n=8, solver='NqueensQueue'):
        self.solver = solver
        self.n = n
        self.domain = np.ones((self.n, self.n), dtype=bool)
        self.primes = []
        self.prime_solutions = {}

    def plot(self):
        fig = plt.figure(figsize=(11, 6))
        fig.suptitle(f'{self.n}-queens problem')
        axes = fig.subplots(1)
        axes.set_title('Board', pad=20)
        for i in range(self.n):
            axes.vlines(i + 0.5, ymin=-0.5, ymax=self.n - 0.5, lw=0.5)
            axes.hlines(i + 0.5, xmin=-0.5, xmax=self.n - 0.5, lw=0.5)
        axes.matshow(self.domain)
        plt.draw()

    @staticmethod
    def prime_decomposition_after_4(n):
        primes = []
        i = 2
        while i*i <= n:
            if (n >= 2 * i) and (n % i == 0):
                primes.append(i)
                n //= i
            else:
                i += 1
        primes.append(n)
        while (primes[0] < 4) and (len(primes) > 1):
            primes = [primes[0] * primes[1]] + primes[2:]
            primes.sort()
        return primes

    def combine(self):
        solutions = [self.prime_solutions[p] for p in self.primes]
        current = solutions[-1]
        for sol in solutions[-2::-1]:
            c = sol.tolist()
            for i in range(len(c)):
                for j in range(len(c[i])):
                    c[i][j] = current * c[i][j]
            current = np.block(c)
        self.domain = current

    def solve(self):
        self.primes = self.prime_decomposition_after_4(self.n)
        for p in self.primes:
            if p in self.prime_solutions:
                continue
            self.prime_solutions[p] = solvers[self.solver](
                n=p, initials=None, verbose=False
            ).solve().domain
            print(f'Solved {p}-queens problem')
        self.combine()
        return self


""" Solve and benchmark
---------------------------- """

solvers = {
    'Nqueens': Nqueens,
    'NqueensQueue': NqueensQueue,
    'NqueensPrime': NqueensPrime
}


def place_queens(solver='Nqueens', **kwargs):
    data = solvers[solver](**kwargs)
    data.solve()
    if data.is_solved():
        print(f'Number of picks: {data.picks}')
    return data


def benchmark(solver='Nqueens', store=False, n=1000, **kwargs):
    global solved, times, picks
    times = np.zeros(n)
    solved = np.zeros(n, dtype=bool)
    picks = np.zeros(n, dtype=int)
    for i in range(n):
        t = time()
        data = place_queens(
            solver=solver,
            n=i,
            initials=None,
            **kwargs
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
