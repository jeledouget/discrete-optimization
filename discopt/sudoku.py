"""
Resolve a Sudoku of size N:
- N rows
- N columns
Affect a value [1...N] to each element of the Sudoku so as to have
no duplicates on any row, any column, any of the N subsquares (sqrt(N) * sqrt(N))
==================================================================================== """

import numpy as np
from matplotlib import pyplot as plt

class Sudoku:

    def __init__(self, n=9, initials=None, plot=False, plot_latency=0.5, final_plot=False, verbose=True):
        if n not in np.array([4, 9, 16, 25, 36, 49, 64, 81, 100]):
            raise ValueError('n should be a square value in {4, 9, ..., 100}')
        self.n = n
        self.domain = np.ones((self.n, self.n, self.n), dtype=bool)
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

    def is_feasible(self):
        """
        - No 2 same values on a row
        - No 2 same values on a column
        - No 2 same values in a subsquare
        - Non-empty domain for each value in each row
        - Non-empty domain for each value in each column
        - Non-empty domain for each value in each subsquare
        """

    def prune(self):
        """ For each value (1 to N):
        - Intersect domains (row - col - subsquare)
        """
        pass

    def split(self):
        """
        Remove half possible values for a random location amongst minimum size domains
        """
        pass

    def solve(self):
        """ If feasible, prune. If pruned, split."""
        pass
