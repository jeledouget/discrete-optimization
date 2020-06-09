from time import time
import numpy as np
from matplotlib import pyplot as plt


class MagicSeries:

    """ Find a Magic Series of a given size. Any series is OK.
    A magic Series is a list such that : c[i] = len([x for x in c if x == i]) for all i """

    def __init__(self, n):
        self.n = n
        self.series = []
        # -1 : unknown. 0: False. 1: True
        self.constraints = -1 * np.ones((self.n, self.n), dtype=int)
        self.solved = False

    def is_solved(self):
        if not self.solved:
            self.solved = all([
                s == sum([_ == i for _ in self.series])
                for i,s in enumerate(self.series)
            ])
        return self.solved


