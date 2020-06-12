#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from time import time
import pandas as pd
import numpy as np
from timeout_decorator import timeout, TimeoutError
from discopt.knapsack import solvers, Item, Data


def solve_it(input_data, solver=None, _timeout=None, **kwargs):

    def _solve_it(input_data, solver):

        solver = solver or 'depth_first_search_no_rec'  # hidden default

        data = parse_data(input_data)

        print(f"\n### Using Solver {solver} ###")
        res = solvers[solver](data, **kwargs)

        # prepare the solution in the specified output format
        output_data = '\n'.join([
            str(res.value) + ' ' + str(int(res.opt)),
            ' '.join([str(int(_)) for _ in res.selection])
        ])
        return output_data

    if _timeout:
        @timeout(_timeout)
        def really_solve_it(*args, **kwargs):
            return _solve_it(*args, **kwargs)
    else:
        really_solve_it = _solve_it

    return really_solve_it(input_data, solver)


def solve_file(file_location, solver=None, _timeout=30):
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    return solve_it(input_data, solver=solver, _timeout=_timeout)


def parse_data(input_data):
    # parse the input
    lines = input_data.split('\n')
    first_line = lines[0].split()
    item_count = int(first_line[0])
    capacity = int(first_line[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    return Data(items=items, capacity=capacity)


def parse_file(input_file):
    with open(input_file, 'r') as input_data_file:
        input_data = input_data_file.read()
    return parse_data(input_data)


def plot_response_time(_solvers=(
            'dynamic_programming',
            'depth_first_search_no_rec',
            'best_first_search_no_rec',
            'least_discrepancy_search',
            'least_discrepancy_search_non_greedy'
        ),
        _n=(4, 8, 12, 16, 19, 23, 25, 30, 50, 100, 200, 300, 400, 500, 1000, 10000),
        _timeout=300):

    times = pd.DataFrame(index=_n, columns=_solvers)
    results = pd.DataFrame(index=_n, columns=_solvers)
    for s in _solvers:
        for n in _n:
            t = time()
            try:
                results.loc[n, s] = solve_file(f'./data/ks_{n}_0', s, _timeout=_timeout)
                times.loc[n, s] = time() - t
            except TimeoutError:
                times.loc[n, s] = np.nan
    times.plot(
        marker='s',
        markerfacecolor="None",
        logy=True,
        xticks=_n,
        title='Time elapsed (seconds) against number of items'
    )
    return times, results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        solver = sys.argv[2].strip() if len(sys.argv) > 2 else None
        print(solve_file(file_location, solver=None))
        print()
    else:
        print('This test requires an input file.  '
              'Please select one from the data directory. '
              '(i.e. python solver.py ./data/ks_4_0)')
