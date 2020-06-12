#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from time import time
from pydantic import BaseModel
from typing import List, Tuple
import pandas as pd
import numpy as np
from timeout_decorator import timeout, TimeoutError


"""
Data structures
------------------------ """

class Data(BaseModel):
    node_count: int
    edges: Tuple[int, int]


class Output(BaseModel):
    obj: int
    opt: bool
    colors: List[int]


"""
Implementations
--------------------- """


"""
Factories / submissions / etc.
------------------------------------ """


solvers = {}


def solve_it(input_data, solver=None, _timeout=None, **kwargs):

    def _solve_it(input_data, solver):

        solver = solver or 'best_first_search_no_rec'  # hidden default

        data = parse_data(input_data)

        print(f"\n### Using Solver {solver} ###")
        res = solvers[solver](data, **kwargs)

        # prepare the solution in the specified output format
        output_data = '\n'.join([
            str(res.obj) + ' ' + str(int(res.opt)),
            ' '.join([str(int(_)) for _ in res.colors])
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
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append(int(parts[0]), int(parts[1]))

    return Data(edges=edges, node_count=node_count)


def parse_file(input_file):
    with open(input_file, 'r') as input_data_file:
        input_data = input_data_file.read()
    return parse_data(input_data)

"""
Benchmark Results
----------------------- """

def benchmark(
        _solvers,
        n=((4, 1), (20, 1), (20, 3), (50, 3)),
        _timeout=300):
    times = pd.DataFrame(index=n, columns=_solvers)
    results = pd.DataFrame(index=n, columns=_solvers)
    for s in _solvers:
        for i,d in n:
            t = time()
            try:
                results.loc[n, s] = solve_file(f'./data/ks_{i}_{d}', s, _timeout=_timeout)
                times.loc[n, s] = time() - t
            except TimeoutError:
                times.loc[n, s] = np.nan
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
              '(i.e. python solver.py ./data/gc_4_1)')
