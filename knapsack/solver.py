#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from time import time
from collections import namedtuple
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from timeout_decorator import timeout, TimeoutError


"""
Data structures
------------------------ """

Item = namedtuple("Item", ['index', 'value', 'weight'])

class Data(BaseModel):
    capacity: int
    items: List[Item]

test_data = Data(
    capacity=11,
    items=[
        Item(index=0, value=8, weight=4),
        Item(index=1, value=10, weight=5),
        Item(index=2, value=15, weight=8),
        Item(index=3, value=4, weight=3)
    ]
)

class Output(BaseModel):
    value: int
    weight: int
    selection: List[bool]


"""
Basic brute force (for loop stores all results)
------------------------------------------------ """

def brute_force(data):
    capacity = data.capacity
    items = data.items
    n = len(items)
    best_sum = 0
    item_selection = [False] * n
    for i in range(1 << n):
        selection = [bool(int(_)) for _ in list(format(i, f'0{n}b'))]
        indices = [item.index for s, item in zip(selection, items) if s]
        weight = sum([items[j].weight for j in indices])
        if weight <= capacity:
            val = sum([items[j].value for j in indices])
            if val > best_sum:
                best_sum = val
                item_selection = selection
                final_weight = weight
    return Output(value=best_sum, weight=final_weight, selection=item_selection)


"""
Brute force with recursion (follow the tree)
--------------------------------------------------- """

def brute_force_rec(data):

    capacity = data.capacity
    items = data.items
    n = len(items)

    def _recursion(_items, cur_val, cur_weight, cur_selection):
        if len(_items) == 0:
            bool_selection = [i in cur_selection for i in range(n)]
            return [Output(value=cur_val, weight=cur_weight, selection=bool_selection)]
        else:
            item = _items[0]
            return [
                *_recursion(
                    _items[1:],
                    cur_val + item.value,
                    cur_weight + item.weight,
                    cur_selection + [item.index]
                ),
                *_recursion(
                    _items[1:],
                    cur_val,
                    cur_weight,
                    cur_selection
                )
            ]

    res = _recursion(items, 0, 0, [])
    res = [v for v in res if v.weight <= capacity]
    res = max(res, key=lambda x: x.value)

    return res


"""
Brute force with recursion (follow the tree - no storage)
------------------------------------------------------------ """

best_val = 0
best_weight = 0
best_selection = []

def brute_force_rec_light(data):
    """ Without storing all intermediate results """
    global best_val, best_weight, best_selection
    capacity = data.capacity
    items = data.items
    n = len(items)

    def _recursion(_items, cur_val, cur_weight, cur_selection):
        global best_val, best_weight, best_selection
        if len(_items) == 0:
            if (cur_weight <= capacity) and (cur_val > best_val):
                best_val = cur_val
                best_weight = cur_weight
                best_selection = [True if i in cur_selection else False for i in range(n)]
        else:
            item = _items[0]
            _recursion(
                _items[1:],
                cur_val + item.value,
                cur_weight + item.weight,
                cur_selection + [item.index]
            )
            _recursion(
                _items[1:],
                cur_val,
                cur_weight,
                cur_selection
            )

    _recursion(items, 0, 0, [])
    res = Output(value=best_val, weight=best_weight, selection=best_selection)

    best_val = 0
    best_weight = 0
    best_selection = []

    return res


"""
Brute force with recursion (follow the tree - no storage - no globals)
---------------------------------------------------------------------- """

class BruteForceRecLightNoGlob:

    def __init__(self, data):
        self.best_val = 0
        self.best_weight = 0
        self.best_selection = []
        self.capacity = data.capacity
        self.items = data.items
        self.n = len(self.items)
        self.solved = False

    def _recursion(self, _items, cur_val, cur_weight, cur_selection):
        if len(_items) == 0:
            if (cur_weight <= self.capacity) and (cur_val > self.best_val):
                self.best_val = cur_val
                self.best_weight = cur_weight
                self.best_selection = [i in cur_selection for i in range(self.n)]
        else:
            item = _items[0]
            self._recursion(
                _items[1:],
                cur_val + item.value,
                cur_weight + item.weight,
                cur_selection + [item.index]
            )
            self._recursion(
                _items[1:],
                cur_val,
                cur_weight,
                cur_selection
            )

    def solve(self):
        self._recursion(self.items, 0, 0, [])
        self.solved = True

    def get(self):
        if not self.solved:
            self.solve()
        return Output(value=self.best_val, weight=self.best_weight, selection=self.best_selection)

def brute_force_rec_light_no_glob(data):
    return BruteForceRecLightNoGlob(data).get()


""" 
Dynamic Programming
----------------------------- """


def dynamic_programming(data):
    # params
    capacity = data.capacity
    items = data.items
    n = len(items)
    # fill array
    arr = np.zeros((capacity+1, n+1), dtype=int)
    for i in range(n):
        for j in range(capacity+1):
            w = items[i].weight
            v = items[i].value
            arr[j, i+1] = max(arr[j, i], arr[j-w, i] + v if w <= j else 0)
    # final optimal value
    value = arr[capacity,n]
    # trace back selected items and used capacity
    selection = [False] * n
    current_capacity = capacity
    current_weight = 0
    for i in range(n, 0, -1):
        item = items[i-1]
        if arr[current_capacity, i] > arr[current_capacity, i-1]:
            selection[i-1] = True
            current_weight += item.weight
            current_capacity -= item.weight
    # return
    return Output(value=value, weight=current_weight, selection=selection)


""" 
Branch and bond: depth first search
------------------------------------ """

class Node(BaseModel):
    value: int
    room: int
    estimate: int
    index: int
    selection: List[int]


class DepthFirstSearch:

    def __init__(self, data):
        self.capacity = data.capacity
        self.items = data.items
        self.n = len(self.items)
        self.visited_nodes = 0
        self.best_node = None
        self.solved = False

    @staticmethod
    def estimation(initial, items):
        return initial + sum([i.value for i in items])

    def recursion(self, node):
        self.visited_nodes += 1
        if node.index == self.n:  # all items visited
            if (node.room >= 0) and (self.best_node.value < node.value):
                self.best_node = node
        else:
            # start left: take next item
            left_room = node.room - self.items[node.index].weight
            if left_room >= 0:
                left_val = node.value + self.items[node.index].value
                left = Node(
                    value=left_val,
                    room=left_room,
                    index=node.index + 1,
                    estimate=self.estimation(left_val, self.items[node.index + 1:]),
                    selection=node.selection + [node.index]
                )
                self.recursion(left)
            # then explore right: do not take next item
            right_val = node.value
            right_estimate = self.estimation(right_val, self.items[node.index+1:])
            if right_estimate > self.best_node.value:
                right = Node(
                    value=node.value,
                    room=node.room,
                    index=node.index + 1,
                    estimate=right_estimate,
                    selection=node.selection + []  # list copy
                )
                self.recursion(right)

    def solve(self):
        start_node = Node(value=0, room=self.capacity, estimate=self.estimation(0, self.items), index=0, selection=[])
        self.best_node = start_node
        self.recursion(start_node)
        print(f'Visited Nodes in Total: {self.visited_nodes}')
        self.solved = True

    def get(self):
        if not self.solved:
            self.solve()
        node = self.best_node
        selection = [i in node.selection for i in range(self.n)]
        return Output(value=node.value, weight=self.capacity - node.room, selection=selection)


def depth_first_search(data):
    return DepthFirstSearch(data).get()


""" ------------ """


def best_first_search(data):
    pass


def least_discrepancy_search(data):
    pass


solvers = {
    'brute_force': brute_force,
    'brute_force_rec': brute_force_rec,
    'brute_force_rec_light': brute_force_rec_light,
    'brute_force_rec_light_no_glob': brute_force_rec_light_no_glob,
    'dynamic_programming': dynamic_programming,
    'depth_first_search': depth_first_search,
    'best_first_search': best_first_search,
    'least_discrepancy_search': least_discrepancy_search
}

def solve_it(input_data, solver=None, _timeout=300):

    @timeout(_timeout)
    def solve_it_timeout(input_data, solver):

        solver = solver or 'depth_first_search'  # hidden default

        # parse the input
        lines = input_data.split('\n')
        first_line = lines[0].split()
        item_count = int(first_line[0])
        capacity = int(first_line[1])

        items = []

        for i in range(1, item_count+1):
            line = lines[i]
            parts = line.split()
            items.append(Item(i-1, int(parts[0]), int(parts[1])))

        data = Data(items=items, capacity=capacity)

        print(f"\n### Using Solver {solver} ###")
        res = solvers[solver](data)

        # prepare the solution in the specified output format
        output_data = '\n'.join([
            str(res.value) + ' ' + str(0),
            ' '.join([str(int(_)) for _ in res.selection])
        ])
        return output_data

    return solve_it_timeout(input_data, solver)


def solve_file(file_location, solver=None, _timeout=30):
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    return solve_it(input_data, solver=solver, _timeout=_timeout)


def plot_response_time(_solvers=(
            'brute_force',
            'brute_force_rec',
            'brute_force_rec_light',
            'brute_force_rec_light_no_glob',
            'dynamic_programming',
            'depth_first_search'
        ),
        _n=(4, 8, 12, 16, 19, 23, 25, 30),
        _timeout=30):

    df = pd.DataFrame(index=_n, columns=_solvers)
    for s in _solvers:
        for n in _n:
            t = time()
            try:
                solve_file(f'./data/ks_{n}_0', s, _timeout=_timeout)
                df.loc[n, s] = time() - t
            except TimeoutError:
                df.loc[n, s] = np.nan
    df.plot(marker='s', markerfacecolor="None", logy=True, xticks=_n,
            title='Time elapsed (seconds) against number of items')
    return df


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
