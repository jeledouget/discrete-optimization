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
    opt: bool
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
    return Output(
        value=best_sum,
        weight=final_weight,
        selection=item_selection,
        opt=True
    )


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
            return [
                Output(
                    value=cur_val,
                    weight=cur_weight,
                    selection=bool_selection,
                    opt=True
                )
            ]
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
                best_selection = [i in cur_selection for i in range(n)]
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
    res = Output(
        value=best_val,
        weight=best_weight,
        selection=best_selection,
        opt=True
    )

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
        return Output(
            value=self.best_val,
            weight=self.best_weight,
            selection=self.best_selection,
            opt=True
        )

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
            arr[j, i+1] = max(
                arr[j, i],
                arr[j-w, i] + v if w <= j else 0
            )
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
    return Output(
        value=value,
        weight=current_weight,
        selection=selection,
        opt=True
    )


""" 
Branch and bond: model and estimation
---------------------------------------- """

class Node(BaseModel):
    value: int
    room: int
    estimate: int
    level: int
    selection: List[int]


def estimation(value, room, items):
    added_value = 0
    added_weight = 0
    for item in items:
        if added_weight + item.weight <= room:
            added_value += item.value
            added_weight += item.weight
        else:
            added_value += item.value * (room - added_weight) / item.weight
            break
    return value + added_value


""" 
Branch and bond: depth first search
---------------------------------------- """


class DepthFirstSearch:

    def __init__(self, data):
        self.capacity = data.capacity
        self.items = data.items
        self.n = len(self.items)
        self.visited_nodes = 0
        self.best_node = None
        self.solved = False

    def recursion(self, node):
        self.visited_nodes += 1
        if node.level == self.n:  # all items visited
            if (node.room >= 0) and (self.best_node.value < node.value):
                self.best_node = node
        else:
            item = self.items[node.level]
            # start left: take next item
            left_room = node.room - item.weight
            if left_room >= 0:
                left_val = node.value + item.value
                left = Node(
                    value=left_val,
                    room=left_room,
                    level=node.level + 1,
                    estimate=estimation(
                        left_val,
                        left_room,
                        self.items[node.level + 1:]
                    ),
                    selection=node.selection + [item.index]
                )
                self.recursion(left)
            # then explore right: do not take next item
            right_val = node.value
            right_estimate = estimation(
                right_val,
                node.room,
                self.items[node.level+1:]
            )
            if right_estimate > self.best_node.value:
                right = Node(
                    value=node.value,
                    room=node.room,
                    level=node.level + 1,
                    estimate=right_estimate,
                    selection=node.selection + []  # list copy
                )
                self.recursion(right)

    def solve(self):
        # sort
        self.items.sort(key=lambda x: x.value / x.weight, reverse=True)
        # init
        start_node = Node(
            value=0,
            room=self.capacity,
            estimate=estimation(0, self.capacity, self.items),
            level=0,
            selection=[]
        )
        self.best_node = start_node
        # solve
        self.recursion(start_node)
        nnodes = f'{2**self.n:.1E}' if self.n < 1024 else f'2**{self.n}'
        print(
            f'Total visited nodes: '
            f'{self.visited_nodes:.1E}/' +
            nnodes +
            f'({100 * self.visited_nodes/(2**self.n):.0f}%)'
        )
        self.solved = True

    def get(self):
        if not self.solved:
            self.solve()
        node = self.best_node
        selection = [i in node.selection for i in range(self.n)]
        return Output(
            value=node.value,
            weight=self.capacity - node.room,
            selection=selection,
            opt=True
        )


def depth_first_search(data):
    return DepthFirstSearch(data).get()


""" 
Without recursion 
----------------------- """

class DepthFirstSearchNoRec:

    def __init__(self, data):
        self.capacity = data.capacity
        self.items = data.items
        self.n = len(self.items)
        self.visited_nodes = 0
        self.node_stack = []
        self.best_node = None
        self.solved = False

    def consume_stack(self):
        while len(self.node_stack) > 0:
            # pop last node
            node = self.node_stack.pop(-1)
            self.visited_nodes += 1
            if node.level == self.n:  # all items visited
                if (node.room >= 0) and (self.best_node.value < node.value):
                    self.best_node = node
            else:
                item = self.items[node.level]
                # append right first (will be explored later): do not take next item
                right_val = node.value
                right_estimate = estimation(
                    right_val,
                    node.room,
                    self.items[node.level+1:]
                )
                if right_estimate > self.best_node.value:
                    right = Node(
                        value=node.value,
                        room=node.room,
                        level=node.level + 1,
                        estimate=right_estimate,
                        selection=node.selection + []  # list copy
                    )
                    self.node_stack.append(right)
                # then append left (will be explored first): take next item
                left_room = node.room - item.weight
                if left_room >= 0:
                    left_val = node.value + item.value
                    left = Node(
                        value=left_val,
                        room=left_room,
                        level=node.level + 1,
                        estimate=estimation(
                            left_val,
                            left_room,
                            self.items[node.level + 1:]
                        ),
                        selection=node.selection + [item.index]
                    )
                    self.node_stack.append(left)

    def solve(self):
        # sort
        self.items.sort(key=lambda x: x.value / x.weight, reverse=True)
        # init
        start_node = Node(
            value=0,
            room=self.capacity,
            estimate=estimation(0, self.capacity, self.items),
            level=0,
            selection=[]
        )
        self.node_stack.append(start_node)
        self.best_node = start_node
        # solve
        self.consume_stack()
        nnodes = f'{2 ** self.n:.1E}' if self.n < 1024 else f'2**{self.n}'
        print(
            f'Total visited nodes: '
            f'{self.visited_nodes:.1E}/' +
            nnodes +
            f'({100 * self.visited_nodes / (2 ** self.n):.0f}%)'
        )
        self.solved = True

    def get(self):
        if not self.solved:
            self.solve()
        node = self.best_node
        selection = [i in node.selection for i in range(self.n)]
        return Output(
            value=node.value,
            weight=self.capacity - node.room,
            selection=selection,
            opt=True
        )


def depth_first_search_no_rec(data):
    return DepthFirstSearchNoRec(data).get()


""" 
Branch and bond: best first search
------------------------------------ """


def insort_position_rec(sorted_values, val):
    if len(sorted_values) == 0:
        return 0
    else:
        half = len(sorted_values) // 2
        if val > sorted_values[half]:
            return half + 1 + insort_position_rec(sorted_values[half+1:], val)
        else:
            return insort_position_rec(sorted_values[:half], val)


def insort_position(sorted_values, val):
    pos = 0
    cur_list = sorted_values
    while cur_list:
        half = len(cur_list) // 2
        if val > cur_list[half]:
            pos += half + 1
            cur_list = cur_list[half + 1:]
        else:
            cur_list = cur_list[:half]
    return pos


def insort_nodes(sorted_nodes, node):
    """ Insert element in list while keeping it sorted """
    sorted_values = [_.estimate for _ in sorted_nodes]
    val = node.estimate
    pos = insort_position(sorted_values, val)
    sorted_nodes.insert(pos, node)


class BestFirstSearchNoRec:

    def __init__(self, data):
        self.capacity = data.capacity
        self.items = data.items
        self.n = len(self.items)
        self.visited_nodes = 0
        self.node_bag = []
        self.best_node = None
        self.solved = False

    def consume_next_node(self):
        # nodes are sorted by estimate (checked at insertion)
        # pop node with best estimate
        node = self.node_bag.pop(-1)
        self.visited_nodes += 1
        if node.level == self.n:  # all items visited
            if (node.room >= 0) and (self.best_node.value < node.value):
                self.best_node = node
        else:
            item = self.items[node.level]
            # append right first (will be explored later): do not take next item
            right_val = node.value
            right_estimate = estimation(
                right_val,
                node.room,
                self.items[node.level + 1:]
            )
            if right_estimate > self.best_node.value:
                right = Node(
                    value=node.value,
                    room=node.room,
                    level=node.level + 1,
                    estimate=right_estimate,
                    selection=node.selection + []  # list copy
                )
                insort_nodes(self.node_bag, right)
            # then append left (will be explored first): take next item
            left_room = node.room - item.weight
            if left_room >= 0:
                left_val = node.value + item.value
                left = Node(
                    value=left_val,
                    room=left_room,
                    level=node.level + 1,
                    estimate=estimation(left_val, left_room, self.items[node.level + 1:]),
                    selection=node.selection + [item.index]
                )
                insort_nodes(self.node_bag, left)

    def consume_bag(self):
        while len(self.node_bag) > 0:
            self.consume_next_node()

    def solve(self):
        # sort
        self.items.sort(key=lambda x: x.value / x.weight, reverse=True)
        # init
        start_node = Node(
            value=0,
            room=self.capacity,
            estimate=estimation(0, self.capacity, self.items),
            level=0,
            selection=[]
        )
        self.node_bag.append(start_node)
        self.best_node = start_node
        # solve
        self.consume_bag()
        nnodes = f'{2 ** self.n:.1E}' if self.n < 1024 else f'2**{self.n}'
        print(
            f'Total visited nodes: '
            f'{self.visited_nodes:.1E}/' +
            nnodes +
            f'({100 * self.visited_nodes / (2 ** self.n):.0f}%)'
        )
        self.solved = True

    def get(self):
        if not self.solved:
            self.solve()
        node = self.best_node
        selection = [i in node.selection for i in range(self.n)]
        return Output(
            value=node.value,
            weight=self.capacity - node.room,
            selection=selection,
            opt=True
        )


def best_first_search_no_rec(data):
    return BestFirstSearchNoRec(data).get()


""" 
Least discrepancy search
------------------------------------ """


class LeastDiscrepancySearch:

    def __init__(self, data, greedy=True):
        self.capacity = data.capacity
        self.items = data.items
        self.n = len(self.items)
        # greedy: tries first all-but-k objects. non-greedy: tries first k-maximum objects
        self.greedy = greedy
        self.visited_nodes = 0
        self.best_node = None
        self.solved = False

    def consume_bag(self, start_node):
        waves = range(self.n, -1, -1) if self.greedy else range(self.n + 1)
        for max_items in waves:
            node_bag = [start_node]
            while len(node_bag) > 0:
                # nodes are previously sorted by estimate
                # pop node with best estimate
                node = node_bag.pop(-1)
                self.visited_nodes += 1
                if node.level == self.n:  # all items visited
                    if (node.room >= 0) and (self.best_node.value < node.value):
                        self.best_node = node
                else:
                    item = self.items[node.level]
                    n_current_items = len(node.selection)
                    n_items_room = max_items - n_current_items
                    n_down_items = self.n - node.level - 1
                    if n_down_items >= n_items_room:
                        # append right first (will be explored later): do not take next item
                        right_val = node.value
                        right_estimate = estimation(
                            right_val,
                            node.room,
                            self.items[node.level+1:]
                        )
                        if right_estimate > self.best_node.value:
                            right = Node(
                                value=node.value,
                                room=node.room,
                                level=node.level + 1,
                                estimate=right_estimate,
                                selection=node.selection + []  # list copy
                            )
                            insort_nodes(node_bag, right)
                    if n_current_items < max_items:
                        # then append left (will be explored first): take next item
                        left_room = node.room - item.weight
                        if left_room >= 0:
                            left_val = node.value + item.value
                            left = Node(
                                value=left_val,
                                room=left_room,
                                level=node.level + 1,
                                estimate=estimation(left_val, left_room, self.items[node.level + 1:]),
                                selection=node.selection + [item.index]
                            )
                            insort_nodes(node_bag, left)

    def solve(self):
        # sort
        self.items.sort(key=lambda x: x.value / x.weight, reverse=True)
        # init
        start_node = Node(
            value=0,
            room=self.capacity,
            estimate=estimation(0, self.capacity, self.items),
            level=0,
            selection=[]
        )
        self.best_node = start_node
        # solve
        self.consume_bag(start_node)
        nnodes = f'{2 ** self.n:.1E}' if self.n < 1024 else f'2**{self.n}'
        print(
            f'Total visited nodes: '
            f'{self.visited_nodes:.1E}/' +
            nnodes +
            f'({100 * self.visited_nodes / (2 ** self.n):.0f}%)'
        )
        self.solved = True

    def get(self):
        if not self.solved:
            self.solve()
        node = self.best_node
        selection = [i in node.selection for i in range(self.n)]
        return Output(
            value=node.value,
            weight=self.capacity - node.room,
            selection=selection,
            opt=True
        )


def least_discrepancy_search(data):
    return LeastDiscrepancySearch(data, greedy=True).get()


def least_discrepancy_search_non_greedy(data):
    return LeastDiscrepancySearch(data, greedy=False).get()


""" 
Factories / submissions / etc.
------------------------------------ """


solvers = {
    'brute_force': brute_force,
    'brute_force_rec': brute_force_rec,
    'brute_force_rec_light': brute_force_rec_light,
    'brute_force_rec_light_no_glob': brute_force_rec_light_no_glob,
    'dynamic_programming': dynamic_programming,
    'depth_first_search': depth_first_search,
    'depth_first_search_no_rec': depth_first_search_no_rec,
    'best_first_search_no_rec': best_first_search_no_rec,
    'least_discrepancy_search': least_discrepancy_search,
    'least_discrepancy_search_non_greedy': least_discrepancy_search_non_greedy
}


def solve_it(input_data, solver=None, _timeout=None, **kwargs):

    def _solve_it(input_data, solver):

        solver = solver or 'best_first_search_no_rec'  # hidden default

        data = parse_data(input_data)

        print(f"\n### Using Solver {solver} ###")
        res = solvers[solver](data, **kwargs)

        # prepare the solution in the specified output format
        output_data = '\n'.join([
            str(res.value) + ' ' + str(res.opt),
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
