#!/usr/bin/python
# -*- coding: utf-8 -*-


from collections import namedtuple
from pydantic import BaseModel
from typing import List

Item = namedtuple("Item", ['index', 'value', 'weight'])


class Data(BaseModel):
    capacity: int
    items: List[Item]


class Output(BaseModel):
    value: int
    weight: int
    selection: List[bool]


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
    return Output(value=best_sum, weight=weight, selection=item_selection)


def brute_force_rec(data):

    capacity = data.capacity
    items = data.items
    n = len(items)

    def _recursion(_items, cur_val, cur_weight, cur_selection):
        if len(_items) == 0:
            bool_selection = [True if i in cur_selection else False for i in range(n)]
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



def dynamic_programming(data):
    pass


def depth_first_search(data):
    pass


def best_first_search(data):
    pass


def least_discrepancy_search(data):
    pass


solvers = {
    'brute_force': brute_force,
    'brute_force_rec': brute_force_rec,
    'dynamic_programming': dynamic_programming,
    'depth_first_search': depth_first_search,
    'best_first_search': best_first_search,
    'least_discrepancy_search': least_discrepancy_search
}


def solve_it(input_data, solver=None):

    solver = solver or 'brute_force'  # hidden default

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


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        solver = sys.argv[2].strip() if len(sys.argv) > 2 else None
        print(solve_it(input_data, solver=solver))
        print()
    else:
        print('This test requires an input file.  '
              'Please select one from the data directory. '
              '(i.e. python solver.py ./data/ks_4_0)')
