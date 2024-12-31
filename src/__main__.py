import math
import re
import timeit
import unittest
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import DefaultDict

from tqdm import tqdm


@dataclass(eq=True, frozen=True)
class Node:
    row: int
    column: int
    value: str
    weight: int
    is_teleport: bool = False


@dataclass
class Path:
    node: tuple
    parent: "Path | None"

    def flatten(self):
        res = []
        p = self
        while p:
            res.append(p.node)
            p = p.parent
        return list(reversed(res))

    def __str__(self):
        return f"{self.node} -> {self.parent.node if self.parent else ''}..."

    def __repr__(self):
        return str(self)


@dataclass
class ResultPath:
    cost: int
    path: Path

    def to_path(self):
        flatten_path = self.path.flatten()
        return "\n".join(f"{x[1]} {x[0]}" for x in flatten_path)


@dataclass(eq=True, frozen=True)
class DijkstraState:
    node: tuple
    generator: bool
    princesses: tuple[bool, ...]
    dragon: bool


@dataclass
class QueueItem:
    cost: int
    curr_state: DijkstraState
    path: Path | None

    def __lt__(self, other):
        return self.cost < other.cost


def load_input(file_name):
    with open(file_name) as f:
        data = f.read()
        if data[0].isdigit():
            lines = data.splitlines()
            sizes, grid = list(map(int, re.findall(r'\d+', lines[0]))), lines[1:]
        else:
            sizes, grid = None, data
        return sizes, [list(line) for line in grid]


def add_node(data, graph, parent_node, curr_node):
    curr_r, curr_c = curr_node
    value = data[curr_r][curr_c]
    if value == 'N':
        return

    weight = 1
    if value == 'H':
        weight = 2

    graph[parent_node].append(Node(curr_r, curr_c, value, weight))


def add_teleport_edges(graph: DefaultDict[tuple, list[Node]], teleports: DefaultDict[str, set[tuple[int, int]]]):
    for teleport_name, nodes in teleports.items():
        for start in nodes:
            for end in nodes:
                if start is end:
                    continue
                graph[start].append(Node(end[0], end[1], teleport_name, weight=1, is_teleport=True))


def create_graph(data) -> tuple[DefaultDict[tuple, list[Node]], dict[tuple, int]]:
    R = len(data)
    C = len(data[0])
    graph = defaultdict(list)
    teleports = defaultdict(set)
    princesses_loc = {}
    for r in range(R):
        for c in range(C):
            parent_node = (r, c)
            val = data[r][c]
            if val.isdigit():
                teleports[val].add((r, c))
            elif val == 'P':
                princesses_loc[parent_node] = len(princesses_loc)

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                rr, cc = r + dy, c + dx
                if 0 <= cc < C and 0 <= rr < R:
                    add_node(data, graph, parent_node, (rr, cc))

    add_teleport_edges(graph, teleports)
    return graph, princesses_loc


def dijkstra(
        graph: DefaultDict[tuple, list[Node]],
        start: tuple,
        princesses_loc: dict[tuple, int]
) -> ResultPath | None:
    start_state = DijkstraState(
        node=start, generator=False, princesses=tuple([False] * len(princesses_loc)), dragon=False
    )
    queue = [QueueItem(0, start_state, None)]
    mins = {start_state: queue[0].cost}
    pbar = tqdm()

    while queue:
        queue_item: QueueItem = heappop(queue)
        curr_state = queue_item.curr_state
        path = Path(curr_state.node, queue_item.path)
        pbar.set_description(f"Cost: {queue_item.cost}, HeapSize: {len(queue)}, Explored nodes: {len(mins)}")
        if curr_state.dragon and all(curr_state.princesses):
            return ResultPath(queue_item.cost, path)

        for target in graph[curr_state.node]:
            if target.is_teleport and curr_state.generator is False:
                continue

            if target.value == 'P' and curr_state.dragon is True:
                princesses = list(curr_state.princesses)
                princess_loc = (target.row, target.column)
                princesses[princesses_loc[princess_loc]] = True
                princesses = tuple(princesses)
            else:
                princesses = curr_state.princesses

            if target.value == 'D' and curr_state.dragon is False:
                dragon = True
            else:
                dragon = curr_state.dragon
            if target.value == 'G' and curr_state.generator is False:
                generator = True
            else:
                generator = curr_state.generator

            target_state = DijkstraState(
                node=(target.row, target.column), generator=generator, princesses=princesses, dragon=dragon
            )

            prev = mins.get(target_state, None)
            next_cost = queue_item.cost + target.weight
            if prev is None or next_cost < prev:
                mins[target_state] = next_cost
                heappush(queue, QueueItem(next_cost, target_state, path))
    return None


# https://github.com/ipython/ipython/blob/main/IPython/core/magics/execution.py
def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""

    if timespan >= 60.0:
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    units = [u"s", u"ms", u'\xb5s', "ns"]
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return "%.*g %s" % (precision, timespan * scaling[order], units[order])


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed_time = timeit.default_timer() - start_time
        print(f"Time spent {_format_time(elapsed_time)}")
        return result

    return wrapper


@timer
def solve():
    sizes, data = load_input("input/test6.txt")
    print(len(data), len(data[0]))
    graph, princesses_loc = create_graph(data)
    start = (0, 0)
    res = dijkstra(graph, start, princesses_loc)
    path = res.to_path()
    print(path)
    print(f"Cost: {res.cost}")

