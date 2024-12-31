import re
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import DefaultDict

from tqdm import tqdm


@dataclass
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

    def to_str_path(self) -> str:
        flatten_path = self.path.flatten()
        return "\n".join(f"{x[1]} {x[0]}" for x in flatten_path)


@dataclass
class Result:
    raw_data: list[list[str]]
    sizes: tuple[int, int, int] | None
    res: ResultPath | None


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
        if " " in data.split('\n')[0]:
            lines = data.splitlines()
            sizes, grid = tuple(map(int, re.findall(r'\d+', lines[0]))), lines[1:]
        else:
            sizes, grid = None, data.split()
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


def create_state(row, column, value, princesses_loc, prev_state):
    if value == 'P' and prev_state.dragon is True:
        princesses = list(prev_state.princesses)
        princesses[princesses_loc[(row, column)]] = True
        princesses = tuple(princesses)
    else:
        princesses = prev_state.princesses

    if value == 'D' and prev_state.dragon is False:
        dragon = True
    else:
        dragon = prev_state.dragon
    if value == 'G' and prev_state.generator is False:
        generator = True
    else:
        generator = prev_state.generator

    return DijkstraState(node=(row, column), generator=generator, princesses=princesses, dragon=dragon)


def dijkstra(
        data: list[list[str]],
        graph: DefaultDict[tuple, list[Node]],
        start: tuple,
        princesses_loc: dict[tuple, int],
        tqdm_enabled=False
) -> ResultPath | None:
    start_state = create_state(start[0], start[0], data[start[0]][start[1]], princesses_loc, DijkstraState(
        node=start, generator=False, princesses=tuple([False] * len(princesses_loc)), dragon=False
    ))
    initial_cost = 2 if data[start[0]][start[1]] == 'H' else 1
    queue = [QueueItem(initial_cost, start_state, None)]
    mins = {start_state: queue[0].cost}
    pbar = tqdm(disable=not tqdm_enabled)

    while queue:
        queue_item: QueueItem = heappop(queue)
        curr_state = queue_item.curr_state
        path = Path(curr_state.node, queue_item.path)
        pbar.set_description(f"Cost: {queue_item.cost}, HeapSize: {len(queue)}, Explored nodes: {len(mins)}")
        if all(curr_state.princesses):
            return ResultPath(queue_item.cost, path)

        for target in graph[curr_state.node]:
            if target.is_teleport and curr_state.generator is False:
                continue
            target_state = create_state(target.row, target.column, target.value, princesses_loc, curr_state)
            prev = mins.get(target_state, None)
            next_cost = queue_item.cost + target.weight
            if prev is None or next_cost < prev:
                mins[target_state] = next_cost
                heappush(queue, QueueItem(next_cost, target_state, path))
    return None


def run(filename, start=(0, 0)):
    sizes, data = load_input(filename)
    graph, princesses_loc = create_graph(data)
    res = dijkstra(data, graph, start, princesses_loc)
    return Result(data, sizes, res)
