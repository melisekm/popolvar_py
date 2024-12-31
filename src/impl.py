import re
from collections import defaultdict
from heapq import heappop, heappush

from tqdm import tqdm

from models import Node, Path, ResultPath, Result, DijkstraState, QueueItem, GridLocation, NodeLoc, DEFAULT_WEIGHT
from utils import is_teleport, get_directions, get_weight, DEFAULT_START_LOC


def parse_input(file_name: str) -> tuple[tuple[int, ...], list[list[str]]]:
    with open(file_name) as f:
        data = f.read()
        if " " in data.split('\n')[0]:
            lines = data.splitlines()
            sizes, grid = tuple(map(int, re.findall(r'\d+', lines[0]))), lines[1:]
        else:
            sizes, grid = None, data.split()
        return sizes, [list(line) for line in grid]


def add_teleport_edges(
        graph: defaultdict[NodeLoc, list[Node]], teleports: defaultdict[str, set[NodeLoc]]
):
    for teleport_name, nodes in teleports.items():
        for start in nodes:
            for end in nodes:
                if start is not end:
                    graph[start].append(Node(end[0], end[1], teleport_name, weight=DEFAULT_WEIGHT, is_teleport=True))


def create_graph(data: list[list[str]]) -> tuple[defaultdict[NodeLoc, list[Node]], dict[NodeLoc, int]]:
    grid_height = len(data)
    grid_width = len(data[0])
    graph = defaultdict(list)
    teleports = defaultdict(set)
    princesses_loc = {}
    for r in range(grid_height):
        for c in range(grid_width):
            parent_node = (r, c)
            node_val = data[r][c]
            if is_teleport(node_val):
                teleports[node_val].add((r, c))
            elif node_val == GridLocation.PRINCESS:
                princesses_loc[parent_node] = len(princesses_loc)
            for dr, dc in get_directions(r, c, grid_height, grid_width):
                edge_val = data[dr][dc]
                if edge_val == GridLocation.WALL:
                    continue
                weight = get_weight(edge_val)
                graph[parent_node].append(Node(dr, dc, edge_val, weight))

    add_teleport_edges(graph, teleports)
    return graph, princesses_loc


def create_state(
        row: int, column: int, value: str, princesses_loc: dict[NodeLoc, int], prev_state: DijkstraState
) -> DijkstraState:
    if value == GridLocation.PRINCESS and prev_state.dragon_dead is True:
        princesses_saved = list(prev_state.princesses_saved)
        # [False, False, False] -> [False, False, True] until [True, True, True]
        princesses_saved[princesses_loc[(row, column)]] = True
        princesses_saved = tuple(princesses_saved)
    else:
        princesses_saved = prev_state.princesses_saved

    dragon_dead = value == GridLocation.DRAGON or prev_state.dragon_dead
    generator_active = value == GridLocation.GENERATOR or prev_state.generator_active

    return DijkstraState(
        node=(row, column), generator_active=generator_active,
        princesses_saved=princesses_saved, dragon_dead=dragon_dead
    )


def dijkstra(
        data: list[list[str]],
        graph: defaultdict[tuple, list[Node]],
        start: NodeLoc,
        princesses_loc: dict[NodeLoc, int],
        tqdm_enabled=False
) -> ResultPath | None:
    start_state = create_state(
        start[0], start[1], data[start[0]][start[1]], princesses_loc,
        DijkstraState(
            node=start, generator_active=False,
            princesses_saved=tuple([False] * len(princesses_loc)), dragon_dead=False
        )
    )
    initial_weight = get_weight(data[start[0]][start[1]])
    mins = {start_state: initial_weight}
    queue = [QueueItem(initial_weight, start_state, None)]

    pbar = tqdm(disable=not tqdm_enabled)
    while queue:
        queue_item: QueueItem = heappop(queue)
        curr_state = queue_item.curr_state
        path = Path(curr_state.node, queue_item.path)
        pbar.set_description(f"Cost: {queue_item.cost}, Heap size: {len(queue)}, Explored nodes: {len(mins)}")

        if curr_state.are_all_princesses_saved:
            return ResultPath(queue_item.cost, path)

        for target in graph[curr_state.node]:
            if target.is_teleport and not curr_state.generator_active:
                continue
            target_state = create_state(target.row, target.column, target.value, princesses_loc, curr_state)
            prev = mins.get(target_state, None)
            next_cost = queue_item.cost + target.weight
            if prev is None or next_cost < prev:
                mins[target_state] = next_cost
                heappush(queue, QueueItem(next_cost, target_state, path))
    return None


def run(filename: Path | str, start: NodeLoc = DEFAULT_START_LOC) -> Result:
    sizes, data = parse_input(filename)
    graph, princesses_loc = create_graph(data)
    res = dijkstra(data, graph, start, princesses_loc)
    return Result(data, sizes, res)
