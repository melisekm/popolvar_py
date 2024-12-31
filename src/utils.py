import math
import timeit
from typing import Callable, Iterator

from models import NodeLoc, GridLocation, GRASS_WEIGHT, DEFAULT_WEIGHT

ALLOWED_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DEFAULT_START_LOC = (0, 0)


def get_directions(r: int, c: int, max_row: int, max_column: int) -> Iterator[tuple[NodeLoc]]:
    for dr, dc in ALLOWED_DIRECTIONS:
        rr, cc = r + dr, c + dc
        if 0 <= cc < max_column and 0 <= rr < max_row:
            yield rr, cc


def get_weight(val: str) -> int:
    return GRASS_WEIGHT if val == GridLocation.GRASS else DEFAULT_WEIGHT


def is_teleport(val: str) -> bool:
    # 0 - 9 are teleports
    return val.isdigit()


# https://github.com/ipython/ipython/blob/main/IPython/core/magics/execution.py
def _format_time(timespan: float, precision=3) -> str:
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


def timer(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed_time = timeit.default_timer() - start_time
        print(f"Time elapsed: {_format_time(elapsed_time)}")
        return result

    return wrapper


def calc_time(data: list[list[str]], path: list[tuple[int, ...]], max_t: int) -> int:
    dragon_dead = False
    res = 0
    for node in path:
        y, x = node[1], node[0]
        if data[y][x] == 'H':
            res += 2
        else:
            res += 1
        if data[y][x] == 'D' and res <= max_t:
            dragon_dead = True
        if data[y][x] == 'D' and res > max_t and dragon_dead is False:
            raise Exception("Did not slay dragon in time!")
        if data[y][x] == 'N':
            raise Exception("Moved through wall!")
    return res
