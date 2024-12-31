import math
import timeit


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


def calc_time(data, path, max_t):
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
