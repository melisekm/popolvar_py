import re
import unittest

from impl import run
from utils import timer, calc_time


@timer
def solve():
    out = run("input/original/test5.txt")
    path = out.res.to_str_path()
    print(path)
    print(calc_time(out.raw_data, [(x[1], x[0]) for x in out.res.path.flatten()], 999))
    print(f"Cost: {out.res.cost}")


class PopolvarTest(unittest.TestCase):
    def test(self):
        for idx in range(1, 18):
            out = run(f"input/no_teleports/scenar{idx}.txt")
            time_to_kill_dragon = out.sizes[-1]
            if not out.res:
                print("Path not found")
                continue

            with open(f"input/no_teleports/expected/scenar{idx}CHECK.txt") as f:
                expected_path = f.read().splitlines()
                expected_path = [tuple(map(int, re.findall(r'\d+', node))) for node in expected_path]
            expected_cost = calc_time(out.raw_data, expected_path, time_to_kill_dragon)
            print(f"Test: {idx} - Expected: {expected_cost}. Real: {out.res.cost}")
            self.assertEqual(expected_cost, out.res.cost, f"{idx}\n{out.res.to_str_path()}")


if __name__ == '__main__':
    unittest.main()
    # raise SystemExit(solve())
