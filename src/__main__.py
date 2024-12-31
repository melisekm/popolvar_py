import re
import unittest

from impl import run
from utils import timer, calc_time


@timer
def solve():
    out = run("input/test6.txt")
    path = out.res.to_path()
    print(path)
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
            flatten_path = [(x[1], x[0]) for x in out.res.path.flatten()]
            expected_cost = calc_time(out.raw_data, expected_path, time_to_kill_dragon)
            real_cost = calc_time(out.raw_data, flatten_path, time_to_kill_dragon)
            print(f"Test: {idx} - Expected: {expected_cost}. Real: {real_cost}")
            self.assertEqual(expected_cost, real_cost, f"{idx}\n{out.res.to_path()}")


if __name__ == '__main__':
    unittest.main()
    # raise SystemExit(solve())
