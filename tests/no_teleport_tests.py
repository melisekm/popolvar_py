import re
import unittest
from pathlib import Path
import sys

from models import NodeLoc

sys.path.append(str(Path(__file__).parent.parent / "src"))

from impl import run
from utils import calc_time


class NoTeleportTest(unittest.TestCase):
    def test(self):
        tests_path = Path(__file__).parent.parent / "input"
        for idx in range(1, 18):
            idx = str(idx).zfill(2)
            out = run(tests_path / 'no_teleports' / f'test{idx}.txt')
            time_to_kill_dragon = out.sizes[-1]
            if not out.result_path:
                print("Path not found")
                continue

            with open(f"{tests_path / 'no_teleports' / 'expected' / f'test{idx}_expected.txt'}") as f:
                expected_path = f.read().splitlines()
                expected_path = [tuple(map(int, re.findall(r'\d+', node))) for node in expected_path]
            expected_cost = calc_time(out.raw_data, expected_path, time_to_kill_dragon)
            print(f"Test: {idx} - Expected: {expected_cost}. Real: {out.result_path.cost}")
            self.assertEqual(expected_cost, out.result_path.cost, f"{idx}\n{out.result_path.to_str_path()}")


if __name__ == '__main__':
    unittest.main()
