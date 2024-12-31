from impl import run
from utils import timer


@timer
def solve():
    out = run("input/original/test09.txt")
    if out.result_path:
        path = out.result_path.to_str_path()
        print(path)
        print(f"Map sizes: {len(out.raw_data)} {len(out.raw_data[0])}")
        print(f"Cost: {out.result_path.cost}")
    else:
        print("Path not found")


if __name__ == '__main__':
    raise SystemExit(solve())
