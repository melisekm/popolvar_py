from dataclasses import dataclass
from enum import StrEnum

NodeLoc = tuple[int, int]

DEFAULT_WEIGHT = 1
GRASS_WEIGHT = 2


class GridLocation(StrEnum):
    PRINCESS = 'P'
    WALL = 'N'
    GRASS = 'H'
    DRAGON = 'D'
    GENERATOR = 'G'


@dataclass
class Node:
    row: int
    column: int
    value: str
    weight: int
    is_teleport: bool = False


@dataclass
class Path:
    node: NodeLoc
    parent: "Path | None"

    def reconstruct(self) -> list[NodeLoc]:
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
        flatten_path = self.path.reconstruct()
        return "\n".join(f"{x[1]} {x[0]}" for x in flatten_path)


@dataclass
class Result:
    raw_data: list[list[str]]
    sizes: tuple[int, ...] | None
    result_path: ResultPath | None


@dataclass(eq=True, frozen=True)
class DijkstraState:
    node: NodeLoc
    generator_active: bool
    princesses_saved: tuple[bool, ...]
    dragon_dead: bool

    @property
    def are_all_princesses_saved(self) -> bool:
        return all(self.princesses_saved)


@dataclass
class QueueItem:
    cost: int
    curr_state: DijkstraState
    path: Path | None

    def __lt__(self, other):
        return self.cost < other.cost
