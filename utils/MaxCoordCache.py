import heapq
import math
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class Node:
    coordination_index: float
    activation_window: tuple[int, int]
    similarity_score: float


class MaxNCoordCache:
    def __init__(self, epsilon: float = 0.1):
        # Heap entries: (key, Node)
        # key = (-bucket, -insertion_id) so higher coord wins, then more recent wins
        self._heap: list[tuple[tuple[float, int], Node]] = []
        self._epsilon = epsilon
        self._insertion_id = 0
        self.predicted_ideal: Node = None
        self.predicted_ideal_data: np.ndarray = None

    def __len__(self) -> int:
        return len(self._heap)

    def add_node(self, coordination_index: float, activation_window: Any, is_predicted_ideal: bool = False) -> None:
        self._insertion_id += 1
        # Bucket by epsilon so coords within epsilon compare by recency
        bucket = math.floor(coordination_index / self._epsilon) * self._epsilon
        key = (-bucket, -self._insertion_id)
        node = Node(coordination_index=coordination_index, activation_window=activation_window, similarity_score=0)
        heapq.heappush(self._heap, (key, node))

    def get_top_n_nodes(self, n: int) -> list[Node]:
        top = heapq.nsmallest(n, self._heap)
        return [node for _key, node in top]
