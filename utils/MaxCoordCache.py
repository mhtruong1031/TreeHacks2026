import heapq
import math
from typing import Any


class MaxNCoordCache:
    """Max-heap by coordination_index with recency tiebreaker within epsilon.

    When two nodes have coordination_index within `epsilon`, the more recently
    added one ranks higher.
    """

    def __init__(self, epsilon: float = 0.1):
        # Heap entries: (key, (coordination_index, activation_window))
        # key = (-bucket, -insertion_id) so higher coord wins, then more recent wins
        self._heap: list[tuple[tuple[float, int], tuple[float, Any]]] = []
        self._epsilon = epsilon
        self._insertion_id = 0
        self.predicted_ideal = None

    def add_node(self, coordination_index: float, activation_window: tuple) -> None:
        self._insertion_id += 1
        # Bucket by epsilon so coords within epsilon compare by recency
        bucket = math.floor(coordination_index / self._epsilon) * self._epsilon
        key = (-bucket, -self._insertion_id)
        heapq.heappush(self._heap, (key, (coordination_index, activation_window)))

    def get_top_n_nodes(self, n: int) -> list[tuple[float, Any]]:
        top = heapq.nsmallest(n, self._heap)
        return [payload for _key, payload in top]
