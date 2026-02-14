class MaxNCoordCache:
    class Node:
        def __init__(self, coordination_index: float, activation_window: tuple):
            self.coordination_index = coordination_index
            self.activation_window = activation_window
            self.next = None
            self.prev = None
        
    # top n coordination indexes and associated activation windows
    def __init__(self, ):

        