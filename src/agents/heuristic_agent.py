from __future__ import annotations

import numpy as np


class HeuristicRoutingAgent:
    """
    Simple rule-based agent:
    - If disruption is high, choose safer route
    - If truck cost and truckload pressure are low, choose efficient route
    - Otherwise choose balanced route
    """

    def act(self, state: np.ndarray) -> int:
        truck_cost, truckload, local_freight, disruption = state

        # action mapping:
        # 0 = standard
        # 1 = balanced
        # 2 = safer / conservative

        if disruption >= 0.75:
            return 2

        if truck_cost < 0.45 and truckload < 0.45:
            return 0

        return 1