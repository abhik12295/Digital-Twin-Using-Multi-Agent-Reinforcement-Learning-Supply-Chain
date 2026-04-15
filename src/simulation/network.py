from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx


@dataclass
class NetworkNode:
    node_id: str
    node_type: str
    x: float
    y: float


class SupplyChainNetwork:
    def __init__(
        self,
        num_suppliers: int = 5,
        num_warehouses: int = 4,
        num_customers: int = 8,
        seed: int = 42,
    ) -> None:
        self.num_suppliers = num_suppliers
        self.num_warehouses = num_warehouses
        self.num_customers = num_customers
        self.seed = seed
        self.graph = nx.DiGraph()
        random.seed(seed)

    def build(self) -> nx.DiGraph:
        suppliers = self._create_nodes("supplier", self.num_suppliers)
        warehouses = self._create_nodes("warehouse", self.num_warehouses)
        customers = self._create_nodes("customer", self.num_customers)

        for node in suppliers + warehouses + customers:
            self.graph.add_node(
                node.node_id,
                node_type=node.node_type,
                pos=(node.x, node.y),
            )

        self._connect_layers(suppliers, warehouses, min_edges=2, max_edges=3)
        self._connect_layers(warehouses, customers, min_edges=2, max_edges=4)

        return self.graph

    def _create_nodes(self, prefix: str, count: int) -> List[NetworkNode]:
        nodes: List[NetworkNode] = []
        for i in range(count):
            nodes.append(
                NetworkNode(
                    node_id=f"{prefix[:1].upper()}{i + 1}",
                    node_type=prefix,
                    x=round(random.uniform(0, 100), 2),
                    y=round(random.uniform(0, 100), 2),
                )
            )
        return nodes

    def _connect_layers(
        self,
        sources: List[NetworkNode],
        targets: List[NetworkNode],
        min_edges: int,
        max_edges: int,
    ) -> None:
        for source in sources:
            edge_count = random.randint(min_edges, min(max_edges, len(targets)))
            selected_targets = random.sample(targets, edge_count)
            for target in selected_targets:
                distance = self._euclidean_distance(source.x, source.y, target.x, target.y)
                self.graph.add_edge(
                    source.node_id,
                    target.node_id,
                    distance_miles=round(distance * 10 + random.uniform(10, 75), 2),
                    base_cost=round(distance * 1.25 + random.uniform(15, 50), 2),
                )

    @staticmethod
    def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def get_node_groups(self) -> Dict[str, List[str]]:
        groups = {"supplier": [], "warehouse": [], "customer": []}
        for node_id, attrs in self.graph.nodes(data=True):
            groups[attrs["node_type"]].append(node_id)
        return groups