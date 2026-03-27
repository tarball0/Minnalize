"""
Graph-Based Feature Extraction
Models entity relationships as graphs to detect unusual connectivity patterns.
"""

import numpy as np
import pandas as pd
import networkx as nx

from nexusguard.config import FeatureConfig


class GraphFeatureExtractor:
    """Extracts anomaly-indicative features from entity relationship graphs."""

    def __init__(self, config: FeatureConfig):
        self.min_edge_weight = config.graph_min_edge_weight
        self.baseline_graph: nx.DiGraph | None = None
        self.baseline_metrics: dict = {}

    def build_baseline(self, df: pd.DataFrame) -> None:
        """Build a baseline communication graph from normal activity."""
        G = nx.DiGraph()
        # Build user -> destination edges weighted by frequency
        edges = df.groupby(["user_id", "dest_ip"]).size().reset_index(name="weight")
        for _, row in edges.iterrows():
            if row["weight"] >= self.min_edge_weight:
                G.add_edge(row["user_id"], row["dest_ip"], weight=row["weight"])

        self.baseline_graph = G

        # Precompute baseline metrics per node
        for node in G.nodes():
            self.baseline_metrics[node] = {
                "degree": G.degree(node),
                "out_degree": G.out_degree(node) if G.is_directed() else 0,
                "in_degree": G.in_degree(node) if G.is_directed() else 0,
                "neighbors": set(G.neighbors(node)),
            }

        # Community detection using greedy modularity (undirected)
        UG = G.to_undirected()
        if len(UG.nodes()) > 0:
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = greedy_modularity_communities(UG)
                self._node_to_community = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        self._node_to_community[node] = i
            except Exception:
                self._node_to_community = {n: 0 for n in UG.nodes()}
        else:
            self._node_to_community = {}

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract graph-based features for each event."""
        df = df.copy()

        graph_scores = np.zeros(len(df))
        new_edge_flags = np.zeros(len(df))
        cross_community_flags = np.zeros(len(df))

        if self.baseline_graph is None:
            df["graph_anomaly_score"] = 0.0
            df["is_new_edge"] = 0
            df["is_cross_community"] = 0
            return df

        for idx, row in df.iterrows():
            user = row["user_id"]
            dest = row["dest_ip"]
            score = 0.0

            # Check if this edge exists in baseline
            if not self.baseline_graph.has_edge(user, dest):
                new_edge_flags[idx] = 1
                score += 0.5

            # Check if destination is in a different community
            user_comm = self._node_to_community.get(user, -1)
            dest_comm = self._node_to_community.get(dest, -2)
            if user_comm != dest_comm:
                cross_community_flags[idx] = 1
                score += 0.3

            # Check for unusual connectivity pattern
            if user in self.baseline_metrics:
                known_neighbors = self.baseline_metrics[user]["neighbors"]
                if dest not in known_neighbors:
                    score += 0.2

            graph_scores[idx] = min(score, 1.0)

        df["graph_anomaly_score"] = graph_scores
        df["is_new_edge"] = new_edge_flags.astype(int)
        df["is_cross_community"] = cross_community_flags.astype(int)
        return df
