"""
Vector-based Fact Clusterer.
Groups atomic facts into dense semantic clusters based on cosine similarity of normalized subject/property strings:
f"{fact.subject} | {fact.property_name}".

Benefits:
- Turns N x K pairwise LLM calls into 1 batched call per cluster.
- Identifies singletons (unique facts with no vector neighbors) which bypass LLM evaluation completely (50-70% cost reduction).
"""
import math
from typing import List, Tuple, Dict
from src.domain.models import Fact
from src.domain.interfaces import IEmbeddingService


class FactCluster:
    """Represents a group of semantically aligned candidate facts."""
    def __init__(self, cluster_id: str, candidate_facts: List[Fact]):
        self.cluster_id = cluster_id
        self.candidate_facts = candidate_facts


class VectorFactClusterer:
    """
    Groups atomic facts into dense semantic clusters using vector cosine similarity.
    """

    def __init__(self, embedding_service: IEmbeddingService, similarity_threshold: float = 0.85):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    def group_into_clusters(self, facts: List[Fact]) -> Tuple[List[FactCluster], List[Fact]]:
        """
        Groups a list of facts into semantic clusters and singletons.

        Returns:
            Tuple of (clusters: List[FactCluster], singletons: List[Fact])
        """
        if not facts:
            return [], []

        if len(facts) == 1:
            return [], facts

        # Step 1: Generate embeddings for normalized string representations
        fact_vectors: List[List[float]] = []
        for fact in facts:
            norm_str = f"{fact.subject} | {fact.property_name}"
            vec = self.embedding_service.generate_embedding(norm_str)
            fact_vectors.append(vec)

        # Step 2: Build adjacency graph using cosine similarity threshold
        n = len(facts)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}

        for i in range(n):
            v_i = fact_vectors[i]
            norm_i = math.sqrt(sum(x * x for x in v_i))
            if norm_i == 0:
                continue

            for j in range(i + 1, n):
                v_j = fact_vectors[j]
                norm_j = math.sqrt(sum(x * x for x in v_j))
                if norm_j == 0:
                    continue

                dot = sum(a * b for a, b in zip(v_i, v_j))
                sim = dot / (norm_i * norm_j)

                if sim >= self.similarity_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        # Step 3: Find connected components (clusters)
        visited = set()
        clusters: List[FactCluster] = []
        singletons: List[Fact] = []
        cluster_counter = 1

        for i in range(n):
            if i in visited:
                continue

            # BFS / DFS traversal
            component_indices = []
            queue = [i]
            visited.add(i)

            while queue:
                curr = queue.pop(0)
                component_indices.append(curr)

                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            component_facts = [facts[idx] for idx in component_indices]
            if len(component_facts) > 1:
                cluster_obj = FactCluster(
                    cluster_id=f"cluster_{cluster_counter:03d}",
                    candidate_facts=component_facts
                )
                clusters.append(cluster_obj)
                cluster_counter += 1
            else:
                singletons.append(component_facts[0])

        return clusters, singletons
