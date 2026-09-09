"""
Vector-based Fact Clusterer.
Groups atomic facts into dense semantic clusters using a two-stage gate:
  Stage 1: Cosine similarity on normalized "subject | property_name" embeddings.
  Stage 2: Structural token-overlap check on subject and property_name strings.

Benefits:
- Turns N x K pairwise LLM calls into 1 batched call per cluster.
- Structural gate prevents semantically adjacent but factually unrelated clusters.
- Identifies singletons which bypass LLM evaluation completely.
"""
import os
import math
import re
from typing import List, Tuple, Dict, Set
from src.domain.models import Fact
from src.domain.interfaces import IEmbeddingService
from src.infrastructure.metrics import PipelineMetricsCollector


SUBJECT_OVERLAP_THRESHOLD = float(os.getenv("SUBJECT_OVERLAP_THRESHOLD", "0.6"))
PROPERTY_OVERLAP_THRESHOLD = float(os.getenv("PROPERTY_OVERLAP_THRESHOLD", "0.5"))


class FactCluster:
    """Represents a group of semantically aligned candidate facts."""
    def __init__(self, cluster_id: str, candidate_facts: List[Fact]):
        self.cluster_id = cluster_id
        self.candidate_facts = candidate_facts


class VectorFactClusterer:
    """
    Groups atomic facts into dense semantic clusters using vector cosine similarity
    with a structural relevance gate.
    """

    def __init__(self, embedding_service: IEmbeddingService, similarity_threshold: float = 0.85):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Tokenizes text into lowercase alphanumeric tokens."""
        return set(re.findall(r'[a-z0-9]+', text.lower()))

    @staticmethod
    def _compute_token_overlap(tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Computes Jaccard-style overlap between two token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    def _structural_relevance_check(self, fact_a: Fact, fact_b: Fact) -> bool:
        """Two-stage structural gate: checks subject and property token overlap."""
        subj_a = self._tokenize(fact_a.subject)
        subj_b = self._tokenize(fact_b.subject)
        subj_overlap = self._compute_token_overlap(subj_a, subj_b)

        if subj_overlap < SUBJECT_OVERLAP_THRESHOLD:
            return False

        prop_a = self._tokenize(fact_a.property_name)
        prop_b = self._tokenize(fact_b.property_name)
        prop_overlap = self._compute_token_overlap(prop_a, prop_b)

        if prop_overlap < PROPERTY_OVERLAP_THRESHOLD:
            return False

        return True

    def group_into_clusters(self, facts: List[Fact]) -> Tuple[List[FactCluster], List[Fact]]:
        """
        Groups a list of facts into semantic clusters and singletons.
        Uses two-stage gate: vector similarity + structural relevance.

        Returns:
            Tuple of (clusters: List[FactCluster], singletons: List[Fact])
        """
        if not facts:
            return [], []

        if len(facts) == 1:
            return [], facts

        metrics = PipelineMetricsCollector()

        # Step 1: Generate embeddings for normalized string representations
        fact_vectors: List[List[float]] = []
        for fact in facts:
            norm_str = f"{fact.subject} | {fact.property_name}"
            vec = self.embedding_service.generate_embedding(norm_str)
            fact_vectors.append(vec)

        # Step 2: Build adjacency graph using two-stage gate
        n = len(facts)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        edges_before_filter = 0
        edges_after_filter = 0
        similarity_values: List[float] = []

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
                    edges_before_filter += 1

                    # Stage 2: Structural relevance check
                    if self._structural_relevance_check(facts[i], facts[j]):
                        adj[i].append(j)
                        adj[j].append(i)
                        edges_after_filter += 1
                        similarity_values.append(sim)

        # Step 3: Find connected components (clusters)
        visited = set()
        clusters: List[FactCluster] = []
        singletons: List[Fact] = []
        cluster_counter = 1

        for i in range(n):
            if i in visited:
                continue

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

        # Update pipeline metrics
        avg_sim = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0
        avg_cluster_size = sum(len(c.candidate_facts) for c in clusters) / len(clusters) if clusters else 0.0
        rejection_rate = (1 - edges_after_filter / edges_before_filter) if edges_before_filter > 0 else 0.0

        metrics.update_cluster_metrics(
            total_facts_clustered=n,
            num_clusters=len(clusters),
            num_singletons=len(singletons),
            avg_cluster_size=avg_cluster_size,
            avg_intra_cluster_similarity=avg_sim,
            edges_before_structural_filter=edges_before_filter,
            edges_after_structural_filter=edges_after_filter,
            structural_filter_rejection_rate=rejection_rate
        )

        return clusters, singletons
