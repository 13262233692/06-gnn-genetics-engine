from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import heapq
import numpy as np
import torch

from .gnn_model import HeterogeneousGNN
from .graph_converter import PyGGraphData

logger = logging.getLogger(__name__)


@dataclass
class ExplanationPathNode:
    node_id: str
    node_type: str
    contribution: float
    layer: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "contribution": float(self.contribution),
            "layer": int(self.layer)
        }


@dataclass
class ExplanationPathEdge:
    source_node_id: str
    target_node_id: str
    edge_type: str
    attention_weight: float
    layer: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "attention_weight": float(self.attention_weight),
            "layer": int(self.layer)
        }


@dataclass
class ExplanationPath:
    path_id: str
    rank: int
    total_flow: float
    nodes: List[ExplanationPathNode]
    edges: List[ExplanationPathEdge]
    path_description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "rank": int(self.rank),
            "total_flow": float(self.total_flow),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "path_description": self.path_description
        }


@dataclass
class SNPExplanation:
    snp_id: str
    total_contribution: float
    top_paths: List[ExplanationPath]
    node_importance: Dict[str, float]
    edge_importance: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snp_id": self.snp_id,
            "total_contribution": float(self.total_contribution),
            "top_paths": [p.to_dict() for p in self.top_paths],
            "node_importance": {k: float(v) for k, v in self.node_importance.items()},
            "edge_importance": {k: float(v) for k, v in self.edge_importance.items()}
        }


class AttentionFlowExplainer:
    def __init__(
        self,
        model: HeterogeneousGNN,
        top_k_paths: int = 5,
        flow_threshold: float = 1e-4,
        max_path_length: int = 6
    ):
        self.model = model
        self.top_k_paths = top_k_paths
        self.flow_threshold = flow_threshold
        self.max_path_length = max_path_length
        self._raw_attention: List[Dict[str, Any]] = []

    def __enter__(self):
        self.model.enable_attention_capture()
        self._raw_attention = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._raw_attention = self.model.collect_captured_attention()
        self.model.disable_attention_capture()

    def capture(self) -> None:
        self._raw_attention = self.model.collect_captured_attention()

    def explain_snp(
        self,
        snp_idx: int,
        pyg_data: PyGGraphData,
        top_k: Optional[int] = None
    ) -> Optional[SNPExplanation]:
        if not self._raw_attention:
            logger.warning("No attention data captured. Run forward pass inside the context manager first.")
            return None

        top_k = top_k or self.top_k_paths

        node_mapping = pyg_data.node_mapping
        idx_to_node = self._build_reverse_mapping(node_mapping)

        snp_id = None
        for sid, idx in node_mapping.get("SNP", {}).items():
            if idx == snp_idx:
                snp_id = sid
                break

        if snp_id is None:
            logger.warning(f"SNP index {snp_idx} not found in node mapping")
            return None

        try:
            node_importance = self._compute_node_flow_contribution(
                target_idx=snp_idx,
                target_type="SNP",
                idx_to_node=idx_to_node
            )

            edge_importance = self._compute_edge_flow_contribution(idx_to_node)

            top_paths = self._extract_top_k_paths(
                start_idx=snp_idx,
                start_type="SNP",
                idx_to_node=idx_to_node,
                top_k=top_k
            )

            total_contribution = sum(p.total_flow for p in top_paths)

            return SNPExplanation(
                snp_id=snp_id,
                total_contribution=total_contribution,
                top_paths=top_paths,
                node_importance=node_importance,
                edge_importance=edge_importance
            )
        except Exception as e:
            logger.error(f"Failed to generate explanation for SNP {snp_id}: {e}")
            return None

    def explain_multiple_snps(
        self,
        snp_indices: List[int],
        pyg_data: PyGGraphData,
        top_k: Optional[int] = None
    ) -> Dict[str, SNPExplanation]:
        results: Dict[str, SNPExplanation] = {}
        for snp_idx in snp_indices:
            explanation = self.explain_snp(snp_idx, pyg_data, top_k)
            if explanation:
                results[explanation.snp_id] = explanation
        return results

    def _build_reverse_mapping(
        self,
        node_mapping: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[int, str]]:
        reverse = {}
        for node_type, mapping in node_mapping.items():
            reverse[node_type] = {idx: nid for nid, idx in mapping.items()}
        return reverse

    def _get_layer_attention(
        self,
        layer_idx: int
    ) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
        result = {}
        for entry in self._raw_attention:
            if entry["layer"] == layer_idx:
                result[entry["edge_type"]] = (entry["edge_index"], entry["alpha"])
        return result

    def _compute_node_flow_contribution(
        self,
        target_idx: int,
        target_type: str,
        idx_to_node: Dict[str, Dict[int, str]]
    ) -> Dict[str, float]:
        num_layers = self.model.num_layers
        node_flows: Dict[Tuple[str, int], float] = defaultdict(float)
        node_flows[(target_type, target_idx)] = 1.0

        for layer in range(num_layers - 1, -1, -1):
            layer_attn = self._get_layer_attention(layer)
            new_flows: Dict[Tuple[str, int], float] = defaultdict(float)

            for (src_type, et, dst_type), (edge_index, alpha) in layer_attn.items():
                for e_idx in range(edge_index.size(1)):
                    src_idx = int(edge_index[0, e_idx].item())
                    dst_idx = int(edge_index[1, e_idx].item())
                    attn_weight = float(alpha[e_idx].item())

                    dst_key = (dst_type, dst_idx)
                    if dst_key in node_flows:
                        propagated_flow = node_flows[dst_key] * attn_weight
                        if propagated_flow >= self.flow_threshold:
                            src_key = (src_type, src_idx)
                            new_flows[src_key] += propagated_flow

            for key, flow in new_flows.items():
                node_flows[key] += flow

        result = {}
        for (n_type, n_idx), flow in node_flows.items():
            if n_idx in idx_to_node.get(n_type, {}):
                n_id = idx_to_node[n_type][n_idx]
                result[n_id] = max(result.get(n_id, 0.0), flow)

        return result

    def _compute_edge_flow_contribution(
        self,
        idx_to_node: Dict[str, Dict[int, str]]
    ) -> Dict[str, float]:
        edge_flows: Dict[str, float] = {}

        for entry in self._raw_attention:
            layer = entry["layer"]
            edge_type = entry["edge_type"]
            edge_index = entry["edge_index"]
            alpha = entry["alpha"]
            src_type, rel_type, dst_type = edge_type

            for e_idx in range(edge_index.size(1)):
                src_idx = int(edge_index[0, e_idx].item())
                dst_idx = int(edge_index[1, e_idx].item())
                attn_weight = float(alpha[e_idx].item())

                src_id = idx_to_node.get(src_type, {}).get(src_idx)
                dst_id = idx_to_node.get(dst_type, {}).get(dst_idx)

                if src_id and dst_id:
                    edge_key = f"{src_id}|{rel_type}|{dst_id}|L{layer}"
                    edge_flows[edge_key] = max(
                        edge_flows.get(edge_key, 0.0),
                        attn_weight
                    )

        return edge_flows

    def _extract_top_k_paths(
        self,
        start_idx: int,
        start_type: str,
        idx_to_node: Dict[str, Dict[int, str]],
        top_k: int
    ) -> List[ExplanationPath]:
        heap: List[Tuple[float, int, List[Dict[str, Any]]]] = []
        visited_paths = set()
        all_paths: List[Tuple[float, List[Dict[str, Any]]]] = []

        start_node_id = idx_to_node.get(start_type, {}).get(start_idx)
        if start_node_id is None:
            return []

        initial_state = {
            "node_type": start_type,
            "node_idx": start_idx,
            "node_id": start_node_id,
            "flow": 1.0,
            "layer": self.model.num_layers,
            "path_nodes": [ExplanationPathNode(
                node_id=start_node_id,
                node_type=start_type,
                contribution=1.0,
                layer=self.model.num_layers
            )],
            "path_edges": []
        }

        heapq.heappush(heap, (-1.0, id(initial_state), [initial_state]))

        max_iterations = 10000
        iterations = 0

        while heap and iterations < max_iterations:
            iterations += 1
            neg_flow, _, path_stack = heapq.heappop(heap)
            current_flow = -neg_flow

            current = path_stack[-1]
            current_layer = current["layer"]

            if current_layer <= 0 or len(path_stack) >= self.max_path_length:
                all_paths.append((current_flow, path_stack))
                if len(all_paths) >= top_k * 10:
                    break
                continue

            layer_attn = self._get_layer_attention(current_layer - 1)
            found_next = False

            for (src_type, et, dst_type), (edge_index, alpha) in layer_attn.items():
                if dst_type != current["node_type"]:
                    continue

                mask = (edge_index[1] == current["node_idx"])
                matching_indices = torch.nonzero(mask).squeeze(-1)

                for pos in matching_indices.tolist():
                    pos_idx = int(pos) if torch.is_tensor(pos) else pos
                    src_idx_int = int(edge_index[0, pos_idx].item())
                    attn_weight = float(alpha[pos_idx].item())

                    if attn_weight < self.flow_threshold:
                        continue

                    next_flow = current_flow * attn_weight
                    if next_flow < self.flow_threshold:
                        continue

                    src_id = idx_to_node.get(src_type, {}).get(src_idx_int)
                    if src_id is None:
                        continue

                    path_key = tuple(
                        (n["node_type"], n["node_idx"]) for n in path_stack
                    ) + ((src_type, src_idx_int),)
                    if path_key in visited_paths:
                        continue
                    visited_paths.add(path_key)

                    edge = ExplanationPathEdge(
                        source_node_id=src_id,
                        target_node_id=current["node_id"],
                        edge_type=et,
                        attention_weight=attn_weight,
                        layer=current_layer - 1
                    )

                    next_node = ExplanationPathNode(
                        node_id=src_id,
                        node_type=src_type,
                        contribution=next_flow,
                        layer=current_layer - 1
                    )

                    next_state = {
                        "node_type": src_type,
                        "node_idx": src_idx_int,
                        "node_id": src_id,
                        "flow": next_flow,
                        "layer": current_layer - 1,
                        "path_nodes": current["path_nodes"] + [next_node],
                        "path_edges": current["path_edges"] + [edge]
                    }

                    heapq.heappush(
                        heap,
                        (-next_flow, id(next_state), path_stack + [next_state])
                    )
                    found_next = True

            if not found_next:
                all_paths.append((current_flow, path_stack))

        unique_paths: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        for flow, stack in all_paths:
            node_ids = tuple(s["node_id"] for s in stack)
            key = node_ids
            if key not in unique_paths or flow > unique_paths[key][0]:
                unique_paths[key] = (flow, stack)

        sorted_paths = sorted(
            unique_paths.values(),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]

        result_paths: List[ExplanationPath] = []
        for rank, (total_flow, stack) in enumerate(sorted_paths, 1):
            nodes = stack[-1]["path_nodes"]
            edges = stack[-1]["path_edges"]

            desc_parts = []
            for n in nodes:
                desc_parts.append(f"{n.node_type}:{n.node_id}")
            path_desc = " ← ".join(desc_parts)

            path_id = f"path_{rank}"
            result_paths.append(ExplanationPath(
                path_id=path_id,
                rank=rank,
                total_flow=total_flow,
                nodes=nodes,
                edges=edges,
                path_description=path_desc
            ))

        return result_paths

    def get_attention_summary(self) -> Dict[str, Any]:
        if not self._raw_attention:
            return {"status": "no_data"}

        stats = {
            "num_layers_captured": len(set(e["layer"] for e in self._raw_attention)),
            "num_edge_types": len(set(str(e["edge_type"]) for e in self._raw_attention)),
            "total_edges_captured": sum(e["edge_index"].size(1) for e in self._raw_attention),
            "per_layer": defaultdict(lambda: {"edges": 0, "types": set(), "mean_alpha": 0.0})
        }

        for entry in self._raw_attention:
            layer = entry["layer"]
            alpha = entry["alpha"]
            stats["per_layer"][layer]["edges"] += alpha.numel()
            stats["per_layer"][layer]["types"].add(str(entry["edge_type"]))
            if alpha.numel() > 0:
                stats["per_layer"][layer]["mean_alpha"] += float(alpha.mean().item()) * alpha.numel()

        for layer in stats["per_layer"]:
            total = stats["per_layer"][layer]["edges"]
            if total > 0:
                stats["per_layer"][layer]["mean_alpha"] /= total
            stats["per_layer"][layer]["types"] = list(stats["per_layer"][layer]["types"])

        stats["per_layer"] = dict(stats["per_layer"])
        return stats
