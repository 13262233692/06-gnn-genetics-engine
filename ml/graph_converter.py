from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NegativeSampling
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PyGGraphData:
    data: HeteroData
    node_mapping: Dict[str, Dict[str, int]]
    edge_mapping: Dict[str, List[int]]
    node_type_encoder: OneHotEncoder
    edge_type_encoder: OneHotEncoder

    def to(self, device: torch.device) -> 'PyGGraphData':
        self.data = self.data.to(device)
        return self


@dataclass
class SampledGraphData:
    data: HeteroData
    node_mapping: Dict[str, Dict[str, int]]
    train_loader: NeighborLoader = None
    val_loader: NeighborLoader = None
    test_loader: NeighborLoader = None
    train_idx: torch.Tensor = None
    val_idx: torch.Tensor = None
    test_idx: torch.Tensor = None
    num_neighbors: List[int] = field(default_factory=lambda: [10, 8, 5])

    def get_train_batches(self):
        if self.train_loader is not None:
            return self.train_loader
        return None

    def get_val_batches(self):
        if self.val_loader is not None:
            return self.val_loader
        return None


class GraphConverter:
    NODE_TYPES = ['Gene', 'SNP', 'GOTerm', 'Phenotype', 'Environment', 'Crop', 'Sample', 'Pathway']
    EDGE_TYPES = [
        'CONTAINS_SNP', 'ASSOCIATED_WITH', 'ANNOTATED_TO', 'EXPRESSES_IN',
        'INTERACTS_WITH', 'CORRELATES_WITH', 'INFLUENCES', 'BELONGS_TO',
        'PARTICIPATES_IN', 'PARENT_OF'
    ]

    def __init__(
        self,
        embedding_dim: int = 128,
        num_neighbors: Optional[List[int]] = None,
        batch_size: int = 256,
        num_workers: int = 0
    ):
        self.embedding_dim = embedding_dim
        self.num_neighbors = num_neighbors or [10, 8, 5]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.node_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.edge_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._fit_encoders()
        self.scaler = StandardScaler()

    def _fit_encoders(self) -> None:
        self.node_type_encoder.fit(np.array(self.NODE_TYPES).reshape(-1, 1))
        self.edge_type_encoder.fit(np.array(self.EDGE_TYPES).reshape(-1, 1))

    def convert_to_pyg(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        target_phenotype_id: Optional[str] = None
    ) -> PyGGraphData:
        node_mapping: Dict[str, Dict[str, int]] = {}
        node_features: Dict[str, List[np.ndarray]] = {}
        node_labels: Dict[str, List[float]] = {}

        for node_type in self.NODE_TYPES:
            node_mapping[node_type] = {}
            node_features[node_type] = []
            node_labels[node_type] = []

        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('label', 'SNP')
            if node_type not in node_mapping:
                node_type = 'SNP'

            idx = len(node_mapping[node_type])
            node_mapping[node_type][node_id] = idx

            features = self._extract_node_features(node, node_type)
            node_features[node_type].append(features)

            if node_type == 'Phenotype' and target_phenotype_id:
                label = 1.0 if node_id == target_phenotype_id else 0.0
            else:
                label = node.get('label_value', 0.0)
            node_labels[node_type].append(label)

        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        edge_mapping: Dict[str, List[int]] = {}

        for edge_type in self.EDGE_TYPES:
            edge_mapping[edge_type] = []

        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')
            edge_type = edge.get('type', 'ASSOCIATED_WITH')

            source_type = self._find_node_type(source_id, nodes, node_mapping)
            target_type = self._find_node_type(target_id, nodes, node_mapping)

            if source_type not in node_mapping or target_type not in node_mapping:
                continue
            if source_id not in node_mapping[source_type] or target_id not in node_mapping[target_type]:
                continue

            src_idx = node_mapping[source_type][source_id]
            tgt_idx = node_mapping[target_type][target_id]

            edge_key = (source_type, edge_type, target_type)
            if edge_key not in edge_index_dict:
                edge_index_dict[edge_key] = [[], []]
                edge_attr_dict[edge_key] = []

            edge_index_dict[edge_key][0].append(src_idx)
            edge_index_dict[edge_key][1].append(tgt_idx)

            edge_features = self._extract_edge_features(edge, edge_type)
            edge_attr_dict[edge_key].append(edge_features)
            edge_mapping[edge_type].append(len(edge_attr_dict[edge_key]) - 1)

        data = HeteroData()

        for node_type in self.NODE_TYPES:
            if len(node_features[node_type]) > 0:
                x = torch.tensor(np.array(node_features[node_type]), dtype=torch.float32)
                data[node_type].x = x

                y = torch.tensor(np.array(node_labels[node_type]), dtype=torch.float32)
                data[node_type].y = y

        for edge_key, (src_list, tgt_list) in edge_index_dict.items():
            if len(src_list) > 0:
                edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
                data[edge_key].edge_index = edge_index

                edge_attr = torch.tensor(np.array(edge_attr_dict[edge_key]), dtype=torch.float32)
                data[edge_key].edge_attr = edge_attr

        return PyGGraphData(
            data=data,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
            node_type_encoder=self.node_type_encoder,
            edge_type_encoder=self.edge_type_encoder
        )

    def convert_to_sampled(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        target_phenotype_id: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        degree_threshold: int = 50,
        random_walk_length: int = 3,
        random_walk_iterations: int = 10
    ) -> SampledGraphData:
        pyg_data = self.convert_to_pyg(nodes, edges, target_phenotype_id)
        data = pyg_data.data

        snp_count = data['SNP'].num_nodes if 'SNP' in data and data['SNP'].num_nodes is not None else 0
        if snp_count == 0:
            logger.warning("No SNP nodes found for sampled data creation")
            return SampledGraphData(
                data=data,
                node_mapping=pyg_data.node_mapping,
                num_neighbors=self.num_neighbors
            )

        if 'SNP' in data and 'y' in data['SNP']:
            labels = data['SNP'].y.long()
        else:
            labels = torch.zeros(snp_count, dtype=torch.long)

        degree_bias = self._compute_degree_biased_sampling(data, 'SNP', degree_threshold)

        all_indices = torch.arange(snp_count)
        weighted_indices = all_indices[degree_bias > 0]

        if len(weighted_indices) < snp_count * 0.3:
            weighted_indices = all_indices

        n = len(weighted_indices)
        perm = torch.randperm(n)
        weighted_indices = weighted_indices[perm]

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = weighted_indices[:train_end]
        val_idx = weighted_indices[train_end:val_end]
        test_idx = weighted_indices[val_end:]

        seed_nodes_for_training = self._random_walk_seed_sampling(
            data, train_idx, 'SNP', random_walk_length, random_walk_iterations
        )

        num_neighbors_list = self._adaptive_num_neighbors(data, degree_threshold)

        try:
            train_loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors_list,
                input_nodes=('SNP', seed_nodes_for_training),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            val_loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors_list,
                input_nodes=('SNP', val_idx),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        except Exception as e:
            logger.warning(f"Failed to create NeighborLoader (falling back to default): {e}")
            train_loader = None
            val_loader = None

        return SampledGraphData(
            data=data,
            node_mapping=pyg_data.node_mapping,
            train_loader=train_loader,
            val_loader=val_loader,
            train_idx=seed_nodes_for_training,
            val_idx=val_idx,
            test_idx=test_idx,
            num_neighbors=num_neighbors_list
        )

    def _compute_degree_biased_sampling(
        self,
        data: HeteroData,
        target_type: str,
        degree_threshold: int
    ) -> torch.Tensor:
        num_nodes = data[target_type].num_nodes
        if num_nodes is None or num_nodes == 0:
            return torch.ones(1)

        degree = torch.zeros(num_nodes, dtype=torch.float32)

        for edge_type in data.edge_types:
            if edge_type[0] == target_type:
                edge_index = data[edge_type].edge_index
                src_deg = torch.zeros(num_nodes, dtype=torch.float32)
                unique_src, counts_src = torch.unique(edge_index[0], return_counts=True)
                src_deg[unique_src[unique_src < num_nodes]] = counts_src[unique_src < num_nodes].float()
                degree += src_deg

            if edge_type[2] == target_type:
                edge_index = data[edge_type].edge_index
                dst_deg = torch.zeros(num_nodes, dtype=torch.float32)
                unique_dst, counts_dst = torch.unique(edge_index[1], return_counts=True)
                dst_deg[unique_dst[unique_dst < num_nodes]] = counts_dst[unique_dst < num_nodes].float()
                degree += dst_deg

        degree = torch.clamp(degree, min=1.0)
        sampling_prob = 1.0 / (1.0 + (degree / degree_threshold).float())
        sampling_prob = sampling_prob / sampling_prob.sum() * len(sampling_prob)

        return sampling_prob

    def _random_walk_seed_sampling(
        self,
        data: HeteroData,
        initial_seeds: torch.Tensor,
        target_type: str,
        walk_length: int,
        iterations: int
    ) -> torch.Tensor:
        if len(initial_seeds) == 0:
            return initial_seeds

        seed_set = set(initial_seeds.tolist())

        node_type_to_idx = {nt: {} for nt in self.NODE_TYPES}
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            if src_type not in node_type_to_idx or dst_type not in node_type_to_idx:
                continue
            edge_index = data[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue

            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                if src not in node_type_to_idx[src_type]:
                    node_type_to_idx[src_type][src] = []
                node_type_to_idx[src_type][src].append((dst_type, dst))
                if dst not in node_type_to_idx[dst_type]:
                    node_type_to_idx[dst_type][dst] = []
                node_type_to_idx[dst_type][dst].append((src_type, src))

        seeds_list = list(seed_set)
        max_seeds = min(len(seeds_list) * 2, len(seeds_list) + 500)

        for _ in range(iterations):
            new_seeds = set()
            for seed in seeds_list:
                current_type = target_type
                current_node = seed

                for step in range(walk_length):
                    neighbors = node_type_to_idx.get(current_type, {}).get(current_node, [])
                    if not neighbors:
                        break

                    idx = np.random.randint(len(neighbors))
                    next_type, next_node = neighbors[idx]

                    if next_type == target_type and next_node not in seed_set:
                        new_seeds.add(next_node)

                    current_type = next_type
                    current_node = next_node

            seed_set.update(new_seeds)
            seeds_list = list(seed_set)

            if len(seeds_list) >= max_seeds:
                break

        result = torch.tensor(list(seed_set), dtype=torch.long)
        return result

    def _adaptive_num_neighbors(
        self,
        data: HeteroData,
        degree_threshold: int
    ) -> List[int]:
        avg_degree = 0.0
        count = 0

        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if edge_index.size(1) > 0:
                avg_degree += edge_index.size(1)
                count += 1

        if count > 0:
            avg_degree /= count

        if avg_degree > degree_threshold * 2:
            num_neighbors = [5, 3, 2]
        elif avg_degree > degree_threshold:
            num_neighbors = [8, 5, 3]
        else:
            num_neighbors = self.num_neighbors

        logger.info(
            f"Adaptive num_neighbors={num_neighbors} "
            f"(avg_degree={avg_degree:.1f}, threshold={degree_threshold})"
        )
        return num_neighbors

    def create_inference_loader(
        self,
        data: HeteroData,
        target_node_type: str = 'SNP',
        target_indices: Optional[torch.Tensor] = None
    ) -> NeighborLoader:
        if target_indices is None:
            num_nodes = data[target_node_type].num_nodes
            if num_nodes is None or num_nodes == 0:
                raise ValueError(f"No {target_node_type} nodes in data")
            target_indices = torch.arange(num_nodes)

        inference_loader = NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            input_nodes=(target_node_type, target_indices),
            batch_size=min(self.batch_size, 512),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return inference_loader

    def create_snp_subgraph(
        self,
        snp_data: List[Dict[str, Any]],
        phenotype_id: str
    ) -> 'PyGGraphData':
        nodes = []
        edges = []

        phenotype_node = {
            'id': phenotype_id,
            'label': 'Phenotype',
            'label_value': 1.0
        }
        nodes.append(phenotype_node)

        for snp in snp_data:
            snp_node = {
                'id': snp['snp']['id'],
                'label': 'SNP',
                **snp['snp']
            }
            nodes.append(snp_node)

            edges.append({
                'source': snp['snp']['id'],
                'target': phenotype_id,
                'type': 'ASSOCIATED_WITH',
                **snp.get('association', {})
            })

            gene_id = snp.get('gene_id')
            if gene_id:
                gene_node = {
                    'id': gene_id,
                    'label': 'Gene'
                }
                if not any(n['id'] == gene_id for n in nodes):
                    nodes.append(gene_node)

                edges.append({
                    'source': gene_id,
                    'target': snp['snp']['id'],
                    'type': 'CONTAINS_SNP'
                })

        return self.convert_to_pyg(nodes, edges, phenotype_id)

    def _extract_node_features(self, node: Dict[str, Any], node_type: str) -> np.ndarray:
        features = []

        type_encoding = self.node_type_encoder.transform([[node_type]])[0]
        features.extend(type_encoding)

        numeric_features = []

        if node_type == 'Gene':
            numeric_features.extend([
                float(node.get('start_position', 0)) / 1e9,
                float(node.get('end_position', 0)) / 1e9,
                1.0 if node.get('strand') == '+' else -1.0 if node.get('strand') == '-' else 0.0
            ])
        elif node_type == 'SNP':
            numeric_features.extend([
                float(node.get('position', 0)) / 1e9,
                float(node.get('maf', 0.0)),
                self._encode_variant_type(node.get('variant_type', 'SNV')),
                self._encode_functional_impact(node.get('functional_impact', 'unknown'))
            ])
        elif node_type == 'Phenotype':
            numeric_features.append(self._encode_trait_category(node.get('trait_category', 'unknown')))
        elif node_type == 'Environment':
            numeric_features.extend([
                float(node.get('temperature', 0.0)) / 50.0,
                float(node.get('precipitation', 0.0)) / 2000.0,
                float(node.get('elevation', 0.0)) / 5000.0
            ])

        features.extend(numeric_features)

        if len(features) < self.embedding_dim:
            padding = np.zeros(self.embedding_dim - len(features))
            features.extend(padding)
        elif len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]

        return np.array(features, dtype=np.float32)

    def _extract_edge_features(self, edge: Dict[str, Any], edge_type: str) -> np.ndarray:
        features = []

        type_encoding = self.edge_type_encoder.transform([[edge_type]])[0]
        features.extend(type_encoding)

        numeric_features = []

        if edge_type == 'ASSOCIATED_WITH':
            p_value = float(edge.get('p_value', 1.0))
            numeric_features.append(-np.log10(p_value + 1e-300) / 30.0)
            numeric_features.append(float(edge.get('odds_ratio', 1.0)) / 10.0)
        elif edge_type == 'CORRELATES_WITH':
            numeric_features.append(float(edge.get('correlation_coefficient', 0.0)))
            p_value = float(edge.get('p_value', 1.0))
            numeric_features.append(-np.log10(p_value + 1e-300) / 30.0)
        elif edge_type == 'CONTAINS_SNP':
            numeric_features.append(float(edge.get('distance', 0.0)) / 1e6)

        features.extend(numeric_features)

        if len(features) < 32:
            padding = np.zeros(32 - len(features))
            features.extend(padding)
        elif len(features) > 32:
            features = features[:32]

        return np.array(features, dtype=np.float32)

    def _find_node_type(
        self,
        node_id: str,
        nodes: List[Dict[str, Any]],
        node_mapping: Dict[str, Dict[str, int]]
    ) -> str:
        for node in nodes:
            if node.get('id') == node_id:
                return node.get('label', 'SNP')

        for node_type, mapping in node_mapping.items():
            if node_id in mapping:
                return node_type

        return 'SNP'

    def _encode_variant_type(self, variant_type: str) -> float:
        encoding_map = {
            'SNV': 1.0,
            'MNP': 0.8,
            'insertion': 0.6,
            'deletion': 0.6,
            'multiallelic': 0.4,
            'complex': 0.2
        }
        return encoding_map.get(variant_type, 0.0)

    def _encode_functional_impact(self, impact: str) -> float:
        impact = impact.lower() if impact else 'unknown'
        if 'high' in impact:
            return 1.0
        elif 'moderate' in impact:
            return 0.75
        elif 'low' in impact:
            return 0.5
        elif 'modifier' in impact:
            return 0.25
        return 0.0

    def _encode_trait_category(self, category: str) -> float:
        category = category.lower() if category else 'unknown'
        if 'drought' in category or 'stress' in category:
            return 1.0
        elif 'disease' in category or 'resistance' in category:
            return 0.9
        elif 'yield' in category:
            return 0.8
        elif 'quality' in category:
            return 0.7
        elif 'maturity' in category:
            return 0.6
        return 0.0
