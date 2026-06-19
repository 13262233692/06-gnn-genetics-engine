from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import torch
from torch_geometric.data import HeteroData, Data
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


class GraphConverter:
    NODE_TYPES = ['Gene', 'SNP', 'GOTerm', 'Phenotype', 'Environment', 'Crop', 'Sample', 'Pathway']
    EDGE_TYPES = [
        'CONTAINS_SNP', 'ASSOCIATED_WITH', 'ANNOTATED_TO', 'EXPRESSES_IN',
        'INTERACTS_WITH', 'CORRELATES_WITH', 'INFLUENCES', 'BELONGS_TO',
        'PARTICIPATES_IN', 'PARENT_OF'
    ]

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
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
