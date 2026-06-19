from typing import Dict, Tuple, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv, GATConv, GCNConv, SAGEConv,
    global_mean_pool, global_max_pool
)
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class HeterogeneousGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        node_types: list,
        edge_types: list,
        dropout: float = 0.3,
        embedding_dim: int = 128
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(embedding_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for src_type, et, dst_type in self._get_edge_keys():
                conv_dict[(src_type, et, dst_type)] = GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // 4,
                    heads=4,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])

        self.skip_lins = nn.ModuleList([
            nn.ModuleDict({nt: nn.Linear(hidden_channels, hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])

        self.snp_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

    def _get_edge_keys(self) -> list:
        keys = []
        node_types = self.node_types
        for src in node_types:
            for et in self.edge_types:
                for dst in node_types:
                    keys.append((src, et, dst))
        return keys

    def forward(
        self,
        data: HeteroData,
        target_node_type: str = 'SNP'
    ) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data and 'x' in data[node_type]:
                x = data[node_type].x
                x = self.node_encoders[node_type](x)
                x_dict[node_type] = x
            else:
                x_dict[node_type] = None

        edge_index_dict = {}
        edge_attr_dict = {}
        for edge_key in data.edge_types:
            if edge_key in data.edge_index_dict:
                edge_index_dict[edge_key] = data[edge_key].edge_index
                if 'edge_attr' in data[edge_key]:
                    edge_attr_dict[edge_key] = data[edge_key].edge_attr

        all_embeddings = {nt: [] for nt in self.node_types if x_dict[nt] is not None}

        for i, conv in enumerate(self.convs):
            x_dict_in = {k: v for k, v in x_dict.items() if v is not None}
            edge_index_in = {k: v for k, v in edge_index_dict.items()
                             if k[0] in x_dict_in and k[2] in x_dict_in}

            x_out = conv(x_dict_in, edge_index_in)

            for node_type in x_dict_in:
                if node_type in x_out:
                    residual = self.skip_lins[i][node_type](x_dict_in[node_type])
                    x = x_out[node_type] + residual
                    x = self.norms[i][node_type](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x_dict[node_type] = x
                    all_embeddings[node_type].append(x)

        final_embeddings = {}
        for node_type in all_embeddings:
            if len(all_embeddings[node_type]) > 0:
                final_embeddings[node_type] = torch.cat(all_embeddings[node_type], dim=-1)

        return final_embeddings

    def predict_snp_importance(
        self,
        data: HeteroData,
        phenotype_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.forward(data)

        if 'SNP' not in embeddings:
            raise ValueError("No SNP embeddings found in graph")

        snp_emb = embeddings['SNP']

        if phenotype_embeddings is not None and 'Phenotype' in embeddings:
            pheno_emb = embeddings['Phenotype']
            if phenotype_embeddings.shape[0] > 0:
                pheno_mean = phenotype_embeddings.mean(dim=0, keepdim=True).expand(snp_emb.shape[0], -1)
            else:
                pheno_mean = pheno_emb.mean(dim=0, keepdim=True).expand(snp_emb.shape[0], -1)
        elif 'Phenotype' in embeddings:
            pheno_emb = embeddings['Phenotype']
            pheno_mean = pheno_emb.mean(dim=0, keepdim=True).expand(snp_emb.shape[0], -1)
        else:
            pheno_mean = torch.zeros_like(snp_emb)

        interaction = snp_emb * pheno_mean
        combined = torch.cat([snp_emb, pheno_mean, interaction], dim=-1)

        logits = self.snp_predictor(combined)
        probabilities = F.softmax(logits, dim=-1)

        return probabilities


class GeneticsGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.initial_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv = SAGEConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr='mean',
                normalize=True
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_encoder(x)
        all_embeddings = [x]

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_residual
            all_embeddings.append(x)

        layer_embeddings = torch.cat(all_embeddings, dim=-1)

        if batch is not None:
            graph_emb = global_mean_pool(x, batch)
        else:
            graph_emb = x.mean(dim=0, keepdim=True)

        return x, graph_emb

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        phenotype_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        node_emb, graph_emb = self.forward(x, edge_index)

        if phenotype_embedding is not None:
            phenotype_expanded = phenotype_embedding.expand(node_emb.shape[0], -1)
            interaction = node_emb * phenotype_expanded
            combined = torch.cat([node_emb, phenotype_expanded, interaction], dim=-1)
        else:
            graph_expanded = graph_emb.expand(node_emb.shape[0], -1)
            interaction = node_emb * graph_expanded
            combined = torch.cat([node_emb, graph_expanded, interaction], dim=-1)

        logits = self.predictor(combined)
        return logits

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        node_emb, _ = self.forward(x, edge_index)
        attn_output, attn_weights = self.attention(
            node_emb.unsqueeze(0),
            node_emb.unsqueeze(0),
            node_emb.unsqueeze(0)
        )
        return attn_weights.squeeze(0)
