import math
from typing import Dict, Tuple, Optional, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv, GATConv, GCNConv, SAGEConv,
    global_mean_pool, global_max_pool
)
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

logger = logging.getLogger(__name__)


class StableGATConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
        add_self_loops: bool = False,
        negative_slope: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self._out_channels = out_channels * heads if concat else out_channels

        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.bias = nn.Parameter(torch.Tensor(self._out_channels)) if concat else nn.Parameter(torch.Tensor(out_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self._attention_weights: Optional[torch.Tensor] = None
        self._edge_index: Optional[torch.Tensor] = None
        self._attn_hooks: list = []
        self._capture_enabled: bool = False

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        gain = nn.init.calculate_gain('leaky_relu', self.negative_slope)
        nn.init.xavier_uniform_(self.lin_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.lin_dst.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_src, gain=gain)
        nn.init.xavier_uniform_(self.att_dst, gain=gain)
        nn.init.zeros_(self.bias)

    def enable_capture(self) -> None:
        self._capture_enabled = True
        self._attention_weights = None
        self._edge_index = None

    def disable_capture(self) -> None:
        self._capture_enabled = False

    def register_attention_hook(self, hook_fn) -> Any:
        handle = self.register_forward_hook(hook_fn)
        self._attn_hooks.append(handle)
        return handle

    def clear_hooks(self) -> None:
        for handle in self._attn_hooks:
            handle.remove()
        self._attn_hooks = []

    def get_last_attention(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._attention_weights is None or self._edge_index is None:
            return None
        return self._edge_index.detach().cpu(), self._attention_weights.detach().cpu()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        num_nodes = x.size(0)

        x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)

        alpha = alpha_src[edge_index[0]] + alpha_dst[edge_index[1]]
        alpha = self.leaky_relu(alpha)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        alpha = alpha - alpha.max(dim=0, keepdim=True).values
        alpha = torch.exp(alpha)
        alpha = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-16)

        alpha = torch.clamp(alpha, min=0.0, max=1.0)

        if self._capture_enabled:
            self._attention_weights = alpha
            self._edge_index = edge_index

        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device, dtype=x.dtype)

        x_src_expanded = x_src[edge_index[0]]
        weighted = x_src_expanded * alpha.unsqueeze(-1)

        out.scatter_reduce_(0, edge_index[1].unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted, reduce='sum')

        if self.concat:
            out = out.view(num_nodes, self._out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias

        return out


class PreNormResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.3,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels, eps=layer_norm_eps)
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(dropout * 0.5)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.gate(x)
        x = x + residual
        x = torch.clamp(x, min=-50.0, max=50.0)
        return x


class HeterogeneousGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        node_types: list,
        edge_types: list,
        dropout: float = 0.3,
        embedding_dim: int = 128,
        residual_alpha: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.residual_alpha = residual_alpha
        self.layer_norm_eps = layer_norm_eps

        self.node_encoders = nn.ModuleDict()
        for node_type in node_types:
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(embedding_dim, hidden_channels),
                nn.LayerNorm(hidden_channels, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.convs = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            actual_edge_types = self._get_existing_edge_keys(edge_types, node_types)

            if len(actual_edge_types) == 0:
                actual_edge_types = self._get_edge_keys()

            for src_type, et, dst_type in actual_edge_types:
                conv_dict[(src_type, et, dst_type)] = StableGATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // 4,
                    heads=4,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

            self.pre_norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_channels, eps=layer_norm_eps)
                for nt in node_types
            }))

            self.ffns.append(nn.ModuleDict({
                nt: PreNormResidualBlock(hidden_channels, dropout, layer_norm_eps)
                for nt in node_types
            }))

        self.snp_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

        self._capture_attention = False
        self._collected_attention: Optional[List[Dict[str, Any]]] = None

        self._reset_all_parameters()

    def enable_attention_capture(self) -> None:
        self._capture_attention = True
        self._collected_attention = []
        for layer_idx, hetero_conv in enumerate(self.convs):
            for edge_key, conv_module in hetero_conv.convs.items():
                if isinstance(conv_module, StableGATConv):
                    conv_module.enable_capture()

    def disable_attention_capture(self) -> None:
        self._capture_attention = False
        for hetero_conv in self.convs:
            for edge_key, conv_module in hetero_conv.convs.items():
                if isinstance(conv_module, StableGATConv):
                    conv_module.disable_capture()

    def collect_captured_attention(self) -> List[Dict[str, Any]]:
        collected = []
        for layer_idx, hetero_conv in enumerate(self.convs):
            for edge_key, conv_module in hetero_conv.convs.items():
                if isinstance(conv_module, StableGATConv):
                    attn_data = conv_module.get_last_attention()
                    if attn_data is not None:
                        edge_index, alpha = attn_data
                        alpha_mean = alpha.mean(dim=-1)
                        collected.append({
                            "layer": layer_idx,
                            "edge_type": edge_key,
                            "edge_index": edge_index,
                            "alpha": alpha_mean,
                            "alpha_per_head": alpha
                        })
        return collected

    def _reset_all_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in self.node_encoders and \
               not any(module is m for m in self.snp_predictor):
                continue

        for module in self.snp_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_edge_keys(self) -> list:
        keys = []
        node_types = self.node_types
        for src in node_types:
            for et in self.edge_types:
                for dst in node_types:
                    keys.append((src, et, dst))
        return keys

    def _get_existing_edge_keys(self, edge_types: list, node_types: list) -> list:
        keys = []
        known_edges = {
            ('Gene', 'CONTAINS_SNP', 'SNP'),
            ('SNP', 'ASSOCIATED_WITH', 'Phenotype'),
            ('Gene', 'ANNOTATED_TO', 'GOTerm'),
            ('Gene', 'EXPRESSES_IN', 'Environment'),
            ('Gene', 'INTERACTS_WITH', 'Gene'),
            ('SNP', 'CORRELATES_WITH', 'SNP'),
            ('Environment', 'INFLUENCES', 'Phenotype'),
            ('Sample', 'BELONGS_TO', 'Crop'),
            ('Gene', 'PARTICIPATES_IN', 'Pathway'),
            ('GOTerm', 'PARENT_OF', 'GOTerm'),
        }
        for edge_key in known_edges:
            if edge_key[0] in node_types and edge_key[2] in node_types and edge_key[1] in edge_types:
                keys.append(edge_key)
        return keys

    def _init_node_encoders(self) -> None:
        for node_type in self.node_types:
            for module in self.node_encoders[node_type].modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        target_node_type: str = 'SNP'
    ) -> Dict[str, torch.Tensor]:
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type] is not None:
                x = x_dict[node_type]
                x = self.node_encoders[node_type](x)
                x = torch.clamp(x, min=-50.0, max=50.0)
                x_dict[node_type] = x
            else:
                x_dict[node_type] = None

        layer_weights = [1.0 / (self.num_layers + 1)] * (self.num_layers + 1)
        all_embeddings = {nt: [] for nt in self.node_types if x_dict.get(nt) is not None}

        for nt in all_embeddings:
            if x_dict[nt] is not None:
                all_embeddings[nt].append(x_dict[nt])

        for i, conv in enumerate(self.convs):
            x_normed = {}
            for node_type in self.node_types:
                if x_dict.get(node_type) is not None:
                    x_normed[node_type] = self.pre_norms[i][node_type](x_dict[node_type])
                else:
                    x_normed[node_type] = None

            x_dict_in = {k: v for k, v in x_normed.items() if v is not None}
            edge_index_in = {k: v for k, v in edge_index_dict.items()
                             if k[0] in x_dict_in and k[2] in x_dict_in}

            if not edge_index_in or not x_dict_in:
                continue

            try:
                x_out = conv(x_dict_in, edge_index_in)
            except Exception as e:
                logger.warning(f"HeteroConv failed at layer {i}: {e}, skipping layer")
                continue

            for node_type in x_dict_in:
                if node_type in x_out and x_dict.get(node_type) is not None:
                    new_emb = x_out[node_type]

                    if new_emb.shape == x_dict[node_type].shape:
                        x = (1.0 - self.residual_alpha) * x_dict[node_type] + self.residual_alpha * new_emb
                    else:
                        x = new_emb

                    x = self.ffns[i][node_type](x)

                    x = torch.clamp(x, min=-50.0, max=50.0)

                    if torch.isnan(x).any() or torch.isinf(x).any():
                        logger.warning(f"NaN/Inf detected at layer {i} for {node_type}, replacing with zeros")
                        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

                    x_dict[node_type] = x
                    all_embeddings[node_type].append(x)

        final_embeddings = {}
        for node_type in all_embeddings:
            if len(all_embeddings[node_type]) > 0:
                stacked = torch.stack(all_embeddings[node_type], dim=0)
                weights = torch.tensor(layer_weights[:stacked.size(0)], device=stacked.device, dtype=stacked.dtype)
                weights = weights / weights.sum()
                final_embeddings[node_type] = (stacked * weights.view(-1, 1, 1)).sum(dim=0)

        return final_embeddings

    def forward_from_heterodata(
        self,
        data: HeteroData,
        target_node_type: str = 'SNP'
    ) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data and 'x' in data[node_type]:
                x = data[node_type].x
                if hasattr(data[node_type], 'n_id'):
                    pass
                x_dict[node_type] = x
            else:
                x_dict[node_type] = None

        edge_index_dict = {}
        for edge_key in data.edge_types:
            if edge_key in data.edge_index_dict:
                edge_index_dict[edge_key] = data[edge_key].edge_index

        return self.forward(x_dict, edge_index_dict, target_node_type)

    def predict_snp_importance(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        phenotype_embeddings: Optional[torch.Tensor] = None,
        snp_n_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.forward(x_dict, edge_index_dict, target_node_type='SNP')

        if 'SNP' not in embeddings or embeddings['SNP'] is None:
            raise ValueError("No SNP embeddings found in graph")

        snp_emb = embeddings['SNP']

        if phenotype_embeddings is not None and 'Phenotype' in embeddings and embeddings['Phenotype'] is not None:
            pheno_emb = embeddings['Phenotype']
            if phenotype_embeddings.shape[0] > 0:
                pheno_mean = phenotype_embeddings.mean(dim=0, keepdim=True).expand(snp_emb.shape[0], -1)
            else:
                pheno_mean = pheno_emb.mean(dim=0, keepdim=True).expand(snp_emb.shape[0], -1)
        elif 'Phenotype' in embeddings and embeddings['Phenotype'] is not None:
            pheno_emb = embeddings['Phenotype']
            pheno_mean = pheno_emb.mean(dim=0, keepdim=True).expand(snp_emb.shape[0], -1)
        else:
            pheno_mean = torch.zeros_like(snp_emb)

        interaction = snp_emb * pheno_mean
        combined = torch.cat([snp_emb, pheno_mean, interaction], dim=-1)

        logits = self.snp_predictor(combined)
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        probabilities = F.softmax(logits, dim=-1)
        probabilities = torch.clamp(probabilities, min=1e-8, max=1.0 - 1e-8)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

        return probabilities


class GeneticsGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        residual_alpha: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual_alpha = residual_alpha

        self.initial_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for i in range(num_layers):
            conv = SAGEConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr='mean',
                normalize=True
            )
            self.convs.append(conv)
            self.pre_norms.append(nn.LayerNorm(hidden_channels, eps=layer_norm_eps))
            self.ffns.append(PreNormResidualBlock(hidden_channels, dropout, layer_norm_eps))

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_channels, eps=layer_norm_eps)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_encoder(x)
        x = torch.clamp(x, min=-50.0, max=50.0)

        layer_weights = [1.0 / (self.num_layers + 1)] * (self.num_layers + 1)
        all_embeddings = [x]

        for i, (conv, pre_norm, ffn) in enumerate(zip(self.convs, self.pre_norms, self.ffns)):
            x_normed = pre_norm(x)
            x_new = conv(x_normed, edge_index)

            if x_new.shape == x.shape:
                x = (1.0 - self.residual_alpha) * x + self.residual_alpha * x_new
            else:
                x = x_new

            x = ffn(x)
            x = torch.clamp(x, min=-50.0, max=50.0)

            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning(f"NaN/Inf at layer {i}, replacing with zeros")
                x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

            all_embeddings.append(x)

        stacked = torch.stack(all_embeddings, dim=0)
        weights = torch.tensor(layer_weights[:stacked.size(0)], device=stacked.device, dtype=stacked.dtype)
        weights = weights / weights.sum()
        x = (stacked * weights.view(-1, 1, 1)).sum(dim=0)

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
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        return logits

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        node_emb, _ = self.forward(x, edge_index)
        node_emb_normed = self.attn_norm(node_emb)
        attn_output, attn_weights = self.attention(
            node_emb_normed.unsqueeze(0),
            node_emb_normed.unsqueeze(0),
            node_emb_normed.unsqueeze(0)
        )
        return attn_weights.squeeze(0)
