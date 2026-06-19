from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score
)
from tqdm import tqdm
from config import settings
from .gnn_model import HeterogeneousGNN, GeneticsGNN
from .graph_converter import GraphConverter, PyGGraphData, SampledGraphData

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    gradient_clip_norm: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_grad_norm_log: float = 100.0
    nan_patience: int = 3
    use_amp: bool = True
    label_smoothing: float = 0.1


@dataclass
class TrainingResult:
    model_path: str
    final_loss: float
    final_accuracy: float
    final_auc: float
    epochs_trained: int
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    incremental: bool = False
    new_nodes_count: int = 0
    new_edges_count: int = 0
    nan_recovery_count: int = 0
    oom_recovery_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "final_auc": self.final_auc,
            "epochs_trained": self.epochs_trained,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "training_history": self.training_history,
            "incremental": self.incremental,
            "new_nodes_count": self.new_nodes_count,
            "new_edges_count": self.new_edges_count,
            "nan_recovery_count": self.nan_recovery_count,
            "oom_recovery_count": self.oom_recovery_count
        }


class GNNTrainer:
    def __init__(
        self,
        model: Optional[HeterogeneousGNN] = None,
        config: Optional[TrainingConfig] = None,
        graph_converter: Optional[GraphConverter] = None
    ):
        self.config = config or TrainingConfig(
            learning_rate=settings.GNN_LEARNING_RATE,
            batch_size=settings.GNN_BATCH_SIZE,
            epochs=settings.GNN_EPOCHS
        )
        self.device = torch.device(self.config.device)
        self.graph_converter = graph_converter or GraphConverter(
            embedding_dim=settings.GNN_EMBEDDING_DIM,
            num_neighbors=settings.SAMPLING_NUM_NEIGHBORS,
            batch_size=self.config.batch_size
        )

        if model is None:
            self.model = HeterogeneousGNN(
                hidden_channels=settings.GNN_HIDDEN_DIM,
                num_layers=settings.GNN_NUM_LAYERS,
                node_types=self.graph_converter.NODE_TYPES,
                edge_types=self.graph_converter.EDGE_TYPES,
                dropout=settings.GNN_DROPOUT,
                embedding_dim=settings.GNN_EMBEDDING_DIM,
                residual_alpha=settings.RESIDUAL_ALPHA,
                layer_norm_eps=settings.LAYER_NORM_EPS
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        class_weights = torch.tensor([1.0, 3.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )

        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.config.use_amp and self.device.type == 'cuda'
        )

        self.last_train_timestamp = None
        self.is_trained = False
        self._nan_counter = 0

    def _extract_batch_data(self, batch):
        x_dict = {}
        edge_index_dict = {}

        for node_type in self.model.node_types:
            if node_type in batch and 'x' in batch[node_type]:
                x_dict[node_type] = batch[node_type].x

        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = batch[edge_type].edge_index

        return x_dict, edge_index_dict

    def _get_input_node_indices(self, batch, node_type='SNP'):
        if node_type in batch and hasattr(batch[node_type], 'n_id'):
            return batch[node_type].n_id
        if hasattr(batch, f'{node_type}_batch'):
            return getattr(batch, f'{node_type}_batch')
        return None

    async def train_sampled(
        self,
        sampled_data: SampledGraphData,
        incremental: bool = False,
        new_nodes_count: int = 0,
        new_edges_count: int = 0
    ) -> TrainingResult:
        logger.info(f"Starting {'incremental ' if incremental else ''}sampled training")

        result = TrainingResult(
            model_path=settings.GNN_MODEL_PATH,
            final_loss=0.0,
            final_accuracy=0.0,
            final_auc=0.0,
            epochs_trained=0,
            incremental=incremental,
            new_nodes_count=new_nodes_count,
            new_edges_count=new_edges_count
        )

        if sampled_data.train_loader is None:
            logger.warning("No train loader available, falling back to full-graph training")
            pyg_data = PyGGraphData(
                data=sampled_data.data,
                node_mapping=sampled_data.node_mapping,
                edge_mapping={},
                node_type_encoder=self.graph_converter.node_type_encoder,
                edge_type_encoder=self.graph_converter.edge_type_encoder
            )
            return await self.train(pyg_data, incremental, new_nodes_count, new_edges_count)

        data = sampled_data.data
        if 'SNP' in data and 'y' in data['SNP']:
            full_labels = data['SNP'].y.long()
        else:
            logger.warning("No labels available for training")
            result.errors.append("No labels available")
            return result

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        self._nan_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses = []
            train_correct = 0
            train_total = 0

            try:
                for batch_idx, batch in enumerate(sampled_data.train_loader):
                    batch = batch.to(self.device)

                    x_dict, edge_index_dict = self._extract_batch_data(batch)

                    if 'SNP' not in x_dict or x_dict['SNP'] is None:
                        continue

                    snp_n_id = self._get_input_node_indices(batch, 'SNP')

                    if snp_n_id is not None and full_labels is not None:
                        valid_mask = snp_n_id < len(full_labels)
                        if valid_mask.sum() == 0:
                            continue
                        labels = full_labels[snp_n_id[valid_mask]]
                        snp_x = x_dict['SNP'][valid_mask]
                    else:
                        labels = torch.zeros(x_dict['SNP'].size(0), dtype=torch.long, device=self.device)
                        snp_x = x_dict['SNP']

                    x_dict_filtered = {k: v for k, v in x_dict.items() if v is not None}
                    edge_index_filtered = {k: v for k, v in edge_index_dict.items()
                                          if k[0] in x_dict_filtered and k[2] in x_dict_filtered}

                    self.optimizer.zero_grad()

                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.config.use_amp and self.device.type == 'cuda'
                    ):
                        probabilities = self.model.predict_snp_importance(
                            x_dict_filtered,
                            edge_index_filtered
                        )

                        snp_probs = probabilities[:len(labels)]
                        if snp_probs.size(0) != labels.size(0):
                            min_len = min(snp_probs.size(0), labels.size(0))
                            snp_probs = snp_probs[:min_len]
                            labels = labels[:min_len]

                        logits = torch.log(snp_probs + 1e-10)
                        loss = self.criterion(logits, labels)

                    if torch.isnan(loss) or torch.isinf(loss):
                        self._nan_counter += 1
                        result.nan_recovery_count += 1
                        logger.warning(
                            f"NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}. "
                            f"NaN recovery count: {self._nan_counter}"
                        )
                        if self._nan_counter >= self.config.nan_patience:
                            logger.error(f"NaN patience ({self.config.nan_patience}) exceeded, stopping training")
                            result.errors.append(f"Training stopped due to persistent NaN loss after {epoch+1} epochs")
                            if best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                            return result
                        continue

                    self.scaler.scale(loss).backward()

                    self.scaler.unscale_(self.optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )

                    if total_norm > self.config.max_grad_norm_log:
                        logger.warning(
                            f"Large gradient norm at epoch {epoch+1}, batch {batch_idx}: {total_norm:.2f}"
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    train_losses.append(loss.item())
                    preds = snp_probs.argmax(dim=-1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)

                    del batch, x_dict, edge_index_dict, probabilities, loss
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    result.oom_recovery_count += 1
                    logger.warning(f"CUDA OOM at epoch {epoch+1}, clearing cache and reducing batch")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    if result.oom_recovery_count > 3:
                        logger.error("OOM recovery limit exceeded, stopping training")
                        result.errors.append("Training stopped due to persistent OOM")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        return result
                    continue
                else:
                    raise

            avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
            train_acc = train_correct / max(train_total, 1)

            self.model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            all_val_probs = []
            all_val_labels = []

            with torch.no_grad():
                for batch in sampled_data.val_loader:
                    batch = batch.to(self.device)

                    x_dict, edge_index_dict = self._extract_batch_data(batch)

                    if 'SNP' not in x_dict or x_dict['SNP'] is None:
                        continue

                    snp_n_id = self._get_input_node_indices(batch, 'SNP')

                    if snp_n_id is not None and full_labels is not None:
                        valid_mask = snp_n_id < len(full_labels)
                        if valid_mask.sum() == 0:
                            continue
                        labels = full_labels[snp_n_id[valid_mask]]
                    else:
                        labels = torch.zeros(x_dict['SNP'].size(0), dtype=torch.long, device=self.device)

                    x_dict_filtered = {k: v for k, v in x_dict.items() if v is not None}
                    edge_index_filtered = {k: v for k, v in edge_index_dict.items()
                                          if k[0] in x_dict_filtered and k[2] in x_dict_filtered}

                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.config.use_amp and self.device.type == 'cuda'
                    ):
                        probabilities = self.model.predict_snp_importance(
                            x_dict_filtered,
                            edge_index_filtered
                        )

                        snp_probs = probabilities[:len(labels)]
                        if snp_probs.size(0) != labels.size(0):
                            min_len = min(snp_probs.size(0), labels.size(0))
                            snp_probs = snp_probs[:min_len]
                            labels = labels[:min_len]

                        logits = torch.log(snp_probs + 1e-10)
                        val_loss = self.criterion(logits, labels)

                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        val_losses.append(val_loss.item())
                        preds = snp_probs.argmax(dim=-1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
                        all_val_probs.append(snp_probs[:, 1].cpu())
                        all_val_labels.append(labels.cpu())

                    del batch
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            val_acc = val_correct / max(val_total, 1)

            val_auc = 0.5
            if all_val_probs and all_val_labels:
                try:
                    all_val_probs_cat = torch.cat(all_val_probs).numpy()
                    all_val_labels_cat = torch.cat(all_val_labels).numpy()
                    if len(np.unique(all_val_labels_cat)) >= 2:
                        val_auc = roc_auc_score(all_val_labels_cat, all_val_probs_cat)
                except Exception:
                    val_auc = 0.5

            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc
            }
            result.training_history.append(epoch_stats)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}"
            )

            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            result.epochs_trained = epoch + 1

            if self._nan_counter > 0 and epoch > 5 and avg_train_loss < float('nan'):
                self._nan_counter = max(0, self._nan_counter - 1)

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        result.final_loss = best_val_loss
        result.final_accuracy = val_acc if val_total > 0 else 0.0
        result.final_auc = val_auc
        result.completed_at = datetime.now()

        self._save_model()
        self.last_train_timestamp = datetime.now().isoformat()
        self.is_trained = True

        logger.info(
            f"Sampled training completed. Val Loss: {best_val_loss:.4f}, "
            f"Val Acc: {result.final_accuracy:.4f}, Val AUC: {val_auc:.4f}, "
            f"NaN recoveries: {result.nan_recovery_count}, OOM recoveries: {result.oom_recovery_count}"
        )

        return result

    async def train(
        self,
        pyg_data: PyGGraphData,
        incremental: bool = False,
        new_nodes_count: int = 0,
        new_edges_count: int = 0
    ) -> TrainingResult:
        logger.info(f"Starting {'incremental ' if incremental else ''}full-graph training")

        result = TrainingResult(
            model_path=settings.GNN_MODEL_PATH,
            final_loss=0.0,
            final_accuracy=0.0,
            final_auc=0.0,
            epochs_trained=0,
            incremental=incremental,
            new_nodes_count=new_nodes_count,
            new_edges_count=new_edges_count
        )

        try:
            data, labels, train_mask, val_mask, test_mask = self._prepare_labels(pyg_data)
            if labels is None:
                logger.warning("No labels available for training")
                result.errors.append("No labels available")
                return result

            data = data.to(self.device)
            labels = labels.to(self.device)

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            self._nan_counter = 0

            self.model.train()

            for epoch in range(self.config.epochs):
                self.optimizer.zero_grad()

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.config.use_amp and self.device.type == 'cuda'
                ):
                    x_dict = {}
                    edge_index_dict = {}
                    for node_type in self.model.node_types:
                        if node_type in data and 'x' in data[node_type]:
                            x_dict[node_type] = data[node_type].x
                    for edge_key in data.edge_types:
                        if edge_key in data.edge_index_dict:
                            edge_index_dict[edge_key] = data[edge_key].edge_index

                    probabilities = self.model.predict_snp_importance(x_dict, edge_index_dict)
                    logits = torch.log(probabilities + 1e-10)
                    loss = self.criterion(logits[train_mask], labels[train_mask])

                if torch.isnan(loss) or torch.isinf(loss):
                    self._nan_counter += 1
                    result.nan_recovery_count += 1
                    logger.warning(f"NaN loss at epoch {epoch+1}, recovery count: {self._nan_counter}")
                    if self._nan_counter >= self.config.nan_patience:
                        logger.error("NaN patience exceeded")
                        result.errors.append("Persistent NaN loss")
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        return result
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss = loss.item()
                train_acc = self._compute_accuracy(probabilities[train_mask], labels[train_mask])

                self.model.eval()
                with torch.no_grad():
                    x_dict = {}
                    edge_index_dict = {}
                    for node_type in self.model.node_types:
                        if node_type in data and 'x' in data[node_type]:
                            x_dict[node_type] = data[node_type].x
                    for edge_key in data.edge_types:
                        if edge_key in data.edge_index_dict:
                            edge_index_dict[edge_key] = data[edge_key].edge_index

                    val_probs = self.model.predict_snp_importance(x_dict, edge_index_dict)
                    val_loss = self.criterion(
                        torch.log(val_probs[val_mask] + 1e-10),
                        labels[val_mask]
                    ).item()
                    val_acc = self._compute_accuracy(val_probs[val_mask], labels[val_mask])
                    val_auc = self._compute_auc(val_probs[val_mask], labels[val_mask])

                self.model.train()

                epoch_stats = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc
                }
                result.training_history.append(epoch_stats)

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
                )

                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                result.epochs_trained = epoch + 1

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                self.model.to(self.device)

            self.model.eval()
            with torch.no_grad():
                x_dict = {}
                edge_index_dict = {}
                for node_type in self.model.node_types:
                    if node_type in data and 'x' in data[node_type]:
                        x_dict[node_type] = data[node_type].x
                for edge_key in data.edge_types:
                    if edge_key in data.edge_index_dict:
                        edge_index_dict[edge_key] = data[edge_key].edge_index

                test_probs = self.model.predict_snp_importance(x_dict, edge_index_dict)
                test_loss = self.criterion(
                    torch.log(test_probs[test_mask] + 1e-10),
                    labels[test_mask]
                ).item()
                test_acc = self._compute_accuracy(test_probs[test_mask], labels[test_mask])
                test_auc = self._compute_auc(test_probs[test_mask], labels[test_mask])

            result.final_loss = test_loss
            result.final_accuracy = test_acc
            result.final_auc = test_auc
            result.completed_at = datetime.now()

            self._save_model()
            self.last_train_timestamp = datetime.now().isoformat()
            self.is_trained = True

            logger.info(
                f"Training completed. Test Loss: {test_loss:.4f}, "
                f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}"
            )

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

        return result

    def _prepare_labels(
        self,
        pyg_data: PyGGraphData
    ):
        data = pyg_data.data
        if 'SNP' in data and 'y' in data['SNP']:
            labels = data['SNP'].y.long()
            train_mask, val_mask, test_mask = self._split_data(labels)
            return data, labels, train_mask, val_mask, test_mask
        return data, None, None, None, None

    def _split_data(
        self,
        labels: torch.Tensor,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        n = len(labels)
        indices = torch.randperm(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_mask = indices[:train_end]
        val_mask = indices[train_end:val_end]
        test_mask = indices[val_end:]

        return train_mask, val_mask, test_mask

    def _compute_accuracy(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        predictions = probabilities.argmax(dim=-1)
        return (predictions == labels).float().mean().item()

    def _compute_auc(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        if len(torch.unique(labels)) < 2:
            return 0.5
        probs_np = probabilities[:, 1].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        try:
            return roc_auc_score(labels_np, probs_np)
        except Exception:
            return 0.5

    def _save_model(self) -> None:
        model_dir = os.path.dirname(settings.GNN_MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'last_train_timestamp': self.last_train_timestamp,
            'config': {
                'hidden_channels': self.model.hidden_channels,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'embedding_dim': self.model.embedding_dim,
                'residual_alpha': self.model.residual_alpha,
                'layer_norm_eps': self.model.layer_norm_eps
            }
        }

        torch.save(checkpoint, settings.GNN_MODEL_PATH)
        logger.info(f"Model saved to {settings.GNN_MODEL_PATH}")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        path = model_path or settings.GNN_MODEL_PATH
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.last_train_timestamp = checkpoint.get('last_train_timestamp')
                self.is_trained = True
                logger.info(f"Model loaded from {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                return False
        return False

    async def train_incremental(
        self,
        new_nodes: List[Dict[str, Any]],
        new_edges: List[Dict[str, Any]],
        existing_pyg_data: PyGGraphData
    ) -> TrainingResult:
        logger.info(f"Starting incremental training with "
                    f"{len(new_nodes)} new nodes and {len(new_edges)} new edges")

        new_nodes_count = len(new_nodes)
        new_edges_count = len(new_edges)

        if new_nodes_count == 0 and new_edges_count == 0:
            logger.info("No new data for incremental training")
            result = TrainingResult(
                model_path=settings.GNN_MODEL_PATH,
                final_loss=0.0,
                final_accuracy=0.0,
                final_auc=0.0,
                epochs_trained=0,
                incremental=True,
                new_nodes_count=0,
                new_edges_count=0
            )
            result.completed_at = datetime.now()
            return result

        all_nodes = existing_pyg_data.data.get('nodes', [])
        all_edges = existing_pyg_data.data.get('edges', [])

        for node in new_nodes:
            all_nodes.append(node)

        for edge in new_edges:
            all_edges.append(edge)

        if settings.SAMPLING_ENABLED:
            sampled_data = self.graph_converter.convert_to_sampled(
                nodes=all_nodes,
                edges=all_edges,
                degree_threshold=settings.SAMPLING_DEGREE_THRESHOLD,
                random_walk_length=settings.SAMPLING_RANDOM_WALK_LENGTH,
                random_walk_iterations=settings.SAMPLING_RANDOM_WALK_ITERATIONS
            )
            return await self.train_sampled(
                sampled_data,
                incremental=True,
                new_nodes_count=new_nodes_count,
                new_edges_count=new_edges_count
            )
        else:
            new_pyg_data = self.graph_converter.convert_to_pyg(all_nodes, all_edges)
            return await self.train(
                new_pyg_data,
                incremental=True,
                new_nodes_count=new_nodes_count,
                new_edges_count=new_edges_count
            )

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "model_type": type(self.model).__name__,
            "hidden_channels": self.model.hidden_channels,
            "num_layers": self.model.num_layers,
            "dropout": self.model.dropout,
            "embedding_dim": self.model.embedding_dim,
            "residual_alpha": self.model.residual_alpha,
            "is_trained": self.is_trained,
            "last_train_timestamp": self.last_train_timestamp,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "config": {
                "gradient_clip_norm": self.config.gradient_clip_norm,
                "use_amp": self.config.use_amp,
                "label_smoothing": self.config.label_smoothing
            }
        }
