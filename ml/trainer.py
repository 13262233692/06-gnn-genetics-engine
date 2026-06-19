from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score
)
from tqdm import tqdm
from config import settings
from .gnn_model import HeterogeneousGNN, GeneticsGNN
from .graph_converter import GraphConverter, PyGGraphData

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
            "new_edges_count": self.new_edges_count
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
            embedding_dim=settings.GNN_EMBEDDING_DIM
        )

        if model is None:
            self.model = HeterogeneousGNN(
                hidden_channels=settings.GNN_HIDDEN_DIM,
                num_layers=settings.GNN_NUM_LAYERS,
                node_types=self.graph_converter.NODE_TYPES,
                edge_types=self.graph_converter.EDGE_TYPES,
                dropout=settings.GNN_DROPOUT,
                embedding_dim=settings.GNN_EMBEDDING_DIM
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
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(self.device))

        self.last_train_timestamp = None
        self.is_trained = False

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

    async def train(
        self,
        pyg_data: PyGGraphData,
        incremental: bool = False,
        new_nodes_count: int = 0,
        new_edges_count: int = 0
    ) -> TrainingResult:
        logger.info(f"Starting {'incremental ' if incremental else ''}training on {'' if incremental else 'full '}training")

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

            self.model.train()

            for epoch in range(self.config.epochs):
                self.optimizer.zero_grad()

                probabilities = self.model.predict_snp_importance(data)
                logits = torch.log(probabilities + 1e-10)

                loss = self.criterion(logits[train_mask], labels[train_mask])

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

                self.optimizer.step()

                train_loss = loss.item()
                train_acc = self._compute_accuracy(probabilities[train_mask], labels[train_mask])

                self.model.eval()
                with torch.no_grad():
                    val_probs = self.model.predict_snp_importance(data)
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
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                result.epochs_trained = epoch + 1

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            self.model.eval()
            with torch.no_grad():
                test_probs = self.model.predict_snp_importance(data)
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
                'embedding_dim': self.model.embedding_dim
            }
        }

        torch.save(checkpoint, settings.GNN_MODEL_PATH)
        logger.info(f"Model saved to {settings.GNN_MODEL_PATH}")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        path = model_path or settings.GNN_MODEL_PATH
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
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

        new_pyg_data = self.graph_converter.convert_to_pyg(all_nodes, all_edges)

        result = await self.train(
            new_pyg_data,
            incremental=True,
            new_nodes_count=new_nodes_count,
            new_edges_count=new_edges_count
        )

        return result

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "model_type": type(self.model).__name__,
            "hidden_channels": self.model.hidden_channels,
            "num_layers": self.model.num_layers,
            "dropout": self.model.dropout,
            "embedding_dim": self.model.embedding_dim,
            "is_trained": self.is_trained,
            "last_train_timestamp": self.last_train_timestamp,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
