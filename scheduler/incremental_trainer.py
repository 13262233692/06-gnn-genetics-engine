from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
from ml import GNNTrainer, TrainingResult, GraphConverter, PyGGraphData
from database import GraphOperations

logger = logging.getLogger(__name__)


@dataclass
class IncrementalTrainingResult:
    task_id: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"
    new_nodes_count: int = 0
    new_edges_count: int = 0
    training_result: Optional[TrainingResult] = None
    errors: List[str] = field(default_factory=list)
    last_check_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "new_nodes_count": self.new_nodes_count,
            "new_edges_count": self.new_edges_count,
            "training_result": self.training_result.to_dict() if self.training_result else None,
            "errors": self.errors,
            "last_check_timestamp": self.last_check_timestamp,
            "success": self.status == "completed" and len(self.errors) == 0
        }


class IncrementalTrainer:
    def __init__(
        self,
        trainer: GNNTrainer,
        graph_ops: GraphOperations,
        graph_converter: Optional[GraphConverter] = None
    ):
        self.trainer = trainer
        self.graph_ops = graph_ops
        self.graph_converter = graph_converter or GraphConverter()
        self.last_training_timestamp = None
        self._training_lock = asyncio.Lock()
        self._is_training = False

    async def check_for_updates(
        self,
        last_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        check_timestamp = datetime.now().isoformat()
        since_timestamp = last_timestamp or self.last_training_timestamp or "1970-01-01T00:00:00"

        logger.info(f"Checking for graph updates since {since_timestamp}")

        new_nodes = await self.graph_ops.get_new_nodes_since(since_timestamp)
        new_relationships = await self.graph_ops.get_new_relationships_since(since_timestamp)

        result = {
            "check_timestamp": check_timestamp,
            "since_timestamp": since_timestamp,
            "new_nodes_count": len(new_nodes),
            "new_relationships_count": len(new_relationships),
            "new_nodes": new_nodes,
            "new_relationships": new_relationships
        }

        logger.info(f"Found {len(new_nodes)} new nodes and {len(new_relationships)} new relationships")
        return result

    async def run_incremental_training(
        self,
        task_id: Optional[str] = None
    ) -> IncrementalTrainingResult:
        task_id = task_id or f"incr_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = IncrementalTrainingResult(task_id=task_id)

        if self._is_training:
            result.status = "skipped"
            result.errors.append("Training already in progress")
            logger.warning("Incremental training skipped: already in progress")
            return result

        async with self._training_lock:
            self._is_training = True
            result.status = "running"

            try:
                updates = await self.check_for_updates()
                result.last_check_timestamp = updates["check_timestamp"]
                result.new_nodes_count = updates["new_nodes_count"]
                result.new_edges_count = updates["new_relationships_count"]

                if updates["new_nodes_count"] == 0 and updates["new_relationships_count"] == 0:
                    result.status = "completed"
                    result.completed_at = datetime.now()
                    logger.info("No new data available for incremental training")
                    return result

                existing_graph_data = await self._build_existing_graph()

                new_nodes_list = []
                for node_data in updates["new_nodes"]:
                    node = {
                        **node_data["properties"],
                        "label": node_data["labels"][0] if node_data["labels"] else "SNP"
                    }
                    new_nodes_list.append(node)

                new_edges_list = []
                for rel_data in updates["new_relationships"]:
                    edge = {
                        "source": rel_data["source"]["id"],
                        "target": rel_data["target"]["id"],
                        "type": rel_data["rel_type"],
                        **rel_data["properties"]
                    }
                    new_edges_list.append(edge)

                if existing_graph_data:
                    training_result = await self.trainer.train_incremental(
                        new_nodes=new_nodes_list,
                        new_edges=new_edges_list,
                        existing_pyg_data=existing_graph_data
                    )
                else:
                    training_result = await self._train_full_graph()

                result.training_result = training_result
                self.last_training_timestamp = updates["check_timestamp"]
                result.status = "completed"

                logger.info(
                    f"Incremental training completed successfully: "
                    f"processed {result.new_nodes_count} new nodes and "
                    f"{result.new_edges_count} new edges"
                )

            except Exception as e:
                result.status = "failed"
                error_msg = f"Incremental training failed: {str(e)}"
                result.errors.append(error_msg)
                logger.error(error_msg)

            finally:
                result.completed_at = datetime.now()
                self._is_training = False

            return result

    async def _build_existing_graph(self) -> Optional[PyGGraphData]:
        try:
            stats = await self.graph_ops.get_graph_statistics()
            total_nodes = sum(ns["count"] for ns in stats.get("node_counts", []))

            if total_nodes == 0:
                return None

            subgraph_data = await self.graph_ops.get_gene_snp_phenotype_subgraph(
                phenotype_name="",
                min_p_value=1.0
            )

            if not subgraph_data["nodes"]:
                return None

            return self.graph_converter.convert_to_pyg(
                nodes=subgraph_data["nodes"],
                edges=subgraph_data["edges"]
            )
        except Exception as e:
            logger.error(f"Error building existing graph: {str(e)}")
            return None

    async def _train_full_graph(self) -> TrainingResult:
        logger.info("Running full graph training (no existing model)")

        subgraph_data = await self.graph_ops.get_gene_snp_phenotype_subgraph(
            phenotype_name="",
            min_p_value=1.0
        )

        pyg_data = self.graph_converter.convert_to_pyg(
            nodes=subgraph_data["nodes"],
            edges=subgraph_data["edges"]
        )

        return await self.trainer.train(pyg_data, incremental=False)

    async def force_full_retraining(self) -> IncrementalTrainingResult:
        task_id = f"full_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = IncrementalTrainingResult(task_id=task_id)

        if self._is_training:
            result.status = "skipped"
            result.errors.append("Training already in progress")
            return result

        async with self._training_lock:
            self._is_training = True
            result.status = "running"

            try:
                logger.info("Forcing full graph retraining")
                training_result = await self._train_full_graph()
                result.training_result = training_result
                result.last_check_timestamp = datetime.now().isoformat()
                result.status = "completed"
                self.last_training_timestamp = datetime.now().isoformat()

            except Exception as e:
                result.status = "failed"
                result.errors.append(f"Full retraining failed: {str(e)}")

            finally:
                result.completed_at = datetime.now()
                self._is_training = False

            return result

    def get_training_status(self) -> Dict[str, Any]:
        return {
            "is_training": self._is_training,
            "last_training_timestamp": self.last_training_timestamp,
            "model_trained": self.trainer.is_trained,
            "model_summary": self.trainer.get_model_summary()
        }
