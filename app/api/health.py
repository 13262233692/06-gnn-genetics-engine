from fastapi import APIRouter, HTTPException, status
import logging
from datetime import datetime
from schemas import HealthResponse
from database import Neo4jDriver
from ml import GNNTrainer
from scheduler import TrainingScheduler

logger = logging.getLogger(__name__)
router = APIRouter(tags=["健康检查"])

neo4j_driver = Neo4jDriver()
gnn_trainer: GNNTrainer = None
training_scheduler: TrainingScheduler = None


def set_dependencies(trainer: GNNTrainer, scheduler: TrainingScheduler) -> None:
    global gnn_trainer, training_scheduler
    gnn_trainer = trainer
    training_scheduler = scheduler


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="系统健康检查",
    description="检查系统各组件的运行状态"
)
async def health_check() -> HealthResponse:
    try:
        neo4j_connected = False
        try:
            await neo4j_driver.connect()
            count = await neo4j_driver.execute_query("RETURN 1 as test")
            neo4j_connected = len(count) > 0
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")

        model_loaded = gnn_trainer.is_trained if gnn_trainer else False
        scheduler_running = training_scheduler._is_running if training_scheduler else False

        overall_status = "healthy"
        if not neo4j_connected:
            overall_status = "degraded"

        return HealthResponse(
            status=overall_status,
            neo4j_connected=neo4j_connected,
            model_loaded=model_loaded,
            scheduler_running=scheduler_running,
            version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/health/live", summary="存活检查")
async def liveness_probe() -> dict:
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get("/health/ready", summary="就绪检查")
async def readiness_probe() -> dict:
    try:
        await neo4j_driver.connect()
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Not ready: {str(e)}"
        )
