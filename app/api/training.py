from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Query
import logging
from typing import List, Optional
from datetime import datetime

from schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatusResponse,
    IncrementalTrainingStatusResponse,
    SchedulerConfigRequest,
    TrainingHistoryResponse,
    TrainingHistoryEntry
)
from ml import GNNTrainer
from scheduler import TrainingScheduler, IncrementalTrainer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["模型训练"])

gnn_trainer: GNNTrainer = None
incremental_trainer: IncrementalTrainer = None
training_scheduler: TrainingScheduler = None


def set_dependencies(
    trainer: GNNTrainer,
    incr_trainer: IncrementalTrainer,
    scheduler: TrainingScheduler
) -> None:
    global gnn_trainer, incremental_trainer, training_scheduler
    gnn_trainer = trainer
    incremental_trainer = incr_trainer
    training_scheduler = scheduler


@router.post(
    "",
    response_model=TrainingResponse,
    summary="触发模型训练",
    description="手动触发模型训练，可选择增量训练或全量重训练"
)
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    if gnn_trainer is None or incremental_trainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training service not initialized"
        )

    try:
        if request.epochs:
            gnn_trainer.config.epochs = request.epochs
        if request.learning_rate:
            for param_group in gnn_trainer.optimizer.param_groups:
                param_group['lr'] = request.learning_rate

        result = await training_scheduler.trigger_immediate_training(
            force_full=request.force_full
        )

        training_result = result.training_result

        return TrainingResponse(
            task_id=result.task_id,
            status=result.status,
            started_at=result.started_at,
            completed_at=result.completed_at,
            final_loss=training_result.final_loss if training_result else None,
            final_accuracy=training_result.final_accuracy if training_result else None,
            final_auc=training_result.final_auc if training_result else None,
            epochs_trained=training_result.epochs_trained if training_result else 0,
            new_nodes_count=result.new_nodes_count,
            new_edges_count=result.new_edges_count,
            incremental=not request.force_full,
            errors=result.errors,
            success=result.status == "completed" and len(result.errors) == 0
        )

    except Exception as e:
        logger.error(f"Training trigger failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training trigger failed: {str(e)}"
        )


@router.get(
    "/status",
    response_model=TrainingStatusResponse,
    summary="获取训练状态",
    description="获取当前模型训练状态和模型摘要"
)
async def get_training_status() -> TrainingStatusResponse:
    if incremental_trainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training service not initialized"
        )

    try:
        status_data = incremental_trainer.get_training_status()

        return TrainingStatusResponse(
            is_training=status_data["is_training"],
            last_training_timestamp=status_data["last_training_timestamp"],
            model_trained=status_data["model_trained"],
            model_summary=status_data["model_summary"],
            device=str(gnn_trainer.device) if gnn_trainer else "cpu"
        )

    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )


@router.get(
    "/scheduler",
    response_model=IncrementalTrainingStatusResponse,
    summary="获取增量训练调度器状态",
    description="获取增量训练调度器的运行状态和配置"
)
async def get_scheduler_status() -> IncrementalTrainingStatusResponse:
    if training_scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not initialized"
        )

    try:
        scheduler_status = training_scheduler.get_scheduler_status()
        training_status = await get_training_status()

        return IncrementalTrainingStatusResponse(
            is_running=scheduler_status["is_running"],
            is_enabled=scheduler_status["is_enabled"],
            interval_seconds=scheduler_status["interval_seconds"],
            next_run_time=scheduler_status["next_run_time"],
            training_status=training_status,
            history_count=scheduler_status["history_count"]
        )

    except Exception as e:
        logger.error(f"Failed to get scheduler status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scheduler status: {str(e)}"
        )


@router.post(
    "/scheduler",
    response_model=IncrementalTrainingStatusResponse,
    summary="配置增量训练调度器",
    description="配置增量训练调度器的运行参数，包括启停、间隔设置等"
)
async def configure_scheduler(request: SchedulerConfigRequest) -> IncrementalTrainingStatusResponse:
    if training_scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not initialized"
        )

    try:
        if request.enabled is not None:
            from config import settings
            settings.INCREMENTAL_TRAIN_ENABLED = request.enabled

        if request.interval_seconds is not None:
            training_scheduler.update_interval(request.interval_seconds)

        if request.cron_expression is not None:
            training_scheduler.set_cron_schedule(request.cron_expression)

        if request.action:
            if request.action == "start":
                await training_scheduler.start()
            elif request.action == "stop":
                await training_scheduler.stop()
            elif request.action == "pause":
                training_scheduler.pause()
            elif request.action == "resume":
                training_scheduler.resume()

        return await get_scheduler_status()

    except Exception as e:
        logger.error(f"Failed to configure scheduler: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure scheduler: {str(e)}"
        )


@router.get(
    "/history",
    response_model=TrainingHistoryResponse,
    summary="获取训练历史",
    description="获取历史训练任务记录"
)
async def get_training_history(
    limit: int = Query(10, ge=1, le=100, description="返回记录数量"),
    status: Optional[str] = Query(None, description="按状态筛选")
) -> TrainingHistoryResponse:
    if training_scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not initialized"
        )

    try:
        history_data = training_scheduler.get_training_history(limit=limit, status=status)

        history_entries = []
        for entry in history_data:
            tr = entry.get("training_result")
            history_entries.append(TrainingHistoryEntry(
                task_id=entry["task_id"],
                started_at=entry["started_at"],
                completed_at=entry["completed_at"],
                status=entry["status"],
                new_nodes_count=entry["new_nodes_count"],
                new_edges_count=entry["new_edges_count"],
                final_auc=tr["final_auc"] if tr else None,
                success=entry["success"]
            ))

        return TrainingHistoryResponse(
            history=history_entries,
            total=len(history_data)
        )

    except Exception as e:
        logger.error(f"Failed to get training history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training history: {str(e)}"
        )


@router.get(
    "/check-updates",
    summary="检查图谱更新",
    description="检查自上次训练以来知识图谱是否有新数据"
)
async def check_for_updates() -> dict:
    if incremental_trainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training service not initialized"
        )

    try:
        updates = await incremental_trainer.check_for_updates()
        return {
            "check_timestamp": updates["check_timestamp"],
            "since_timestamp": updates["since_timestamp"],
            "new_nodes_count": updates["new_nodes_count"],
            "new_relationships_count": updates["new_relationships_count"],
            "has_updates": updates["new_nodes_count"] > 0 or updates["new_relationships_count"] > 0
        }

    except Exception as e:
        logger.error(f"Failed to check for updates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check for updates: {str(e)}"
        )
