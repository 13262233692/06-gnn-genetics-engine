from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class TrainingRequest(BaseModel):
    force_full: bool = Field(
        False,
        description="是否强制全量训练（否则为增量训练）"
    )
    epochs: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="训练轮数（覆盖配置值）"
    )
    learning_rate: Optional[float] = Field(
        None,
        gt=0,
        le=1,
        description="学习率（覆盖配置值）"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "force_full": False,
                "epochs": 50,
                "learning_rate": 0.001
            }
        }


class TrainingResponse(BaseModel):
    task_id: str = Field(..., description="训练任务ID")
    status: str = Field(..., description="任务状态")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    final_loss: Optional[float] = Field(None, description="最终损失")
    final_accuracy: Optional[float] = Field(None, description="最终准确率")
    final_auc: Optional[float] = Field(None, description="最终AUC")
    epochs_trained: Optional[int] = Field(None, description="实际训练轮数")
    new_nodes_count: int = Field(0, description="处理的新节点数")
    new_edges_count: int = Field(0, description="处理的新边数")
    incremental: bool = Field(..., description="是否为增量训练")
    errors: List[str] = Field(default_factory=list, description="错误列表")
    success: bool = Field(..., description="是否成功")


class TrainingStatusResponse(BaseModel):
    is_training: bool = Field(..., description="是否正在训练")
    last_training_timestamp: Optional[str] = Field(None, description="上次训练时间")
    model_trained: bool = Field(..., description="模型是否已训练")
    model_summary: Dict[str, Any] = Field(..., description="模型摘要")
    device: str = Field(..., description="计算设备")


class IncrementalTrainingStatusResponse(BaseModel):
    is_running: bool = Field(..., description="调度器是否运行中")
    is_enabled: bool = Field(..., description="增量训练是否启用")
    interval_seconds: int = Field(..., description="训练间隔（秒）")
    next_run_time: Optional[str] = Field(None, description="下次运行时间")
    training_status: TrainingStatusResponse = Field(..., description="训练状态")
    history_count: int = Field(..., description="历史记录数量")


class SchedulerConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(None, description="是否启用增量训练")
    interval_seconds: Optional[int] = Field(
        None,
        ge=60,
        le=86400 * 7,
        description="训练间隔秒数"
    )
    cron_expression: Optional[str] = Field(
        None,
        description="Cron表达式（覆盖interval设置）"
    )
    action: Optional[str] = Field(
        None,
        pattern="^(start|stop|pause|resume)$",
        description="调度器操作"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "interval_seconds": 3600,
                "action": "start"
            }
        }


class TrainingHistoryEntry(BaseModel):
    task_id: str = Field(..., description="任务ID")
    started_at: str = Field(..., description="开始时间")
    completed_at: Optional[str] = Field(None, description="完成时间")
    status: str = Field(..., description="状态")
    new_nodes_count: int = Field(..., description="新节点数")
    new_edges_count: int = Field(..., description="新边数")
    final_auc: Optional[float] = Field(None, description="最终AUC")
    success: bool = Field(..., description="是否成功")


class TrainingHistoryResponse(BaseModel):
    history: List[TrainingHistoryEntry] = Field(..., description="训练历史列表")
    total: int = Field(..., description="总记录数")
