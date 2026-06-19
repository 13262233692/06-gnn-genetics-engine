from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.job import Job
from config import settings
from .incremental_trainer import IncrementalTrainer, IncrementalTrainingResult

logger = logging.getLogger(__name__)


class TrainingScheduler:
    def __init__(
        self,
        incremental_trainer: IncrementalTrainer,
        auto_start: bool = True
    ):
        self.incremental_trainer = incremental_trainer
        self.scheduler = AsyncIOScheduler()
        self._scheduled_job: Optional[Job] = None
        self._training_history: List[IncrementalTrainingResult] = []
        self._max_history = 100
        self._is_running = False

        if auto_start and settings.INCREMENTAL_TRAIN_ENABLED:
            asyncio.create_task(self.start())

    async def start(self) -> None:
        if self._is_running:
            logger.warning("Training scheduler already running")
            return

        logger.info("Starting incremental training scheduler")

        self._schedule_incremental_training()
        self.scheduler.start()
        self._is_running = True

        logger.info(
            f"Incremental training scheduler started with interval: "
            f"{settings.INCREMENTAL_TRAIN_INTERVAL}s"
        )

    async def stop(self) -> None:
        if not self._is_running:
            return

        logger.info("Stopping incremental training scheduler")
        self.scheduler.shutdown(wait=False)
        self._is_running = False
        logger.info("Incremental training scheduler stopped")

    def _schedule_incremental_training(self) -> None:
        if self._scheduled_job:
            self._scheduled_job.remove()

        trigger = IntervalTrigger(seconds=settings.INCREMENTAL_TRAIN_INTERVAL)

        self._scheduled_job = self.scheduler.add_job(
            self._run_scheduled_training,
            trigger=trigger,
            id="incremental_training",
            name="Incremental Graph Training",
            replace_existing=True,
            misfire_grace_time=300,
            coalesce=True,
            max_instances=1
        )

        logger.info(
            f"Scheduled incremental training with interval: "
            f"{settings.INCREMENTAL_TRAIN_INTERVAL}s"
        )

    async def _run_scheduled_training(self) -> None:
        try:
            logger.info("Starting scheduled incremental training")
            result = await self.incremental_trainer.run_incremental_training()
            self._add_to_history(result)

            if result.status == "completed" and result.training_result:
                logger.info(
                    f"Scheduled training completed successfully. "
                    f"Epochs: {result.training_result.epochs_trained}, "
                    f"AUC: {result.training_result.final_auc:.4f}"
                )
            elif result.status == "failed":
                logger.error(
                    f"Scheduled training failed: {result.errors}"
                )
            else:
                logger.info(
                    f"Scheduled training skipped: {result.status}"
                )

        except Exception as e:
            logger.error(f"Error in scheduled training: {str(e)}")

    async def trigger_immediate_training(
        self,
        force_full: bool = False
    ) -> IncrementalTrainingResult:
        logger.info(f"Triggering immediate {'full' if force_full else 'incremental'} training")

        if force_full:
            result = await self.incremental_trainer.force_full_retraining()
        else:
            result = await self.incremental_trainer.run_incremental_training()

        self._add_to_history(result)
        return result

    def _add_to_history(self, result: IncrementalTrainingResult) -> None:
        self._training_history.append(result)
        if len(self._training_history) > self._max_history:
            self._training_history = self._training_history[-self._max_history:]

    def get_training_history(
        self,
        limit: int = 10,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        history = self._training_history

        if status:
            history = [h for h in history if h.status == status]

        history = sorted(history, key=lambda x: x.started_at, reverse=True)
        return [h.to_dict() for h in history[:limit]]

    def get_scheduler_status(self) -> Dict[str, Any]:
        next_run_time = None
        if self._scheduled_job and self._scheduled_job.next_run_time:
            next_run_time = self._scheduled_job.next_run_time.isoformat()

        return {
            "is_running": self._is_running,
            "is_enabled": settings.INCREMENTAL_TRAIN_ENABLED,
            "interval_seconds": settings.INCREMENTAL_TRAIN_INTERVAL,
            "next_run_time": next_run_time,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                }
                for job in self.scheduler.get_jobs()
            ],
            "training_status": self.incremental_trainer.get_training_status(),
            "history_count": len(self._training_history)
        }

    def update_interval(self, interval_seconds: int) -> None:
        logger.info(f"Updating training interval to {interval_seconds}s")
        settings.INCREMENTAL_TRAIN_INTERVAL = interval_seconds
        self._schedule_incremental_training()

    def pause(self) -> None:
        if self._scheduled_job:
            self._scheduled_job.pause()
            logger.info("Incremental training paused")

    def resume(self) -> None:
        if self._scheduled_job:
            self._scheduled_job.resume()
            logger.info("Incremental training resumed")

    def set_cron_schedule(self, cron_expression: str) -> None:
        if self._scheduled_job:
            self._scheduled_job.remove()

        trigger = CronTrigger.from_crontab(cron_expression)
        self._scheduled_job = self.scheduler.add_job(
            self._run_scheduled_training,
            trigger=trigger,
            id="incremental_training",
            name="Incremental Graph Training",
            replace_existing=True
        )

        logger.info(f"Set cron schedule: {cron_expression}")
