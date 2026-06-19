from .predict import router as predict_router
from .ingestion import router as ingestion_router
from .training import router as training_router
from .graph import router as graph_router
from .health import router as health_router

__all__ = [
    "predict_router",
    "ingestion_router",
    "training_router",
    "graph_router",
    "health_router"
]
