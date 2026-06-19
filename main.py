import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import Neo4jDriver, GraphSchema, GraphOperations
from ml import HeterogeneousGNN, GNNTrainer, SNPredictor, GraphConverter
from scheduler import IncrementalTrainer, TrainingScheduler
from app.api import (
    predict_router,
    ingestion_router,
    training_router,
    graph_router,
    health_router
)
from app.api import predict as predict_api
from app.api import ingestion as ingestion_api
from app.api import training as training_api
from app.api import graph as graph_api

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

neo4j_driver: Optional[Neo4jDriver] = None
graph_ops: Optional[GraphOperations] = None
gnn_model: Optional[HeterogeneousGNN] = None
gnn_trainer: Optional[GNNTrainer] = None
sn_predictor: Optional[SNPredictor] = None
incremental_trainer: Optional[IncrementalTrainer] = None
training_scheduler: Optional[TrainingScheduler] = None


def init_neo4j() -> None:
    global neo4j_driver, graph_ops
    logger.info("Initializing Neo4j connection...")
    
    neo4j_driver = Neo4jDriver(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE
    )
    
    graph_schema = GraphSchema(neo4j_driver)
    graph_schema.create_constraints_and_indexes()
    
    graph_ops = GraphOperations(neo4j_driver)
    logger.info("Neo4j initialized successfully")


def init_gnn() -> None:
    global gnn_model, gnn_trainer, sn_predictor
    logger.info("Initializing GNN model...")
    
    os.makedirs(os.path.dirname(settings.GNN_MODEL_PATH), exist_ok=True)
    
    graph_converter = GraphConverter(
        embedding_dim=settings.GNN_EMBEDDING_DIM,
        num_neighbors=settings.SAMPLING_NUM_NEIGHBORS,
        batch_size=settings.GNN_BATCH_SIZE
    )
    
    gnn_model = HeterogeneousGNN(
        hidden_channels=settings.GNN_HIDDEN_DIM,
        num_layers=settings.GNN_NUM_LAYERS,
        node_types=graph_converter.NODE_TYPES,
        edge_types=graph_converter.EDGE_TYPES,
        dropout=settings.GNN_DROPOUT,
        embedding_dim=settings.GNN_EMBEDDING_DIM,
        residual_alpha=settings.RESIDUAL_ALPHA,
        layer_norm_eps=settings.LAYER_NORM_EPS
    )
    
    gnn_trainer = GNNTrainer(
        model=gnn_model,
        graph_converter=graph_converter
    )
    
    if os.path.exists(settings.GNN_MODEL_PATH):
        logger.info(f"Loading existing model from {settings.GNN_MODEL_PATH}")
        gnn_trainer.load_model()
    
    sn_predictor = SNPredictor(
        trainer=gnn_trainer,
        graph_ops=graph_ops,
        graph_converter=graph_converter
    )
    
    logger.info("GNN model initialized successfully")


def init_scheduler() -> None:
    global incremental_trainer, training_scheduler
    logger.info("Initializing training scheduler...")
    
    incremental_trainer = IncrementalTrainer(
        graph_ops=graph_ops,
        trainer=gnn_trainer,
        graph_converter=GraphConverter(
            embedding_dim=settings.GNN_EMBEDDING_DIM,
            num_neighbors=settings.SAMPLING_NUM_NEIGHBORS,
            batch_size=settings.GNN_BATCH_SIZE
        )
    )
    
    training_scheduler = TrainingScheduler(
        incremental_trainer=incremental_trainer,
        interval_seconds=settings.INCREMENTAL_TRAIN_INTERVAL
    )
    
    if settings.INCREMENTAL_TRAIN_ENABLED:
        training_scheduler.start()
        logger.info("Incremental training scheduler started")
    else:
        logger.info("Incremental training scheduler is disabled")


def set_api_dependencies() -> None:
    logger.info("Setting API dependencies...")
    predict_api.set_predictor(sn_predictor)
    ingestion_api.set_dependencies(graph_ops)
    training_api.set_dependencies(gnn_trainer, training_scheduler, incremental_trainer)
    graph_api.set_dependencies(graph_ops)
    logger.info("API dependencies set successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_neo4j()
        init_gnn()
        init_scheduler()
        set_api_dependencies()
        logger.info("Application started successfully")
        yield
    finally:
        logger.info("Shutting down application...")
        
        if training_scheduler:
            training_scheduler.stop()
            logger.info("Training scheduler stopped")
        
        if neo4j_driver:
            neo4j_driver.close()
            logger.info("Neo4j connection closed")
        
        logger.info("Application shutdown complete")


app = FastAPI(
    title="农业育种多态性图谱分析系统",
    description="基于图神经网络的农业育种SNP靶点预测系统",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(ingestion_router)
app.include_router(training_router)
app.include_router(graph_router)


@app.get("/", summary="根路径", description="系统欢迎信息")
async def root():
    return {
        "name": "农业育种多态性图谱分析系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": "/api/v1"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True
    )
