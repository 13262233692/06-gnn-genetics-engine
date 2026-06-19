from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "genetics"

    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    GNN_MODEL_PATH: str = "ml/models/gnn_model.pt"
    GNN_EMBEDDING_DIM: int = 128
    GNN_HIDDEN_DIM: int = 256
    GNN_NUM_LAYERS: int = 3
    GNN_DROPOUT: float = 0.3
    GNN_LEARNING_RATE: float = 0.001
    GNN_BATCH_SIZE: int = 256
    GNN_EPOCHS: int = 100

    RESIDUAL_ALPHA: float = 0.1
    LAYER_NORM_EPS: float = 1e-5
    GRADIENT_CLIP_NORM: float = 5.0
    LABEL_SMOOTHING: float = 0.1
    NAN_PATIENCE: int = 3

    SAMPLING_ENABLED: bool = True
    SAMPLING_NUM_NEIGHBORS: List[int] = [10, 8, 5]
    SAMPLING_DEGREE_THRESHOLD: int = 50
    SAMPLING_RANDOM_WALK_LENGTH: int = 3
    SAMPLING_RANDOM_WALK_ITERATIONS: int = 10
    SAMPLING_INFERENCE_BATCH_SIZE: int = 512

    INCREMENTAL_TRAIN_INTERVAL: int = 3600
    INCREMENTAL_TRAIN_ENABLED: bool = True
    INCREMENTAL_BATCH_SIZE: int = 1000

    DATA_DIR: str = "data"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
