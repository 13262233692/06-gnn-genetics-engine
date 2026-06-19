from pydantic_settings import BaseSettings
from typing import Optional


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
    GNN_BATCH_SIZE: int = 64
    GNN_EPOCHS: int = 100

    INCREMENTAL_TRAIN_INTERVAL: int = 3600
    INCREMENTAL_TRAIN_ENABLED: bool = True
    INCREMENTAL_BATCH_SIZE: int = 1000

    DATA_DIR: str = "data"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
