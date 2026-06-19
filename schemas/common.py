from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    neo4j_connected: Optional[bool] = Field(None, description="Neo4j connection status")
    model_loaded: Optional[bool] = Field(None, description="GNN model loaded status")
    scheduler_running: Optional[bool] = Field(None, description="Training scheduler status")
    version: str = Field("1.0.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "neo4j_connected": True,
                "model_loaded": True,
                "scheduler_running": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[int] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Phenotype not found in database",
                "error_code": 404
            }
        }


class PaginatedRequest(BaseModel):
    limit: int = Field(100, ge=1, le=10000, description="Maximum number of items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")
