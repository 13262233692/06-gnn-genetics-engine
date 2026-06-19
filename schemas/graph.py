from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class NodeCount(BaseModel):
    label: List[str] = Field(..., description="节点标签")
    count: int = Field(..., description="节点数量")


class RelationshipCount(BaseModel):
    type: str = Field(..., description="关系类型")
    count: int = Field(..., description="关系数量")


class GraphStatsResponse(BaseModel):
    node_counts: List[NodeCount] = Field(..., description="各类型节点计数")
    relationship_counts: List[RelationshipCount] = Field(..., description="各类型关系计数")
    total_nodes: int = Field(..., description="节点总数")
    total_relationships: int = Field(..., description="关系总数")
    last_updated: datetime = Field(default_factory=datetime.now, description="统计时间")

    class Config:
        json_schema_extra = {
            "example": {
                "node_counts": [
                    {"label": ["Gene"], "count": 12000},
                    {"label": ["SNP"], "count": 500000},
                    {"label": ["GOTerm"], "count": 15000},
                    {"label": ["Phenotype"], "count": 250}
                ],
                "relationship_counts": [
                    {"type": "CONTAINS_SNP", "count": 480000},
                    {"type": "ASSOCIATED_WITH", "count": 15000},
                    {"type": "ANNOTATED_TO", "count": 45000}
                ],
                "total_nodes": 527250,
                "total_relationships": 540000
            }
        }


class NodeResponse(BaseModel):
    id: str = Field(..., description="节点ID")
    labels: List[str] = Field(..., description="节点标签")
    properties: Dict[str, Any] = Field(..., description="节点属性")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")


class SubgraphRequest(BaseModel):
    phenotype_name: str = Field(..., description="表型名称")
    max_depth: int = Field(3, ge=1, le=5, description="子图深度")
    limit: int = Field(10000, ge=100, le=100000, description="节点数量限制")
    min_p_value: float = Field(1e-5, ge=1e-30, le=1.0, description="关联P值阈值")


class SubgraphEdge(BaseModel):
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    type: str = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")


class SubgraphResponse(BaseModel):
    phenotype_name: str = Field(..., description="表型名称")
    nodes: List[NodeResponse] = Field(..., description="节点列表")
    edges: List[SubgraphEdge] = Field(..., description="边列表")
    node_count: int = Field(..., description="节点总数")
    edge_count: int = Field(..., description="边总数")


class SNPAssociationResponse(BaseModel):
    snp: NodeResponse = Field(..., description="SNP节点")
    association: Dict[str, Any] = Field(..., description="关联属性")
    rank: int = Field(..., description="按P值排序的排名")
