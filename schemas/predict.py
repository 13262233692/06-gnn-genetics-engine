from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    phenotype_name: str = Field(
        ...,
        description="目标表型名称，如 '抗旱性'、'抗病性' 等",
        examples=["抗旱性", "抗倒伏性", "产量"]
    )
    min_p_value: float = Field(
        1e-5,
        ge=1e-30,
        le=1.0,
        description="SNP关联显著性P值阈值，越小越严格"
    )
    top_k: int = Field(
        50,
        ge=1,
        le=1000,
        description="返回的靶点SNP数量"
    )
    max_depth: int = Field(
        3,
        ge=1,
        le=10,
        description="子图遍历最大深度"
    )
    include_uncertainty: bool = Field(
        False,
        description="是否包含不确定性估计（Monte Carlo采样）"
    )
    uncertainty_samples: int = Field(
        10,
        ge=1,
        le=100,
        description="不确定性估计的采样次数"
    )
    include_explanation: bool = Field(
        True,
        description="是否包含基于注意力流的可解释性路径分析"
    )
    explanation_top_paths: int = Field(
        5,
        ge=1,
        le=20,
        description="每个SNP返回的Top-K关键贡献路径数量"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "phenotype_name": "抗旱性",
                "min_p_value": 1e-5,
                "top_k": 50,
                "max_depth": 3,
                "include_uncertainty": False,
                "uncertainty_samples": 10,
                "include_explanation": True,
                "explanation_top_paths": 5
            }
        }


class ExplanationPathNodeResponse(BaseModel):
    node_id: str = Field(..., description="图谱节点ID")
    node_type: str = Field(..., description="节点类型 (Gene/SNP/Phenotype/GOTerm/Environment等)")
    contribution: float = Field(..., description="该节点在当前路径中的贡献度(归一化0-1)")
    layer: int = Field(..., description="GNN网络层号(从输出层反向追溯)")


class ExplanationPathEdgeResponse(BaseModel):
    source_node_id: str = Field(..., description="源节点ID")
    target_node_id: str = Field(..., description="目标节点ID")
    edge_type: str = Field(..., description="关系类型 (CONTAINS_SNP/ASSOCIATED_WITH/ANNOTATED_TO等)")
    attention_weight: float = Field(..., description="GAT注意力权重(0-1)")
    layer: int = Field(..., description="所在GNN网络层号")


class ExplanationPathResponse(BaseModel):
    path_id: str = Field(..., description="路径唯一标识")
    rank: int = Field(..., description="路径贡献度排名")
    total_flow: float = Field(..., description="该路径从靶点SNP到该路径末端节点的总注意力流通量")
    nodes: List[ExplanationPathNodeResponse] = Field(
        ...,
        description="路径上的节点列表(从SNP出发向源方向回溯)"
    )
    edges: List[ExplanationPathEdgeResponse] = Field(
        ...,
        description="路径上的关系列表"
    )
    path_description: str = Field(
        ...,
        description="人类可读的路径描述，如 'SNP:rs123 ← Gene:AT1G01010 ← GOTerm:GO:0000001'"
    )


class SNPExplanationResponse(BaseModel):
    snp_id: str = Field(..., description="SNP唯一标识符")
    total_contribution: float = Field(
        ...,
        description="该SNP在所有Top路径中的累计注意力流通量"
    )
    top_paths: List[ExplanationPathResponse] = Field(
        ...,
        description="对预测结果贡献最大的Top-K条真实图谱关系路径"
    )
    node_importance: Dict[str, float] = Field(
        ...,
        description="各图谱节点对该SNP预测的相对重要性字典 {node_id: score}"
    )
    edge_importance: Dict[str, float] = Field(
        ...,
        description="各图谱关系对该SNP预测的相对重要性字典 {edge_key: score}"
    )


class TargetSNPResponse(BaseModel):
    snp_id: str = Field(..., description="SNP唯一标识符")
    rs_id: Optional[str] = Field(None, description="dbSNP rs编号")
    chromosome: str = Field(..., description="染色体编号")
    position: int = Field(..., description="染色体位置")
    ref_allele: str = Field(..., description="参考等位基因")
    alt_allele: str = Field(..., description="变异等位基因")
    confidence: float = Field(..., ge=0.0, le=1.0, description="GNN预测置信度")
    probability: float = Field(..., ge=0.0, le=1.0, description="正向关联概率")
    probability_distribution: Dict[str, float] = Field(
        ...,
        description="完整概率分布 {'positive': ..., 'negative': ...}"
    )
    associated_genes: List[str] = Field(
        ...,
        description="关联的基因ID列表"
    )
    go_terms: List[Dict[str, str]] = Field(
        ...,
        description="关联的GO术语列表"
    )
    p_value: Optional[float] = Field(None, description="GWAS关联P值")
    odds_ratio: Optional[float] = Field(None, description="优势比")
    variant_type: Optional[str] = Field(None, description="变异类型")
    functional_impact: Optional[str] = Field(None, description="功能影响预测")
    rank: int = Field(..., description="预测排名")
    explanation: Optional[SNPExplanationResponse] = Field(
        None,
        description="该SNP的可解释性分析结果(当include_explanation=True时返回)"
    )


class PredictionSummary(BaseModel):
    top_snps_count: int = Field(..., description="返回的靶点SNP数量")
    average_confidence: float = Field(..., description="平均置信度")
    max_confidence: float = Field(..., description="最高置信度")
    min_confidence: float = Field(..., description="最低置信度")
    explanations_included: bool = Field(..., description="是否包含可解释性分析")


class PredictionResponse(BaseModel):
    phenotype_name: str = Field(..., description="查询的表型名称")
    target_snps: List[TargetSNPResponse] = Field(..., description="预测的靶点SNP列表")
    total_snps_analyzed: int = Field(..., description="分析的SNP总数")
    prediction_timestamp: datetime = Field(..., description="预测时间戳")
    model_version: str = Field(..., description="GNN模型版本")
    graph_subset_size: int = Field(..., description="子图节点数量")
    inference_time_ms: float = Field(..., description="推理耗时（毫秒）")
    sampling_used: bool = Field(..., description="是否使用了NeighborLoader采样推理")
    explanation_enabled: bool = Field(..., description="可解释性分析是否启用")
    summary: PredictionSummary = Field(..., description="预测结果摘要")


class BatchPredictionRequest(BaseModel):
    phenotype_names: List[str] = Field(
        ...,
        description="表型名称列表",
        examples=[["抗旱性", "抗病性", "产量"]]
    )
    min_p_value: float = Field(1e-5, ge=1e-30, le=1.0)
    top_k: int = Field(50, ge=1, le=1000)
    include_explanation: bool = Field(True, description="是否包含可解释性路径分析")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="各表型的预测结果列表")


class UncertaintyEstimate(BaseModel):
    mean_confidence: float = Field(..., description="平均置信度")
    std_confidence: float = Field(..., description="置信度标准差")
    variance: float = Field(..., description="方差")
    ci_lower: float = Field(..., description="95%置信区间下限")
    ci_upper: float = Field(..., description="95%置信区间上限")
    samples: int = Field(..., description="采样次数")


class UncertaintyPredictionResponse(BaseModel):
    base_prediction: PredictionResponse = Field(..., description="基础预测结果")
    uncertainty_estimates: Dict[str, UncertaintyEstimate] = Field(
        ...,
        description="各SNP的不确定性估计"
    )
    monte_carlo_samples: int = Field(..., description="Monte Carlo采样次数")
