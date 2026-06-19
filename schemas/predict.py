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

    class Config:
        json_schema_extra = {
            "example": {
                "phenotype_name": "抗旱性",
                "min_p_value": 1e-5,
                "top_k": 50,
                "max_depth": 3,
                "include_uncertainty": False,
                "uncertainty_samples": 10
            }
        }


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


class PredictionSummary(BaseModel):
    top_snps_count: int = Field(..., description="返回的靶点SNP数量")
    average_confidence: float = Field(..., description="平均置信度")
    max_confidence: float = Field(..., description="最高置信度")
    min_confidence: float = Field(..., description="最低置信度")


class PredictionResponse(BaseModel):
    phenotype_name: str = Field(..., description="查询的表型名称")
    target_snps: List[TargetSNPResponse] = Field(..., description="预测的靶点SNP列表")
    total_snps_analyzed: int = Field(..., description="分析的SNP总数")
    prediction_timestamp: datetime = Field(..., description="预测时间戳")
    model_version: str = Field(..., description="GNN模型版本")
    graph_subset_size: int = Field(..., description="子图节点数量")
    inference_time_ms: float = Field(..., description="推理耗时（毫秒）")
    summary: PredictionSummary = Field(..., description="预测结果摘要")


class BatchPredictionRequest(BaseModel):
    phenotype_names: List[str] = Field(
        ...,
        description="表型名称列表",
        examples=[["抗旱性", "抗病性", "产量"]]
    )
    min_p_value: float = Field(1e-5, ge=1e-30, le=1.0)
    top_k: int = Field(50, ge=1, le=1000)


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
