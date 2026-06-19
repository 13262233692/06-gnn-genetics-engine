from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class IngestionRequest(BaseModel):
    file_path: str = Field(..., description="数据文件路径")
    data_type: str = Field(
        ...,
        description="数据类型: snp, go, phenotype, sample, environment, crop",
        pattern="^(snp|go|phenotype|sample|environment|crop)$"
    )
    delimiter: str = Field(",", description="CSV文件分隔符")

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "data/snps/wheat_snps.vcf",
                "data_type": "snp",
                "delimiter": ","
            }
        }


class IngestionResponse(BaseModel):
    success: bool = Field(..., description="导入是否成功")
    source_file: str = Field(..., description="源文件路径")
    nodes_created: int = Field(..., description="创建的节点数")
    nodes_updated: int = Field(..., description="更新的节点数")
    relationships_created: int = Field(..., description="创建的关系数")
    relationships_updated: int = Field(..., description="更新的关系数")
    errors: List[str] = Field(default_factory=list, description="错误信息列表")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: datetime = Field(..., description="完成时间")
    duration_seconds: float = Field(..., description="耗时（秒）")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "source_file": "data/snps/wheat_snps.vcf",
                "nodes_created": 15000,
                "nodes_updated": 2000,
                "relationships_created": 18000,
                "relationships_updated": 1500,
                "errors": [],
                "duration_seconds": 45.5
            }
        }


class SNPIngestionRequest(BaseModel):
    snp_data: List[Dict[str, Any]] = Field(..., description="SNP数据列表")

    class Config:
        json_schema_extra = {
            "example": {
                "snp_data": [
                    {
                        "CHROM": "1A",
                        "POS": 12345,
                        "ID": "rs123456",
                        "REF": "A",
                        "ALT": "G",
                        "MAF": 0.35,
                        "INFO": "GENE=TraesCS1A01G000100;ANN=missense_variant"
                    }
                ]
            }
        }


class GOIngestionRequest(BaseModel):
    go_data: List[Dict[str, Any]] = Field(..., description="GO注释数据列表")


class PhenotypeIngestionRequest(BaseModel):
    phenotype_data: List[Dict[str, Any]] = Field(..., description="表型数据列表")
    sample_data: Optional[List[Dict[str, Any]]] = Field(None, description="样本数据列表")
    environment_data: Optional[List[Dict[str, Any]]] = Field(None, description="环境数据列表")
    crop_data: Optional[List[Dict[str, Any]]] = Field(None, description="作物数据列表")


class AssociationIngestionRequest(BaseModel):
    snp_id: str = Field(..., description="SNP ID")
    phenotype_name: str = Field(..., description="表型名称")
    p_value: float = Field(..., ge=0, le=1, description="关联P值")
    odds_ratio: Optional[float] = Field(None, description="优势比")
    confidence_interval: Optional[str] = Field(None, description="置信区间")
    study_id: Optional[str] = Field(None, description="研究ID")
