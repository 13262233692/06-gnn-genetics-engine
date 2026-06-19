from fastapi import APIRouter, HTTPException, status, Query
import logging
from typing import List
from datetime import datetime
import asyncio

from schemas import (
    PredictionRequest,
    PredictionResponse,
    TargetSNPResponse,
    PredictionSummary,
    BatchPredictionRequest,
    BatchPredictionResponse,
    UncertaintyPredictionResponse
)
from ml import SNPredictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/predict", tags=["靶点预测"])

snp_predictor: SNPredictor = None


def set_predictor(predictor: SNPredictor) -> None:
    global snp_predictor
    snp_predictor = predictor


@router.post(
    "",
    response_model=PredictionResponse,
    summary="预测靶点SNP",
    description="""
    根据目标表型特征（如抗旱性、抗病性），
    在子图空间中进行GNN聚合推演，预测高权重靶点SNP组合。
    输出包含每个靶点的置信概率分布。
    """
)
async def predict_target_snps(request: PredictionRequest) -> PredictionResponse:
    if snp_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )

    try:
        if request.include_uncertainty:
            result = await snp_predictor.predict_with_uncertainty(
                phenotype_name=request.phenotype_name,
                num_samples=request.uncertainty_samples,
                min_p_value=request.min_p_value,
                top_k=request.top_k
            )
            base_result = result["base_prediction"]
        else:
            result = await snp_predictor.predict(
                phenotype_name=request.phenotype_name,
                min_p_value=request.min_p_value,
                top_k=request.top_k,
                max_depth=request.max_depth
            )
            base_result = result

        target_snps = [
            TargetSNPResponse(**snp.to_dict())
            for snp in base_result.target_snps
        ]

        summary = PredictionSummary(
            top_snps_count=base_result.summary["top_snps_count"],
            average_confidence=base_result.summary["average_confidence"],
            max_confidence=base_result.summary["max_confidence"],
            min_confidence=base_result.summary["min_confidence"]
        )

        return PredictionResponse(
            phenotype_name=base_result.phenotype_name,
            target_snps=target_snps,
            total_snps_analyzed=base_result.total_snps_analyzed,
            prediction_timestamp=base_result.prediction_timestamp,
            model_version=base_result.model_version,
            graph_subset_size=base_result.graph_subset_size,
            inference_time_ms=base_result.inference_time_ms,
            summary=summary
        )

    except ValueError as e:
        logger.warning(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="批量预测",
    description="对多个表型进行批量SNP靶点预测"
)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if snp_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )

    try:
        results = await snp_predictor.predict_batch(
            phenotype_names=request.phenotype_names,
            min_p_value=request.min_p_value,
            top_k=request.top_k
        )

        response_predictions = []
        for result in results:
            target_snps = [
                TargetSNPResponse(**snp.to_dict())
                for snp in result.target_snps
            ]
            summary = PredictionSummary(
                top_snps_count=result.summary["top_snps_count"],
                average_confidence=result.summary["average_confidence"],
                max_confidence=result.summary["max_confidence"],
                min_confidence=result.summary["min_confidence"]
            )
            response_predictions.append(PredictionResponse(
                phenotype_name=result.phenotype_name,
                target_snps=target_snps,
                total_snps_analyzed=result.total_snps_analyzed,
                prediction_timestamp=result.prediction_timestamp,
                model_version=result.model_version,
                graph_subset_size=result.graph_subset_size,
                inference_time_ms=result.inference_time_ms,
                summary=summary
            ))

        return BatchPredictionResponse(predictions=response_predictions)

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/drought",
    response_model=PredictionResponse,
    summary="抗旱性靶点预测",
    description="专门针对抗旱性表型的SNP靶点预测"
)
async def predict_drought_resistance(
    top_k: int = Query(50, ge=1, le=1000, description="返回的靶点数量"),
    min_p_value: float = Query(1e-5, ge=1e-30, le=1.0, description="P值阈值")
) -> PredictionResponse:
    request = PredictionRequest(
        phenotype_name="抗旱性",
        min_p_value=min_p_value,
        top_k=top_k,
        max_depth=3,
        include_uncertainty=False
    )
    return await predict_target_snps(request)


@router.get(
    "/disease-resistance",
    response_model=PredictionResponse,
    summary="抗病性靶点预测",
    description="专门针对抗病性表型的SNP靶点预测"
)
async def predict_disease_resistance(
    disease_type: str = Query(..., description="病害类型，如 '锈病'、'白粉病' 等"),
    top_k: int = Query(50, ge=1, le=1000, description="返回的靶点数量"),
    min_p_value: float = Query(1e-5, ge=1e-30, le=1.0, description="P值阈值")
) -> PredictionResponse:
    phenotype_name = f"抗{disease_type}"
    request = PredictionRequest(
        phenotype_name=phenotype_name,
        min_p_value=min_p_value,
        top_k=top_k,
        max_depth=3,
        include_uncertainty=False
    )
    return await predict_target_snps(request)


@router.post(
    "/uncertainty",
    response_model=UncertaintyPredictionResponse,
    summary="带不确定性估计的预测",
    description="使用蒙特卡洛Dropout进行不确定性估计的SNP靶点预测"
)
async def predict_with_uncertainty(
    request: PredictionRequest
) -> UncertaintyPredictionResponse:
    if snp_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized"
        )

    try:
        result = await snp_predictor.predict_with_uncertainty(
            phenotype_name=request.phenotype_name,
            num_samples=request.uncertainty_samples,
            min_p_value=request.min_p_value,
            top_k=request.top_k
        )

        return UncertaintyPredictionResponse(**result)

    except Exception as e:
        logger.error(f"Uncertainty prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Uncertainty prediction failed: {str(e)}"
        )
