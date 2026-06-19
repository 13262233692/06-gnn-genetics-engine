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
    UncertaintyPredictionResponse,
    ExplanationPathResponse,
    ExplanationPathNodeResponse,
    ExplanationPathEdgeResponse,
    SNPExplanationResponse
)
from ml import SNPredictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/predict", tags=["靶点预测"])

snp_predictor: SNPredictor = None


def set_predictor(predictor: SNPredictor) -> None:
    global snp_predictor
    snp_predictor = predictor


def _build_explanation(snp_dict: dict) -> SNPExplanationResponse:
    explanation_raw = snp_dict.get("explanation")
    if explanation_raw is None:
        return None

    paths = []
    for p in explanation_raw.get("top_paths", []):
        nodes = [ExplanationPathNodeResponse(**n) for n in p.get("nodes", [])]
        edges = [ExplanationPathEdgeResponse(**e) for e in p.get("edges", [])]
        paths.append(ExplanationPathResponse(
            path_id=p.get("path_id", ""),
            rank=p.get("rank", 0),
            total_flow=p.get("total_flow", 0.0),
            nodes=nodes,
            edges=edges,
            path_description=p.get("path_description", "")
        ))

    return SNPExplanationResponse(
        snp_id=explanation_raw.get("snp_id", snp_dict.get("snp_id", "")),
        total_contribution=explanation_raw.get("total_contribution", 0.0),
        top_paths=paths,
        node_importance=explanation_raw.get("node_importance", {}),
        edge_importance=explanation_raw.get("edge_importance", {})
    )


def _build_prediction_response(result) -> PredictionResponse:
    result_dict = result.to_dict() if hasattr(result, 'to_dict') else result

    target_snps = []
    for snp in result_dict.get("target_snps", []):
        explanation = _build_explanation(snp)
        snp_payload = {k: v for k, v in snp.items() if k != "explanation"}
        snp_payload["explanation"] = explanation
        target_snps.append(TargetSNPResponse(**snp_payload))

    summary_data = result_dict.get("summary", {})
    summary = PredictionSummary(
        top_snps_count=summary_data.get("top_snps_count", len(target_snps)),
        average_confidence=summary_data.get("average_confidence", 0.0),
        max_confidence=summary_data.get("max_confidence", 0.0),
        min_confidence=summary_data.get("min_confidence", 0.0),
        explanations_included=summary_data.get("explanations_included", False)
    )

    return PredictionResponse(
        phenotype_name=result_dict.get("phenotype_name", ""),
        target_snps=target_snps,
        total_snps_analyzed=result_dict.get("total_snps_analyzed", 0),
        prediction_timestamp=result_dict.get("prediction_timestamp", datetime.now()),
        model_version=result_dict.get("model_version", "1.0.0"),
        graph_subset_size=result_dict.get("graph_subset_size", 0),
        inference_time_ms=result_dict.get("inference_time_ms", 0.0),
        sampling_used=result_dict.get("sampling_used", False),
        explanation_enabled=result_dict.get("explanation_enabled", False),
        summary=summary
    )


@router.post(
    "",
    response_model=PredictionResponse,
    summary="预测靶点SNP",
    description="""
    根据目标表型特征（如抗旱性、抗病性），
    在子图空间中进行GNN聚合推演，预测高权重靶点SNP组合。
    输出包含每个靶点的置信概率分布以及可解释性路径分析。
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
                top_k=request.top_k,
                include_explanation=request.include_explanation,
                explanation_top_paths=request.explanation_top_paths
            )
            base_result = result["base_prediction"]
        else:
            base_result = await snp_predictor.predict(
                phenotype_name=request.phenotype_name,
                min_p_value=request.min_p_value,
                top_k=request.top_k,
                max_depth=request.max_depth,
                include_explanation=request.include_explanation,
                explanation_top_paths=request.explanation_top_paths
            )

        return _build_prediction_response(base_result)

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
    description="对多个表型进行批量SNP靶点预测（支持可解释性分析）"
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
            top_k=request.top_k,
            include_explanation=request.include_explanation
        )

        response_predictions = [
            _build_prediction_response(result)
            for result in results
        ]

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
    description="专门针对抗旱性表型的SNP靶点预测（默认包含可解释性分析）"
)
async def predict_drought_resistance(
    top_k: int = Query(50, ge=1, le=1000, description="返回的靶点数量"),
    min_p_value: float = Query(1e-5, ge=1e-30, le=1.0, description="P值阈值"),
    include_explanation: bool = Query(True, description="是否包含可解释性路径分析"),
    explanation_top_paths: int = Query(5, ge=1, le=20, description="每个SNP返回的关键路径数量")
) -> PredictionResponse:
    request = PredictionRequest(
        phenotype_name="抗旱性",
        min_p_value=min_p_value,
        top_k=top_k,
        max_depth=3,
        include_uncertainty=False,
        include_explanation=include_explanation,
        explanation_top_paths=explanation_top_paths
    )
    return await predict_target_snps(request)


@router.get(
    "/disease-resistance",
    response_model=PredictionResponse,
    summary="抗病性靶点预测",
    description="专门针对抗病性表型的SNP靶点预测（默认包含可解释性分析）"
)
async def predict_disease_resistance(
    disease_type: str = Query(..., description="病害类型，如 '锈病'、'白粉病' 等"),
    top_k: int = Query(50, ge=1, le=1000, description="返回的靶点数量"),
    min_p_value: float = Query(1e-5, ge=1e-30, le=1.0, description="P值阈值"),
    include_explanation: bool = Query(True, description="是否包含可解释性路径分析"),
    explanation_top_paths: int = Query(5, ge=1, le=20, description="每个SNP返回的关键路径数量")
) -> PredictionResponse:
    phenotype_name = f"抗{disease_type}"
    request = PredictionRequest(
        phenotype_name=phenotype_name,
        min_p_value=min_p_value,
        top_k=top_k,
        max_depth=3,
        include_uncertainty=False,
        include_explanation=include_explanation,
        explanation_top_paths=explanation_top_paths
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
            top_k=request.top_k,
            include_explanation=request.include_explanation,
            explanation_top_paths=request.explanation_top_paths
        )

        base_result = _build_prediction_response(result["base_prediction"])
        result["base_prediction"] = base_result

        return UncertaintyPredictionResponse(**result)

    except Exception as e:
        logger.error(f"Uncertainty prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Uncertainty prediction failed: {str(e)}"
        )
