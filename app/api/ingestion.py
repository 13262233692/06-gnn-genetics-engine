from fastapi import APIRouter, HTTPException, status, UploadFile, File
import logging
import os
import pandas as pd
from typing import List, Optional
from datetime import datetime
import aiofiles

from schemas import (
    IngestionRequest,
    IngestionResponse,
    SNPIngestionRequest,
    GOIngestionRequest,
    PhenotypeIngestionRequest,
    AssociationIngestionRequest
)
from data_ingestion import (
    SNPIngestor,
    GOIngestor,
    PhenotypeIngestor
)
from database import GraphOperations

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["数据导入"])

graph_ops: GraphOperations = None
snp_ingestor: SNPIngestor = None
go_ingestor: GOIngestor = None
phenotype_ingestor: PhenotypeIngestor = None


def set_dependencies(
    graph_operations: GraphOperations,
    snp: SNPIngestor,
    go: GOIngestor,
    phenotype: PhenotypeIngestor
) -> None:
    global graph_ops, snp_ingestor, go_ingestor, phenotype_ingestor
    graph_ops = graph_operations
    snp_ingestor = snp
    go_ingestor = go
    phenotype_ingestor = phenotype


async def _ensure_ingestors() -> None:
    global snp_ingestor, go_ingestor, phenotype_ingestor, graph_ops
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not initialized"
        )
    if snp_ingestor is None:
        snp_ingestor = SNPIngestor(graph_ops)
    if go_ingestor is None:
        go_ingestor = GOIngestor(graph_ops)
    if phenotype_ingestor is None:
        phenotype_ingestor = PhenotypeIngestor(graph_ops)


@router.post(
    "",
    response_model=IngestionResponse,
    summary="从文件导入数据",
    description="从指定文件路径导入SNP、GO注释、表型等数据到知识图谱"
)
async def ingest_from_file(request: IngestionRequest) -> IngestionResponse:
    await _ensure_ingestors()

    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}"
        )

    try:
        if request.data_type == "snp":
            result = await snp_ingestor.ingest(request.file_path)
        elif request.data_type == "go":
            result = await go_ingestor.ingest(request.file_path)
        elif request.data_type == "phenotype":
            result = await phenotype_ingestor.ingest(request.file_path)
        elif request.data_type == "sample":
            result = await phenotype_ingestor.ingest_samples(request.file_path)
        elif request.data_type == "environment":
            result = await phenotype_ingestor.ingest_environments(request.file_path)
        elif request.data_type == "crop":
            result = await phenotype_ingestor.ingest_crops(request.file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported data type: {request.data_type}"
            )

        return IngestionResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post(
    "/upload",
    response_model=IngestionResponse,
    summary="上传并导入文件",
    description="上传文件并导入数据到知识图谱"
)
async def ingest_uploaded_file(
    data_type: str,
    file: UploadFile = File(..., description="数据文件")
) -> IngestionResponse:
    await _ensure_ingestors()

    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"

    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        request = IngestionRequest(
            file_path=file_path,
            data_type=data_type
        )
        return await ingest_from_file(request)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"File upload ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


@router.post(
    "/snp",
    response_model=IngestionResponse,
    summary="导入SNP数据",
    description="直接从JSON数据导入SNP变异信息"
)
async def ingest_snps(request: SNPIngestionRequest) -> IngestionResponse:
    await _ensure_ingestors()

    try:
        df = pd.DataFrame(request.snp_data)
        result = await snp_ingestor.ingest_from_dataframe(df, source_name="api_upload")
        return IngestionResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"SNP ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SNP ingestion failed: {str(e)}"
        )


@router.post(
    "/snp/associations",
    response_model=IngestionResponse,
    summary="导入SNP-表型关联数据",
    description="导入GWAS分析得到的SNP与表型关联数据"
)
async def ingest_snp_associations(
    associations: List[AssociationIngestionRequest]
) -> IngestionResponse:
    await _ensure_ingestors()

    try:
        data = [{
            'snp_id': assoc.snp_id,
            'phenotype_name': assoc.phenotype_name,
            'p_value': assoc.p_value,
            'odds_ratio': assoc.odds_ratio,
            'confidence_interval': assoc.confidence_interval,
            'study_id': assoc.study_id
        } for assoc in associations]

        df = pd.DataFrame(data)
        temp_file = f"data/temp/associations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs("data/temp", exist_ok=True)
        df.to_csv(temp_file, index=False)

        result = await snp_ingestor.ingest_associations(temp_file)
        os.remove(temp_file)

        return IngestionResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"Association ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Association ingestion failed: {str(e)}"
        )


@router.post(
    "/go",
    response_model=IngestionResponse,
    summary="导入GO注释数据",
    description="导入基因本体论注释数据"
)
async def ingest_go_terms(request: GOIngestionRequest) -> IngestionResponse:
    await _ensure_ingestors()

    try:
        df = pd.DataFrame(request.go_data)
        temp_file = f"data/temp/go_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs("data/temp", exist_ok=True)
        df.to_csv(temp_file, index=False)

        result = await go_ingestor.ingest(temp_file)
        os.remove(temp_file)

        return IngestionResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"GO ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GO ingestion failed: {str(e)}"
        )


@router.post(
    "/phenotype",
    response_model=IngestionResponse,
    summary="导入表型数据",
    description="导入作物表型记录、样本、环境和作物数据"
)
async def ingest_phenotypes(request: PhenotypeIngestionRequest) -> IngestionResponse:
    await _ensure_ingestors()

    try:
        result = None

        if request.phenotype_data:
            df_pheno = pd.DataFrame(request.phenotype_data)
            temp_file = f"data/temp/pheno_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs("data/temp", exist_ok=True)
            df_pheno.to_csv(temp_file, index=False)
            result = await phenotype_ingestor.ingest(temp_file)
            os.remove(temp_file)

        if request.sample_data:
            df_sample = pd.DataFrame(request.sample_data)
            temp_file = f"data/temp/sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_sample.to_csv(temp_file, index=False)
            sample_result = await phenotype_ingestor.ingest_samples(temp_file)
            os.remove(temp_file)
            if result:
                result.nodes_created += sample_result.nodes_created
                result.relationships_created += sample_result.relationships_created

        if request.environment_data:
            df_env = pd.DataFrame(request.environment_data)
            temp_file = f"data/temp/env_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_env.to_csv(temp_file, index=False)
            env_result = await phenotype_ingestor.ingest_environments(temp_file)
            os.remove(temp_file)
            if result:
                result.nodes_created += env_result.nodes_created
                result.relationships_created += env_result.relationships_created

        if request.crop_data:
            df_crop = pd.DataFrame(request.crop_data)
            temp_file = f"data/temp/crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_crop.to_csv(temp_file, index=False)
            crop_result = await phenotype_ingestor.ingest_crops(temp_file)
            os.remove(temp_file)
            if result:
                result.nodes_created += crop_result.nodes_created
                result.relationships_created += crop_result.relationships_created

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for ingestion"
            )

        result.complete()
        return IngestionResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"Phenotype ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phenotype ingestion failed: {str(e)}"
        )


@router.post(
    "/go/ontology",
    response_model=IngestionResponse,
    summary="导入GO本体结构",
    description="从OBO格式文件导入完整的Gene Ontology层级结构"
)
async def ingest_go_ontology(file: UploadFile = File(..., description="OBO格式的GO本体文件")) -> IngestionResponse:
    await _ensure_ingestors()

    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/go_ontology_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obo"

    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        result = await go_ingestor.ingest_ontology(file_path)
        os.remove(file_path)

        return IngestionResponse(**result.to_dict())

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"GO ontology ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GO ontology ingestion failed: {str(e)}"
        )
