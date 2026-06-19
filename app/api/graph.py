from fastapi import APIRouter, HTTPException, status, Query
import logging
from typing import List, Optional
from datetime import datetime

from schemas import (
    GraphStatsResponse,
    NodeCount,
    RelationshipCount,
    NodeResponse,
    SubgraphResponse,
    SubgraphEdge,
    SNPAssociationResponse,
    PaginatedRequest
)
from database import GraphOperations

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/graph", tags=["图谱查询"])

graph_ops: GraphOperations = None


def set_dependencies(operations: GraphOperations) -> None:
    global graph_ops
    graph_ops = operations


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    summary="获取图谱统计",
    description="获取知识图谱的节点和关系统计信息"
)
async def get_graph_statistics() -> GraphStatsResponse:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        stats = await graph_ops.get_graph_statistics()

        node_counts = [
            NodeCount(label=ns["label"], count=ns["count"])
            for ns in stats.get("node_counts", [])
        ]
        relationship_counts = [
            RelationshipCount(type=rc["type"], count=rc["count"])
            for rc in stats.get("relationship_counts", [])
        ]

        total_nodes = sum(nc.count for nc in node_counts)
        total_rels = sum(rc.count for rc in relationship_counts)

        return GraphStatsResponse(
            node_counts=node_counts,
            relationship_counts=relationship_counts,
            total_nodes=total_nodes,
            total_relationships=total_rels
        )

    except Exception as e:
        logger.error(f"Failed to get graph stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph statistics: {str(e)}"
        )


@router.get(
    "/nodes/{label}",
    response_model=List[NodeResponse],
    summary="按类型查询节点",
    description="根据节点类型（如Gene、SNP、Phenotype等）分页查询节点列表"
)
async def get_nodes_by_label(
    label: str,
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0)
) -> List[NodeResponse]:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        nodes = await graph_ops.get_nodes_by_label(label, limit=limit, offset=offset)

        return [
            NodeResponse(
                id=node.get("id", ""),
                labels=[label],
                properties={k: v for k, v in node.items() if k not in ["id"]},
                created_at=node.get("created_at"),
                updated_at=node.get("updated_at")
            )
            for node in nodes
        ]

    except Exception as e:
        logger.error(f"Failed to get nodes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get nodes: {str(e)}"
        )


@router.get(
    "/node/{node_id}",
    response_model=NodeResponse,
    summary="查询单个节点",
    description="根据节点ID查询节点详情"
)
async def get_node_by_id(node_id: str) -> NodeResponse:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """
        result = await graph_ops.driver.execute_query(
            query, {"node_id": node_id}
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node not found: {node_id}"
            )

        node = dict(result[0]["n"])
        labels = result[0]["labels"]

        return NodeResponse(
            id=node.get("id", node_id),
            labels=labels,
            properties={k: v for k, v in node.items() if k not in ["id"]},
            created_at=node.get("created_at"),
            updated_at=node.get("updated_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get node: {str(e)}"
        )


@router.get(
    "/subgraph",
    response_model=SubgraphResponse,
    summary="查询表型子图",
    description="查询与特定表型相关的子图，包括基因、SNP、GO术语等"
)
async def get_phenotype_subgraph(
    phenotype_name: str,
    max_depth: int = Query(3, ge=1, le=5),
    limit: int = Query(10000, ge=100, le=100000),
    min_p_value: float = Query(1e-5, ge=1e-30, le=1.0)
) -> SubgraphResponse:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        subgraph_data = await graph_ops.get_gene_snp_phenotype_subgraph(
            phenotype_name=phenotype_name,
            min_p_value=min_p_value
        )

        nodes_dict = {}
        for node in subgraph_data["nodes"]:
            node_id = node.get("id")
            if node_id:
                nodes_dict[node_id] = NodeResponse(
                    id=node_id,
                    labels=[node.get("label", "Unknown")],
                    properties={k: v for k, v in node.items() if k not in ["id", "label"]},
                    created_at=node.get("created_at"),
                    updated_at=node.get("updated_at")
                )

        edges = []
        for edge in subgraph_data["edges"]:
            edges.append(SubgraphEdge(
                source=edge.get("source"),
                target=edge.get("target"),
                type=edge.get("type", "UNKNOWN"),
                properties={k: v for k, v in edge.items()
                            if k not in ["source", "target", "type"]}
            ))

        return SubgraphResponse(
            phenotype_name=phenotype_name,
            nodes=list(nodes_dict.values()),
            edges=edges,
            node_count=len(nodes_dict),
            edge_count=len(edges)
        )

    except Exception as e:
        logger.error(f"Failed to get subgraph: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get subgraph: {str(e)}"
        )


@router.get(
    "/snps/associated/{phenotype_name}",
    response_model=List[SNPAssociationResponse],
    summary="查询表型关联SNP",
    description="查询与特定表型显著关联的SNP列表，按P值排序"
)
async def get_snps_associated_with_phenotype(
    phenotype_name: str,
    min_p_value: float = Query(1e-5, ge=1e-30, le=1.0),
    limit: int = Query(1000, ge=1, le=10000)
) -> List[SNPAssociationResponse]:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        snp_associations = await graph_ops.get_snps_associated_with_phenotype(
            phenotype_name=phenotype_name,
            min_p_value=min_p_value,
            limit=limit
        )

        return [
            SNPAssociationResponse(
                snp=NodeResponse(
                    id=item["snp"]["id"],
                    labels=["SNP"],
                    properties={k: v for k, v in item["snp"].items() if k != "id"},
                    created_at=item["snp"].get("created_at"),
                    updated_at=item["snp"].get("updated_at")
                ),
                association=item["association"],
                rank=i + 1
            )
            for i, item in enumerate(snp_associations)
        ]

    except Exception as e:
        logger.error(f"Failed to get SNP associations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get SNP associations: {str(e)}"
        )


@router.get(
    "/search",
    summary="搜索节点",
    description="根据关键词搜索节点"
)
async def search_nodes(
    keyword: str,
    label: Optional[str] = None,
    limit: int = Query(50, ge=1, le=1000)
) -> List[NodeResponse]:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        label_filter = f":{label}" if label else ""
        query = f"""
        MATCH (n{label_filter})
        WHERE n.name CONTAINS $keyword OR n.id CONTAINS $keyword
        RETURN n, labels(n) as labels
        LIMIT $limit
        """
        result = await graph_ops.driver.execute_query(
            query, {"keyword": keyword, "limit": limit}
        )

        return [
            NodeResponse(
                id=record["n"].get("id", ""),
                labels=record["labels"],
                properties={k: v for k, v in dict(record["n"]).items() if k != "id"},
                created_at=record["n"].get("created_at"),
                updated_at=record["n"].get("updated_at")
            )
            for record in result
        ]

    except Exception as e:
        logger.error(f"Failed to search nodes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search nodes: {str(e)}"
        )


@router.get(
    "/neighbors/{node_id}",
    summary="查询节点邻居",
    description="查询指定节点的所有邻居节点和关系"
)
async def get_node_neighbors(
    node_id: str,
    relationship_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> dict:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        rel_filter = f":{relationship_type}" if relationship_type else ""
        query = f"""
        MATCH (n {{id: $node_id}})-[r{rel_filter}]->(neighbor)
        RETURN neighbor, type(r) as rel_type, r as rel_props
        LIMIT $limit
        """
        result = await graph_ops.driver.execute_query(
            query, {"node_id": node_id, "limit": limit}
        )

        neighbors = []
        for record in result:
            neighbor = dict(record["neighbor"])
            neighbors.append({
                "node": NodeResponse(
                    id=neighbor.get("id", ""),
                    labels=list(record["neighbor"].labels),
                    properties={k: v for k, v in neighbor.items() if k != "id"},
                    created_at=neighbor.get("created_at"),
                    updated_at=neighbor.get("updated_at")
                ),
                "relationship_type": record["rel_type"],
                "relationship_properties": dict(record["rel_props"])
            })

        return {
            "node_id": node_id,
            "neighbor_count": len(neighbors),
            "neighbors": neighbors
        }

    except Exception as e:
        logger.error(f"Failed to get node neighbors: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get node neighbors: {str(e)}"
        )


@router.get(
    "/phenotypes",
    summary="获取所有表型",
    description="获取数据库中所有表型列表"
)
async def get_all_phenotypes(
    trait_category: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> List[NodeResponse]:
    if graph_ops is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph operations not initialized"
        )

    try:
        where_clause = "WHERE p.trait_category = $category" if trait_category else ""
        query = f"""
        MATCH (p:Phenotype)
        {where_clause}
        RETURN p
        ORDER BY p.name
        LIMIT $limit
        """
        params = {"limit": limit}
        if trait_category:
            params["category"] = trait_category
        result = await graph_ops.driver.execute_query(query, params)

        return [
            NodeResponse(
                id=record["p"].get("id", ""),
                labels=["Phenotype"],
                properties={k: v for k, v in dict(record["p"]).items() if k != "id"},
                created_at=record["p"].get("created_at"),
                updated_at=record["p"].get("updated_at")
            )
            for record in result
        ]

    except Exception as e:
        logger.error(f"Failed to get phenotypes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get phenotypes: {str(e)}"
        )
