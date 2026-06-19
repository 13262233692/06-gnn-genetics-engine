from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from .neo4j_driver import Neo4jDriver
from .graph_schema import GraphSchema

logger = logging.getLogger(__name__)


class GraphOperations:
    def __init__(self):
        self.driver = Neo4jDriver()
        self.schema = GraphSchema()

    async def initialize_schema(self) -> None:
        logger.info("Initializing graph schema constraints and indexes")
        for query in self.schema.get_create_constraints_queries():
            await self.driver.execute_write(query)
        for query in self.schema.get_create_indexes_queries():
            await self.driver.execute_write(query)
        logger.info("Graph schema initialized successfully")

    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        update_if_exists: bool = True
    ) -> str:
        node_label = self.schema.get_node_label(label)
        props = properties.copy()
        node_id = props.get("id")
        if not node_id:
            raise ValueError("Node must have an 'id' property")

        if update_if_exists:
            query = f"""
            MERGE (n:{node_label} {{id: $id}})
            SET n += $props
            SET n.updated_at = $updated_at
            RETURN n.id as id
            """
        else:
            query = f"""
            CREATE (n:{node_label} $props)
            SET n.created_at = $created_at
            SET n.updated_at = $updated_at
            RETURN n.id as id
            """

        result = await self.driver.execute_query(
            query,
            {
                "id": node_id,
                "props": props,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        )
        return result[0]["id"] if result else node_id

    async def create_relationship(
        self,
        source_id: str,
        source_label: str,
        target_id: str,
        target_label: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        update_if_exists: bool = True
    ) -> None:
        src_label = self.schema.get_node_label(source_label)
        tgt_label = self.schema.get_node_label(target_label)
        relation_type = self.schema.get_rel_type(rel_type)
        props = properties or {}

        if update_if_exists:
            query = f"""
            MATCH (s:{src_label} {{id: $source_id}}), (t:{tgt_label} {{id: $target_id}})
            MERGE (s)-[r:{relation_type}]->(t)
            SET r += $props
            SET r.updated_at = $updated_at
            """
        else:
            query = f"""
            MATCH (s:{src_label} {{id: $source_id}}), (t:{tgt_label} {{id: $target_id}})
            CREATE (s)-[r:{relation_type} $props]->(t)
            SET r.created_at = $created_at
            SET r.updated_at = $updated_at
            """

        await self.driver.execute_write(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "props": props,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        )

    async def batch_create_nodes(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        node_label = self.schema.get_node_label(label)
        created_count = 0
        timestamp = datetime.now().isoformat()

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            query = f"""
            UNWIND $batch as props
            MERGE (n:{node_label} {{id: props.id}})
            SET n += props
            SET n.updated_at = '{timestamp}'
            ON CREATE SET n.created_at = '{timestamp}'
            """
            await self.driver.execute_write(query, {"batch": batch})
            created_count += len(batch)
            logger.info(f"Created/updated {created_count} {node_label} nodes")

        return created_count

    async def batch_create_relationships(
        self,
        source_label: str,
        target_label: str,
        rel_type: str,
        relationships: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        src_label = self.schema.get_node_label(source_label)
        tgt_label = self.schema.get_node_label(target_label)
        relation_type = self.schema.get_rel_type(rel_type)
        created_count = 0
        timestamp = datetime.now().isoformat()

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            query = f"""
            UNWIND $batch as rel
            MATCH (s:{src_label} {{id: rel.source_id}}), (t:{tgt_label} {{id: rel.target_id}})
            MERGE (s)-[r:{relation_type}]->(t)
            SET r += rel.properties
            SET r.updated_at = '{timestamp}'
            ON CREATE SET r.created_at = '{timestamp}'
            """
            await self.driver.execute_write(query, {"batch": batch})
            created_count += len(batch)
            logger.info(f"Created/updated {created_count} {relation_type} relationships")

        return created_count

    async def get_nodes_by_label(
        self,
        label: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        node_label = self.schema.get_node_label(label)
        query = f"""
        MATCH (n:{node_label})
        RETURN n
        ORDER BY n.id
        SKIP $offset LIMIT $limit
        """
        result = await self.driver.execute_query(
            query, {"limit": limit, "offset": offset}
        )
        return [record["n"] for record in result]

    async def get_subgraph_for_phenotype(
        self,
        phenotype_name: str,
        max_depth: int = 3,
        limit: int = 10000
    ) -> Dict[str, Any]:
        query = """
        MATCH (p:Phenotype {name: $phenotype_name})
        CALL apoc.path.subgraphAll(p, {
            maxLevel: $max_depth,
            relationshipFilter: ">",
            limit: $limit
        })
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        result = await self.driver.execute_query(
            query,
            {
                "phenotype_name": phenotype_name,
                "max_depth": max_depth,
                "limit": limit
            }
        )

        if not result:
            return {"nodes": [], "relationships": []}

        return {
            "nodes": [dict(n) for n in result[0]["nodes"]],
            "relationships": [
                {
                    "source": r.start_node.id,
                    "target": r.end_node.id,
                    "type": r.type,
                    "properties": dict(r)
                }
                for r in result[0]["relationships"]
            ]
        }

    async def get_snps_associated_with_phenotype(
        self,
        phenotype_name: str,
        min_p_value: float = 1e-5,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (s:SNP)-[r:ASSOCIATED_WITH]->(p:Phenotype {name: $phenotype_name})
        WHERE r.p_value <= $min_p_value
        RETURN s, r
        ORDER BY r.p_value ASC
        LIMIT $limit
        """
        result = await self.driver.execute_query(
            query,
            {
                "phenotype_name": phenotype_name,
                "min_p_value": min_p_value,
                "limit": limit
            }
        )
        return [
            {
                "snp": dict(record["s"]),
                "association": dict(record["r"])
            }
            for record in result
        ]

    async def get_gene_snp_phenotype_subgraph(
        self,
        phenotype_name: str,
        min_p_value: float = 1e-5
    ) -> Dict[str, Any]:
        query = """
        MATCH path = (g:Gene)-[:CONTAINS_SNP]->(s:SNP)-[a:ASSOCIATED_WITH]->(p:Phenotype {name: $phenotype_name})
        WHERE a.p_value <= $min_p_value
        OPTIONAL MATCH (g)-[:ANNOTATED_TO]->(go:GOTerm)
        RETURN g, s, a, p, collect(DISTINCT go) as go_terms
        """
        result = await self.driver.execute_query(
            query,
            {
                "phenotype_name": phenotype_name,
                "min_p_value": min_p_value
            }
        )

        nodes = {}
        edges = []

        for record in result:
            gene = dict(record["g"])
            snp = dict(record["s"])
            phenotype = dict(record["p"])
            association = dict(record["a"])
            go_terms = [dict(go) for go in record["go_terms"]]

            nodes[gene["id"]] = {**gene, "label": "Gene"}
            nodes[snp["id"]] = {**snp, "label": "SNP"}
            nodes[phenotype["id"]] = {**phenotype, "label": "Phenotype"}
            for go in go_terms:
                nodes[go["id"]] = {**go, "label": "GOTerm"}

            edges.append({
                "source": gene["id"],
                "target": snp["id"],
                "type": "CONTAINS_SNP"
            })
            edges.append({
                "source": snp["id"],
                "target": phenotype["id"],
                "type": "ASSOCIATED_WITH",
                "p_value": association.get("p_value"),
                "odds_ratio": association.get("odds_ratio")
            })
            for go in go_terms:
                edges.append({
                    "source": gene["id"],
                    "target": go["id"],
                    "type": "ANNOTATED_TO"
                })

        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }

    async def get_new_nodes_since(
        self,
        timestamp: str,
        labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        label_filter = " OR ".join([f"n:{l}" for l in labels]) if labels else ""
        where_clause = f"WHERE ({label_filter}) AND " if label_filter else "WHERE "
        where_clause += "(n.created_at >= $timestamp OR n.updated_at >= $timestamp)"

        query = f"""
        MATCH (n)
        {where_clause}
        RETURN DISTINCT labels(n) as labels, n
        ORDER BY n.id
        """
        result = await self.driver.execute_query(query, {"timestamp": timestamp})
        return [
            {
                "labels": record["labels"],
                "properties": dict(record["n"])
            }
            for record in result
        ]

    async def get_new_relationships_since(
        self,
        timestamp: str,
        rel_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        type_filter = " OR ".join([f"type(r) = '{t}'" for t in rel_types]) if rel_types else ""
        where_clause = f"WHERE ({type_filter}) AND " if type_filter else "WHERE "
        where_clause += "(r.created_at >= $timestamp OR r.updated_at >= $timestamp)"

        query = f"""
        MATCH (s)-[r]->(t)
        {where_clause}
        RETURN DISTINCT s, type(r) as rel_type, t, r
        """
        result = await self.driver.execute_query(query, {"timestamp": timestamp})
        return [
            {
                "source": dict(record["s"]),
                "source_labels": list(record["s"].labels),
                "rel_type": record["rel_type"],
                "target": dict(record["t"]),
                "target_labels": list(record["t"].labels),
                "properties": dict(record["r"])
            }
            for record in result
        ]

    async def get_graph_statistics(self) -> Dict[str, Any]:
        query = """
        MATCH (n)
        WITH DISTINCT labels(n) as label, count(n) as count
        RETURN collect({label: label, count: count}) as node_stats
        """
        node_result = await self.driver.execute_query(query)

        query = """
        MATCH ()-[r]->()
        WITH DISTINCT type(r) as rel_type, count(r) as count
        RETURN collect({type: rel_type, count: count}) as rel_stats
        """
        rel_result = await self.driver.execute_query(query)

        return {
            "node_counts": node_result[0]["node_stats"] if node_result else [],
            "relationship_counts": rel_result[0]["rel_stats"] if rel_result else []
        }

    async def get_snp_gene_phenotype_edges(
        self,
        phenotype_name: Optional[str] = None
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        where_clause = "WHERE p.name = $phenotype_name" if phenotype_name else ""
        query = f"""
        MATCH (g:Gene)-[cs:CONTAINS_SNP]->(s:SNP)-[aw:ASSOCIATED_WITH]->(p:Phenotype)
        {where_clause}
        RETURN g.id as gene_id, s.id as snp_id, p.id as phenotype_id,
               cs as contains_rel, aw as assoc_rel
        """
        params = {"phenotype_name": phenotype_name} if phenotype_name else {}
        result = await self.driver.execute_query(query, params)

        edges = []
        for record in result:
            edges.append((
                record["gene_id"],
                record["snp_id"],
                "CONTAINS_SNP",
                dict(record["contains_rel"])
            ))
            edges.append((
                record["snp_id"],
                record["phenotype_id"],
                "ASSOCIATED_WITH",
                dict(record["assoc_rel"])
            ))
        return edges
