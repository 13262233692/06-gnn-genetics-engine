from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np
import torch
from .gnn_model import HeterogeneousGNN
from .graph_converter import GraphConverter, PyGGraphData
from .trainer import GNNTrainer
from database import GraphOperations

logger = logging.getLogger(__name__)


@dataclass
class TargetSNP:
    snp_id: str
    rs_id: Optional[str]
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    confidence: float
    probability: float
    probability_distribution: Dict[str, float]
    associated_genes: List[str]
    go_terms: List[Dict[str, str]]
    p_value: Optional[float]
    odds_ratio: Optional[float]
    variant_type: Optional[str]
    functional_impact: Optional[str]
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snp_id": self.snp_id,
            "rs_id": self.rs_id,
            "chromosome": self.chromosome,
            "position": self.position,
            "ref_allele": self.ref_allele,
            "alt_allele": self.alt_allele,
            "confidence": float(self.confidence),
            "probability": float(self.probability),
            "probability_distribution": {
                k: float(v) for k, v in self.probability_distribution.items()
            },
            "associated_genes": self.associated_genes,
            "go_terms": self.go_terms,
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "odds_ratio": float(self.odds_ratio) if self.odds_ratio is not None else None,
            "variant_type": self.variant_type,
            "functional_impact": self.functional_impact,
            "rank": self.rank
        }


@dataclass
class PredictionResult:
    phenotype_name: str
    target_snps: List[TargetSNP]
    total_snps_analyzed: int
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0.0"
    graph_subset_size: int = 0
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phenotype_name": self.phenotype_name,
            "target_snps": [snp.to_dict() for snp in self.target_snps],
            "total_snps_analyzed": self.total_snps_analyzed,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "model_version": self.model_version,
            "graph_subset_size": self.graph_subset_size,
            "inference_time_ms": float(self.inference_time_ms),
            "summary": {
                "top_snps_count": len(self.target_snps),
                "average_confidence": float(np.mean([s.confidence for s in self.target_snps])) if self.target_snps else 0.0,
                "max_confidence": float(max([s.confidence for s in self.target_snps])) if self.target_snps else 0.0,
                "min_confidence": float(min([s.confidence for s in self.target_snps])) if self.target_snps else 0.0
            }
        }


class SNPredictor:
    def __init__(
        self,
        trainer: GNNTrainer,
        graph_ops: GraphOperations,
        graph_converter: Optional[GraphConverter] = None
    ):
        self.trainer = trainer
        self.graph_ops = graph_ops
        self.graph_converter = graph_converter or GraphConverter()
        self.device = trainer.device

        if not self.trainer.is_trained:
            logger.warning("Model not trained. Attempting to load...")
            self.trainer.load_model()

    async def predict(
        self,
        phenotype_name: str,
        min_p_value: float = 1e-5,
        top_k: int = 50,
        max_depth: int = 3
    ) -> PredictionResult:
        logger.info(f"Predicting target SNPs for phenotype: {phenotype_name}")
        start_time = datetime.now()

        snp_associations = await self.graph_ops.get_snps_associated_with_phenotype(
            phenotype_name=phenotype_name,
            min_p_value=min_p_value,
            limit=10000
        )

        if not snp_associations:
            logger.warning(f"No SNPs found associated with phenotype: {phenotype_name}")
            return PredictionResult(
                phenotype_name=phenotype_name,
                target_snps=[],
                total_snps_analyzed=0,
                inference_time_ms=0.0
            )

        subgraph_data = await self.graph_ops.get_gene_snp_phenotype_subgraph(
            phenotype_name=phenotype_name,
            min_p_value=min_p_value
        )

        phenotype_id = None
        for node in subgraph_data["nodes"]:
            if node.get("label") == "Phenotype" and node.get("name") == phenotype_name:
                phenotype_id = node["id"]
                break

        if phenotype_id is None:
            phenotype_id = f"PHENO:{phenotype_name.replace(' ', '_')}"

        pyg_data = self.graph_converter.convert_to_pyg(
            nodes=subgraph_data["nodes"],
            edges=subgraph_data["edges"],
            target_phenotype_id=phenotype_id
        )

        probabilities = self._run_inference(pyg_data)

        snp_mapping = pyg_data.node_mapping.get("SNP", {})
        snp_details = await self._get_snp_details(list(snp_mapping.keys()))

        target_snps = []
        for snp_id, idx in snp_mapping.items():
            if idx < len(probabilities):
                prob_positive = probabilities[idx, 1].item()
                prob_negative = probabilities[idx, 0].item()

                detail = snp_details.get(snp_id, {})
                association = next(
                    (a for a in snp_associations if a["snp"]["id"] == snp_id),
                    None
                )

                if association:
                    detail = {**detail, **association["snp"]}

                go_terms = await self._get_go_terms_for_snp(snp_id, subgraph_data["edges"])
                associated_genes = self._get_associated_genes(snp_id, subgraph_data["edges"])

                target_snp = TargetSNP(
                    snp_id=snp_id,
                    rs_id=detail.get("rs_id"),
                    chromosome=str(detail.get("chromosome", "")),
                    position=int(detail.get("position", 0)),
                    ref_allele=detail.get("ref_allele", ""),
                    alt_allele=detail.get("alt_allele", ""),
                    confidence=prob_positive,
                    probability=prob_positive,
                    probability_distribution={
                        "positive": prob_positive,
                        "negative": prob_negative
                    },
                    associated_genes=associated_genes,
                    go_terms=go_terms,
                    p_value=association["association"].get("p_value") if association else None,
                    odds_ratio=association["association"].get("odds_ratio") if association else None,
                    variant_type=detail.get("variant_type"),
                    functional_impact=detail.get("functional_impact"),
                    rank=0
                )
                target_snps.append(target_snp)

        target_snps.sort(key=lambda x: x.confidence, reverse=True)
        target_snps = target_snps[:top_k]

        for i, snp in enumerate(target_snps):
            snp.rank = i + 1

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        result = PredictionResult(
            phenotype_name=phenotype_name,
            target_snps=target_snps,
            total_snps_analyzed=len(snp_mapping),
            graph_subset_size=len(subgraph_data["nodes"]),
            inference_time_ms=inference_time
        )

        logger.info(f"Prediction completed for {phenotype_name}: "
                    f"found {len(target_snps)} target SNPs in {inference_time:.2f}ms")

        return result

    def _run_inference(self, pyg_data: PyGGraphData) -> torch.Tensor:
        self.trainer.model.eval()
        data = pyg_data.data.to(self.device)

        with torch.no_grad():
            probabilities = self.trainer.model.predict_snp_importance(data)

        return probabilities

    async def _get_snp_details(self, snp_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        details = {}
        query = """
        MATCH (s:SNP)
        WHERE s.id IN $snp_ids
        RETURN s
        """
        results = await self.graph_ops.driver.execute_query(
            query, {"snp_ids": snp_ids}
        )
        for record in results:
            snp = dict(record["s"])
            details[snp["id"]] = snp
        return details

    async def _get_go_terms_for_snp(
        self,
        snp_id: str,
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        gene_ids = []
        for edge in edges:
            if edge.get("target") == snp_id and edge.get("type") == "CONTAINS_SNP":
                gene_ids.append(edge.get("source"))

        go_terms = []
        for gene_id in gene_ids:
            for edge in edges:
                if edge.get("source") == gene_id and edge.get("type") == "ANNOTATED_TO":
                    go_id = edge.get("target")
                    go_terms.append({"go_id": go_id, "gene_id": gene_id})

        return go_terms

    def _get_associated_genes(
        self,
        snp_id: str,
        edges: List[Dict[str, Any]]
    ) -> List[str]:
        gene_ids = []
        for edge in edges:
            if edge.get("target") == snp_id and edge.get("type") == "CONTAINS_SNP":
                gene_ids.append(edge.get("source"))
        return list(set(gene_ids))

    async def predict_batch(
        self,
        phenotype_names: List[str],
        min_p_value: float = 1e-5,
        top_k: int = 50
    ) -> List[PredictionResult]:
        results = []
        for phenotype_name in phenotype_names:
            try:
                result = await self.predict(
                    phenotype_name=phenotype_name,
                    min_p_value=min_p_value,
                    top_k=top_k
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for {phenotype_name}: {str(e)}")
        return results

    async def predict_with_uncertainty(
        self,
        phenotype_name: str,
        num_samples: int = 10,
        min_p_value: float = 1e-5,
        top_k: int = 50
    ) -> Dict[str, Any]:
        logger.info(f"Predicting with uncertainty for phenotype: {phenotype_name}")

        base_result = await self.predict(
            phenotype_name=phenotype_name,
            min_p_value=min_p_value,
            top_k=top_k
        )

        self.trainer.model.train()

        all_predictions = []
        for _ in range(num_samples):
            result = await self.predict(
                phenotype_name=phenotype_name,
                min_p_value=min_p_value,
                top_k=top_k
            )
            all_predictions.append(result)

        self.trainer.model.eval()

        snp_uncertainties = {}
        for target_snp in base_result.target_snps:
            confidences = []
            for pred in all_predictions:
                for s in pred.target_snps:
                    if s.snp_id == target_snp.snp_id:
                        confidences.append(s.confidence)
                        break

            if confidences:
                snp_uncertainties[target_snp.snp_id] = {
                    "mean_confidence": float(np.mean(confidences)),
                    "std_confidence": float(np.std(confidences)),
                    "variance": float(np.var(confidences)),
                    "ci_lower": float(np.percentile(confidences, 2.5)),
                    "ci_upper": float(np.percentile(confidences, 97.5)),
                    "samples": num_samples
                }

        return {
            "base_prediction": base_result.to_dict(),
            "uncertainty_estimates": snp_uncertainties,
            "monte_carlo_samples": num_samples
        }

    def get_snp_network_centrality(
        self,
        pyg_data: PyGGraphData
    ) -> Dict[str, Dict[str, float]]:
        data = pyg_data.data
        centrality_scores = {}

        if 'SNP' not in data or 'edge_index' not in data.edge_types:
            return centrality_scores

        snp_mapping = pyg_data.node_mapping.get("SNP", {})
        num_snps = len(snp_mapping)

        if num_snps == 0:
            return centrality_scores

        edge_index = None
        for et in data.edge_types:
            if 'SNP' in et[0] or 'SNP' in et[2]:
                edge_index = data[et].edge_index
                break

        if edge_index is None:
            return centrality_scores

        degree = torch.zeros(num_snps, device=self.device)
        src, dst = edge_index
        unique_src, counts_src = torch.unique(src, return_counts=True)
        unique_dst, counts_dst = torch.unique(dst, return_counts=True)

        for i, node_idx in enumerate(unique_src):
            if node_idx < num_snps:
                degree[node_idx] += counts_src[i]
        for i, node_idx in enumerate(unique_dst):
            if node_idx < num_snps:
                degree[node_idx] += counts_dst[i]

        for snp_id, idx in snp_mapping.items():
            if idx < len(degree):
                deg_centrality = degree[idx].item() / max(num_snps - 1, 1)
                centrality_scores[snp_id] = {
                    "degree_centrality": float(deg_centrality),
                    "raw_degree": int(degree[idx].item())
                }

        return centrality_scores
