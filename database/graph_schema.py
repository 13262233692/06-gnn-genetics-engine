from typing import Dict, List, Any


class GraphSchema:
    NODE_LABELS = {
        "GENE": "Gene",
        "SNP": "SNP",
        "GO_TERM": "GOTerm",
        "PHENOTYPE": "Phenotype",
        "ENVIRONMENT": "Environment",
        "CROP": "Crop",
        "SAMPLE": "Sample",
        "PATHWAY": "Pathway"
    }

    RELATIONSHIP_TYPES = {
        "CONTAINS_SNP": "CONTAINS_SNP",
        "ASSOCIATED_WITH": "ASSOCIATED_WITH",
        "ANNOTATED_TO": "ANNOTATED_TO",
        "EXPRESSES_IN": "EXPRESSES_IN",
        "INTERACTS_WITH": "INTERACTS_WITH",
        "CORRELATES_WITH": "CORRELATES_WITH",
        "INFLUENCES": "INFLUENCES",
        "BELONGS_TO": "BELONGS_TO",
        "PARTICIPATES_IN": "PARTICIPATES_IN",
        "PARENT_OF": "PARENT_OF"
    }

    NODE_PROPERTIES: Dict[str, List[str]] = {
        "Gene": [
            "id", "name", "chromosome", "start_position", "end_position",
            "strand", "description", "organism", "gene_type"
        ],
        "SNP": [
            "id", "rs_id", "chromosome", "position", "ref_allele",
            "alt_allele", "maf", "variant_type", "functional_impact"
        ],
        "GOTerm": [
            "id", "name", "namespace", "definition", "ontology"
        ],
        "Phenotype": [
            "id", "name", "description", "trait_category",
            "measurement_unit", "species"
        ],
        "Environment": [
            "id", "name", "description", "location", "temperature",
            "precipitation", "soil_type", "elevation"
        ],
        "Crop": [
            "id", "name", "species", "cultivar", "breeding_line"
        ],
        "Sample": [
            "id", "sample_id", "collection_date", "genotype_id",
            "phenotype_values", "quality_score"
        ],
        "Pathway": [
            "id", "name", "description", "pathway_source"
        ]
    }

    RELATIONSHIP_PROPERTIES: Dict[str, List[str]] = {
        "CONTAINS_SNP": ["distance", "region_type"],
        "ASSOCIATED_WITH": ["p_value", "odds_ratio", "confidence_interval", "study_id"],
        "ANNOTATED_TO": ["evidence_code", "annotation_date"],
        "EXPRESSES_IN": ["expression_level", "tissue", "condition"],
        "INTERACTS_WITH": ["interaction_type", "confidence_score"],
        "CORRELATES_WITH": ["correlation_coefficient", "p_value", "sample_size"],
        "INFLUENCES": ["effect_size", "direction", "significance"],
        "BELONGS_TO": ["taxonomic_level"],
        "PARTICIPATES_IN": ["role", "evidence"],
        "PARENT_OF": ["relationship_type"]
    }

    @classmethod
    def get_create_constraints_queries(cls) -> List[str]:
        queries = []
        for label in cls.NODE_LABELS.values():
            queries.append(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                f"REQUIRE n.id IS UNIQUE"
            )
        return queries

    @classmethod
    def get_create_indexes_queries(cls) -> List[str]:
        index_definitions = [
            ("Gene", "name"),
            ("Gene", "chromosome"),
            ("SNP", "rs_id"),
            ("SNP", "chromosome"),
            ("SNP", "position"),
            ("GOTerm", "namespace"),
            ("Phenotype", "trait_category"),
            ("Phenotype", "species"),
            ("Environment", "location"),
            ("Crop", "species"),
            ("Sample", "genotype_id"),
            ("ASSOCIATED_WITH", "p_value"),
            ("CORRELATES_WITH", "correlation_coefficient")
        ]
        queries = []
        for i, (label, prop) in enumerate(index_definitions):
            if "_" in label:
                queries.append(
                    f"CREATE INDEX IF NOT EXISTS FOR ()-[r:{label}]-() "
                    f"ON (r.{prop})"
                )
            else:
                queries.append(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) "
                    f"ON (n.{prop})"
                )
        return queries

    @classmethod
    def get_node_label(cls, node_type: str) -> str:
        return cls.NODE_LABELS.get(node_type.upper(), node_type)

    @classmethod
    def get_rel_type(cls, rel_type: str) -> str:
        return cls.RELATIONSHIP_TYPES.get(rel_type.upper(), rel_type)
