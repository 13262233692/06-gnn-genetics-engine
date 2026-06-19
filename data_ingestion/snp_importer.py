from typing import List, Dict, Any
import logging
import numpy as np
from .importer_base import BaseIngestor, IngestionResult
from database import GraphOperations

logger = logging.getLogger(__name__)


class SNPIngestor(BaseIngestor):
    REQUIRED_COLUMNS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'INFO']

    def __init__(self, graph_ops: GraphOperations):
        super().__init__(graph_ops)

    async def ingest(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_vcf(file_path) if file_path.endswith('.vcf') else self._read_csv(file_path)
            self._validate_dataframe(df, self.REQUIRED_COLUMNS)

            snp_nodes = []
            gene_snp_rels = []

            for _, row in df.iterrows():
                try:
                    snp_id = self._generate_id('SNP', row['CHROM'], row['POS'], row['REF'], row['ALT'])
                    info_dict = self._parse_info_field(row['INFO'])

                    snp_props = {
                        'id': snp_id,
                        'rs_id': row['ID'] if row['ID'] != '.' else None,
                        'chromosome': str(row['CHROM']),
                        'position': int(row['POS']),
                        'ref_allele': row['REF'],
                        'alt_allele': row['ALT'],
                        'maf': float(info_dict.get('MAF', info_dict.get('AF', 0.0))),
                        'variant_type': self._determine_variant_type(row['REF'], row['ALT']),
                        'functional_impact': info_dict.get('ANN', info_dict.get('EFF', 'unknown')),
                        'quality': float(row['QUAL']) if row['QUAL'] != '.' else None,
                        'filter': row['FILTER']
                    }
                    snp_nodes.append(snp_props)

                    gene_id = info_dict.get('GENE', info_dict.get('Gene'))
                    if gene_id:
                        gene_snp_rels.append({
                            'source_id': f'GENE:{gene_id}',
                            'target_id': snp_id,
                            'properties': {
                                'distance': int(info_dict.get('DISTANCE', 0)),
                                'region_type': info_dict.get('REGION', 'intergenic')
                            }
                        })

                except Exception as e:
                    result.add_error(f"Error processing row {row.name}: {str(e)}")
                    continue

            if snp_nodes:
                created = await self.graph_ops.batch_create_nodes('SNP', snp_nodes)
                result.nodes_created = created
                result.nodes_updated = len(snp_nodes) - created
                logger.info(f"Ingested {len(snp_nodes)} SNP nodes")

            if gene_snp_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'GENE', 'SNP', 'CONTAINS_SNP', gene_snp_rels
                )
                result.relationships_created = created
                result.relationships_updated = len(gene_snp_rels) - created
                logger.info(f"Ingested {len(gene_snp_rels)} GENE-SNP relationships")

            result.complete()
        except Exception as e:
            result.add_error(f"Ingestion failed: {str(e)}")

        return result

    def _parse_info_field(self, info_str: str) -> Dict[str, Any]:
        info_dict = {}
        for entry in info_str.split(';'):
            if '=' in entry:
                key, value = entry.split('=', 1)
                info_dict[key] = value
            else:
                info_dict[entry] = True
        return info_dict

    def _determine_variant_type(self, ref: str, alt: str) -> str:
        ref_len = len(ref)
        alt_len = len(alt)

        if ',' in alt:
            return 'multiallelic'
        if ref_len == 1 and alt_len == 1:
            return 'SNV'
        elif ref_len == alt_len:
            return 'MNP'
        elif ref_len > alt_len:
            return 'deletion'
        elif ref_len < alt_len:
            return 'insertion'
        else:
            return 'complex'

    async def ingest_associations(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path)
            required = ['snp_id', 'phenotype_name', 'p_value']
            self._validate_dataframe(df, required)

            associations = []
            for _, row in df.iterrows():
                try:
                    snp_id = row['snp_id'] if row['snp_id'].startswith('SNP:') else f'SNP:{row["snp_id"]}'
                    phenotype_id = f'PHENO:{row["phenotype_name"].replace(" ", "_")}'

                    associations.append({
                        'source_id': snp_id,
                        'target_id': phenotype_id,
                        'properties': {
                            'p_value': float(row['p_value']),
                            'odds_ratio': float(row.get('odds_ratio', 1.0)),
                            'confidence_interval': str(row.get('confidence_interval', '')),
                            'study_id': str(row.get('study_id', ''))
                        }
                    })
                except Exception as e:
                    result.add_error(f"Error processing association row {row.name}: {str(e)}")
                    continue

            if associations:
                created = await self.graph_ops.batch_create_relationships(
                    'SNP', 'PHENOTYPE', 'ASSOCIATED_WITH', associations
                )
                result.relationships_created = created
                result.relationships_updated = len(associations) - created

            result.complete()
        except Exception as e:
            result.add_error(f"Association ingestion failed: {str(e)}")

        return result

    async def ingest_from_dataframe(self, df: 'pd.DataFrame', source_name: str = 'dataframe') -> IngestionResult:
        result = IngestionResult(source_file=source_name)
        try:
            self._validate_dataframe(df, self.REQUIRED_COLUMNS)

            snp_nodes = []
            for _, row in df.iterrows():
                try:
                    snp_id = self._generate_id('SNP', row['CHROM'], row['POS'], row['REF'], row['ALT'])
                    snp_props = {
                        'id': snp_id,
                        'rs_id': row.get('ID', ''),
                        'chromosome': str(row['CHROM']),
                        'position': int(row['POS']),
                        'ref_allele': row['REF'],
                        'alt_allele': row['ALT'],
                        'maf': float(row.get('MAF', row.get('AF', 0.0))),
                        'variant_type': self._determine_variant_type(row['REF'], row['ALT']),
                        'functional_impact': row.get('functional_impact', 'unknown')
                    }
                    snp_nodes.append(snp_props)
                except Exception as e:
                    result.add_error(f"Error processing row: {str(e)}")
                    continue

            if snp_nodes:
                created = await self.graph_ops.batch_create_nodes('SNP', snp_nodes)
                result.nodes_created = created
                result.nodes_updated = len(snp_nodes) - created

            result.complete()
        except Exception as e:
            result.add_error(f"DataFrame ingestion failed: {str(e)}")

        return result
