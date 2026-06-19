from typing import List, Dict, Any, Optional
import logging
import json
from .importer_base import BaseIngestor, IngestionResult
from database import GraphOperations

logger = logging.getLogger(__name__)


class PhenotypeIngestor(BaseIngestor):
    REQUIRED_COLUMNS = ['phenotype_id', 'name', 'description', 'trait_category']

    def __init__(self, graph_ops: GraphOperations):
        super().__init__(graph_ops)

    async def ingest(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path)
            self._validate_dataframe(df, self.REQUIRED_COLUMNS)

            phenotype_nodes = []
            sample_phenotype_rels = []
            environment_phenotype_rels = []

            for _, row in df.iterrows():
                try:
                    phenotype_id = row['phenotype_id'] if str(row['phenotype_id']).startswith('PHENO:') \
                        else f'PHENO:{row["phenotype_id"]}'

                    phenotype_props = {
                        'id': phenotype_id,
                        'name': row['name'],
                        'description': row.get('description', ''),
                        'trait_category': row['trait_category'],
                        'measurement_unit': row.get('measurement_unit', ''),
                        'species': row.get('species', '')
                    }
                    phenotype_nodes.append(phenotype_props)

                    sample_id = row.get('sample_id', row.get('genotype_id'))
                    if sample_id:
                        sample_phenotype_rels.append({
                            'source_id': f'SAMPLE:{sample_id}',
                            'target_id': phenotype_id,
                            'properties': {
                                'phenotype_value': float(row.get('phenotype_value', 0.0)),
                                'measurement_date': str(row.get('measurement_date', '')),
                                'replicate': int(row.get('replicate', 1)),
                                'quality_score': float(row.get('quality_score', 1.0))
                            }
                        })

                    environment_id = row.get('environment_id', row.get('location'))
                    if environment_id:
                        env_id = environment_id if str(environment_id).startswith('ENV:') \
                            else f'ENV:{environment_id}'
                        corr_coef = float(row.get('correlation_coefficient', 0.0))
                        if abs(corr_coef) > 0:
                            environment_phenotype_rels.append({
                                'source_id': env_id,
                                'target_id': phenotype_id,
                                'properties': {
                                    'correlation_coefficient': corr_coef,
                                    'p_value': float(row.get('p_value', 1.0)),
                                    'sample_size': int(row.get('sample_size', 0))
                                }
                            })

                except Exception as e:
                    result.add_error(f"Error processing phenotype row {row.name}: {str(e)}")
                    continue

            if phenotype_nodes:
                created = await self.graph_ops.batch_create_nodes('PHENOTYPE', phenotype_nodes)
                result.nodes_created = created
                result.nodes_updated = len(phenotype_nodes) - created
                logger.info(f"Ingested {len(phenotype_nodes)} phenotype nodes")

            if sample_phenotype_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'SAMPLE', 'PHENOTYPE', 'EXPRESSES_IN', sample_phenotype_rels
                )
                result.relationships_created += created
                result.relationships_updated += len(sample_phenotype_rels) - created
                logger.info(f"Ingested {len(sample_phenotype_rels)} SAMPLE-PHENOTYPE relationships")

            if environment_phenotype_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'ENVIRONMENT', 'PHENOTYPE', 'CORRELATES_WITH', environment_phenotype_rels
                )
                result.relationships_created += created
                result.relationships_updated += len(environment_phenotype_rels) - created
                logger.info(f"Ingested {len(environment_phenotype_rels)} ENV-PHENO relationships")

            result.complete()
        except Exception as e:
            result.add_error(f"Phenotype ingestion failed: {str(e)}")

        return result

    async def ingest_samples(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path)
            required = ['sample_id', 'genotype_id', 'phenotype_values']
            self._validate_dataframe(df, required)

            sample_nodes = []
            sample_crop_rels = []

            for _, row in df.iterrows():
                try:
                    sample_id = row['sample_id'] if str(row['sample_id']).startswith('SAMPLE:') \
                        else f'SAMPLE:{row["sample_id"]}'

                    phenotype_values = row['phenotype_values']
                    if isinstance(phenotype_values, str):
                        try:
                            phenotype_values = json.loads(phenotype_values)
                        except json.JSONDecodeError:
                            phenotype_values = {}

                    sample_props = {
                        'id': sample_id,
                        'sample_id': row['sample_id'],
                        'collection_date': str(row.get('collection_date', '')),
                        'genotype_id': row['genotype_id'],
                        'phenotype_values': phenotype_values if isinstance(phenotype_values, dict) else {},
                        'quality_score': float(row.get('quality_score', 1.0))
                    }
                    sample_nodes.append(sample_props)

                    crop_id = row.get('crop_id', row.get('cultivar_id'))
                    if crop_id:
                        cid = crop_id if str(crop_id).startswith('CROP:') else f'CROP:{crop_id}'
                        sample_crop_rels.append({
                            'source_id': sample_id,
                            'target_id': cid,
                            'properties': {
                                'taxonomic_level': 'variety'
                            }
                        })

                except Exception as e:
                    result.add_error(f"Error processing sample row {row.name}: {str(e)}")
                    continue

            if sample_nodes:
                created = await self.graph_ops.batch_create_nodes('SAMPLE', sample_nodes)
                result.nodes_created = created
                result.nodes_updated = len(sample_nodes) - created

            if sample_crop_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'SAMPLE', 'CROP', 'BELONGS_TO', sample_crop_rels
                )
                result.relationships_created = created
                result.relationships_updated = len(sample_crop_rels) - created

            result.complete()
        except Exception as e:
            result.add_error(f"Sample ingestion failed: {str(e)}")

        return result

    async def ingest_environments(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path)
            required = ['environment_id', 'name', 'location']
            self._validate_dataframe(df, required)

            env_nodes = []
            for _, row in df.iterrows():
                try:
                    env_id = row['environment_id'] if str(row['environment_id']).startswith('ENV:') \
                        else f'ENV:{row["environment_id"]}'

                    env_props = {
                        'id': env_id,
                        'name': row['name'],
                        'description': row.get('description', ''),
                        'location': row['location'],
                        'temperature': float(row.get('temperature', 0.0)),
                        'precipitation': float(row.get('precipitation', 0.0)),
                        'soil_type': row.get('soil_type', ''),
                        'elevation': float(row.get('elevation', 0.0))
                    }
                    env_nodes.append(env_props)
                except Exception as e:
                    result.add_error(f"Error processing environment row {row.name}: {str(e)}")
                    continue

            if env_nodes:
                created = await self.graph_ops.batch_create_nodes('ENVIRONMENT', env_nodes)
                result.nodes_created = created
                result.nodes_updated = len(env_nodes) - created

            result.complete()
        except Exception as e:
            result.add_error(f"Environment ingestion failed: {str(e)}")

        return result

    async def ingest_crops(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path)
            required = ['crop_id', 'name', 'species']
            self._validate_dataframe(df, required)

            crop_nodes = []
            for _, row in df.iterrows():
                try:
                    crop_id = row['crop_id'] if str(row['crop_id']).startswith('CROP:') \
                        else f'CROP:{row["crop_id"]}'

                    crop_props = {
                        'id': crop_id,
                        'name': row['name'],
                        'species': row['species'],
                        'cultivar': row.get('cultivar', ''),
                        'breeding_line': row.get('breeding_line', '')
                    }
                    crop_nodes.append(crop_props)
                except Exception as e:
                    result.add_error(f"Error processing crop row {row.name}: {str(e)}")
                    continue

            if crop_nodes:
                created = await self.graph_ops.batch_create_nodes('CROP', crop_nodes)
                result.nodes_created = created
                result.nodes_updated = len(crop_nodes) - created

            result.complete()
        except Exception as e:
            result.add_error(f"Crop ingestion failed: {str(e)}")

        return result
