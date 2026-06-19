from typing import List, Dict, Any
import logging
from .importer_base import BaseIngestor, IngestionResult
from database import GraphOperations

logger = logging.getLogger(__name__)


class GOIngestor(BaseIngestor):
    REQUIRED_COLUMNS = ['go_id', 'name', 'namespace', 'definition']

    def __init__(self, graph_ops: GraphOperations):
        super().__init__(graph_ops)

    async def ingest(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path)
            self._validate_dataframe(df, self.REQUIRED_COLUMNS)

            go_nodes = []
            gene_go_rels = []

            for _, row in df.iterrows():
                try:
                    go_id = self._normalize_go_id(row['go_id'])

                    go_props = {
                        'id': go_id,
                        'name': row['name'],
                        'namespace': row['namespace'],
                        'definition': row['definition'],
                        'ontology': row.get('ontology', row['namespace'])
                    }
                    go_nodes.append(go_props)

                    gene_id = row.get('gene_id', row.get('gene'))
                    if gene_id:
                        gene_go_rels.append({
                            'source_id': f'GENE:{gene_id}',
                            'target_id': go_id,
                            'properties': {
                                'evidence_code': str(row.get('evidence_code', 'IEA')),
                                'annotation_date': str(row.get('annotation_date', ''))
                            }
                        })

                except Exception as e:
                    result.add_error(f"Error processing GO term row {row.name}: {str(e)}")
                    continue

            if go_nodes:
                created = await self.graph_ops.batch_create_nodes('GO_TERM', go_nodes)
                result.nodes_created = created
                result.nodes_updated = len(go_nodes) - created
                logger.info(f"Ingested {len(go_nodes)} GO term nodes")

            if gene_go_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'GENE', 'GO_TERM', 'ANNOTATED_TO', gene_go_rels
                )
                result.relationships_created = created
                result.relationships_updated = len(gene_go_rels) - created
                logger.info(f"Ingested {len(gene_go_rels)} GENE-GO relationships")

            result.complete()
        except Exception as e:
            result.add_error(f"GO ingestion failed: {str(e)}")

        return result

    def _normalize_go_id(self, go_id: str) -> str:
        go_id = str(go_id).strip().upper()
        if not go_id.startswith('GO:'):
            go_id = f'GO:{go_id.zfill(7)}'
        return go_id

    async def ingest_ontology(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            go_nodes = []
            parent_rels = []

            for term in root.findall('.//term'):
                try:
                    go_id = term.find('id').text if term.find('id') is not None else ''
                    if not go_id:
                        continue

                    go_name = term.find('name').text if term.find('name') is not None else ''
                    namespace = term.find('namespace').text if term.find('namespace') is not None else ''
                    definition_elem = term.find('def/defstr')
                    definition = definition_elem.text if definition_elem is not None else ''

                    go_props = {
                        'id': go_id,
                        'name': go_name,
                        'namespace': namespace,
                        'definition': definition,
                        'ontology': namespace,
                        'is_obsolete': term.find('is_obsolete') is not None
                    }
                    go_nodes.append(go_props)

                    for is_a in term.findall('is_a'):
                        parent_id = is_a.text
                        if parent_id:
                            parent_rels.append({
                                'source_id': go_id,
                                'target_id': parent_id,
                                'properties': {
                                    'relationship_type': 'is_a'
                                }
                            })

                except Exception as e:
                    result.add_error(f"Error processing OBO term: {str(e)}")
                    continue

            if go_nodes:
                created = await self.graph_ops.batch_create_nodes('GO_TERM', go_nodes)
                result.nodes_created = created
                result.nodes_updated = len(go_nodes) - created

            if parent_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'GO_TERM', 'GO_TERM', 'PARENT_OF', parent_rels
                )
                result.relationships_created = created
                result.relationships_updated = len(parent_rels) - created

            result.complete()
        except Exception as e:
            result.add_error(f"Ontology ingestion failed: {str(e)}")

        return result

    async def ingest_gene_annotations(self, file_path: str) -> IngestionResult:
        result = IngestionResult(source_file=file_path)
        try:
            df = self._read_csv(file_path, sep='\t', comment='!')
            gaf_columns = [
                'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
                'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect',
                'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
                'Taxon', 'Date', 'Assigned_By'
            ]
            df.columns = gaf_columns[:len(df.columns)]

            gene_go_rels = []
            for _, row in df.iterrows():
                try:
                    gene_id = row['DB_Object_ID']
                    go_id = self._normalize_go_id(row['GO_ID'])

                    gene_go_rels.append({
                        'source_id': f'GENE:{gene_id}',
                        'target_id': go_id,
                        'properties': {
                            'evidence_code': row['Evidence_Code'],
                            'qualifier': row.get('Qualifier', ''),
                            'annotation_date': row.get('Date', ''),
                            'assigned_by': row.get('Assigned_By', '')
                        }
                    })
                except Exception as e:
                    result.add_error(f"Error processing GAF row {row.name}: {str(e)}")
                    continue

            if gene_go_rels:
                created = await self.graph_ops.batch_create_relationships(
                    'GENE', 'GO_TERM', 'ANNOTATED_TO', gene_go_rels
                )
                result.relationships_created = created
                result.relationships_updated = len(gene_go_rels) - created

            result.complete()
        except Exception as e:
            result.add_error(f"GAF ingestion failed: {str(e)}")

        return result
