from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import pandas as pd
from database import GraphOperations

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    source_file: str
    nodes_created: int = 0
    nodes_updated: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    success: bool = False

    def complete(self) -> None:
        self.completed_at = datetime.now()
        self.success = len(self.errors) == 0

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        logger.error(error)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "relationships_created": self.relationships_created,
            "relationships_updated": self.relationships_updated,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "success": self.success
        }


class BaseIngestor(ABC):
    def __init__(self, graph_ops: GraphOperations):
        self.graph_ops = graph_ops

    @abstractmethod
    async def ingest(self, file_path: str) -> IngestionResult:
        pass

    def _read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        logger.info(f"Reading CSV file: {file_path}")
        return pd.read_csv(file_path, **kwargs)

    def _read_vcf(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Reading VCF file: {file_path}")
        skiprows = 0
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('##'):
                    skiprows += 1
                else:
                    break

        df = pd.read_csv(file_path, sep='\t', skiprows=skiprows, comment='#')
        if df.columns[0] == '#CHROM':
            df.columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', *df.columns[9:]]
        return df

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _generate_id(self, prefix: str, *parts: str) -> str:
        return f"{prefix}:{':'.join(str(p) for p in parts)}"
