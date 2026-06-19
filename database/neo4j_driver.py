from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Dict, Any
import logging
from config import settings

logger = logging.getLogger(__name__)


class Neo4jDriver:
    _instance: Optional["Neo4jDriver"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._driver = None

    async def connect(self) -> None:
        if self._driver is None:
            logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI}")
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                database=settings.NEO4J_DATABASE
            )
            await self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._driver is None:
            await self.connect()
        async with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            yield session

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncTransaction, None]:
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                yield tx

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            return [record.data() for record in await result.fetch()]

    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        async with self.transaction() as tx:
            await tx.run(query, parameters or {})

    async def execute_batch_write(
        self,
        query: str,
        parameters_list: List[Dict[str, Any]]
    ) -> None:
        async with self.transaction() as tx:
            for params in parameters_list:
                await tx.run(query, params)

    async def get_node_count(self, label: str) -> int:
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
        result = await self.execute_query(query)
        return result[0]["count"] if result else 0

    async def get_relationship_count(self, rel_type: str) -> int:
        query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        result = await self.execute_query(query)
        return result[0]["count"] if result else 0
