from .index import GraphRAGIndexer
from .query import GraphRAGQuerier
from .config import load_config

class GraphRAG:
    def __init__(self, workspace="ragtest", api_key=None, ):
        config = load_config(api_key)
        self.indexer = GraphRAGIndexer(workspace, config=config)
        self.querier = GraphRAGQuerier(workspace, config=config)

    async def insert_pdf(self, pdf_path):
        await self.indexer.insert_pdf(pdf_path)

    async def aquery(self, question, callbacks=[], system_prompt=None):
        return await self.querier.query(question, callbacks=callbacks, system_prompt=system_prompt)