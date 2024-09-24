from .index import GraphRAGIndexer
from .query import GraphRAGQuerier
from .config import load_config
from .db import DB
from grag_api.extract.json_extract import process_json_content
from grag_api.extract.pdf_extract import PDFProcessor
import os
from datetime import datetime

class GraphRAG:
    def __init__(self, workspace="ragtest", api_key=None):
        config = load_config(api_key)
        self.indexer = GraphRAGIndexer(workspace, config=config)
        self.querier = GraphRAGQuerier(workspace, config=config)
        self.db = DB()
        self.pdf_processor = PDFProcessor(config)
        self.workspace = workspace

    def upsert_pdf(self, pdf_path):
        pdf_data = self.pdf_processor.run(pdf_path)
        self.db.batch_upsert_data(pdf_data)

    def delete_pdf(self, filename):
        self.db.delete_data_by_title(filename)

    def upsert_json(self, json_elements):
        json_data = process_json_content(json_elements)
        self.db.batch_upsert_data(json_data)

    def delete_item(self, id):
        self.db.delete_data([id])

    def get_all_files(self):
        return self.db.get_all_titles()

    async def aindex(self):
        dataset = self.db.load_data()
        await self.indexer.run(dataset)

    async def aquery(self, question, callbacks=[], system_prompt=None):
        return await self.querier.query(question, callbacks=callbacks, system_prompt=system_prompt)

    def get_last_training_time(self):
        index_file_path = os.path.join(self.workspace, "_index")
        if not os.path.exists(index_file_path):
            return None

        with open(index_file_path, 'r') as f:
            timestamp = f.read().strip()

        try:
            timestamp = float(timestamp)
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return None
