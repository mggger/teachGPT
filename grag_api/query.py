from pathlib import Path
import pandas as pd
from graphrag.query import indexer_adapters, llm
import tiktoken
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.input.loaders import dfs
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from grag_api.custom_search import LocalSearch
from graphrag.query.structured_search.local_search.system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from graphrag.vector_stores import LanceDBVectorStore


class GraphRAGQuerier:
    def __init__(self, workspace="ragtest", config=None):
        self.workspace = workspace
        self.config = config
        self.api_key = self.config['llm']['api_key']
        self.index_file_path = Path(self.workspace) / "_index"
        self.last_loaded_timestamp = None
        self.entities = None
        self.reports = None
        self.relationships = None
        self.text_units = None
        self.search_engine = None

    def load_data(self):
        INPUT_DIR = Path(self.workspace) / "output" / "graph" / "artifacts"
        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        RELATIONSHIP_TABLE = "create_final_relationships"
        TEXT_UNIT_TABLE = "create_final_text_units"
        COMMUNITY_LEVEL = 2

        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")

        self.entities = indexer_adapters.read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
        self.reports = indexer_adapters.read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        self.relationships = indexer_adapters.read_indexer_relationships(relationship_df)
        self.text_units = indexer_adapters.read_indexer_text_units(text_unit_df)

    def setup_llm_and_embeddings(self):
        llm_instance = ChatOpenAI(
            api_key=self.api_key,
            model=self.config['llm']['model'],
            api_type=llm.oai.typing.OpenaiApiType.OpenAI,
            max_retries=20,
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        text_embedder = OpenAIEmbedding(
            api_key=self.api_key,
            api_type=llm.oai.typing.OpenaiApiType.OpenAI,
            model=self.config['embeddings']['llm']['model'],
            deployment_name=self.config['embeddings']['llm']['model'],
            max_retries=20,
        )

        return llm_instance, token_encoder, text_embedder

    def setup_vector_store(self):
        LANCEDB_URI = "lancedb"
        description_embedding_store = LanceDBVectorStore(
            collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)

        dfs.store_entity_semantic_embeddings(
            entities=self.entities,
            vectorstore=description_embedding_store
        )

        return description_embedding_store

    def setup_local_search(self, llm_instance, token_encoder, text_embedder, description_embedding_store):
        context_builder_instance = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": 12_000,
        }

        llm_params = {
            "max_tokens": 2_000,
            "temperature": 0.0,
        }

        return LocalSearch(
            llm=llm_instance,
            context_builder=context_builder_instance,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            response_type='Single Paragraph',
        )

    def check_and_reload_data(self):
        if self.index_file_path.exists():
            with self.index_file_path.open("r") as f:
                current_timestamp = f.read().strip()
            if current_timestamp != self.last_loaded_timestamp:
                self.load_data()
                self.last_loaded_timestamp = current_timestamp
                return True
        return False

    async def query(self, question, conversation_history=[], callbacks=[], system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT):
        data_reloaded = self.check_and_reload_data()
        if data_reloaded or self.search_engine is None:
            llm_instance, token_encoder, text_embedder = self.setup_llm_and_embeddings()
            description_embedding_store = self.setup_vector_store()
            self.search_engine = self.setup_local_search(
                llm_instance, token_encoder, text_embedder, description_embedding_store
            )

        self.search_engine.callbacks = callbacks
        self.search_engine.system_prompt = system_prompt
        result = await self.search_engine.asearch(question, conversation_history=conversation_history)
        return result
