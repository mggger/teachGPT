import time
from pathlib import Path
import yaml
import pandas as pd
from graphrag.config import create_graphrag_config
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.community_reports.prompts import COMMUNITY_REPORT_PROMPT
from graphrag.index.graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
from graphrag.index.progress import PrintProgressReporter
from graphrag.index.run import run_pipeline_with_config
from graphrag.index.utils import gen_md5_hash
from .extract import PDFProcessor
from config import Config


class GraphRAGIndexer:
    def __init__(self, workspace="ragtest", config=None):
        self.workspace = workspace
        self.config = config
        self.reporter = PrintProgressReporter("GraphRAG Indexer: ")
        self.dataset_path = Path(self.workspace) / "dataset.parquet"
        self.index_file_path = Path(self.workspace) / "_index"
        self._check_and_init()
        self.pdf_processer = PDFProcessor(Config)

    def _check_and_init(self):
        root = Path(self.workspace)
        if not root.exists():
            self.reporter.info("No existing workspace found. Initializing workspace.")
            self._init()
        else:
            self.reporter.info("Found existing workspace.")

    def _init(self):
        self.reporter.info(f"Initializing project at {self.workspace}")
        root = Path(self.workspace)
        root.mkdir(parents=True, exist_ok=True)

        settings_yaml = root / "settings.yaml"
        with settings_yaml.open("w") as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

        prompts_dir = root / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompts = {
            "entity_extraction.txt": GRAPH_EXTRACTION_PROMPT,
            "summarize_descriptions.txt": SUMMARIZE_PROMPT,
            "claim_extraction.txt": CLAIM_EXTRACTION_PROMPT,
            "community_report.txt": COMMUNITY_REPORT_PROMPT
        }

        for filename, content in prompts.items():
            file_path = prompts_dir / filename
            if not file_path.exists():
                with file_path.open("wb") as file:
                    file.write(content.encode(encoding="utf-8", errors="strict"))

        if not self.dataset_path.exists():
            empty_df = pd.DataFrame(columns=["id", "text", "title"])
            empty_df.to_parquet(self.dataset_path)

    def _update_index(self):
        timestamp = str(int(time.time()))
        with self.index_file_path.open("w") as f:
            f.write(timestamp)
        self.reporter.info(f"Updated index timestamp: {timestamp}")

    def _process_documents(self, documents):
        processed_docs = []
        for doc in documents:
            doc_data = {
                "text": doc['doc_content'],
                "title": doc['filename'],
            }
            doc_data["id"] = gen_md5_hash(doc_data, doc_data.keys())
            processed_docs.append(doc_data)
        return pd.DataFrame(processed_docs)

    def _read_dataset(self):
        if self.dataset_path.exists():
            return pd.read_parquet(self.dataset_path)
        return pd.DataFrame(columns=["id", "text", "title"])

    def _write_dataset(self, df):
        df.to_parquet(self.dataset_path, index=False)

    def _update_dataset(self, new_docs):
        existing_df = self._read_dataset()
        new_df = pd.concat([existing_df, new_docs], ignore_index=True)

        # Check for duplicates
        duplicates = new_df[new_df.duplicated(subset=["title"], keep=False)]
        if not duplicates.empty:
            duplicate_files = duplicates['title'].unique().tolist()
            self.reporter.info(f"Duplicate files detected: {', '.join(duplicate_files)}")
            new_df.drop_duplicates(subset=["title"], keep="first", inplace=True)

        self._write_dataset(new_df)
        return new_df, duplicates

    async def insert_pdf(self, pdf_path):
        if isinstance(pdf_path, str):
            pdf_paths = [pdf_path]
        else:
            pdf_paths = pdf_path

        documents = []
        existing_dataset = self._read_dataset()
        existing_filenames = set(existing_dataset['title'])

        for pdf_path in pdf_paths:
            filename = Path(pdf_path).name
            if filename in existing_filenames:
                self.reporter.info(f"Skipping {filename} as it has already been processed.")
                continue

            document = self.pdf_processer.run(pdf_path)
            documents.append(document)
            self.reporter.info(f"Successfully processed: {pdf_path}")

        if documents:
            await self._ainsert(documents)

    async def _ainsert(self, documents):
        output_dir = Path(self.workspace) / "output" / "graph"
        output_dir.mkdir(parents=True, exist_ok=True)

        settings_yaml = Path(self.workspace) / "settings.yaml"
        with settings_yaml.open("r") as file:
            config_data = yaml.safe_load(file)

        graphrag_config = create_graphrag_config(config_data, self.workspace)
        pipeline_config: PipelineConfig = create_pipeline_config(graphrag_config, verbose=True)

        pipeline_config.storage.base_dir = str(output_dir / "artifacts")
        pipeline_config.reporting.base_dir = str(output_dir / "reports")

        new_docs = self._process_documents(documents)
        dataset, duplicates = self._update_dataset(new_docs)

        if not duplicates.empty:
            duplicate_files = duplicates['title'].unique().tolist()
            self.reporter.info(
                f"The following files already exist and will not be reprocessed: {', '.join(duplicate_files)}")
            return

        async for output in run_pipeline_with_config(
                pipeline_config,
                dataset=dataset,
                run_id="graph",
                progress_reporter=self.reporter,
        ):
            if output.errors:
                self.reporter.error(f"{output.workflow}: {output.errors}")
            else:
                self.reporter.success(output.workflow)
        self._update_index()
        self.reporter.success("All workflows completed successfully.")
