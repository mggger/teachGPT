import time
from pathlib import Path

import yaml
from graphrag.config import create_graphrag_config
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.community_reports.prompts import COMMUNITY_REPORT_PROMPT
from graphrag.index.graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from graphrag.index.graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
from graphrag.index.progress import NullProgressReporter
from graphrag.index.run import run_pipeline_with_config


class GraphRAGIndexer:
    def __init__(self, workspace="ragtest", config=None):
        self.workspace = workspace
        self.config = config
        self.reporter = NullProgressReporter()
        self.dataset_path = Path(self.workspace) / "dataset.parquet"
        self.index_file_path = Path(self.workspace) / "_index"
        self._check_and_init()

    def _check_and_init(self):
        root = Path(self.workspace)
        if not root.exists():
            self.reporter.info("No existing workspace found. Initializing workspace.")
            self._init()
        else:
            self.reporter.info("Found existing workspace.")
            root = Path(self.workspace)
            settings_yaml = root / "settings.yaml"
            with settings_yaml.open("w") as file:
                yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

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

    def _update_index(self):
        timestamp = str(int(time.time()))
        with self.index_file_path.open("w") as f:
            f.write(timestamp)
        self.reporter.info(f"Updated index timestamp: {timestamp}")

    async def run(self, dataset):
        await self._ainsert(dataset)

    async def _ainsert(self, dataset):
        output_dir = Path(self.workspace) / "output" / "graph"
        output_dir.mkdir(parents=True, exist_ok=True)

        settings_yaml = Path(self.workspace) / "settings.yaml"
        with settings_yaml.open("r") as file:
            config_data = yaml.safe_load(file)

        graphrag_config = create_graphrag_config(config_data, self.workspace)
        pipeline_config: PipelineConfig = create_pipeline_config(graphrag_config, verbose=True)

        pipeline_config.storage.base_dir = str(output_dir / "artifacts")
        pipeline_config.reporting.base_dir = str(output_dir / "reports")

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
