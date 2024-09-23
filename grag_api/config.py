import copy

DEFAULT_CONFIG = {
    "encoding_model": "cl100k_base",
    "skip_workflows": [],
    "llm": {
        "api_key": 'sk-xx',
        "type": "openai_chat",
        "model": "gpt-4o-mini",
        "model_supports_json": True,
    },
    "parallelization": {
        "stagger": 0.3,
    },
    "async_mode": "threaded",
    "embeddings": {
        "async_mode": "threaded",
        "llm": {
            "api_key": 'sk-xx',
            "type": "openai_embedding",
            "model": "text-embedding-3-small",
            "max_tokens": 8192,
        },
    },
    "chunks": {
        "size": 7000,
        "overlap": 500,
        "group_by_columns": ["id"],
    },
    "input": {
        "type": "file",
        "file_type": "text",
        "base_dir": "input",
        "file_encoding": "utf-8",
        "file_pattern": r".*\.txt$",
    },
    "cache": {
        "type": "file",
        "base_dir": "cache",
    },
    "storage": {
        "type": "file",
        "base_dir": "output/${timestamp}/artifacts",
    },
    "reporting": {
        "type": "file",
        "base_dir": "output/${timestamp}/reports",
    },
    "entity_extraction": {
        "prompt": "prompts/entity_extraction.txt",
        "entity_types": ["organization", "person", "geo", "event"],
        "max_gleanings": 1,
    },
    "summarize_descriptions": {
        "prompt": "prompts/summarize_descriptions.txt",
        "max_length": 500,
    },
    "claim_extraction": {
        "prompt": "prompts/claim_extraction.txt",
        "description": "Any claims or facts that could be relevant to information discovery.",
        "max_gleanings": 1,
    },
    "community_reports": {
        "prompt": "prompts/community_report.txt",
        "max_length": 2000,
        "max_input_length": 8000,
    },
    "cluster_graph": {
        "max_cluster_size": 10,
    },
    "embed_graph": {
        "enabled": False,
    },
    "umap": {
        "enabled": False,
    },
    "snapshots": {
        "graphml": False,
        "raw_entities": False,
        "top_level_nodes": False,
    },
    "local_search": {},
    "global_search": {},
}


def load_config(api_key=None):
    """
    Load the configuration and optionally replace the API key.

    Args:
        api_key (str, optional): The API key to use. If None, the default key will be used.

    Returns:
        dict: The configuration dictionary with the API key replaced if provided.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if api_key is not None:
        # Replace the API key in the main LLM configuration
        config['llm']['api_key'] = api_key

        # Replace the API key in the embeddings LLM configuration
        config['embeddings']['llm']['api_key'] = api_key

    return config
