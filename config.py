import os

Config = {
    "unstructured_api_endpoint":  os.environ.get("unstructured_api_endpoint"),
    "unstructured_api_key": os.environ.get("unstructured_api_key"),
    "r2_access_key": os.environ.get("r2_access_key"),
    "r2_secret_key": os.environ.get("r2_secret_key"),
    "r2_endpoint_url": os.environ.get("r2_endpoint_url"),
    "r2_bucket_name": os.environ.get("r2_bucket_name"),
    "r2_public_url": os.environ.get("r2_public_url"),
    "openai_api_key": os.environ.get("openai_api_key")
}
