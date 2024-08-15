import os
from urllib.parse import urlparse


def is_s3_url(path: str) -> bool:
    if (path[:2] == "s3") and ("://" in path[:6]):
        return True
    return False


def get_bucket_key(s3_uri: str) -> tuple[str, str]:
    if not is_s3_url(s3_uri):
        raise ValueError("Invalid S3 URI scheme. It should be 's3'.")

    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip("/")

    return bucket_name, object_key


def download_folder(bucket: str, prefix: str, local_dir: str):
    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    # List objects with the given prefix (s3_folder)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]

            # Skip the folder itself
            if s3_key == prefix:
                continue

            # Construct the local file path
            local_file_path = os.path.join(local_dir, os.path.relpath(s3_key, prefix))
            local_file_dir = os.path.dirname(local_file_path)

            # Create local directory if it doesn't exist
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            print(f"Downloading {s3_key} to {local_file_path}")
            s3.download_file(bucket, s3_key, local_file_path)


def list_bucket_s3(bucket, prefix: str = "", suffix: str = "") -> list[str]:
    import boto3

    filters = {}
    if prefix:
        filters["Prefix"] = prefix

    if suffix is None:
        suffix = ""

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        objects.extend([obj["Key"] for obj in contents if obj["Key"].endswith(suffix)])

    return objects
