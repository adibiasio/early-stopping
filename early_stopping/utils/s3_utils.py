import os
from urllib.parse import urlparse


def is_s3_url(path: str) -> bool:
    """
    Checks if path is a s3 uri.

    Params:
    -------
    path: str
        The path to check.

    Returns:
    --------
    bool: whether the path is a s3 uri.
    """
    if (path[:2] == "s3") and ("://" in path[:6]):
        return True
    return False


def get_bucket_prefix(s3_uri: str) -> tuple[str, str]:
    """
    Retrieves the bucket and prefix from a s3 uri.

    Params:
    -------
    origin_path: str
        The path (s3 uri) to be parsed.

    Returns:
    --------
    bucket_name: str
        the associated bucket name
    object_prefix: str
        the associated prefix (key)
    """
    if not is_s3_url(s3_uri):
        raise ValueError("Invalid S3 URI scheme. It should be 's3'.")

    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip("/")

    return bucket_name, object_key


def download_folder(path: str, local_dir: str) -> str:
    """
    Downloads s3 file from path to local_dir.

    Parameters:
    -----------
    path: str
        The path (s3 uri) to the original location of the object
    local_dir: str
        The local path to the intended destination location of the object

    Returns:
    --------
    str: the local_dir path where the downloaded folder is located.
    """
    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    bucket, prefix = get_bucket_prefix(path)

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

    return local_dir
