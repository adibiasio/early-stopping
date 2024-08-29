import json

from .s3_utils import get_bucket_prefix, is_s3_url


def load_json(path: str) -> dict:
    """
    Loads a json file from the provided path.

    Params:
    -------
    path: str
        The path of the json file.
        May be s3 or local.

    Returns:
    --------
    dict: the json object.
    """
    if is_s3_url(path):
        import boto3

        bucket, key = get_bucket_prefix(path)
        s3_client = boto3.client("s3")
        result = s3_client.get_object(Bucket=bucket, Key=key)
        out = json.loads(result["Body"].read())
    else:
        with open(path, "r") as f:
            out = json.load(f)
    return out


def save_json(path: str, obj):
    """
    Saves an object to the the provided path as a json file.

    Params:
    -------
    path: str
        The path for the the json file to be saved to.
        May be s3 or local.
    obj: Any
        An object that can be saved to json format.
    """
    is_s3_path = is_s3_url(path)
    if is_s3_path:
        import boto3

        data = json.dumps(obj).encode("UTF-8")
        bucket, key = get_bucket_prefix(path)
        s3_client = boto3.client("s3")
        s3_client.put_object(Body=data, Bucket=bucket, Key=key)
    else:
        import os

        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(obj, fp, indent=2)
