import json

from .s3_utils import get_bucket_key, is_s3_url


def load_json(path: str) -> dict:
    if is_s3_url(path):
        import boto3

        bucket, key = get_bucket_key(path)
        s3_client = boto3.client("s3")
        result = s3_client.get_object(Bucket=bucket, Key=key)
        out = json.loads(result["Body"].read())
    else:
        with open(path, "r") as f:
            out = json.load(f)
    return out
