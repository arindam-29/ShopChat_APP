import sys  # Provides access to some variables used or maintained by the Python interpreter
import os  # Provides a way of using operating system dependent functionality
import json  # Enables encoding and decoding JSON data
from s3fs import S3FileSystem  # Interface to S3, providing a file-like interface
import boto3  # Amazon Web Services (AWS) SDK for Python, allows Python developers to write software that uses services like Amazon S3
from src.exception import CustomException  # Import CustomException class for handling exceptions

# Define a function to save a JSON object to a local file
def save_json_to_local(file_path, object):
    try:
        with open(file_path, "w") as file:
            json.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)

# Define a function to load a JSON object from a local file
def load_json_from_local(file_path):
    try:
        with open(file_path, "r") as file:
            obj = json.load(file)
        return obj
    except Exception as e:
        raise CustomException(e, sys)

# Define a function to save a JSON object to an S3 bucket
def save_json_to_s3(file_path, object):
    try:
        fs = S3FileSystem()
        with fs.open(file_path, "w") as file:
            json.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)

# Define a function to load a JSON object from an S3 bucket
def load_json_from_s3(file_path):
    try:
        fs = S3FileSystem()
        with fs.open(file_path, "r") as file:
            obj = json.load(file)
        return obj
    except Exception as e:
        raise CustomException(e, sys)

# Define a function to upload a directory to an S3 bucket
def upload_directory_to_s3(bucket_name, s3_folder, local_directory):
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_folder, relative_path)
            
            s3_client.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {s3_path} to S3 bucket {bucket_name}")

# Define a function to download a directory from an S3 bucket
def download_directory_from_s3(bucket_name, s3_folder, local_directory):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name) 
    for obj in bucket.objects.filter(Prefix = s3_folder):
        local_file_path = os.path.join(local_directory, obj.key)
        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path))
        bucket.download_file(obj.key, local_file_path)
        print(f"Download {s3_folder} from S3 bucket {bucket_name}")
