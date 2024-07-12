import os
import sys
import shutil
import pandas as pd
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import logging
from src.utils import upload_directory_to_s3
from dataclasses import dataclass,field
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
import grpc
from langchain.globals import set_verbose
set_verbose(True)

# Adjust the root logging level if necessary (this affects all logging)
logging.basicConfig(level=logging.ERROR)

# Specifically adjust gRPC logging level to suppress informational messages
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Define a dataclass for configuration settings related to data embedding
@dataclass
class DataEmbeddingConfig:
    model: str = None
    
    # Load environment variables and configure API keys
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Configuration variables for S3 and local directories
    ABO_BUCKET_NAME:      str = os.getenv("ABO_BUCKET_NAME")
    YOUR_S3_BUCKET_NAME:  str = os.getenv("YOUR_S3_BUCKET_NAME")
    ARTIFACTS_FOLDER:     str = os.getenv("ARTIFACTS_FOLDER")
    WORKING_DIR:          str = os.getenv("WORKING_DIR")

    # Create a working directory if it does not exist
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    # File paths for JSON data and vector databases
    json_file_s3:     str = f"s3://{YOUR_S3_BUCKET_NAME}/PROCESSED/dataset.json"
    json_file_local:  str = f"{ARTIFACTS_FOLDER}/dataset.json"
    vector_db_s3:     str = field(init=False)
    embeddings:   object  = field(init=False)

    # Initialize embeddings and vector database paths based on the selected model
    def __post_init__(self):
        if self.model == "google":
            logging.info("Data Embedding - Embedding Model selected is Google.")
            self.vector_db_s3: str = "GOOGLE_FAISS_DB"
            self.vector_db_local: str = f"{self.ARTIFACTS_FOLDER}GOOGLE_FAISS_DB"
            self.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")         
        elif self.model == "openai":
            logging.info("Data Embedding - Embedding Model selected is OpenAI.")
            self.vector_db_s3: str = "OPENAI_FAISS_DB"
            self.vector_db_local: str = f"{self.ARTIFACTS_FOLDER}OPENAI_FAISS_DB"
            self.embeddings = OpenAIEmbeddings()
        elif self.model == "finetune":
            logging.info("Data Embedding - Embedding Model selected is HuggingFace Finetune model.")
            self.vector_db_s3: str = "FINETUNE_FAISS_DB"
            self.vector_db_local: str = f"{self.ARTIFACTS_FOLDER}FINETUNE_FAISS_DB"
            FINETUNE_MODEL_PATH: str = f"s3://{self.YOUR_S3_BUCKET_NAME}/FINETUNE/finetuned_model"
            self.embeddings = HuggingFaceEmbeddings(model_name = FINETUNE_MODEL_PATH)

# Main class for data embedding process
class DataEmbedding:
    def __init__(self, model):
        self.embedding_config=DataEmbeddingConfig(model)
    
    # Start the data embedding process
    def initiate_data_embedding(self):
        logging.info("Data Embedding - started")
        try:
            # Load dataset from S3 and save locally
            file_path = self.embedding_config.json_file_s3
            dataset = pd.read_json(file_path)
            dataset.to_json(self.embedding_config.json_file_local, orient='records')
            logging.info("Data Embedding - Json file loaded from AWS S3 to dataframe")

            # Load documents and create FAISS vector database
            loader = JSONLoader(file_path=self.embedding_config.json_file_local, jq_schema=".[]", text_content=False)
            documents = loader.load()
            vectors_db = FAISS.from_documents(documents, self.embedding_config.embeddings)
            logging.info("Data Embedding - FIASS DB created successfully")

            # Save vector database locally and upload to S3
            vectors_db.save_local(self.embedding_config.vector_db_local)
            logging.info("Data Embedding - FIASS DB Loading to AWS S3 started")
            upload_directory_to_s3(self.embedding_config.YOUR_S3_BUCKET_NAME, self.embedding_config.vector_db_s3, self.embedding_config.vector_db_local)

            # Remove working directory
            if os.path.exists(self.embedding_config.WORKING_DIR):
                shutil.rmtree(self.embedding_config.WORKING_DIR)
                print(f"{self.embedding_config.WORKING_DIR} folder removed successfully.")    

            logging.info("Data Embedding - completed")
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    # Select the model for embedding
    model = "google"
    #model = "openai"
    #model = "finetune"
    obj=DataEmbedding(model)
    obj.initiate_data_embedding()
