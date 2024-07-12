import os
import sys
from src.exception import CustomException  # Custom exception module for handling exceptions
from src.logger import logging  # Custom logging module for structured logging
import logging  # Python's built-in logging module
import requests
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from dataclasses import dataclass  # Dataclass for defining simple data structures

@dataclass
class DataCaptioningConfig:
    # Load environment variables at the class level
    load_dotenv()
    
    # Configuration variables loaded from environment variables
    ABO_BUCKET_NAME:        str = os.getenv("ABO_BUCKET_NAME")
    YOUR_S3_BUCKET_NAME:    str = os.getenv("YOUR_S3_BUCKET_NAME")

    # Paths for data files
    imgae_url:            str = f"https://{ABO_BUCKET_NAME}.s3.amazonaws.com/images/small/"
    json_file_s3:         str = f"s3://{YOUR_S3_BUCKET_NAME}/PROCESSED/dataset.json"

    # Path for saving captioned data
    save_json_path:       str = f"s3://{YOUR_S3_BUCKET_NAME}/PROCESSED/dataset.json"

    # Define model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

class DataCaptioning:
    def __init__(self):
        # Initialize DataIngestion with configuration
        DataCaptioningConfig()
    
    def initiate_data_captioning(self):
        # Start the data ingestion process
        logging.info("Data Captioning - started")
        try:
            logging.info("Data Captioning - file loading from AWS s3 server - started")
            # Load the JSON file               
            dataset = pd.read_json(DataCaptioningConfig.json_file_s3)

            logging.info("Data Captioning - file loading from AWS s3 server - completed")

            # Function to extract the caption from the image using the Blip model
            def generate_caption_blip(raw_image):
                inputs = DataCaptioningConfig.processor(raw_image, return_tensors="pt")
                out = DataCaptioningConfig.model.generate(**inputs)
                return (DataCaptioningConfig.processor.decode(out[0], skip_special_tokens=True))
            
            # Function to get for the image the URL and generate caption
            def func_generate_(x):
                if x:
                    try: 
                        url: str = f"{DataCaptioningConfig.imgae_url}{x}"
                        raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
                        return generate_caption_blip(raw_image)
                    except UnidentifiedImageError:
                        return None
                else:
                    return None

            logging.info("Data Captioning - generating caption using Blip model- started")

            dataset = dataset.assign(image_caption=dataset.path.apply(func_generate_))
            
            logging.info("Data Captioning - caption generation - completed")

            logging.info("Data Captioning - saving to S3 - started")
            
            # Save the captioned dataset back to S3
            dataset.to_json(DataCaptioningConfig.save_json_path, orient='records')
           
            logging.info("Data Captioning - completed")

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    # Main execution block to start data captioning
    obj=DataCaptioning()
    obj.initiate_data_captioning()
    