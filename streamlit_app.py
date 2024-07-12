import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.utils import download_directory_from_s3
import pandas as pd
from streamlit_feedback import streamlit_feedback
from streamlit_option_menu import option_menu
import os
import logging
import time
import json
from PIL import Image
import requests
from io import BytesIO
import grpc
from langchain.globals import set_verbose
set_verbose(True)

# Adjust the root logging level if necessary (this affects all logging)
logging.basicConfig(level=logging.ERROR)

# Specifically adjust gRPC logging level to suppress informational messages
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

load_dotenv()
# Set the API key for Google GenAI LLM
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))      # Set the API key for Google GenAI LLM
# Set the API key for Google Embeddings
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Set the API key for Google Embeddings
# Set the API key for OpenAI LLM and Embeddings
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set the API key for OpenAI LLM and Embeddings

# Define constants for S3 bucket names and local folder paths
ABO_BUCKET_NAME:      str = os.getenv("ABO_BUCKET_NAME")
SHOPCHAT_BUCKET_NAME: str = os.getenv("SHOPCHAT_BUCKET_NAME")
YOUR_S3_BUCKET_NAME:  str = os.getenv("YOUR_S3_BUCKET_NAME")

app_folder_project:   str = "./artifacts/project/"
app_folder_your:      str = "./artifacts/your/"
folder_path_project:  str = "artifacts/project/"
folder_path_your:     str = "artifacts/your/"

# Define URLs for image and JSON file locations
imgae_url:            str = f"https://{ABO_BUCKET_NAME}.s3.amazonaws.com/images/small/"
json_s3_project:      str = f"s3://{SHOPCHAT_BUCKET_NAME}/PROCESSED/dataset.json"
json_s3_your:         str = f"s3://{YOUR_S3_BUCKET_NAME}/PROCESSED/dataset.json"
json_local_project:   str = f"{folder_path_project}/dataset.json"
json_local_your:      str = f"{folder_path_your}/dataset.json"

# Define constants for FAISS database names
GOOGLE_FIASS_DB:      str = "GOOGLE_FAISS_DB"
OPENAI_FAISS_DB:      str = "OPENAI_FAISS_DB"
FINETUNE_FAISS_DB:    str = "FINETUNE_FAISS_DB"

# Create the app artifacts folder if it doesn't exist
if not os.path.exists(app_folder_project):
    os.makedirs(app_folder_project)
if not os.path.exists(app_folder_your):
    os.makedirs(app_folder_your)

# Define the chat prompt template
prompt     = ChatPromptTemplate.from_template(
"""

You are a Helpful sales recommender who will have a casual and friendly chat with the buyer to find out what they are looking for.
Always introduce yourself as "ShopChat recommender" but Instead of asking directly what kind of products they are looking for, 
have a casual conversation to understanding the need of the buyer. Don't start the conversation like a salesperson. 
When you are confident that you have enough information, recommend the product to the buyer. You also return possible recommendations 
when you are not fully confident but somewhat confident. 

You will always respond in the JSON schema presented below, always return all fields:

Limit your answer to 16,385 tokens.

{{
    "confidence": "string", # This is where you explain how confident you are in your recommendation, this is only for you, can be low, medium, high
    "recommendation": "list", # This is where you recommend a list of the products to the buyer, only return if your confidence is medium or high. this can only be the name of the product and nothing else. if you are at medium or high confidence always return a recommendation
    "your_response_to_buyer": "string" # This is where you respond to the buyer, this is what the buyer will see. Only recommend the product to the user in this response as list of products if your confidence is high. Always have a response for the user in all cases
    "item_id": "list", # This is where you will add the list of "item_id" from the context documents for the recomanded products.
    "domain_name": "list", # This is where you will add the list of "domain_name" from the context documents for the recomanded products.
    "image_id": "list", # This is where you will add the list of "path" from the context documents for the recomanded products.
}}

Here are the products in stock with Amazon:

Answer the questions based on the provided context only, do not make it up if not in context documents.

<context>
{context}
<context>
Questions:{input}

"""
)

# Function to get an image from a given path
def get_image(image_path):
   
    url = imgae_url + image_path
    response = requests.get(url)
    image_pil = Image.open(BytesIO(response.content))

    return image_pil

# Function to download the data from S3 bucket
def load_data(json_s3, json_local, local_folder_path, bucket_name, app_folder):
    try:
        if not os.path.exists(json_local):
            json_file = pd.read_json(json_s3)
            json_file.to_json(json_local, orient='records')
        else:
            json_file = pd.read_json(json_local)

        if not os.path.exists(os.path.join(local_folder_path, GOOGLE_FIASS_DB)):
            download_directory_from_s3(bucket_name, GOOGLE_FIASS_DB, app_folder)
            
        if not os.path.exists(os.path.join(local_folder_path, OPENAI_FAISS_DB)):
            download_directory_from_s3(bucket_name, OPENAI_FAISS_DB, app_folder)
        return json_file
    except Exception as e:
        st.markdown(f"Error in loading data from S3 bucket: {bucket_name}. {e}")

# Function to Load the Embeddings and LLM models
def load_llm_vectordb(local_folder_path):
    # Google Geai LLM
    embeddings   = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector       = FAISS.load_local(f"{local_folder_path}{GOOGLE_FIASS_DB}", embeddings, allow_dangerous_deserialization=True)
    retriever_g  = vector.as_retriever()
    LLM_g        = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    # OpenAI LLM
    embeddings   = OpenAIEmbeddings()
    vector       = FAISS.load_local(f"{local_folder_path}{OPENAI_FAISS_DB}", embeddings, allow_dangerous_deserialization=True)
    retriever_o  = vector.as_retriever(ssearch_type="mmr", search_kwargs={'k': 2})
    LLM_o        = OpenAI(temperature=0)
    return retriever_g, LLM_g, retriever_o, LLM_o

# START STREAMLIT APP ############################################################################################
# Streamlit app title
st.title("Welcome to ShopChat by Amazon!!")

# Sidebar to select File and DB option
with st.sidebar:
    selected = option_menu("Selection Menu", ["Home", "Upload Your S3"], 
        icons=['house', 'cloud-upload'], menu_icon="cast", default_index=0)
    
    # By default streamlit selects "Home" option
    if selected == 'Home':
        st.markdown(f"Bucket used: {SHOPCHAT_BUCKET_NAME}")
        json_file = load_data(json_s3_project, json_local_project, folder_path_project, SHOPCHAT_BUCKET_NAME, app_folder_project)
        retriever_g, LLM_g, retriever_o, LLM_o = load_llm_vectordb(folder_path_project)
   
    if selected == "Upload Your S3":    
        json_file = load_data(json_s3_your, json_local_your, folder_path_your, YOUR_S3_BUCKET_NAME, app_folder_your)
        retriever_g, LLM_g, retriever_o, LLM_o = load_llm_vectordb(folder_path_your)
        st.markdown("**Refreshed data your S3 bucket**")
        st.markdown(f"Bucket name: {YOUR_S3_BUCKET_NAME}")
        st.markdown(f"To reload {SHOPCHAT_BUCKET_NAME} clich Home")

# Define and select the model options
options = ["Google Gen AI", "Open AI", "Finetune"]  # Define the options for the selection
selected_option = st.radio("Choose the Model:", options)  # Create the radio button and store the selected option

# Model selection and setup
if selected_option == "Google Gen AI" or selected_option == "Finetune":
    retriever  = retriever_g
    LLM        = LLM_g
elif selected_option == "Open AI":
    retriever  = retriever_o
    LLM        = LLM_o

# Streamlit session state initialization
if 'stage' not in st.session_state:
    st.session_state.stage = 0

# User query input
query   = st.text_input("What are you buying today.........?")

# Search button
if st.button("Search"):
    st.session_state.stage = 1

# Response handling
if st.session_state.stage == 1:

    document_chain  = create_stuff_documents_chain(LLM,prompt)
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start           = time.process_time()
    response        = retrieval_chain.invoke({'input':query})

    json_response   = json.loads(response['answer'])

    print("Response time :",time.process_time()-start)

    st.markdown(json_response['your_response_to_buyer'])

    if json_response["item_id"]:

        for item_id, domain_name, image_id, recomand in zip(json_response["item_id"], json_response["domain_name"], json_response["image_id"], json_response["recommendation"]):
            
            if image_id:
                with st.form(key=item_id):
                    # Get the image from the image path
                    image_pil = get_image(image_id)
                    st.markdown(f"***{recomand}***")
                    st.image(image=image_pil, width=300)
 
                    # Display more images if available
                    more_images_paths = json_file[json_file["item_id"] == item_id]["other_image_id_path"].to_list()[0]
                    if more_images_paths:
                        with st.expander(f"More Images of {recomand}"):
                            more_images = []
                            for more_images_path in more_images_paths:
                                image_pil = get_image(more_images_path)
                                more_images.append(image_pil)
                            st.image(more_images, width=75)
                    
                    # Display the product URL
                    st.markdown(f"*Click to buy :  https://www.{domain_name}/dp/{item_id}*")
                    
                    # Display the feedback form
                    feedback = streamlit_feedback(feedback_type="thumbs",
                                       optional_text_label="[Optional] Enter your comments here", 
                                       align="flex-start", 
                                       key=f"feedback_{item_id}",)
                    st.form_submit_button('If you like the product, give us 👍 and provide your valuable feedback to help us improve!!!!', disabled=True)
