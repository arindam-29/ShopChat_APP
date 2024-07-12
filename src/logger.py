import logging  # Used for logging messages in a standardized format
import os  # Provides a way of using operating system dependent functionality
from datetime import datetime  # Used to work with dates as date objects

# Generate a log file name based on the current date and time
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create a path for the logs directory and the log file
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

# Ensure the directory for the log file exists; create it if it does not
os.makedirs(logs_path,exist_ok=True)

# Correct the LOG_FILE_PATH to point to the directory, not the file inside itself
LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

# Configure the logging module to write to the log file with a specific format and level
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO) # Sets the logging level to INFO

# Main block to execute a test log message if this script is run directly
if __name__=="__main__":
    logging.info("Log written")