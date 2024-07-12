# create custom error message handler 

import sys  # Used for accessing system-specific parameters and functions
from src.logger import logging  # Import custom logging module for logging

# Define a function to extract error message details
def error_message_details(error, error_details:sys):
    # Extract traceback information
    _,_,exc_tb=error_details.exc_info()
    # Retrieve the filename where the error occurred
    file_name=exc_tb.tb_frame.f_code.co_filename #get the error file name name; look at the python custom exception handling doc in the internet
    # Format the error message with file name, line number, and error message
    error_message="Error occured in script [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message

# Define a custom exception class
class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        # Store the detailed error message
        self.error_message=error_message_details(error=error_message,error_details=error_details)
    
    def __str__(self):
        # Return the detailed error message when the exception is printed
        return self.error_message

# Testing 
# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Division by zero")
#         raise CustomException(e, sys)