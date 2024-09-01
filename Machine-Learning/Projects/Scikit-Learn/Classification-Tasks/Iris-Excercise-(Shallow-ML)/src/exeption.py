import sys
from logger import logging

def error_message_detail(error, error_detial:sys):
    _, _, exc_tb = error_detial.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in Python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{error}]"

    return error_message

class CustomExeption(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Divide by zero")
#         raise CustomExeption(e, sys)