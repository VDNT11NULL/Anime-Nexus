import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__()
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    # Staticmethod to avoid passing self
    def get_detailed_error_message(error_message, error_detail:sys):
        # returns 3 things, but we interested in 3rd :--> TraceBack
        _,_, exc_tb = traceback.sys.exc_info()  
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error occured in {file_name}, Line no. {line_number} : {error_message}"

    def __str__(self):
        return self.error_message