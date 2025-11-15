import sys
from src.logger import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exx_info()
    file_name=exc_tb.tb_frame.f_co_filename