import sys
import traceback

'''
class customexception(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()
        
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename 
    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
'''       

class customexception(Exception):
    def __init__(self, sys, error_details=None):
        self.error_details = error_details
        if error_details:
            self.traceback = traceback.format_exc()
        else:
            self.traceback = None
        super().__init__(sys)

    def __str__(self):
        if self.error_details:
            return f"Custom Exception: {self.error_details}\n{self.traceback}"
        else:
            return f"Custom Exception: {self.args}"


if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        raise customexception(e,sys)