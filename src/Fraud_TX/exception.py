import sys
import traceback   

class customexception(Exception):
    def __init__(self, original_exception, error_details=None):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.error_details = error_details


    def __str__(self):
        if self.error_details:
            return f"Custom Exception: {self.error_details}\n{self.traceback}"
        else:
            return f"Custom Exception: {self.args}"


if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        raise customexception(e,sys)