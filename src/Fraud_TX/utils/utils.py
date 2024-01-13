# utils.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error("Error occured at save object function in utils.py file")
        raise customexception(e, sys)
    


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Exception Occured in load_object function in utils.py file')
        raise customexception(e,sys)