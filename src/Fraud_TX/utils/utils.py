# utils.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score





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
    

def evaluate_model(x_train,x_test,y_train,y_test,classifiers,param_dist):
    try:

        report = {}

        for i in range(len(list(classifiers))):
            model = list(classifiers.values())[i]
            parameter = param_dist[list(classifiers.keys())[i]]

            logging.info(f"Hyper-parameter tunning started")
            random_search = RandomizedSearchCV(model, parameter, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
            random_search.fit(x_train,y_train)

            model.set_params(**random_search.best_params_)
            model.fit(x_train,y_train)

            
            logging.info(f"Cross Validation for {model} is started")
            CV_score = cross_val_score(model, x_train, y_train, cv=5)
            logging.info(f"Cross Validation for {model} is completed")
            print(f"{model} Cross Validation Score: {round(CV_score.mean() * 100, 2).astype(str)} % ")

            logging.info(f"Training part for {model} is completed")

            #y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            #train_clf_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(classifiers.keys())[i]] = test_model_score
        
        return report


    except Exception as e:
        logging.error("Exception occured at evaluate_model stage in utils.py file")
        raise customexception(e,sys)