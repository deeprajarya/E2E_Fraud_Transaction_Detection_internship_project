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
import joblib

def evaluate_model(x_train, x_test, y_train, y_test, classifiers):
    model_report = {}
    
    best_test_accuracy = 0.0
    best_clf_name = None

    for clf_name, clf in classifiers.items():
        # Training accuracy
        train_pred = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Testing accuracy
        test_pred = clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Other metrics (precision, recall, f1-score)
        precision = precision_score(y_test, test_pred)
        recall = recall_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred)
        
        model_report[clf_name] = {
            'Training Accuracy': train_accuracy,
            'Testing Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        
        # Save the best model based on testing accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_clf_name = clf_name
            best_model_filename = f"artifacts/{clf_name}_Best_Model.pkl"
            joblib.dump(clf, best_model_filename)

    return model_report, best_test_accuracy, best_clf_name, best_model_filename




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Exception Occured in load_object function in utils.py file')
        raise customexception(e,sys)