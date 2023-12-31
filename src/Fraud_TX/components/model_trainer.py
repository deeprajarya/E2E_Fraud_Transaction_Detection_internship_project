import os
import sys
import pandas as pd
import numpy as np


from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
from dataclasses import dataclass
from src.Fraud_TX.utils.utils import save_object
from src.Fraud_TX.utils.utils import evaluate_model


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import warnings
import joblib
from typing import Dict, Any




@dataclass 

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','Trained_Model.pkl')
    



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

        

    def initiate_model_training(self,train_array,test_array):

        try:
            logging.info('Model Training stage started')
            logging.info('Splitting Dependent and Independent variables from train and test data')

            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            classifiers = {
                "LogisticRegression": LogisticRegression(),
                "KNearest": KNeighborsClassifier(),
                "Support Vector Classifier": SVC(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier()
            }

            print("Shape of train_array:", train_array.shape)
            print("Shape of test_array:", test_array.shape)


            logging.info("Training of multiple classifiers in model_training stage started")


            logging.info("To prevent over fitting, let's find best parameter and train again")
        
            
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,classifiers)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')



            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = classifiers[best_model_name]


            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model Training stage is completed")


            

        except Exception as e:
            logging.error('Exception occured at Model Training stage')
            raise customexception(e,sys)
            


print(" 'mode_trainer.py' file run successfully")