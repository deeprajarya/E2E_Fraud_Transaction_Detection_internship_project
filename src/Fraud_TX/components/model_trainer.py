import sys
import os
import joblib
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
from sklearn.model_selection import cross_val_score
from src.Fraud_TX.utils.utils import save_object, evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        best_model = None
        try:
            logging.info('Model Training stage started')

            logging.info('Splitting data into x_train, x_test, y_train, y_test for Model Training ')

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info('Splitting data into x_train, x_test, y_train, y_test for Model Training is completed')

            classifiers = {
                "LogisticRegression": LogisticRegression(solver='liblinear'),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                #"svm" : SVC(),
                #"RandomForestClassifier": RandomForestClassifier()
            }

            param_dist = {
                'LogisticRegression': {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                'DecisionTreeClassifier': {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                                            "min_samples_leaf": list(range(5, 7, 1))},
                #'svm': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
                
            }

            model_report:dict = evaluate_model(x_train,x_test,y_train,y_test,classifiers,param_dist)
            
            # print model report
            print(model_report)

            # to get the best model score from the dictonary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = classifiers[best_model_name]
            print('\n====================================================================================')
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('======================================================================================')
            logging.info(f'Best Model Found \n - Model Name : {best_model_name} ,\n - Accuracy Score : {best_model_score}')


    

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


        except Exception as e:
            logging.error('Exception occurred at Model Training stage')
            raise customexception(e, sys)

print("'model_trainer.py' file run successfully")
