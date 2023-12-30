
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
from dataclasses import dataclass
from src.Fraud_TX.utils.utils import save_object
from src.Fraud_TX.utils.utils import evaluate_model
from src.Fraud_TX.components.data_ingestion import DataIngestion

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import os
import sys



@dataclass

class ModelValidationConfig:
    transformed_data_path = os.path.join('artifacts', 'transformed_data.csv')


class ModelValidation:
    
    def __init__(self):
        self.model_validation_config = ModelValidationConfig()
    
    

    def eval_metrics(self,y_test, y_pred):
        accu_score = accuracy_score(y_test, y_pred)
        return accu_score

        def initiate_model_validation(self,x_train,y_train):
            try:
                
                print("\n\n==============================================================")
                for clf_name, clf in .classifiers.items():
                    clf_score = cross_val_score(clf, x_train, y_train, cv=5)
        
                    print(f'Cross Validation Score of {clf} : \n-- {round(clf_score.mean() * 100, 2)}%')
                print("\n================================================================")

                # Assuming x_test and y_test are your test data
                for clf_name, clf in classifiers.items():
                    # Make predictions on the test set
                    y_pred = clf.predict(x_test)

                    # Evaluate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f'Test Accuracy of {clf} : \n-- {round(accuracy * 100, 2)}%')

            except Exception:
                return customexception(sys)

    