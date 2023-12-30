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




@dataclass 

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    transformed_data_path = os.path.join('artifacts', 'transformed_data.csv')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.classifiers = {
            "LogisiticRegression": LogisticRegression(),
            "KNearest": KNeighborsClassifier(),
            "Support Vector Classifier": SVC(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier()
        }

    def initiate_model_training(self):

        try:
            logging.info('Model training initiated')

            # Read the transformed data
            transformed_data = pd.read_csv(self.model_trainer_config.transformed_data_path)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X = transformed_data.drop('Class', axis=1)
            Y = transformed_data['Class']

            logging.info("Split the data into training and testing sets")
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

            # Turn the values into an array for feeding the classification algorithms.
            x_train = x_train.values
            x_test = x_test.values
            y_train = y_train.values
            y_test = y_test.values

            # Initialize the scaler
            scaler = RobustScaler()
        

            # Fit and transform the training data
            x_train = scaler.fit_transform(x_train)

            # Transform the test data
            x_test = scaler.fit_transform(x_test)

            logging.info("Train multiple classifiers in model_training stage")

            classifiers = self.classifiers

        

            for key, classifier in classifiers.items():
                classifier.fit(x_train,y_train)
                training_score = cross_val_score(classifier, x_train, y_train,cv=10)
                print()
                print(classifier, " : Training accuracy score is", round(training_score.mean(),2)*100, "%")

        

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



            # Save the best model using joblib
            model_file = os.path.join('artifacts', best_model_name + ".joblib")
            joblib.dump(best_model, model_file)




            def bestparameter():
                param_list = {
                    'log_reg': {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                    'knears_neighbors': {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                    'svc': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
                    'tree_clf': {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)), "min_samples_leaf": list(range(5, 7, 1))},
                    'forest_clf': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                }

                for clf_name, clf in classifiers.items():
                    random_search = RandomizedSearchCV(clf, param_distributions=param_list[clf_name], n_iter=10, cv=5, scoring='accuracy', random_state=42)
                    random_search.fit(x_train, y_train)
                    print(f"Best Parameters for {clf_name}: {random_search.best_params_}")

                    # Update the best estimator in classifiers dictionary
                    classifiers[clf_name] = random_search.best_estimator_

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Metrics Evaluation (accuracy score) started")

            def eval_metrics(self,y_test, y_pred):
                accu_score = accuracy_score(y_test, y_pred)
                return accu_score
            
            logging.info("Model Validation stage started")

            def initiate_model_validation(self,x_train,y_train):
                print("\n\n==============================================================")
                for clf_name, clf in classifiers.items():
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


            

        except Exception as e:
            logging.info('Exception occured at Model Training stage')
            custom_exception = customexception(sys, error_details=str(e))
            return custom_exception
            
        return self
            
        
        




    def best_model():

        model_trainer_instance = ModelTrainer()
        trained_models = model_trainer_instance.best_model()
        # we can save the trained models 
        '''
        for clf_name, clf in trained_models.items():
            joblib.dump(clf, f"artifacts/clf_name.pkl")
        '''
            





    

print(" 'mode'_trainer.py' file run successfully")