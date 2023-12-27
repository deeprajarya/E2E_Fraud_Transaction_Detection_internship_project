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
from sklearn.model_selection import RandomizedSearchCV


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    transformed_data_path = os.path.join('artifacts', 'transformed_data.csv')




class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self):

        try:
            logging.info('Model training initiated')

            # Read the transformed data
            transformed_data = pd.read_csv(self.model_trainer_config.transformed_data_path)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X = transformed_data.drop('Class', axis=1)
            y = transformed_data['Class']

            logging.info("Split the data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Train multiple classifiers in model_training stage")

            trained_classifiers = {
                
                "RandomForestClassifier": self.train_random_forest(X_train, y_train),
                "KNeighborsClassifier": self.train_kneighbors(X_train, y_train),
                "SVC": self.train_svc(X_train, y_train),
                "DecisionTreeClassifier": self.train_decision_tree(X_train, y_train),
                "LogisticRegression": self.train_logistic_regression(X_train, y_train)
                # Add more classifiers as needed
                }
            

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,classifiers)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')



            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = trained_classifiers[best_model_name]


            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training stage')
            raise customexception(e,sys)



            



    def train_random_forest(self, X_train, y_train):
        # Set hyperparameters for Random Forest using RandomizedSearchCV
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize Random Forest Classifier
        random_forest = RandomForestClassifier()

        # Run RandomizedSearchCV
        random_search = RandomizedSearchCV(random_forest, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42,n_jobs=-1)
        random_search.fit(X_train, y_train)

        # Get the best estimator
        best_random_forest = random_search.best_estimator_

        return best_random_forest

    def train_kneighbors(self, X_train, y_train):
        # Initialize KNeighbors Classifier
        knn = KNeighborsClassifier(n_neighbors=5)

        # Fit the model
        knn.fit(X_train, y_train)

        return knn

    def train_svc(self, X_train, y_train):
        # Initialize Support Vector Classifier
        svc = SVC(kernel='rbf', C=1)

        # Fit the model
        svc.fit(X_train, y_train)

        return svc

    def train_decision_tree(self, X_train, y_train):
        # Set hyperparameters for Decision Tree using RandomizedSearchCV
        param_dist = {
            "criterion": ["gini", "entropy"],
            "max_depth": list(range(2, 4, 1)),
            "min_samples_leaf": list(range(5, 7, 1))
        }

        # Initialize Decision Tree Classifier
        decision_tree = DecisionTreeClassifier()

        # Run RandomizedSearchCV
        random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
        random_search.fit(X_train, y_train)

        # Get the best estimator
        best_decision_tree = random_search.best_estimator_

        return best_decision_tree

    def train_logistic_regression(self, X_train, y_train):
        # Initialize Logistic Regression Classifier
        log_reg = LogisticRegression(solver='liblinear')

        # Fit the model
        log_reg.fit(X_train, y_train)

        return log_reg
    
    def best_model():

        model_trainer_instance = ModelTrainer(transformed_data_path)
        trained_models = model_trainer_instance.best_model()
        # we can save the trained models 
        for clf_name, clf in trained_models.items():
            joblib.dump(clf, f"path/to/save/{clf_name}.pkl")




print(" 'mode'_trainer.py' file run successfully")