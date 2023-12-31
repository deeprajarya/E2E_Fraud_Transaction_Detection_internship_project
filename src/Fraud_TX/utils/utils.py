import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

def evaluate_model(x_train, y_train, x_test, y_test, classifiers):
    try:
        report = {}

        param_list = {
            'LogisticRegression': {"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            'KNearest': {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'SupportVectorClassifier': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
            'DecisionTreeClassifier': {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),"min_samples_leaf": list(range(5, 7, 1))},
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                }
            }


        for clf_name, clf in classifiers.items():
            try:
                random_search = RandomizedSearchCV(clf, param_distributions=param_list[clf_name], n_iter=10, cv=5, scoring='accuracy', random_state=42)
                random_search.fit(x_train, y_train)
                print(f"Best Parameters for {clf_name}: {random_search.best_params_}")

                # Update the best estimator in classifiers dictionary
                classifiers[clf_name] = random_search.best_estimator_

                # Train the model
                classifiers[clf_name].fit(x_train, y_train)

                # Predict Testing data
                y_test_pred = classifiers[clf_name].predict(x_test)
                test_model_score = accuracy_score(y_test, y_test_pred)
                [clf_name] = test_model_score

                print("\n------------------------------------------------------------------")

                for key, clf in classifiers.items():
                    print(f'Test Accuracy of {clf} : \n-- {round(test_model_score * 100, 2)}%')
                    y_pred_train = clf.predict(x_train)
                    y_pred_test = clf.predict(x_test)
                print("\n------------------------------------------------------------------")

            except Exception as clf_error:
                print(f"Error occurred while training {clf_name} classifier: {clf_error}")

        return report, y_pred_train, y_pred_test

    except Exception as e:
        logging.error('Exception occurred during model training in utils.py file')
        raise customexception(e, sys)




'''def evaluate_model(x_train,y_train,x_test,y_test,classifiers):
    try:
        report = {}
        for i in range(len(classifiers)):
            model = list(classifiers.values())[i]
            # Train model
            model.fit(x_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(x_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(classifiers.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    



    
def evaluate_model(x_train, y_train, x_test, y_test, classifiers):
    try:
        report = {}

        param_list = {
            'LogisticRegression': {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            'KNearest': {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'SupportVectorClassifier': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
            'DecisionTreeClassifier': {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                                       "min_samples_leaf": list(range(5, 7, 1))},
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }

        for clf_name, clf in classifiers.items():
            y_pred_train = None
            y_pred_test = None
            try:
                random_search = RandomizedSearchCV(clf, param_distributions=param_list[clf_name], n_iter=10, cv=5,
                                           scoring='accuracy', random_state=42)
                print(f"Best Parameters for {clf_name}: {random_search.best_params_}")
                random_search.fit(x_train, y_train)

                # Update the best estimator in classifiers dictionary
                classifiers[clf_name] = random_search.best_estimator_

                logging.info(" Here prediction stage is started (in utils.py file)")
                # Predict Testing data
                y_test_pred = random_search.predict(x_test)
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[clf_name] = test_model_score

                print("\n------------------------------------------------------------------")

                for clf_name,clf in classifiers.items():
                    print(f'Test Accuracy of {clf} : \n-- {round(test_model_score * 100, 2)}%')
                    y_pred_train = clf.predict(x_train)
                    y_pred_test = clf.predict(x_test)
                print("\n------------------------------------------------------------------")

                logging.info("Prediction stage is completed")

            except Exception as clf_error:
                print(f"Error occurred while training {clf_name} classifier: {clf_error}")

        return report, y_pred_train, y_pred_test

    except Exception as e:
        logging.error('Exception occurred during model training in utils.py file')
        raise customexception(e, sys)




def evaluate_model(x_train,y_train,x_test,y_test,classifiers):
    try:
        report = {}

        param_list = {
                    'LogisticRegression': {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                    'KNearest': {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                    'SupportVectorClassifier': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
                    'DecisionTreeClassifier': {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)), "min_samples_leaf": list(range(5, 7, 1))},
                    'RandomForestClassifier': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                }
        

        for i in range(len(classifiers)):
            model = list(classifiers.values())[i]
            # Hyper-Parameter tuning for Train model
            random_search = RandomizedSearchCV(clf, param_distributions=param_list[i], n_iter=10, cv=5, scoring='accuracy', random_state=42)
            print(f"Best Parameters for {i}: {random_search.best_params_}")
            
            # Update the best estimator in classifiers dictionary
            classifiers[i] = random_search.best_estimator_
            
            model.fit(x_train,y_train)

            

            # Predict Testing data
            y_test_pred = model.predict(x_test)
            test_model_score = accuracy_score(y_test,y_test_pred)
            report[list(classifiers.keys())[i]] =  test_model_score

            print("\n------------------------------------------------------------------")

            for key, clf in classifiers.values():
                print(f'Test Accuracy of {clf} : \n-- {round(test_model_score * 100, 2)}%')
                y_pred_train = clf.predict(x_train)
                y_pred_test = clf.predict(x_test)
            print("\n------------------------------------------------------------------")

        return report

    except Exception as e:
        logging.error('Exception occured during model training in utils.py file')
        raise customexception(e,sys)
'''   


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Exception Occured in load_object function in utils.py file')
        raise customexception(e,sys)

    