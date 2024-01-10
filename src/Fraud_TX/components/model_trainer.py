
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception




class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self, train_array,test_array):
        try:
            logging.info('Model Training stage started')

            logging.info('Splitting data into x_train, x_test, y_train, y_test for Model Training ')

            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Splitting data into x_train, x_test, y_train, y_test for Model Training is completed')

    

            classifiers = {
                "RandomForestClassifier": RandomForestClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "LogisticRegression": LogisticRegression(solver='liblinear')
            }

            results = {}  # Dictionary to store results

            logging.info("Hyper-paramter tunning stage is started")

            for clf_name, clf in classifiers.items():
                param_dist = self.get_param_dist(clf_name)

                # Use RandomizedSearchCV for hyperparameter tuning
                random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', random_state=42,n_jobs=4)
                random_search.fit(x_train, y_train)

                best_estimator = random_search.best_estimator_

                # Get Training and Testing Accuracy
                train_accuracy = best_estimator.score(x_train, y_train)
                test_accuracy = accuracy_score(y_test, best_estimator.predict(x_test))

                logging.info(f"{clf_name} - Training Accuracy: {train_accuracy * 100:.2f}%, Testing Accuracy: {test_accuracy * 100:.2f}%")

                '''# Save the model
                model_filename = f"{artifacts}/{clf_name}_Best_Model.pkl"
                joblib.dump(best_estimator, model_filename)'''

                # Save the best model from RandomizedSearchCV
                joblib.dump(best_estimator, f"artifacts/{clf_name}_Best_Model_RandomizedSearchCV.pkl")

                # Print additional information
                print(f"  Best Parameters from RandomizedSearchCV for {clf_name}: {random_search.best_params_}")
                print(f"  Best Cross-Validated Accuracy from RandomizedSearchCV for {clf_name}: {random_search.best_score_:.4f}")
                print()

                results[clf_name] = {
                    'training_accuracy': train_accuracy,
                    'testing_accuracy': test_accuracy,
                    #'model_filename': model_filename
                }

            logging.info("Model Training stage is completed")

            return results

        except Exception as e:
            logging.error('Exception occurred at Model Training stage')
            raise customexception(e, sys)

    def get_param_dist(self, clf_name):
        print("Entering get_param_dist")
        logging.info(" Defining hyperparameter search space based on classifier")
        if clf_name == "RandomForestClassifier":
            print("Entering get_param_dist for RandomForestClassifier")
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif clf_name == "KNeighborsClassifier":
            print("Entering get_param_dist for KNeighborsClassifier")
            return {
                'n_neighbors': list(range(2, 5, 1)),
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

        elif clf_name == "SVC":
            print("Entering get_param_dist for SVC")
            return {
                'C': [0.5, 0.7, 0.9, 1],
                'kernel': ['rbf', 'linear']
            }
        elif clf_name == "DecisionTreeClassifier":
            print("Entering get_param_dist for DecisionTreeClassifier")
            return {
                "criterion": ["gini", "entropy"],
                "max_depth": list(range(2, 4, 1)),
                "min_samples_leaf": list(range(5, 7, 1))
            }
        elif clf_name == "LogisticRegression":
            print("Entering get_param_dist for LogisticRegression")
            return {
                'penalty': ['l1', 'l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
        logging.info(" Defining hyperparameter search space is completed")
