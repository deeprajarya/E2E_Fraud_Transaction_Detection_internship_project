import sys
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
from sklearn.model_selection import cross_val_score

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self, train_array, test_array):
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
            }

            param_dist = {
                'LogisticRegression': {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                'DecisionTreeClassifier': {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                                            "min_samples_leaf": list(range(5, 7, 1))},
            }

            best_model_name = None
            best_model = None
            best_test_accuracy = 0  # Initialize to a lower value

            logging.info("Hyper-parameter tuning stage is started \n")

            for clf_name, clf in classifiers.items():

                # Use RandomizedSearchCV for hyperparameter tuning
                param_dist_for_clf = param_dist.get(clf_name, {})  # Get parameters for the classifier, or an empty dictionary if not found

                if param_dist_for_clf:

                    grid_search = RandomizedSearchCV(clf, param_dist_for_clf, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
                    logging.info(f"\n -- Defining hyperparameter search space for {clf_name} is completed")

                    grid_search.fit(x_train, y_train)

                    best_estimator = grid_search.best_estimator_

                    CV_score = cross_val_score(best_estimator, x_train, y_train, cv=5)
                    logging.info(f"Cross Validation for {clf_name} is completed")
                    print(f"{clf_name} Cross Validation Score: {round(CV_score.mean() * 100, 2).astype(str)} % ")

                    # Get Training and Testing Accuracy
                    train_accuracy = best_estimator.score(x_train, y_train)
                    test_accuracy = accuracy_score(y_test, best_estimator.predict(x_test))

                    logging.info(f"{clf_name} : Training Accuracy: {train_accuracy * 100:.2f}%, Testing Accuracy: {test_accuracy * 100:.2f}%")

                    # Save the best model
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        best_model_name = clf_name
                        best_model = best_estimator
                        

                    # Print additional information
                    print(f"  Best Parameters from RandomizedSearchCV for {clf_name}: {grid_search.best_params_}")
                    logging.info(f"All steps of Training and Validation completed for {clf_name}. Now moving to next Classifier")

            logging.info("Model Training stage is completed")

            if best_model_name:
                joblib.dump(best_model, f"artifacts/best_model.joblib")
                joblib.dump(best_model, f"artifacts/best_model.pkl")
                print(f"\nBest Model: {best_model_name}, Testing Accuracy: {best_test_accuracy * 100:.2f}%")
                print(f"Best Model saved in 'artifacts' folder as 'best_model.joblib' and 'best_model.pkl'")
            else:
                print("No best model found.")

            return best_model_name, best_test_accuracy

        except Exception as e:
            logging.error('Exception occurred at Model Training stage')
            raise customexception(e, sys)

print("'model_trainer.py' file run successfully")
