import os
import sys
from pathlib import Path



from Fraud_TX.logger import logging


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)





class ModelTrainer:
    def __init__(self, transformed_data_path):
        self.transformed_data_path = transformed_data_path

    def train_models(self):
        # Read the transformed data
        transformed_data = pd.read_csv(self.transformed_data_path)

        # Split the data into features (X) and target (y)
        X = transformed_data.drop('Class', axis=1)
        y = transformed_data['Class']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train multiple classifiers
        classifiers = {
            "RandomForestClassifier": self.train_random_forest(X_train, y_train),
            "KNeighborsClassifier": self.train_kneighbors(X_train, y_train),
            "SVC": self.train_svc(X_train, y_train),
            "DecisionTreeClassifier": self.train_decision_tree(X_train, y_train),
            "LogisticRegression": self.train_logistic_regression(X_train, y_train)
            # Add more classifiers as needed
        }

        # Evaluate and log the performance of each classifier
        for clf_name, clf in classifiers.items():
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"{clf_name} Accuracy: {accuracy * 100:.2f}%")

        return classifiers

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
        random_search = RandomizedSearchCV(random_forest, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
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

# Example Usage
if __name__ == "__main__":
    transformed_data_path = "path/to/your/transformed_data.csv"  # Replace with the actual path
    model_trainer_instance = ModelTrainer(transformed_data_path)
    trained_models = model_trainer_instance.train_models()
    # You can save the trained models if needed
    # for clf_name, clf in trained_models.items():
    #     joblib.dump(clf, f"path/to/save/{clf_name}.pkl")
