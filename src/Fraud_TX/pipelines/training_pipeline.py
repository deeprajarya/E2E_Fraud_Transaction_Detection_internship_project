from src.Fraud_TX.components.data_ingestion import DataIngestion
from src.Fraud_TX.components.data_transformation import perform_data_transformation
from src.Fraud_TX.components.model_trainer import ModelTrainer
from src.Fraud_TX.components.model_evaluation import ModelEvaluation

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
import pandas as pd


# Load and ingest data
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

# Read ingested data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Perform data transformation
data_transformation=DataTransformation()

transformed_train_data = perform_data_transformation(train_data)
transformed_test_data = perform_data_transformation(test_data)

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)


# Split data into features and target
X_train = transformed_train_data.drop('Class', axis=1)
y_train = transformed_train_data['Class']
X_test = transformed_test_data.drop('Class', axis=1)
y_test = transformed_test_data['Class']

# Initialize ModelTrainer
model_trainer = ModelTrainer(X_train.values, y_train.values, X_test.values, y_test.values)
model_trainer_obj.initate_model_training(train_arr,test_arr)


model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)


# Train models and get best parameters
best_model = model_trainer.train_models()

# Save the best model to a file for future reference
with open("best_parameters.json", "w") as json_file:
    json.dump(best_model, json_file)
