from src.Fraud_TX.components.data_ingestion import DataIngestion
from src.Fraud_TX.components.data_transformation import DataTransformation
from src.Fraud_TX.components.model_trainer import ModelTrainer
#from src.Fraud_TX.components.model_training import ModelTrainer

import os
import sys
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
import json
import joblib

# Load and ingest data
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

# Perform data transformation
train_path = "artifacts/train.csv"
test_path = "artifacts/test.csv"
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(train_path,test_path)

# Initialize ModelTrainer (Training and Validation)
model_trainer = ModelTrainer()  
model_trainer.initiate_model_training(train_arr,test_arr)



# Evaluate models and save the best one
model_report, best_test_accuracy, best_clf_name, best_model_filename = evaluate_model(train_arr['features'], test_arr['features'], train_arr['labels'], test_arr['labels'], results)

for clf_name, metrics in model_report.items():
    print(f"{clf_name}:")
    print(f"  Training Accuracy: {metrics['Training Accuracy']:.4f}")
    print(f"  Testing Accuracy: {metrics['Testing Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print()

print(f"Best Classifier: {best_clf_name}")
print(f"Best Classifier Testing Accuracy: {best_test_accuracy:.4f}")
print(f"Best Classifier Model Filename: {best_model_filename}")