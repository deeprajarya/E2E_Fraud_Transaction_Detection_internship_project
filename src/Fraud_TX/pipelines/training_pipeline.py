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
