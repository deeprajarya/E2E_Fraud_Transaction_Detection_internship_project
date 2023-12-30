import pandas as pd
from src.Fraud_TX.components.data_ingestion import DataIngestion
from src.Fraud_TX.components.data_transformation import DataTransformation
from src.Fraud_TX.components.model_trainer import ModelTrainer

import os
import sys
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
import json
import joblib



# Load and ingest data
data_ingestion = DataIngestion()
raw_data_path = data_ingestion.initiate_data_ingestion()

# Read ingested data
raw_data = pd.read_csv(raw_data_path)

# Perform data transformation
data_transformation=DataTransformation()
raw_data_arr = data_transformation.initialize_data_transformation(raw_data_path)



# Initialize ModelTrainer ( Training and Validation)
model_trainer = ModelTrainer()
model_trainer_instance = model_trainer.initiate_model_training()   # train_arr,test_arr

# Access the classifiers attribute (it's a dictionary, not a callable function)
classifiers = model_trainer_instance.classifiers

# Train models and get best parameters
best_model = model_trainer_instance.classifiers


# Save the best model to a file for future reference
best_model_name = "best_model.joblib"
with open(best_model_name, "wb") as model_file:
    joblib.dump(best_model, f"artifacts/best_model.joblib")





