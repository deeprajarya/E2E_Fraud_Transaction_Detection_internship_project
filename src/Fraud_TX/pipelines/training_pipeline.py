
from src.Fraud_TX.components.data_ingestion import DataIngestion
from src.Fraud_TX.components.data_transformation import DataTransformation
from src.Fraud_TX.components.model_trainer import ModelTrainer
#from src.Fraud_TX.components.model_evaluation import ModelEvaluator


import os
import sys
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

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)



# Initialize ModelTrainer
model_trainer = ModelTrainer()
model_trainer.initiate_model_training()   # train_arr,test_arr


#model_eval_obj = ModelEvaluation()
#model_eval_obj.initiate_model_evaluation(train_arr,test_arr)


# Train models and get best parameters
best_model = model_trainer.trained_classifiers()

# Save the best model to a file for future reference
with open("best_model.json", "w") as json_file:
    json.dump(best_model, json_file)
