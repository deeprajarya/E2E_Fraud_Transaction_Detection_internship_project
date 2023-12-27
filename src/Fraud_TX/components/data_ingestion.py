import pandas as pd
import numpy as np
import os
import sys

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Move two levels up to the project root
project_root = os.path.abspath(os.path.join(script_path, "..", ".."))
sys.path.append(project_root)

from Fraud_TX.logger import logging
from Fraud_TX.exception import customexception




from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","creditcard.csv")))
            logging.info(" I have read dataset as dataframe")
            
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" I have saved the raw dataset in artifact folder")
            
        
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
           logging.info("exception occured at data ingestion stage")
           raise customexception(e,sys)
    