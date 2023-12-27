
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Fraud_TX.exception import customexception
from src.Fraud_TX.logger import logging


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from src.Fraud_TX.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    transformed_data_path = "artifacts/transformed_data.csv"

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        

    def perform_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            # Read the raw data
            raw_data = pd.read_csv(self.raw_data_path)

            # Perform data transformations
            scaled_data = self.scale_features(raw_data)

            # Save the transformed data
            scaled_data.to_csv(self.transformation_config.transformed_data_path, index=False)

            return self.transformation_config.transformed_data_path
        

        except:
            logging.info("Exception occured in the perform_data_transformation function")

            raise customexception(e,sys)
        



    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data ")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.perform_data_transformation()

            target_column_name = 'Class'
            
            std_scaler = StandardScaler()
            rob_scaler = RobustScaler()

            df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
            df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

            df.drop(['Time', 'Amount'], axis=1, inplace=True)

            return df
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation stage")

            raise customexception(e,sys)




print(" 'data_transformation.py' File Run sucessfully")