


import os
import sys
import numpy as np
import pandas as pd


from dataclasses import dataclass
from src.Fraud_TX.exception import customexception
from src.Fraud_TX.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from src.Fraud_TX.utils.utils import save_object
from pathlib import Path

import warnings

# Ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
   


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.train_df_num_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

    def get_data_transformation(self):
        try:
            logging.info("get_data_transformation stage started")


            logging.info("Pipeline initiated")

            num_pipeline=Pipeline(
                steps=[
                    ('rob_scaler',RobustScaler()),
                    ('std_scaler',StandardScaler())

                ]

            )

            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,self.train_df_num_cols)]
            )
            
            return preprocessor
            
        
        except Exception as e:
            logging.error(f"Exception occurred in the initialize_data_transformation stage: {e}")
            raise customexception(e, sys)
            
        

    def initialize_data_transformation(self,train_path,test_path):
        try:

            logging.info(" initiate_data_transformation stage initiated ")

            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(f'Train Dataframe Head : \n{train_df.head(2).to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head(2).to_string()}')

            

            # Print or log column names for debugging
            logging.info("Column names in the dataframe: %s", train_df.columns)
            logging.info("Columns specified in numerical_cols: %s", self.train_df_num_cols)
            

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'Class'
            drop_columns = ['Time', 'Amount']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

        

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            
            logging.info("Taking array of dataframe for better learning (fast processing)")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]




            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("preprocessing pickle file saved")

            return (
                train_arr,
                test_arr
            ) 

        except Exception as e:
            logging.error(f"Exception occurred in the initialize_data_transformation stage")
            raise customexception(e,sys)



print(" 'data_transformation.py' file run sucessfully")