import os
import sys
import numpy as np
import pandas as pd


from dataclasses import dataclass
from src.Fraud_TX.exception import customexception
from src.Fraud_TX.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.Fraud_TX.utils.utils import save_object
import warnings

# Ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
   


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def get_data_transformation(self):
        try:
            """ This function is responsible to perform data transformation """

            logging.info("get_data_transformation stage started")

            numerical_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']
            
            categorical_columns = []
            
            logging.info("num_pipeline initiate")
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),      # Useful to handel missing values
                ('scalar',StandardScaler()),
                ('rob_scaler',RobustScaler())
            ])
            
            logging.info("cat_pipeline initiate")
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),      # Useful to handel missing values
                ('one_hot_encoder',OneHotEncoder()),
                ('scalar',StandardScaler(with_mean=False))
            ])

            logging.info(f" Numerical Columns : {numerical_columns}")
            logging.info(f" Categorical Columns : {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline,numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.error("Exception occured at get_data_transformation stage")
            raise customexception(e,sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f'Train Dataframe Head : \n{train_df.head(2).to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head(2).to_string()}')

            logging.info("Train and test data read as DataFrame")

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'Class'

            numerical_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']
            
            
            logging.info(f" Dividing train data into dependent and independent features")

            '''input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]'''

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]



            logging.info(f" Dividing test data into dependent and independent features")

            '''input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]'''

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]


            '''# printing following for debuging purpose
            print(f"Shape of input_feature_train_df: {input_feature_train_df.shape}")
            print(f"Shape of input_feature_test_df: {input_feature_test_df.shape}")

            print(f"Columns of input_feature_train_df: {input_feature_train_df.columns}")
            print(f"Columns of input_feature_test_df: {input_feature_test_df.columns}")'''


            logging.info("Applying preproccessing on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) # we can store this data as after applying preproccessing obj we get it in array format
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)    # use transform instead of fit_transform due to the concept of data leakage

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f" Saved Preproccessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



            
        except Exception as e:
            logging.error("Exception occured at initiate_data_transformation stage")
            raise customexception(e,sys)
            



print(" 'data_transformation.py' file run sucessfully")