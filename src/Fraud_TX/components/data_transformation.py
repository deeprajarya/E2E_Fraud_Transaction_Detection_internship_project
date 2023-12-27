
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Fraud_TX.exception import customexception
from src.Fraud_TX.logger import logging


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.Fraud_TX.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    transformed_data_file_path = os.path.join('artifacts', 'transformed_data.csv')



class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        

    def get_data_transformation(self,train_path, test_path):
        
        try:
            logging.info('Data Transformation initiated')

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            std_scaler = StandardScaler()
            rob_scaler = RobustScaler()

            train_df['scaled_amount'] = rob_scaler.fit_transform(train_df['Amount'].values.reshape(-1, 1))
            train_df['scaled_time'] = rob_scaler.fit_transform(train_df['Time'].values.reshape(-1, 1))

            test_df['scaled_amount'] = rob_scaler.fit_transform(test_df['Amount'].values.reshape(-1, 1))
            test_df['scaled_time'] = rob_scaler.fit_transform(test_df['Time'].values.reshape(-1, 1))

            # Save the transformed data to CSV
            train_df.to_csv(os.path.join('artifacts', 'transformed_data.csv'), index=False)

            return rob_scaler
        

        except Exception as e:
            logging.info("Exception occured in the get_data_transformation function")

            raise customexception(sys)
        



    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data ")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation(train_path, test_path)

            

            target_column_name = 'Class'
            drop_columns = ['Time', 'Amount']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return train_arr, test_arr

        
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation stage")
            raise customexception(sys)




print(" 'data_transformation.py' file run sucessfully")