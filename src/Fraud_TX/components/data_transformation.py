
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Fraud_TX.exception import customexception
from src.Fraud_TX.logger import logging


from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
from src.Fraud_TX.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    transformed_data_file_path = os.path.join('artifacts', 'transformed_data.csv')



class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformation(self, raw_data_path):
        try:
            logging.info("data transformation stage initiated")

            data = pd.read_csv(raw_data_path)

            rob_scaler = RobustScaler()

            data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
            data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))

            data.drop(['Time','Amount'], axis=1, inplace=True)

            # Lets shuffle the data before creating the subsamples

            data = data.sample(frac=1)

            fraud_df = data.loc[data['Class'] == 1]
            valid_df = data.loc[data['Class'] == 0][:492]
            normal_distributed_df = pd.concat([fraud_df, valid_df])

            # Shuffle dataframe rows
            data = normal_distributed_df.sample(frac=1, random_state=42)
            logging.info("I have read data head(2) after scaling data ")
            data.head(2)

            # Save the transformed data to CSV
            data.to_csv(os.path.join('artifacts', 'transformed_data.csv'), index=False)
        
            return rob_scaler
        
        except:
            logging.info("Exception occured in the get_data_transformation stage")

            raise customexception(sys)
            
        
  

    def initialize_data_transformation(self,raw_data_path):
        try:
            logging.info(" Initialize data transformation stage initiated ")
            logging.info("read raw data ")
            data = pd.read_csv(raw_data_path)

            logging.info(f'Raw Dataframe Head : \n{data.head().to_string()}')


            preprocessing_obj = self.get_data_transformation(raw_data_path)

            target_column_name = 'Class'
            drop_columns = ['Time', 'Amount']

            raw_data_df = data.drop(columns=drop_columns, axis=1)

            input_feature_arr=preprocessing_obj.fit_transform(raw_data_df)

            logging.info("Apply preprocessing object on transformed datasets")
            transformed_data_df = pd.DataFrame(input_feature_arr, columns=raw_data_df.columns)

            '''logging.info("Applying preprocessing object on transformed datasets.")
            raw_data_arr =  np.array(transformed_data_df)
            '''

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return preprocessing_obj, transformed_data_df

            
        except Exception:
            logging.info("Exception occured in the initiate_data_transformation stage")
            return customexception(sys)
            


print(" 'data_transformation.py' file run sucessfully")