import sys
import os
import pandas as pd
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
from src.Fraud_TX.utils.utils import load_object



class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            
            scaled_data = preprocessor.transform(features)
            predictions = model.predict(scaled_data)

            return predictions
        

        except Exception as e:
            logging.error("Exception occured at Model_preiction stage")
            raise customexception(e,sys)

class CustomData:
    def __init__(self,Time:float,
                V1:float, V2:float, V3:float, V4:float, V5:float, V6:float, V7:float, V8:float, V9:float, V10:float,
                V11:float, V12:float, V13:float, V14:float, V15:float, V16:float, V17:float, V18:float, V19:float, V20:float,
                V21:float, V22:float, V23:float, V24:float, V25:float, V26:float, V27:float, V28:float
                ):
        self.Time=Time
        self.V1=V1
        self.V2=V2
        self.V3=V3
        self.V4=V4
        self.V5=V5
        self.V6=V6
        self.V7=V7
        self.V8=V8
        self.V9=V9
        self.V10=V10
        self.V11=V11
        self.V12=V12
        self.V13=V13
        self.V14=V14
        self.V15=V15
        self.V16=V16
        self.V17=V17
        self.V18=V18
        self.V19=V19
        self.V20=V20
        self.V21=V21
        self.V22=V22
        self.V23=V23
        self.V24=V24
        self.V25=V25
        self.V26=V26
        self.V27=V27
        self.V28=V28

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                    'Time':[self.Time],
                    'V1' :[self.V1],
                    'V2' :[self.V2],
                    'V3' :[self.V3],
                    'V4' :[self.V4],
                    'V5' :[self.V5],
                    'V6' :[self.V6],
                    'V7' :[self.V7],
                    'V8' :[self.V8],
                    'V9' :[self.V9],
                    'V10' :[self.V10],
                    'V11' :[self.V11],
                    'V12' :[self.V12],
                    'V13' :[self.V13],
                    'V14' :[self.V14],
                    'V15' :[self.V15],
                    'V16' :[self.V16],
                    'V17' :[self.V17],
                    'V18' :[self.V18],
                    'V19' :[self.V19],
                    'V20' :[self.V20],
                    'V21' :[self.V21],
                    'V22' :[self.V22],
                    'V23' :[self.V23],
                    'V24' :[self.V24],
                    'V25' :[self.V25],
                    'V26' :[self.V26],
                    'V27' :[self.V27],
                    'V28' :[self.V28]
                
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
            
        except Exception as e:
            logging.error(f"Exception occured at get_data_as_dataframe stage in prediction_pipeline.py file")
            raise customexception(e,sys)