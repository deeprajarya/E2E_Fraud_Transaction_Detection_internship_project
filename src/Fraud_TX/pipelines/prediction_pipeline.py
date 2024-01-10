import os
import pandas as pd
from src.Fraud_TX.logger import logging
from src.Fraud_TX.exception import customexception
import joblib
from src.Fraud_TX.components.data_transformation import DataTransformation
from dataclasses import dataclass



def data_transformation():
    train_df_num_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    
    num_pipeline=Pipeline(
        steps=[
                ('rob_scaler',RobustScaler()),
                ('std_scaler',StandardScaler())

                ]

            )

        preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,self.train_df_num_cols)
            ])

@dataclass
class PredictionPipelineConfig:
    trained_model_file_path = os.path.join('artifacts', 'KNeighborsClassifier_Best_Model.pkl')
    predictions_output_file = os.path.join('artifacts', 'predictions_output.csv')

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def load_trained_model(self):
        try:
            logging.info('Loading trained model...')
            model = joblib.load(self.prediction_pipeline_config.trained_model_file_path)
            logging.info('Trained model loaded successfully.')
            return model
        except Exception as e:
            logging.error('Error loading trained model.')
            raise customexception(e)

    def make_predictions(self, input_data_path):
        try:
            logging.info('Making predictions...')
            model = self.load_trained_model()

            # Load input data for prediction
            input_data = pd.read_csv(input_data_path)

            # Transform the input data for prediction
            transformed_data_for_prediction = self.transform_data_for_prediction(input_data)

            # Assuming 'Class' column is the target variable
            X = transformed_data_for_prediction.drop('Class', axis=1)

            # Make predictions
            predictions = model.predict(X)

            # Save predictions to output file
            predictions_df = pd.DataFrame({'Predictions': predictions})
            predictions_df.to_csv(self.prediction_pipeline_config.predictions_output_file, index=False)

            logging.info('Predictions saved successfully.')
        except Exception as e:
            logging.error('Error making predictions.')
            raise customexception(e)

    def transform_data_for_prediction(self, input_data):
        # Perform the same data transformation as done during training
        transformed_data = DataTransformation.get_data_transformation(input_data)
        return transformed_data

# Example Usage
if __name__ == "__main__":
    input_data_path = "artifacts/test.csv"  # Replace with the actual path
    prediction_pipeline_instance = PredictionPipeline()
    prediction_pipeline_instance.make_predictions(input_data_path)
