import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler



class DataTransformationConfig:
    transformed_data_path = "artifacts/transformed_data.csv"

class DataTransformation:
    def __init__(self, raw_data_path):
        self.transformation_config = DataTransformationConfig()
        self.raw_data_path = raw_data_path

    def perform_data_transformation(self):
        # Read the raw data
        raw_data = pd.read_csv(self.raw_data_path)

        # Perform data transformations
        scaled_data = self.scale_features(raw_data)

        # Save the transformed data
        scaled_data.to_csv(self.transformation_config.transformed_data_path, index=False)

        return self.transformation_config.transformed_data_path

    def scale_features(self, df):
        rob_scaler = RobustScaler()

        df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        return df

# Example Usage
if __name__ == "__main__":
    raw_data_path = "artifacts/raw.csv"

    # Instantiate DataTransformation class
    data_transformation_instance = DataTransformation(raw_data_path)

    # Run the data transformation process
    transformed_data_path = data_transformation_instance.perform_data_transformation()

    print(f"Transformed Data Path: {transformed_data_path}")




