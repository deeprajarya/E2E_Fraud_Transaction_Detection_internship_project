import pandas as pd
from src.Fraud_TX.components.data_transformation import DataTransformation
import joblib


# Specify the path to your 'model.pkl' file
model_path = "artifacts/model.pkl"

# Load the model
best_model = joblib.load(model_path)


# Load and preprocess new data for prediction
new_data = pd.read_csv("artifacts/test.csv")  # Provide path to the new data file
transformed_new_data = initialize_data_transformation(new_data)



# Train the best model on the entire dataset
best_model.fit(X_train, y_train)

# Save the trained model to a file for future predictions
joblib.dump(best_model, "trained_model.joblib")

# Make predictions on new data
new_data_predictions = best_model.predict(transformed_new_data.drop('Class', axis=1))



# Save the predictions to a file or any other desired output
new_data_predictions.to_csv("new_data_predictions.csv", index=False)
