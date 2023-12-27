import pandas as pd
from src.Fraud_TX.components.data_transformation import get_data_transformation
from sklearn.externals import joblib

# Load saved best parameters
with open("best_model.json", "r") as json_file:
    best_model = json.load(json_file)

# Load and preprocess new data for prediction
new_data = pd.read_csv("test.csv")  # Provide path to the new data file
transformed_new_data = get_data_transformation(new_data)

# Load the best model (choose the best model based on your evaluation)
best_model = best_model(**best_parameters['forest_clf'])  

# Train the best model on the entire dataset
best_model.fit(X_train, y_train)

# Save the trained model to a file for future predictions
joblib.dump(best_model, "trained_model.joblib")

# Make predictions on new data
new_data_predictions = best_model.predict(transformed_new_data.drop('Class', axis=1))



# Save the predictions to a file or any other desired output
new_data_predictions.to_csv("new_data_predictions.csv", index=False)
