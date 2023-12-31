import pandas as pd
from data_transformation import DataTransformation

import joblib


# Specify the path to your 'model.pkl' file
model_path = "artifacts/model.pkl"

# Load the model
best_model = joblib.load(model_path)


# Load and preprocess new data for prediction
new_data = pd.read_csv("artifacts/test.csv")  

#create an instance of data transformation
data_transformation_instance = DataTransformation()

# Call the method to initialize data transformation
result = data_transformation_instance.initialize_data_transformation(new_data)

# Check the result and handle the customexception
if isinstance(result, tuple) and len(result) == 2:
    preprocessing_obj, transformed_new_data = result

    # Make predictions on new data
    new_data_predictions = best_model.predict(transformed_new_data.drop('Class', axis=1))

    # Save the trained model to a file for future predictions
    joblib.dump(best_model, "trained_model.joblib")

    # Save the predictions to a file or any other desired output
    new_data_predictions.to_csv("new_data_predictions.csv", index=False)

else:
    # Handle the customexception, print an error message, or take appropriate action
    print(f"Error: {result}")

'''
# Make predictions on new data
new_data_predictions = best_model.predict(transformed_new_data.drop('Class', axis=1))


# Save the trained model to a file for future predictions
joblib.dump(best_model, "trained_model.joblib")

# Save the predictions to a file or any other desired output
new_data_predictions.to_csv("new_data_predictions.csv", index=False)


# Save the predictions to a file or any other desired output
new_data_predictions.to_csv("new_data_predictions.csv", index=False)
'''