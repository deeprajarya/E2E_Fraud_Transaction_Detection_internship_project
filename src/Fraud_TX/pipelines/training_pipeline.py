from src.Fraud_TX.components.data_ingestion import DataIngestion
from src.Fraud_TX.components.data_transformation import perform_data_transformation
from src.Fraud_TX.components.model_trainer import ModelTrainer

# Load and ingest data
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

# Read ingested data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Perform data transformation
transformed_train_data = perform_data_transformation(train_data)
transformed_test_data = perform_data_transformation(test_data)

# Split data into features and target
X_train = transformed_train_data.drop('Class', axis=1)
y_train = transformed_train_data['Class']
X_test = transformed_test_data.drop('Class', axis=1)
y_test = transformed_test_data['Class']

# Initialize ModelTrainer
model_trainer = ModelTrainer(X_train.values, y_train.values, X_test.values, y_test.values)

# Train models and get best parameters
best_parameters = model_trainer.train_models()

# Save the best parameters to a file for future reference
with open("best_parameters.json", "w") as json_file:
    json.dump(best_parameters, json_file)
