# End to End Project : Credit Card Fraud Transaction Detection

## Overview

The E2E Credit Card Fraud Detection System is designed to provide end-to-end solutions for credit card fraud detection. The system comprises data ingestion, transformation, model training, and real-time prediction through a Flask-based web interface.

## Project Structure

- [`src`](src): Contains the source code for the project.
  - [`Fraud_TX`](src/Fraud_TX): Main module for the fraud detection system.
    - [`components`](src/Fraud_TX/components): Components for data ingestion, data transformation, and model training.
    - [`pipelines`](src/Fraud_TX/pipelines): Pipelines for data processing and prediction.
  - [`app.py`](app.py): Flask application for web interface.
  - [`utils.py`](utils.py): Utility functions.
- [`notebooks`](notebooks): Contains Jupyter notebooks (e.g., for exploratory data analysis).

## Components

### 1. Data Ingestion

File: [`data_ingestion.py`](src/Fraud_TX/components/data_ingestion.py)

Responsible for reading raw data, saving it, and splitting it into training and testing datasets.

### 2. Data Transformation

File: [`data_transformation.py`](src/Fraud_TX/components/data_transformation.py)

Performs data preprocessing, including imputation, scaling, and one-hot encoding.

### 3. Model Trainer

File: [`model_trainer.py`](src/Fraud_TX/components/model_trainer.py)

Trains machine learning models (e.g., Logistic Regression, Decision Tree) using the preprocessed data.

### 4. Training Pipeline

File: [`training_pipeline.py`](src/Fraud_TX/pipelines/training_pipeline.py)

Executes the end-to-end process of data ingestion, transformation, and model training.

### 5. Prediction Pipeline

File: [`prediction_pipeline.py`](src/Fraud_TX/pipelines/prediction_pipeline.py)

Handles real-time predictions using a trained model and a preprocessor.

### 6. Web Interface

File: [`app.py`](app.py)

A Flask application for the web interface, allowing users to input credit card data and receive fraud predictions.

### 7. Utilities

File: [`utils.py`](utils.py)

Includes utility functions such as saving and loading objects, and model evaluation.

## Usage

1. **Data Ingestion and Transformation:**
   - Run [`data_ingestion.py`](src/Fraud_TX/components/data_ingestion.py) to ingest and save raw data.
   - Run [`data_transformation.py`](src/Fraud_TX/components/data_transformation.py) to preprocess data.

2. **Model Training:**
   - Run [`model_trainer.py`](src/Fraud_TX/components/model_trainer.py) to train machine learning models.

3. **Training Pipeline:**
   - Run [`training_pipeline.py`](src/Fraud_TX/pipelines/training_pipeline.py) to execute the end-to-end process.

4. **Web Interface:**
   - Run [`app.py`](app.py) to start the Flask application for real-time predictions.

## Dependencies

- Python 3.x
- Flask
- NumPy
- Pandas
- Scikit-learn
- ...

## Contributing

If you'd like to contribute to the project, please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Mention any libraries, tools, or resources you used or were inspired by.
- ...

