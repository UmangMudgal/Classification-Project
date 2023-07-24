import os
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME


SAVED_MODEL_DIR = os.path.join("saved_model")

#Defining Constant Variables for the training pipeline
TARGET_COLUMN = 'class'
PIPELINE_NAME: str = "sensor"
ARTIFACT_DIR: str = "artifact"
FILE_NAME:str = "sensor.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", 'schema.yaml')
SCHEMA_DROP_COLUMN = "drop_columns"


# Constants Relatated to the Data Ingestion Component
DATA_INGESTION_COLLECTION_NAME: str = "Sensor_Data"
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float=0.2


#Constant Related to Data Validataion Component
DATA_VALIDATION_DIR_NAME: str = "Data_Validation"
DATA_VALIDATION_VALID_DATA_DIR: str = "Valid_Data"
DATA_VALIDATION_INVALID_DATA_DIR: str = "Invalid_Data"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "Drift_Report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


#Constant Related to Data Transformation
DATA_TRANSFORMATION_DIR_NAME:str = "Data_Transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME:str = "Transformed" 
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "Transformed_Object"

#Constant Related to Model Trainer
MODEL_TRAINER_DIR_NAME: str = "Model_Trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "Trained_Model"
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITING_THRESHOLD: float = 0.05

#Constant Related to Model Evaluation 
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_DIR_NAME: str = "Model_Evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "report.yaml"

#Constant Related to Model Pusher
MODEL_PUSHER_DIR_NAME = 'Model_Pusher'
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR

