import os
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME

#Defining Constant Variables for the training pipeline
TRAGET_COLUM = 'class'
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

