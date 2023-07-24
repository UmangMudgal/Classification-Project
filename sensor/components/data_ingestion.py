from sensor.exception import CustomException
from sensor.logger import logging
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.data_access.sensor_data import SensorData
from sensor.constant.training_pipeline import SCHEMA_DROP_COLUMN, SCHEMA_FILE_PATH
from sensor.utils.main_utils import read_yaml
import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self)-> pd.DataFrame:
        """
        Description : Export the MongoDB data to local System as DataFrame
        """
        try:
            logging.info("Exporting Data from MongoDb to Feature Store")
            sensor_data = SensorData()
            dataframe = sensor_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            feature_store_file_path  = self.data_ingestion_config.feature_store

            #Creating Folder 
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def split_data_as_train_test(self, dataframe:pd.DataFrame):
        """
        Description : Exported data from feature store is seprated as train and test data
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed Train Test Split on the DataFrame")

            logging.info("Exited the split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info("Exported train and test file path")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            _schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
            dataframe = dataframe.drop(_schema_config[SCHEMA_DROP_COLUMN], axis=1)
            logging.info("Got Data From the MongoDb")   
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                   test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
    