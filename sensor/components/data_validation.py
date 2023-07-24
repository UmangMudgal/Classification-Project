from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.utils.main_utils import read_yaml
from sensor.exception import CustomException
from sensor.logger import logging
from sensor.utils.main_utils import write_yaml_file
import os, sys
from scipy.stats import ks_2samp
import pandas as pd


class DataValidation:

    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        

    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        
        except Exception as e:
            raise CustomException(e, sys)

    def is_numerical_columns_exist(self, dataframe : pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            missing_numerical_columns = []
            numerical_columns_present = True
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(num_column)
            logging.info(f"Missing Numerical Columns : {missing_numerical_columns}")
            return numerical_columns_present


        except Exception as e:
            raise CustomException(e, sys)
        
    def zero_standard_deviation_column(self, dataframe:pd.DataFrame):
        try:
            numerical_columns = len(self._schema_config['numeric_columns'])
            standard_devaition_zero_columns = []
            for num_column in numerical_columns:
                if dataframe[num_column].std==0:
                    standard_devaition_zero_columns.append(num_column)
                    
            logging.info(f"Standard Deviation Numerical Columns : [{standard_devaition_zero_columns}]")
            return standard_devaition_zero_columns
            
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold:float= 0.05)->bool:
        try: 
            report = {}
            status = True
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold<=is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column:{
                            "p-Value": float(is_same_dist.pvalue),
                            "drif_status":is_found
                        }
                    }
                )

            drift_report_file_path = self.data_validation_config.data_drift_report_file_path
            #Create Directory 
            dir_path = os.path.dirname(drift_report_file_path)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            #Reading Train and Test Files
            logging.info("Reading Train and Test File under Initiate Data Validation")
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            logging.info("Reading Completed - Train and Test File under Initiate Data Validation  & Validation of Column Started")

            #Validate Number of Columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            logging.info(f'All the columns present in Train DataFrame: {status}')
            if not status:
                error_message = f"{error_message}Train Dataframe doesnot contains all the columns\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            logging.info(f'All the columns present in Test DataFrame: {status}')
            if not status:
                error_message = f"{error_message}Test Dataframe doesnot contains all the columns\n"
            logging.info("Validation of Column Finished and Validataion of Numerical Column Started")

            #Validation of Numerical Columns
            numerical_status = self.is_numerical_columns_exist(dataframe=train_dataframe)
            if not numerical_status:
                error_message = f"{error_message}Train DataFrame doesnot contains all the numerical columns"

            numerical_status = self.is_numerical_columns_exist(dataframe=test_dataframe)
            if not numerical_status:
                error_message = f"{error_message}Test DataFrame doesnot contains all the numerical columns"
            
            #Data Drift 
            validation_status = True
            if len(error_message)==0:
                drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
                if drift_status:
                    logging.info("Drift Detected")
            else:
                validation_status =False
                raise Exception(error_message)

            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status,
                valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path  = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.data_drift_report_file_path
            )

            logging.info(f"Data Validation Artifact : {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
