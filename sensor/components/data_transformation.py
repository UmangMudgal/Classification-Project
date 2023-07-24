import sys, os
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import CustomException
from sensor.logger import logging
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    
    def __init__(
            self, 
            data_validation_artifact: DataValidationArtifact,
            data_transfomation_config:DataTransformationConfig):
        
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transfomation_config
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
           return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer),
                    ("RobustScaler", robust_scaler)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_dataframe = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_dataframe = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            #Train Dataset
            input_feature_train_dataframe = train_dataframe.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_dataframe = train_dataframe[TARGET_COLUMN]
            #Mapping target feature with target mapping value
            target_feature_train_dataframe = target_feature_train_dataframe.replace(TargetValueMapping().to_dict())

            #Test Dataset
            input_feature_test_dataframe = test_dataframe.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_dataframe = test_dataframe[TARGET_COLUMN]
            #Mapping target feature with target mapping value
            target_feature_test_dataframe = target_feature_test_dataframe.replace(TargetValueMapping().to_dict())

            # Transforming the data with the preprocessor object
            preprocessor_obj = preprocessor.fit(input_feature_train_dataframe)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_dataframe)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_dataframe)

            #Handling Imbalance Data
            smt = SMOTETomek(sampling_strategy='minority')

            #Handling Train Dataset
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_dataframe
            )

            #Handling Test Dataset
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_dataframe
            )

            #Combining the input feature and target feature
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]


            #Saving the numpy array
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array= train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array= test_arr)

            #Saving Preprocessor Object
            save_object(self.data_transformation_config.transformed_object_file_path, obj=preprocessor_obj)

            #Data Transformation Artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_obj_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path= self.data_transformation_config.transformed_test_file_path,
            )


            logging.info(f"Data Transformation Artifact : {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)