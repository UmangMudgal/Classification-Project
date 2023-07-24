from sensor.exception import CustomException
from sklearn.pipeline import Pipeline
import sys,os
from sensor.logger import logging
import pandas as pd
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME




class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    

class SensorModel:

    def __init__(self, preprocessor_object:Pipeline, trianed_model_object:object):
        self.preprocessor_object = preprocessor_object
        self.trianed_model_object = trianed_model_object
    
    def predict(self, dataframe:pd.DataFrame)->pd.DataFrame:
        try:
            transformed_feature = self.preprocessor_object.transform(dataframe)
            y_hat = self.trianed_model_object.predict(transformed_feature)
            
            return y_hat
        
        except Exception as e:
            raise e


class ModelResolver:
    def __init__(self, model_dir=SAVED_MODEL_DIR) -> None:
        try:
            self.model_dir = model_dir
        
        except Exception as e:
            raise e
        
    def get_best_model_path(self, )->str:
        try:
            timestamp = list(map(int, os.listdir(self.model_dir)))
            latest_timestamp = max(timestamp)
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e
        
    def is_model_exists(self):
        try:
            if not os.path.exists(self.model_dir):
                return False
            timestamp = os.listdir(self.model_dir)
            if len(timestamp)==0:
                return False
            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False
            return True
        except Exception as e:
            raise e