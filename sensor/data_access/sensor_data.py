from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.exception import CustomException
from sensor.logger import logging
import sys
import pandas as pd
import numpy as np
from typing import Optional

class SensorData:
    """
    Description: This class helps to export entire mongodb record as pandas dataframe
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str]=None )-> pd.DataFrame:
        """
        Description : Export Entire Collection as DataFrame
        Return : pd.DataFrame of collection
        """
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na":np.nan}, inplace=True)

            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    
        
    