import pymongo
from sensor.constant.env_variables import MONGODB_URL_KEY
from sensor.constant.database import DATABASE_NAME, COLLECTION_NAME
from sensor.exception import CustomException
import certifi
import os 
ca = certifi.where()

class MongoDBClient:
    client = None 

    def __init__(self, database_name=DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                mongo_db_url = MONGODB_URL_KEY
                if mongo_db_url is None:
                    raise Exception(f"Enviornment key : {mongo_db_url} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise CustomException(e)
