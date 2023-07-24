from sensor.exception import CustomException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from sensor.entity.config_entity import ModelPusherConfig
import sys, os
import pandas as pd
import shutil 



class ModelPusher:
    def __init__(self,model_evaluation_artifact:ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            logging.info("Entered Initiate Model Pusher")
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)
            logging.info("Copied trained model loc to model file location")
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)
            logging.info("Copied trained model loc to saved model file location")

            #prepare artifacts
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path, model_file_path=model_file_path
            )

            logging.info(f"Model Pusher Artifact : {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)
