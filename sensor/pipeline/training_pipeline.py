from sensor.entity.config_entity import (
    TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelEvaluationConfig, ModelTrainerConfig, ModelPusherConfig
)
from sensor.exception import CustomException
from sensor.logger import logging
import sys, os
from sensor.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact,
    DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact, ModelEvaluationArtifact)
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.constant.training_pipeline import TRAINING_BUCKET_NAME, SAVED_MODEL_DIR
from sensor.cloud_storage.s3_syncer import S3Sync
class TrainPipeline:
    is_pipeline_running = False
    def __init__(self) :
        self.training_pipeline_config = TrainingPipelineConfig()
        self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
        self.s3_sync = S3Sync()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
            
            logging.info(f"Data Ingestion Completed and artifact {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info("Data Validation Started")
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e,sys)
    
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transfomation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_trainer_config )
            model_artifact = model_trainer.initiate_model_trainer()
            return model_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_evaluation(self, data_validation_artifact:DataValidationArtifact, model_trainer_artifact:ModelTrainerArtifact):

        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config=model_eval_config,
                                        data_validation_artifact=data_validation_artifact, 
                                        model_trainer_artifact=model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact

        except Exception as e:
            raise CustomException(e,sys)
    
    def start_model_pusher(self, model_evaluation_artifact:ModelEvaluationArtifact):
        
        try:
            model_pusher_config = ModelPusherConfig(self.training_pipeline_config)
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact, model_pusher_config=model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def sync_artifact_dir_to_s3(self):
        """
        Description: This will sync artifact folder to s3 bucket
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,
                                           aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise CustomException(e, sys)
        
    def sync_saved_model_dir_to_s3(self):
        """
        Description: This will sync saved model folder to s3 bucket
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder=SAVED_MODEL_DIR,
                                           aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True
            logging.info("Data Ingestion Started")
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            logging.info("Data Validation Started")
            data_validation_artifact:DataValidationArtifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Data Transformation Started")
            data_transformation_artifact:DataTransformationArtifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            logging.info("Model Training Started")
            model_trainer_artifact:ModelTrainerArtifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("Model Evaluation Started")
            model_evaluation_artifact: ModelEvaluationArtifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact, model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Model is not better than base model")
            logging.info("Model Pusher Started")
            model_pusher_artifact: ModelPusherArtifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            TrainPipeline.is_pipeline_running = False
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
        except Exception as e:
            TrainPipeline.is_pipeline_running = False
            raise CustomException(e,sys)
    