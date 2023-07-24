from sensor.exception import CustomException
from sensor.logger import logging
from sensor.utils.main_utils import load_numpy_array_data
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.ml.metrics.classification_metric import get_classification_score
import sys, os
from xgboost import XGBClassifier
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import load_object, save_object
class ModelTrainer:

    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_model(self, x_train, y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            #loading the training and testing array
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            #Splitting data into input and target columns
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model= self.train_model(x_train=x_train, y_train=y_train)
            y_train_pred = model.predict(x_train)
            classification_metric_train = get_classification_score(y_train, y_train_pred)
            
            if classification_metric_train.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained Model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)
            classification_metric_test = get_classification_score(y_test, y_test_pred)
            
            #Overfitting and Underfitting
            diff = abs(classification_metric_train.f1_score - classification_metric_test.f1_score)
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation")
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_obj_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.model_trainer_trained_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            sensor_model = SensorModel(preprocessor_object=preprocessor, trianed_model_object=model)
            save_object(self.model_trainer_config.model_trainer_trained_file_path, obj=sensor_model)

            # Model trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.model_trainer_trained_file_path,
                train_metric_artifact=classification_metric_train,
                test_metric_artifact=classification_metric_test
            )

            logging.info(f"Model Trainer Artifact {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
