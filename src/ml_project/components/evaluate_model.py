"""Module containing the basic functionalities to evaluate the model."""

import mlflow
import tensorflow as tf
from ml_project import logger
from ml_project.utils import common
from ml_project.entity.common_entities import ModelEvaluationConfig
from pathlib import Path
from urllib.parse import urlparse


class EvaluateModel:
    """Class to evaluate a model."""
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.validation_generator = None
        self.score = None

    def load_model(self):
        """Load and store a pretrained model."""
        self.model = tf.keras.models.load_model(
            self.config.model_path
        )
        logger.info("Trained model correctly loaded.")

    def create_data_generator(self):
        """Create validation data generator."""
        # common parameters for both training and validation generators
        data_generator_kwargs = {
            "rescale": 1/255,
            "validation_split": 0.20,
        }

        data_flow_kwargs = {
            "target_size": self.config.params.INPUT_SIZE[:-1],
            "batch_size": self.config.params.BATCH_SIZE,
            "interpolation": "bilinear",
        }

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **data_generator_kwargs
        )

        self.validation_generator = data_generator.flow_from_directory(
            directory=self.config.validation_data_path,
            subset="validation",
            shuffle=False,
            **data_flow_kwargs
        )

    def evaluate_model(self):
        """Evaluate the model."""

        if not self.model:
            self.load_model()

        if not self.validation_generator:
            self.create_data_generator()

        self.score = self.model.evaluate(self.validation_generator)
        common.save_json(path=Path("scores.json"),
                  data={
                      "loss": self.score[0],
                      "accuracy": self.score[1],
                  })

    def log_to_mlflow(self):
        """Tracks experiments and logs results in MLFlow."""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.params)
            mlflow.log_metrics({
                      "loss": self.score[0],
                      "accuracy": self.score[1],
                  })
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
