"""Module containing the basic functionalities to train the model."""

import tensorflow as tf
from ml_project import logger
from ml_project.entity.common_entities import TrainModelConfig


class TrainModel:
    """Class to train a model."""
    def __init__(self, config: TrainModelConfig):
        self.config = config
        self.model = None
        self.training_generator = None
        self.validation_generator = None

    def load_model(self):
        """Load and store a pretrained model."""
        self.model = tf.keras.models.load_model(
            self.config.full_model_path
        )
        logger.info("Untrained model correctly loaded.")

    def create_data_generator(self):
        """Split data into training and validation and augment them."""
        # common parameters for both training and validation generators
        data_generator_kwargs = {
            "rescale": 1/255,
            "validation_split": 0.20,
        }

        data_flow_kwargs = {
            "target_size": self.config.params_input_size[:-1],
            "batch_size": self.config.params_batch_size,
            "interpolation": "bilinear",
        }

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **data_generator_kwargs
        )

        self.validation_generator = data_generator.flow_from_directory(
            directory=self.config.training_data_path,
            subset="validation",
            shuffle=False,
            **data_flow_kwargs
        )

        if self.config.params_augmentation:
            train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **data_generator_kwargs
            )
        else:
            train_data_generator = data_generator

        self.training_generator = train_data_generator.flow_from_directory(
            directory=self.config.training_data_path,
            subset="training",
            shuffle=True,
            **data_flow_kwargs
        )

    def save_trained_model(self):
        """Stores model so that we can later use it for inference."""
        self.model.save(self.config.trained_model_path)
        logger.info(f"Model correctly trained and"
                    f" stored in {self.config.trained_model_path}.")

    def train_model(self):
        """Trains the model using the data generators."""
        steps_per_epoch = self.training_generator.samples // self.training_generator.batch_size
        validation_steps = self.validation_generator.samples // self.validation_generator.batch_size

        if not self.training_generator:
            self.create_data_generator()

        self.model.fit(
            x=self.training_generator,
            epochs=self.config.params_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.validation_generator,
        )

        self.save_trained_model()
