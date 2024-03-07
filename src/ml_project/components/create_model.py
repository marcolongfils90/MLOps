"""Module containing the basic functionalities to create a base model."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop
from ml_project import logger
from ml_project.entity.common_entities import BaseModelConfig


class BaseModel:
    """Base model class to load a pretrained model."""
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def load_model(self):
        """Load and store a pretrained model."""
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_input_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
        )
        self.model.save(self.config.model_path)
        logger.info(f"Pretrained model correctly loaded and stored in {self.config.model_path}")

    @staticmethod
    def create_untrained_model(model,
                               num_classes,
                               learning_rate,
                               freeze: bool = True) -> Model:
        """Modify pretrained model to fit problem at hand."""
        if freeze:
            for layer in model.layers:
                layer.trainable = False

        flattening_layer = layers.Flatten()(model.output)
        fc_layer = layers.Dense(units=512, activation="relu")(flattening_layer)
        output = layers.Dense(units=num_classes, activation="softmax")(fc_layer)

        full_model = Model(inputs=model.input, outputs=output)
        full_model.compile(
            optimizer=RMSprop(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return full_model

    def save_untrained_model(self):
        """Store full model so that we can later train it."""
        self.full_model = self.create_untrained_model(
            model=self.model,
            num_classes=self.config.params_num_classes,
            learning_rate=self.config.params_learning_rate,
            freeze=True,
        )
        self.full_model.save(self.config.untrained_model_path)
        logger.info(f"Full model correctly compiled and"
                    f" stored in {self.config.untrained_model_path}.")
