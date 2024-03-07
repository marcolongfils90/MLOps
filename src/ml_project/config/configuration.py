"""Module with the utilities to handle configurations."""
import os
from pathlib import Path
from ml_project import constants
from ml_project.utils import common
from ml_project.entity import common_entities


class ConfigurationManager:
    """Class to store and handle the configurations for the pipelines."""
    def __init__(self,
                 config_filepath: Path = constants.CONFIG_FILE_PATH,
                 params_filepath: Path = constants.PARAMS_FILE_PATH):
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)

        common.create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> common_entities.DataIngestionConfig:
        """Extracts the data ingestion pipeline configuration."""
        config = self.config.data_ingestion
        common.create_directories([config.root_dir])

        return common_entities.DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_model_config(self) -> common_entities.BaseModelConfig:
        """Extracts the base model creation pipeline configuration."""
        config = self.config.create_model
        common.create_directories([config.root_dir])

        return common_entities.BaseModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            untrained_model_path=Path(config.untrained_model_path),
            params_input_size=self.params.INPUT_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_num_classes=self.params.NUM_CLASSES,
            params_weights=self.params.WEIGHTS,
            params_learning_rate=self.params.LEARNING_RATE
        )

    def get_training_config(self) -> common_entities.TrainModelConfig:
        """Extracts the config to train the full model."""
        config = self.config.train_model
        common.create_directories([config.root_dir])
        # TODO(marcolongfils) make the training path general, so the folder should always
        #  be named "training_data_folder"
        training_data_path = os.path.join(self.config.data_ingestion.unzip_dir,
                                          "kidney-ct-scan-image")

        return common_entities.TrainModelConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            full_model_path=Path(self.config.create_model.untrained_model_path),
            training_data_path=Path(training_data_path),
            params_input_size=self.params.INPUT_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_augmentation=self.params.AUGMENTATION,
            params_epoch=self.params.EPOCHS,
        )
