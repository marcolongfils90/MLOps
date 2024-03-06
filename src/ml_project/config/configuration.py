from pathlib import Path
from ml_project import constants
from ml_project.utils import common
from ml_project.entity.common_entities import DataIngestionConfig


class ConfigurationManager:
    def __init__(self,
                 config_filepath: Path = constants.CONFIG_FILE_PATH,
                 params_filepath: Path = constants.PARAMS_FILE_PATH):
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)

        common.create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        common.create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )