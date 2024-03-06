"""Module for the base model creation pipeline."""
from ml_project.config import configuration
from ml_project.components import create_model
from ml_project import logger

STAGE_NAME = "Base Model Creation"


class BaseModelPipeline:
    """Class for base model creation pipeline."""
    def __init__(self):
        pass

    def run(self):
        """Run the base model creation pipeline."""
        try:
            config = configuration.ConfigurationManager().get_model_config()
            model_config = create_model.BaseModel(config=config)
            model_config.load_model()
            model_config.save_untrained_model()
        except Exception as exc:
            raise exc


if __name__ == "__main__":
    try:
        logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
        BaseModelPipeline().run()
        logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
    except Exception as e:
        logger.exception(e)
