"""Module for the model evaluation pipeline."""
from ml_project.config import configuration
from ml_project.components import evaluate_model
from ml_project import logger

STAGE_NAME = "Model Evaluation"


class EvaluateModelPipeline:
    """Class for model evaluation pipeline."""
    def __init__(self):
        pass

    def run(self):
        """Run the model evaluation pipeline."""
        try:
            config = configuration.ConfigurationManager().get_evaluation_config()
            model_config = evaluate_model.EvaluateModel(config=config)
            model_config.evaluate_model()
            model_config.log_to_mlflow()
        except Exception as exc:
            raise exc


if __name__ == "__main__":
    try:
        logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
        EvaluateModelPipeline().run()
        logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
    except Exception as e:
        logger.exception(e)
