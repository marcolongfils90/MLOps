from ml_project.pipeline import base_model_pipeline
from ml_project.pipeline import data_ingestion_pipeline
from ml_project import logger


STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
    get_data = data_ingestion_pipeline.DataIngestionPipeline().run()
    logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
except Exception as e:
    logger.exception(e)





STAGE_NAME = "Base Model Creation"
try:
    logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
    get_data = base_model_pipeline.BaseModelPipeline().run()
    logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
except Exception as e:
    logger.exception(e)