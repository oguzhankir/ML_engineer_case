from ..operators.inference import Inference
from ..utils.exceptions import *
from ..utils.logger import AppLogger
from ..utils.utils import get_config

logger = AppLogger(__name__).get_logger()


def get_prediction_service(data):
    config = get_config()
    inference = Inference(config)

    return inference.test_inference(data)


def retrain_service(data):
    try:
        config = get_config()
        inference = Inference(config)
        return inference.train_inference(data)
    except Exception as e:
        logger.error(f"Error in retrain service: {str(e)}")
        raise ModelTrainingException("Failed to retrain model") from e
