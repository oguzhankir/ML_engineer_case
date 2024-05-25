import dill as pickle
import pandas as pd
from ..utils.utils import get_config
from ..operators.inference import Inference
def get_prediction_service(data):

    config = get_config()
    inference = Inference(config)

    return inference.test_inference(data)