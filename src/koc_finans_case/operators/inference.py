from .model_manager import ModelManager
import pandas as pd
from ..utils.utils import get_config
from .preprocess import Preprocess
import numpy as np

class Inference:
    def __init__(self, config):
        self.config = config

    def train_inference(self, data):
        pass

    def test_inference(self, data):
        config = get_config()
        model_manager = ModelManager(config["best_model_name"])
        model_manager.load_models()
        data = pd.DataFrame([data.dict()])
        preprocessor = Preprocess(data, inference=True)

        data = preprocessor.execute()[config["feat_cols"]]

        probas = []
        preds = []
        for idx, model in enumerate(model_manager.models):
            proba = model.predict_proba(data)[:, 1][0]

            probas.append(proba)
            preds.append(int(proba > config["fold_thresholds"][idx]))
        return int(np.mean(preds))



