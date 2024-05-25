from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import pickle


class ModelManager:
    def __init__(self, model_name, model_directory="src/koc_finans_case/artifacts/models", models=[]):
        self.model_name = model_name
        self.models = models
        self.model_directory = model_directory

    def train_models(self, data, target, params, cv_splits):
        for train_idx, val_idx in cv_splits:
            if self.model_name == "CatBoost":
                model = CatBoostClassifier(**params)
            elif self.model_name == "LGBM":
                model = LGBMClassifier(**params)
            elif self.model_name == "XGB":
                model = XGBClassifier(**params)

            model.fit(data.loc[train_idx], target.loc[train_idx])
            self.models.append(model)

    def get_models(self):
        return self.models

    def export_models(self):
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

        for idx, model in enumerate(self.models):
            filename = os.path.join(self.model_directory, f"Model_{idx}.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

    def load_models(self):
        self.models = []
        for filename in os.listdir(self.model_directory):
            if "Model" in filename:
                with open(os.path.join(self.model_directory, filename), 'rb') as file:
                    model = pickle.load(file)
                    self.models.append(model)


