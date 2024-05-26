import os
import pickle

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from ..utils.exceptions import *
from ..utils.logger import AppLogger
from ..utils.utils import get_project_root

logger = AppLogger(__name__).get_logger()


class ModelManager:
    """
    Class for managing machine learning models.

    Attributes:
    - model_name (str): Name of the model.
    - models (list): List of trained models.
    - model_directory (str): Directory path for storing and loading models.
    """

    def __init__(self, model_name, models=[]):
        """
        Initialize the ModelManager with the provided model name and list of models.

        Args:
        - model_name (str): Name of the model.
        - models (list): List of trained models. Default is an empty list.
        """
        self.model_name = model_name
        self.models = models
        self.model_directory = os.path.join(get_project_root(), "artifacts", 'models')

    def train_models(self, data, target, params, cv_splits):
        """
        Train multiple models using cross-validation.

        This method trains multiple models using the provided data, target variable,
        parameters, and cross-validation splits. The trained models are stored in
        the models attribute.

        Args:
        - data (DataFrame): Input features for training.
        - target (Series): Target variable for training.
        - params (dict): Hyperparameters for the models.
        - cv_splits (list): List of tuples containing indices for cross-validation splits.

        Raises:
        - ModelTrainingException: If an error occurs during model training.
        """
        try:
            for train_idx, val_idx in cv_splits:
                if self.model_name == "CatBoost":
                    model = CatBoostClassifier(**params)
                elif self.model_name == "LGBM":
                    model = LGBMClassifier(**params)
                elif self.model_name == "XGB":
                    model = XGBClassifier(**params)

                model.fit(data.loc[train_idx], target.loc[train_idx])
                self.models.append(model)
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise ModelTrainingException("Failed to train models") from e

    def get_models(self):
        """
           Get the list of trained models.

           Returns:
           - list: List of trained models.
        """
        return self.models

    def export_models(self):
        """
        Export trained models to the model directory.

        Raises:
        - ModelTrainingException: If an error occurs during model exporting.
        """
        try:
            if not os.path.exists(self.model_directory):
                os.makedirs(self.model_directory)
            else:
                for filename in os.listdir(self.model_directory):
                    file_path = os.path.join(self.model_directory, filename)
                    os.remove(file_path)

            for idx, model in enumerate(self.models):
                filename = os.path.join(self.model_directory, f"Model_{idx}.pkl")
                with open(filename, 'wb') as file:
                    pickle.dump(model, file)

        except Exception as e:
            logger.error(f"Model exporting error: {str(e)}")
            raise ModelTrainingException("Failed to export models") from e

    def load_models(self):
        """
        Load trained models from the model directory.

        Raises:
        - ModelLoadingException: If an error occurs during model loading.
        """
        try:
            self.models = []
            for filename in os.listdir(self.model_directory):
                if "Model" in filename:
                    with open(os.path.join(self.model_directory, filename), 'rb') as file:
                        model = pickle.load(file)
                        self.models.append(model)
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise ModelLoadingException("Failed to load models from directory") from e
