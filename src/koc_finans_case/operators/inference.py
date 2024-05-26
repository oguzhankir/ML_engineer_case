import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .model_manager import ModelManager
from .preprocess import Preprocess
from ..operators.feature_selection import select_features_with_cv
from ..operators.hyperparameter_search import HyperparameterSearch, param_space
from ..utils.exceptions import *
from ..utils.logger import AppLogger
from ..utils.utils import get_config, get_train_data, optimize_thresholds, get_project_root

logger = AppLogger(__name__).get_logger()


class Inference:
    """
    Class for performing model inference and training.

    Attributes:
    - config (dict): Configuration dictionary containing model parameters.
    """

    def __init__(self, config):
        """
        Initialize the Inference class with the provided configuration.

        Args:
        - config (dict): Configuration dictionary containing model parameters.
        """
        self.config = config

    def train_inference(self, data):
        """
        Train the model and export the trained model.

        This method trains the model using the provided data and exports the trained model.
        It preprocesses the data, evaluates different encoding methods using cross-validation,
        and trains the model using the best encoding method.

        Args:
        - data: Data required for training.

        Returns:
        - dict: Dictionary containing the name of the best model.
        """
        try:
            config = get_config()
            df = get_train_data()
            logger.info("Preprocess started")
            preprocessor = Preprocess(df, inference=False)
            df = preprocessor.execute()
            logger.info("Preprocess completed")

            cv = StratifiedKFold(n_splits=data.FOLD_CNT, shuffle=True, random_state=data.SEED)
            cv_splits = list(cv.split(df.index, df.loc[:, config["label"]].astype(str)))

            if data.TUNE:
                X = df.drop(config["id_cols"] + [config["label"]], axis=1).reset_index(drop=True)
                y = df["loan_status"].reset_index(drop=True)

                search_results = {}

                for model in ["CatBoost", "LGBM", "XGB"]:
                    hyperparam_search = HyperparameterSearch(model, X, y)
                    best_params, best_score = hyperparam_search.optimize(
                        cv_splits, y, n_trials=data.N_TRIAL
                    )
                    search_results[model] = {}
                    search_results[model]["best_params"] = best_params
                    search_results[model]["best_score"] = best_score
                best_model_name = max(search_results, key=lambda x: search_results[x]["best_score"])
                params = {
                    key: value
                    for key, value in search_results[best_model_name]["best_params"].items()
                    if key in param_space(best_model_name)
                }
                logger.info(f"Hyperparameter tuning completed for {best_model_name}")

            else:
                best_model_name = "LGBM"
                params = {
                    "learning_rate": 0.10627432163833375,
                    "depth": 9,
                    "bagging_temperature": 4,
                    "subsample": 0.6823010174465818,
                    "min_data_in_leaf": 4,
                    "grow_policy": "Lossguide",
                    "l2_leaf_reg": 4,
                    "auto_class_weights": "Balanced",
                    "model_shrink_mode": "Decreasing",
                    "penalties_coefficient": 4.42205792371293,
                    "model_shrink_rate": 0.24628398605729596,
                    "random_strength": 9.747542108241445,
                    "max_depth": 3,
                    "min_child_samples": 15,
                    "colsample_bytree": 0.2763230246302888,
                    "reg_alpha": 0.4137251316562447,
                    "reg_lambda": 0.760936843867543,
                    "min_child_weight": 5,
                }
                logger.info(f"Using default parameters for {best_model_name}")

            if data.FEATURE_ELIMINATION:
                selected_features = select_features_with_cv(df)
                feat_cols = selected_features.copy()
                logger.info("Feature elimination completed")

            else:
                feat_cols = list(df.columns.difference(set(config["id_cols"] + [config["label"]])))

            model_manager = ModelManager(best_model_name, models=[])
            model_manager.train_models(df[feat_cols], df[config["label"]], params, cv_splits)
            model_manager.export_models()
            logger.info(f"Model training and export completed for {best_model_name}")

            fold_thresholds = optimize_thresholds(
                df, models=model_manager.models, cv_splits=cv_splits, feat_cols=feat_cols
            )
            logger.info("Threshold optimization completed")

            new_config = {
                "best_model_name": best_model_name,
                "feat_cols": feat_cols,
                "params": params,
                "label": config["label"],
                "id_cols": config["id_cols"],
                "FOLD_CNT": data.FOLD_CNT,
                "SEED": data.SEED,
                "DEVICE": config["DEVICE"],
                "ITERATION": data.ITERATION,
                "TUNE": data.TUNE,
                "EVAL_METRIC": data.EVAL_METRIC,
                "N_TRIAL": data.N_TRIAL,
                "FEATURE_ELIMINATION": data.FEATURE_ELIMINATION,
                "USE_CIBIL": config["USE_CIBIL"],
                "fold_thresholds": fold_thresholds,
            }
            project_root = get_project_root()
            config_path = os.path.join(project_root, 'artifacts', 'config.json')

            with open(config_path, "w") as f:
                json.dump(new_config, f)
            logger.info("Configuration saved successfully")

            return {"best_model_name": best_model_name}

        except PreprocessingException as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise PreprocessingException("Error during preprocessing") from e
        except HyperparameterOptimizationException as e:
            logger.error(f"Hyperparameter optimization error: {str(e)}")
            raise HyperparameterOptimizationException("Error during hyperparameter optimization") from e
        except FeatureSelectionException as e:
            logger.error(f"Feature selection error: {str(e)}")
            raise FeatureSelectionException("Error during feature selection") from e
        except ModelTrainingException as e:
            logger.error(f"Model training error: {str(e)}")
            raise ModelTrainingException("Error during model training") from e
        except ConfigSavingException as e:
            logger.error(f"Configuration saving error: {str(e)}")
            raise ConfigSavingException("Error saving configuration") from e
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}")
            raise ModelTrainingException("Unhandled error during training inference") from e

    def test_inference(self, data):
        """
        Perform inference on new data.

        This method performs inference on new data using the trained model.
        It loads the trained model, preprocesses the new data, and performs prediction using the loaded model.

        Args:
        - data: New data for which prediction is required.

        Returns:
        - dict: Dictionary containing the prediction result.
        """
        try:
            config = get_config()
            model_manager = ModelManager(config["best_model_name"])

            logger.info("Model loading started")
            model_manager.load_models()

            if len(model_manager.models) == 0:
                raise ModelNotLoadedException()

            data = pd.DataFrame([data.dict()])

            logger.info("Preprocess started")
            preprocessor = Preprocess(data, inference=True)

            data = preprocessor.execute()[config["feat_cols"]]
            logger.info("Preprocess completed")

            logger.info("Prediction started")
            probas = []
            preds = []
            for idx, model in enumerate(model_manager.models):
                proba = model.predict_proba(data)[:, 1][0]

                probas.append(proba)
                preds.append(int(proba > config["fold_thresholds"][idx]))
            logger.info("Prediction completed")
            return {"prediction": int(np.mean(preds))}

        except PreprocessingException as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise PreprocessingException("Failed to preprocess data") from e
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise PredictionException("Prediction failed") from e
