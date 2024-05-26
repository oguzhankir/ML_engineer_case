import warnings

import numpy as np
import optuna
import sklearn.metrics
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class HyperparameterSearch:
    """
    Class for hyperparameter optimization using Optuna.

    Attributes:
    - model_name (str): Name of the model to optimize hyperparameters for.
    - best_params (dict): Best hyperparameters found during optimization.
    - best_score (float): Best score achieved during optimization.
    """

    def __init__(self, model_name, X, y):
        """
        Initialize the HyperparameterSearch object.

        Args:
        - model_name (str): Name of the model to optimize hyperparameters for.
        """

        self.model_name = model_name
        self.best_params = None
        self.best_score = None
        self.X = X
        self.y = y

    def objective(self, trial, data, target):
        """
        Objective function for hyperparameter optimization.

        Args:
        - trial (optuna.Trial): The current trial.
        - data (list of tuples): List of tuples containing train and validation indices.
        - target (str): Name of the target column.

        Returns:
        - float: Mean F1 score for the given hyperparameters.
        """

        params = {
            "CatBoost": {
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
                "depth": trial.suggest_int("depth", 1, 10),
                "bagging_temperature": trial.suggest_int("bagging_temperature", 1, 7),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
                ),
                "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
                "auto_class_weights": trial.suggest_categorical(
                    "auto_class_weights", ["SqrtBalanced", "Balanced"]
                ),
                "model_shrink_mode": trial.suggest_categorical(
                    "model_shrink_mode", ["Constant", "Decreasing"]
                ),
                "penalties_coefficient": trial.suggest_float(
                    "penalties_coefficient", 0, 6
                ),
                "model_shrink_rate": trial.suggest_float("model_shrink_rate", 0, 1),
                "random_strength": trial.suggest_float("random_strength", 0, 10),
                "task_type": "CPU",
                "devices": "0:1",
                "iterations": 1000,
                "random_state": 42,
                "early_stopping_rounds": 200,
                "thread_count": -1,
                "allow_writing_files": False,
                "eval_metric": "F1:use_weights=False",
                "has_time": False,
                "verbose": False,
            },
            "LGBM": {
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "n_estimators": 1000,
                "random_state": 42,
                "early_stopping_rounds": 200,
                "eval_metric": "F1",
                "verbosity": -1,
            },
            "XGB": {
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "n_estimators": 1000,
                "random_state": 42,
                "early_stopping_rounds": 200,
                "eval_metric": "auc",
            },
        }

        model = eval(f"{self.model_name}Classifier")(**params[self.model_name])

        scores = []
        for train_idx, val_idx in data:
            model.fit(
                self.X.loc[train_idx],
                self.y.loc[train_idx],
                eval_set=[(self.X.loc[val_idx], self.y.loc[val_idx])],
            )

            preds = model.predict(self.X.loc[val_idx])

            score = sklearn.metrics.f1_score(self.y.loc[val_idx], preds)
            scores.append(score)

        cv_score = np.mean(scores)

        return cv_score

    def optimize(self, data, target, n_trials=100):
        """
        Perform hyperparameter optimization.

        Args:
        - data (list of tuples): List of tuples containing train and validation indices.
        - target (str): Name of the target column.
        - n_trials (int): Number of optimization trials.

        Returns:
        - tuple: A tuple containing the best hyperparameters and the best score.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, data, target), n_trials=n_trials
        )
        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"Best {self.model_name} params: {self.best_params}")
        print(f"Best {self.model_name} score: {self.best_score}")

        return self.best_params, self.best_score


def param_space(model_name):
    space = {
        "CatBoost": [
            "learning_rate",
            "depth",
            "bagging_temperature",
            "subsample",
            "min_data_in_leaf",
            "grow_policy",
            "l2_leaf_reg",
            "auto_class_weights",
            "model_shrink_mode",
            "penalties_coefficient",
            "model_shrink_rate",
            "random_strength",
            "task_type",
            "devices",
            "iterations",
            "random_state",
            "early_stopping_rounds",
            "thread_count",
            "allow_writing_files",
            "eval_metric",
            "has_time",
            "verbose",
        ],
        "LGBM": [
            "learning_rate",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "n_estimators",
            "random_state",
            "early_stopping_rounds",
            "eval_metric",
            "verbosity",
        ],
        "XGB": [
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "n_estimators",
            "random_state",
            "early_stopping_rounds",
            "eval_metric",
        ],
    }
    return space[model_name]
