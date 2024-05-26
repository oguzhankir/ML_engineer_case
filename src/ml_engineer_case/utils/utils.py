import json
import os
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
def get_project_root():
    """
    Get the project root directory.

    Returns:
    - str: Path to the project root directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_train_data():
    """
    Get the training data.

    This function reads the training data from a CSV file and returns it as a DataFrame.

    Returns:
    - DataFrame: Training data.
    """
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'artifacts', 'data', 'loan_approval_dataset.csv')

    return pd.read_csv(data_path)

def optimize_thresholds(df, models, cv_splits, feat_cols):
    """
    Optimize prediction thresholds for each fold.

    This function optimizes the prediction thresholds for each fold based on the F1 score.

    Args:
    - df (DataFrame): Input DataFrame containing the data.
    - models (list): List of trained models.
    - cv_splits (list): List of tuples containing indices for cross-validation splits.
    - feat_cols (list): List of feature columns.

    Returns:
    - list: List of optimized thresholds for each fold.
    """
    config = get_config()

    fold_thresholds = []

    fold_scores = []
    thresholds = np.arange(0.1, 0.7, 0.002)

    for idx, (split_train, split_val) in enumerate(tqdm(cv_splits)):

        proba = models[idx].predict_proba(df[feat_cols].iloc[split_val])[:, 1]

        scores = []

        for t in thresholds:
            scores.append(f1_score(df[config["label"]].iloc[split_val], proba >= t))

        ix = np.argmax(scores)


        fold_scores.append(scores[ix])

        fold_thresholds.append(thresholds[ix])
    return fold_thresholds



def get_config():
    """
    Get the configuration settings.

    This function reads the configuration settings from a JSON file and returns them as a dictionary.

    Returns:
    - dict: Configuration settings.
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'artifacts', 'config.json')

    with open(config_path) as json_file:
        data = json.load(json_file)

    return data