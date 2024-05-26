from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold


def select_features_with_cv(df, n_splits=5):
    """
    Selects features using recursive feature elimination (RFE) with cross-validation.

    Args:
    - df (DataFrame): The DataFrame containing the data.
    - n_splits (int): Number of splits for cross-validation.

    Returns:
    - list: A list of selected feature names.
    """

    X = df.drop(["loan_status", "loan_id"], axis=1)
    y = df["loan_status"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_importances = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = CatBoostClassifier()

        rfe_dict = model.select_features(
            X=Pool(X_train, y_train),
            eval_set=Pool(X_val, y_val),
            features_for_select=f"0-{len(X.columns.tolist()) - 1}",
            num_features_to_select=len(X.columns.tolist()) - 3,
            steps=5,
            verbose=False,
            train_final_model=False,
            plot=False,
        )

        feature_importances.append(rfe_dict["selected_features_names"])

    selected_features = set.intersection(*map(set, feature_importances))

    return list(selected_features)
