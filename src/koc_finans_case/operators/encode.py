import category_encoders as ce
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class EncoderEvaluator:
    """
    Class for evaluating different encoding methods for categorical columns.

    Attributes:
    - dataframe (DataFrame): The DataFrame containing the data.
    - target_column (str): The name of the target column.
    - categorical_columns (list): List of column names to encode.
    """

    def __init__(self, dataframe, target_column, categorical_columns):
        """
        Initialize the EncoderEvaluator with the provided DataFrame, target column,
        and list of categorical columns.

        Args:
        - dataframe (DataFrame): The DataFrame containing the data.
        - target_column (str): The name of the target column.
        - categorical_columns (list): List of column names to encode.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.X = dataframe.drop(columns=[target_column])
        self.y = dataframe[target_column]
        self.encoders = {
            # 'onehot': OneHotEncoder(),
            "label": LabelEncoder(),
            "ordinal": ce.OrdinalEncoder(),
            "catboost": ce.CatBoostEncoder(),
            "target": ce.TargetEncoder(),
        }

    def evaluate_encoders(self):
        """
        Evaluates different encoding methods using cross-validation.

        Returns:
        - tuple: A tuple containing the best encoder name and the mean accuracy scores for each encoder.
        """

        results = {}
        for encoder_name, encoder in self.encoders.items():
            if encoder_name == "label":
                X_encoded = self.X.copy()
                for col in self.categorical_columns:
                    X_encoded[col] = LabelEncoder().fit_transform(self.X[col])
                scores = self._evaluate_model(X_encoded)
            else:
                preprocessor = ColumnTransformer(
                    transformers=[("encoder", encoder, self.categorical_columns)],
                    remainder="passthrough",
                )
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", CatBoostClassifier(silent=True)),
                    ]
                )
                scores = cross_val_score(
                    pipeline, self.X, self.y, cv=5, scoring="accuracy"
                )

            results[encoder_name] = scores.mean()

        best_encoder = max(results, key=results.get)
        return best_encoder, results

    def _evaluate_model(self, X):
        """
        Evaluates a model using cross-validation.

        Args:
        - X (DataFrame): The input features.

        Returns:
        - array: An array of accuracy scores for each fold in cross-validation.
        """
        model = CatBoostClassifier(silent=True)
        scores = cross_val_score(model, X, self.y, cv=5, scoring="accuracy")
        return scores
