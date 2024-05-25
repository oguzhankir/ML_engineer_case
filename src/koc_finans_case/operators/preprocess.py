import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import pickle


import os
def select_features_with_cv(df, n_splits=5):
    X = df.drop(["loan_status", "loan_id"], axis=1)
    y = df["loan_status"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_importances = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = CatBoostClassifier()

        rfe_dict = model.select_features(X=Pool(X_train, y_train),
                                         eval_set=Pool(X_val, y_val),
                                         features_for_select=f"0-{len(X.columns.tolist()) - 1}",
                                         num_features_to_select=len(X.columns.tolist()) - 3,
                                         steps=5,
                                         verbose=False,
                                         train_final_model=False,
                                         plot=True)

        feature_importances.append(rfe_dict['selected_features_names'])

    selected_features = set.intersection(*map(set, feature_importances))

    return list(selected_features)


class EncoderEvaluator:
    def __init__(self, dataframe, target_column, categorical_columns):
        self.dataframe = dataframe
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.X = dataframe.drop(columns=[target_column])
        self.y = dataframe[target_column]
        self.encoders = {
            # 'onehot': OneHotEncoder(),
            'label': LabelEncoder(),
            'ordinal': ce.OrdinalEncoder(),
            'catboost': ce.CatBoostEncoder(),
            'target': ce.TargetEncoder()
        }

    def evaluate_encoders(self):
        results = {}
        for encoder_name, encoder in self.encoders.items():
            print(f"Evaluating with {encoder_name} encoder...")
            if encoder_name == 'label':
                # LabelEncoder only applies to a single column at a time, so we need to encode them separately
                X_encoded = self.X.copy()
                for col in self.categorical_columns:
                    X_encoded[col] = LabelEncoder().fit_transform(self.X[col])
                scores = self._evaluate_model(X_encoded)
            else:
                # For other encoders we can use ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('encoder', encoder, self.categorical_columns)
                    ],
                    remainder='passthrough'  # passthrough to keep the other columns unchanged
                )
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', CatBoostClassifier(silent=True))
                ])
                scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')

            results[encoder_name] = scores.mean()

        best_encoder = max(results, key=results.get)
        return best_encoder, results

    def _evaluate_model(self, X):
        model = CatBoostClassifier(silent=True)
        scores = cross_val_score(model, X, self.y, cv=5, scoring='accuracy')
        return scores


class Preprocess:
    def __init__(self, df, inference=False, encoder_path="src/koc_finans_case/artifacts/encoders/encoder.pkl"):
        self.df = df.copy()
        self.categorical_features = ["education", "self_employed"]
        self.encoder_eval = EncoderEvaluator
        self.inference = inference
        self.encoder_path = encoder_path
        self.encoders = {
            # 'onehot': OneHotEncoder(),
            'label': LabelEncoder(),
            'ordinal': ce.OrdinalEncoder(),
            'catboost': ce.CatBoostEncoder(),
            'target': ce.TargetEncoder()
        }

    def fix_column_names(self):
        self.df.columns = [col.strip() for col in self.df.columns]

    def fix_categorical_features(self):
        for col in self.categorical_features:
            self.df[col] = self.df[col].str.strip()
        if "loan_status" in self.df.columns:
            self.df["loan_status"] = self.df["loan_status"].str.strip()

    def _object_to_category(self):
        cols = self.df.select_dtypes("object").columns.tolist()
        self.df[cols] = self.df[cols].astype("category")

    def _evaluate_encoder(self, cats):
        evaluator = self.encoder_eval(self.df, "loan_status", cats)
        best_encoder, results = evaluator.evaluate_encoders()
        self.best_encoder = best_encoder
        return best_encoder

    def get_encoder(self, col):
        with open("/".join(self.encoder_path.split("/")[:-1] + [f"{col}_{self.encoder_path.split('/')[-1]}"]), "rb") as f:
            loaded_encoder = pickle.load(f)

        return loaded_encoder

    def write_encoder(self, encoder, filename):
        with open(filename, "wb") as f:
            pickle.dump(encoder, f)

    def encode_categorical(self):
        if "loan_status" in self.df.columns.tolist():
            self.df["loan_status"] = self.df["loan_status"].str.strip().replace({"Approved": 1, "Rejected": 0})

        cats = self.df.select_dtypes(["object", "category"]).columns.tolist()

        if self.inference:
            for col in cats:
                encoder = self.get_encoder(col)
                try:
                    self.df[col] = encoder.transform(self.df[col])
                except:
                    self.df[col] = encoder.transform(self.df[col], self.df["loan_status"])

        else:
            best_encoder = self._evaluate_encoder(cats)

            try:
                for col in cats:
                    encoder = self.encoders[best_encoder]
                    encoder.fit(self.df[col])
                    self.df[col] = encoder.transform(self.df[col])
                    self.write_encoder(encoder, f"{col}_{self.encoder_path}")
            except:
                for col in cats:
                    encoder = self.encoders[best_encoder]
                    encoder.fit(self.df[col], self.df["loan_status"])
                    self.df[col] = encoder.transform(self.df[col], self.df["loan_status"])
                    self.write_encoder(encoder, f"{col}_{self.encoder_path}")

    def extract_features(self):
        self.df['debt_to_income_ratio'] = self.df['loan_amount'] / self.df['income_annum']
        self.df['total_assets_value'] = self.df['residential_assets_value'] + self.df['commercial_assets_value'] + \
                                        self.df['luxury_assets_value'] + self.df['bank_asset_value']
        self.df['ltv_ratio'] = self.df['loan_amount'] / self.df['total_assets_value']
        self.df['income_stability'] = self.df['self_employed'].apply(lambda x: 1 if x == 'No' else 0)
        self.df['dependents_to_income_ratio'] = self.df['no_of_dependents'] / self.df['income_annum']
        self.df['cibil_score_category'] = pd.cut(self.df['cibil_score'], bins=[0, 300, 600, 750, 900],
                                                 labels=['Poor', 'Fair', 'Good', 'Excellent'])
        self.df['loan_term_years'] = self.df['loan_term'] / 12
        self.df['high_value_assets'] = self.df['total_assets_value'].apply(lambda x: 1 if x > 10000000 else 0)
        self.df['emi'] = self.df['loan_amount'] / self.df['loan_term']
        self.df['income_to_loan_amount_ratio'] = self.df['income_annum'] / self.df['loan_amount']
        self.df['individual_asset_ratio'] = (self.df['residential_assets_value'] + self.df['luxury_assets_value']) / \
                                            self.df['total_assets_value']
        self.df['debt_service_ratio'] = (self.df['loan_amount'] / self.df['loan_term']) / self.df['income_annum']

    def execute(self):
        self.fix_column_names()
        self.fix_categorical_features()
        self.extract_features()
        self.encode_categorical()
        # self._object_to_category()

        return self.df

