import unittest

from src.ml_engineer_case.api.request_dto import *
from src.ml_engineer_case.operators.inference import Inference
from src.ml_engineer_case.utils.utils import get_config


class InferenceTest(unittest.TestCase):
    def test_train_inference(self):
        """
            Test the training and inference functionality with sample configuration.

            This method prepares a sample configuration dictionary, converts it into a
            ReTrainDTO object, retrieves the configuration, initializes an Inference
            object, and tests the train_inference method with the provided data.

            The result of the training and inference process is printed.

            Sample Configuration:
                FOLD_CNT (int): Number of folds for cross-validation.
                SEED (int): Random seed for reproducibility.
                ITERATION (int): Number of iterations for training.
                TUNE (bool): Flag to indicate whether hyperparameter tuning should be performed.
                EVAL_METRIC (str): Evaluation metric to be used.
                N_TRIAL (int): Number of trials for hyperparameter tuning.
                FEATURE_ELIMINATION (bool): Flag to indicate whether feature elimination should be performed.

            Steps:
                1. Prepare sample configuration data as a dictionary.
                2. Convert the dictionary into a ReTrainDTO object.
                3. Retrieve the configuration using get_config().
                4. Initialize an Inference object with the retrieved configuration.
                5. Call the train_inference method with the ReTrainDTO object.
                6. Print the result of the training and inference process.

            """
        data = {'FOLD_CNT': 5,
                'SEED': 42,
                'ITERATION': 2000,
                'TUNE': True,
                'EVAL_METRIC': 'f1_score',
                'N_TRIAL': 2,
                'FEATURE_ELIMINATION': True}
        data = ReTrainDTO(**data)
        config = get_config()
        inference = Inference(config)
        res = inference.train_inference(data)
        print(res)

    def test_inference(self):
        """
            Test the inference functionality with sample data.

            This method prepares a sample input data dictionary, converts it into a
            TestPredictionDTO object, retrieves the configuration, initializes an
            Inference object, and tests the inference method with the provided data.

            The result of the inference is printed.

            Sample Data:
                loan_id (int): Unique identifier for the loan.
                no_of_dependents (int): Number of dependents.
                education (str): Education level of the applicant.
                self_employed (str): Employment status of the applicant.
                income_annum (int): Annual income of the applicant.
                loan_amount (int): Amount of the loan.
                loan_term (int): Term of the loan in months.
                cibil_score (int): CIBIL score of the applicant.
                residential_assets_value (int): Value of residential assets.
                commercial_assets_value (int): Value of commercial assets.
                luxury_assets_value (int): Value of luxury assets.
                bank_asset_value (int): Value of bank assets.

            Steps:
                1. Prepare sample input data as a dictionary.
                2. Convert the dictionary into a TestPredictionDTO object.
                3. Retrieve the configuration using get_config().
                4. Initialize an Inference object with the retrieved configuration.
                5. Call the test_inference method with the TestPredictionDTO object.
                6. Print the result of the inference.

            """
        data = {"loan_id": 1, "no_of_dependents": 2, "education": " Graduate", "self_employed": " No",
                "income_annum": 9600000, "loan_amount": 29900000, "loan_term": 12, "cibil_score": 778,
                "residential_assets_value": 2400000, "commercial_assets_value": 17600000,
                "luxury_assets_value": 22700000, "bank_asset_value": 8000000}

        data = TestPredictionDTO(**data)

        config = get_config()

        inference = Inference(config)
        res = inference.test_inference(data)
        print(res)
