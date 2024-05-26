from typing import Literal

from pydantic import BaseModel


class TestPredictionDTO(BaseModel):
    """
        Data transfer object for test prediction requests.

        This class defines the structure of the input data required for making a prediction.
        It includes the following fields:
        - loan_id (int): Unique identifier for the loan.
        - no_of_dependents (int): Number of dependents.
        - education (Literal[' Graduate', 'Graduate', ' Not Graduate', 'Not Graduate']):
          Education level of the applicant.
        - self_employed (Literal[' Yes', 'Yes', ' No', 'No']): Employment status of the applicant.
        - income_annum (int): Annual income of the applicant.
        - loan_amount (int): Amount of the loan.
        - loan_term (int): Term of the loan in months.
        - cibil_score (int): CIBIL score of the applicant.
        - residential_assets_value (int): Value of residential assets.
        - commercial_assets_value (int): Value of commercial assets.
        - luxury_assets_value (int): Value of luxury assets.
        - bank_asset_value (int): Value of bank assets.

        """

    loan_id: int
    no_of_dependents: int
    education: Literal[' Graduate', 'Graduate', ' Not Graduate', 'Not Graduate']
    self_employed: Literal[' Yes', 'Yes', ' No', 'No']
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


class ReTrainDTO(BaseModel):
    """
        Data transfer object for retraining requests.

        This class defines the structure of the input data required for retraining the model.
        It includes the following fields:
        - FOLD_CNT (int): Number of folds for cross-validation. Default is 5.
        - SEED (int): Random seed for reproducibility. Default is 42.
        - ITERATION (int): Number of iterations for training. Default is 2000.
        - TUNE (bool): Flag to indicate whether hyperparameter tuning should be performed. Default is True.
        - EVAL_METRIC (Literal["f1_score", "accuracy_score", "precision_score"]): Evaluation metric to be used.
          Default is "f1_score".
        - N_TRIAL (int): Number of trials for hyperparameter tuning. Default is 2.
        - FEATURE_ELIMINATION (bool): Flag to indicate whether feature elimination should be performed.
          Default is True.

        """
    FOLD_CNT: int = 5
    SEED: int = 42
    ITERATION: int = 2000
    TUNE: bool = True
    EVAL_METRIC: Literal["f1_score", "accuracy_score", "precision_score"] = "f1_score"
    N_TRIAL: int = 2
    FEATURE_ELIMINATION: bool = True
