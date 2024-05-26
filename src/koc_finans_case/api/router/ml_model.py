from fastapi import APIRouter, HTTPException

from ..request_dto import *
from ...services.ml_model import *
from ...utils.exceptions import *
from ...utils.logger import AppLogger

ml_model_router = APIRouter(prefix="/inference", tags=['inference'])

logger = AppLogger(__name__).get_logger()


@ml_model_router.post("/prediction")
def get_prediction_api(data: TestPredictionDTO) -> dict:
    """
        Handle prediction requests for the ML model.

        This endpoint receives a prediction request in the form of a TestPredictionDTO object,
        logs the receipt of the request, and calls the get_prediction_service function to
        obtain the prediction result. If a CustomException occurs, it logs the error and
        raises an HTTPException with the appropriate status code and message. If any other
        exception occurs, it logs the error and raises a generic HTTPException with a 500
        status code indicating an internal server error.

        Args:
            data (TestPredictionDTO): The input data required for making the prediction.
                - loan_id (int): Unique identifier for the loan.
                - no_of_dependents (int): Number of dependents.
                - education (str): Education level of the applicant.
                - self_employed (str): Employment status of the applicant.
                - income_annum (int): Annual income of the applicant.
                - loan_amount (int): Amount of the loan.
                - loan_term (int): Term of the loan in months.
                - cibil_score (int): CIBIL score of the applicant.
                - residential_assets_value (int): Value of residential assets.
                - commercial_assets_value (int): Value of commercial assets.
                - luxury_assets_value (int): Value of luxury assets.
                - bank_asset_value (int): Value of bank assets.

        Returns:
            dict: A dictionary containing the prediction result.

        Raises:
            HTTPException: If a CustomException occurs, it raises an HTTPException with the
            status code and message from the CustomException. For any other exception, it
            raises an HTTPException with a 500 status code and "Internal Server Error" message.
        """
    logger.info("Received prediction request")
    try:
        return get_prediction_service(data)
    except CustomException as ce:
        logger.error(f"Custom exception occurred: {str(ce)}")
        raise HTTPException(status_code=ce.status_code, detail=ce.message)
    except Exception as e:
        logger.error(f"Unhandled exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@ml_model_router.post("/re_train")
def retrain_api(data: ReTrainDTO) -> dict:
    """
        Handle retraining requests for the ML model.

        This endpoint receives a retraining request in the form of a ReTrainDTO object,
        logs the receipt of the request, and calls the retrain_service function to
        perform the retraining. If a CustomException occurs, it logs the error and
        raises an HTTPException with the appropriate status code and message. If any
        other exception occurs, it logs the error and raises a generic HTTPException
        with a 500 status code indicating an internal server error.

        Args:
            data (ReTrainDTO): The input data required for retraining the model.
                - FOLD_CNT (int): Number of folds for cross-validation.
                - SEED (int): Random seed for reproducibility.
                - ITERATION (int): Number of iterations for training.
                - TUNE (bool): Flag to indicate whether hyperparameter tuning should be performed.
                - EVAL_METRIC (str): Evaluation metric to be used.
                - N_TRIAL (int): Number of trials for hyperparameter tuning.
                - FEATURE_ELIMINATION (bool): Flag to indicate whether feature elimination should be performed.

        Returns:
            dict: A dictionary containing the result of the retraining process.

        Raises:
            HTTPException: If a CustomException occurs, it raises an HTTPException with the
            status code and message from the CustomException. For any other exception, it
            raises an HTTPException with a 500 status code and "Internal Server Error" message.

        """
    logger.info("Received retrain request")
    try:
        return retrain_service(data)
    except CustomException as ce:
        logger.error(f"Custom exception occurred: {str(ce)}")
        raise HTTPException(status_code=ce.status_code, detail=ce.message)
    except Exception as e:
        logger.error(f"Unhandled exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
