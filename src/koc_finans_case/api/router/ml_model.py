from fastapi import APIRouter
from ..request_dto import *
from ...services.ml_model import *
ml_model_router = APIRouter(prefix="/inference", tags=['inference'])


@ml_model_router.post("/prediction")
def get_prediction_api(data: TestPredictionDTO) -> float:
    return get_prediction_service(data)


