import unittest

from src.koc_finans_case.api.request_dto import *
from src.koc_finans_case.operators.inference import Inference
from src.koc_finans_case.utils.utils import get_config


class InferenceTest(unittest.TestCase):
    def test_inference(self):
        data = {"loan_id": 1, "no_of_dependents": 2, "education": " Graduate", "self_employed": " No",
                "income_annum": 9600000, "loan_amount": 29900000, "loan_term": 12, "cibil_score": 778,
                "residential_assets_value": 2400000, "commercial_assets_value": 17600000,
                "luxury_assets_value": 22700000, "bank_asset_value": 8000000}

        data = TestPredictionDTO(**data)

        config = get_config()

        inference = Inference(config)
        res = inference.test_inference(data)
        print(res)
