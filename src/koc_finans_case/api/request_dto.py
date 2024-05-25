from typing import Literal

from pydantic import BaseModel


class TestPredictionDTO(BaseModel):
    loan_id: int
    no_of_dependents: int
    education: Literal[' Graduate', ' Not Graduate']
    self_employed: Literal[' Yes', ' No']
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
