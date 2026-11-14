from datetime import date
from typing import Dict

from pydantic import BaseModel


class ForecastRequest(BaseModel):
    Store: int
    Date: date
    DayOfWeek: int
    Promo: int
    StateHoliday: int
    SchoolHoliday: int
    StoreType: int
    Assortment: int
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: int
    Promo2SinceWeek: float
    Promo2SinceYear: float
    Year: int
    Month: int
    Day: int
    WeekOfYear: int

class ForecastResponse(BaseModel):
    predicted_sales: float

class CompareRequest(ForecastRequest):
    pass

class ModelComparisonResponse(BaseModel):
    predictions: Dict[str, float]