from pydantic import BaseModel
from typing import Dict

class ForecastRequest(BaseModel):
    Store: int
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

class CompareRequest(BaseModel):
    Store: int
    Date: str  # format: "YYYY-MM-DD" - needed for Prophet/SARIMA which forecast by date

class ModelComparisonResponse(BaseModel):
    predictions: Dict[str, float]