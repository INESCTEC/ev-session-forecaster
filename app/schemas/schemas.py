import datetime as dt

from typing import List, Optional
from pydantic import BaseModel


class ForecastComputePayload(BaseModel):
    evse_id: str
    user_id: int
    session_id: Optional[str] = None
    launch_time: Optional[dt.datetime] = None


class TimeSeriesData(BaseModel):
    datetime: dt.datetime
    evse_id: str
    session_id: str
    value: float
    unit: str


class TimeSeries(BaseModel):
    data: List[TimeSeriesData]


class ForecastReturnHorizon(BaseModel):
    launch_time: dt.datetime
    session_id: str
    d0: float
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    d6: float
    d7: float
    d7_plus: float = None


class ForecastReturn(BaseModel):
    launch_time: dt.datetime
    session_id: str
    next_start_time: dt.datetime
    expected_returnal: List[ForecastReturnHorizon]


class TimeSeriesReturn(BaseModel):
    data: List[ForecastReturn]