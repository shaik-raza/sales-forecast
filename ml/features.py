"""Leakage-safe feature engineering shared by training and inference."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


LAG_DAYS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)
FEATURE_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "DaysSincePromo",
    "DaysUntilPromo",
    "RollingMean7",
    "RollingStd7",
    "RollingMean28",
    "RollingStd28",
    "MonthStart",
    "MonthEnd",
    "DaysSincePromo2",
    "DaysUntilPromo2",
    "DaysToNearestHoliday",
    "CompetitionOpenDays",
    "Promo2Active",
    "SalesLag7",
    "SalesLag14",
    "SalesLag28",
]


def _nearest_distances(
    dates: pd.Series,
    event_dates: Iterable[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative days since and until the nearest event."""
    event_values = np.array(sorted(set(pd.to_datetime(list(event_dates)))), dtype="datetime64[ns]")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    if event_values.size == 0:
        missing = np.full(len(dates), np.nan)
        return missing, missing.copy()

    positions = np.searchsorted(event_values, date_values)
    previous_positions = np.maximum(positions - 1, 0)
    next_positions = np.minimum(positions, len(event_values) - 1)
    previous = event_values[previous_positions]
    following = event_values[next_positions]
    since = (date_values - previous).astype("timedelta64[D]").astype(float)
    until = (following - date_values).astype("timedelta64[D]").astype(float)
    since[positions == 0] = np.nan
    until[positions == len(event_values)] = np.nan
    return since, until


def _add_event_distances(
    frame: pd.DataFrame,
    event_column: str,
    since_column: str,
    until_column: str,
) -> None:
    """Add event distances independently for each store."""
    frame[since_column] = np.nan
    frame[until_column] = np.nan
    for store, indices in frame.groupby("Store", sort=False).groups.items():
        store_rows = frame.loc[indices]
        event_dates = store_rows.loc[store_rows[event_column].eq(1), "Date"]
        since, until = _nearest_distances(store_rows["Date"], event_dates)
        frame.loc[indices, since_column] = since
        frame.loc[indices, until_column] = until


def _promo2_active(row: pd.Series) -> int:
    """Return whether Promo2 is active for a row's calendar date."""
    if row["Promo2"] != 1 or pd.isna(row["Promo2SinceYear"]) or pd.isna(row["Promo2SinceWeek"]):
        return 0
    start = pd.to_datetime(
        f"{int(row['Promo2SinceYear'])}-{int(row['Promo2SinceWeek'])}-1",
        format="%G-%V-%u",
    )
    interval = row.get("PromoInterval")
    if pd.isna(interval) or not str(interval).strip() or str(interval) == "None":
        return int(row["Date"] >= start)
    active_months = {month.strip() for month in str(interval).split(",")}
    return int(row["Date"] >= start and row["Date"].strftime("%b") in active_months)


def build_features(data: pd.DataFrame, require_sales: bool = True) -> pd.DataFrame:
    """Build model features while ensuring rolling values use only past sales."""
    required = {"Store", "Date", "Promo", "StateHoliday"}
    if require_sales:
        required.add("Sales")
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.sort_values(["Store", "Date"]).reset_index(drop=True)

    if require_sales:
        grouped_sales = frame.groupby("Store", sort=False)["Sales"]
        for lag in LAG_DAYS:
            frame[f"SalesLag{lag}"] = grouped_sales.shift(lag)
        past_sales = grouped_sales.shift(1)
        rolling = past_sales.groupby(frame["Store"], sort=False)
        for window in ROLLING_WINDOWS:
            frame[f"RollingMean{window}"] = rolling.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            frame[f"RollingStd{window}"] = rolling.transform(
                lambda values: values.rolling(window, min_periods=2).std()
            ).fillna(0)
    else:
        for column in (
            "SalesLag7",
            "SalesLag14",
            "SalesLag28",
            "RollingMean7",
            "RollingStd7",
            "RollingMean28",
            "RollingStd28",
        ):
            frame[column] = np.nan

    _add_event_distances(frame, "Promo", "DaysSincePromo", "DaysUntilPromo")
    frame[["DaysSincePromo", "DaysUntilPromo"]] = frame[
        ["DaysSincePromo", "DaysUntilPromo"]
    ].fillna(0)
    frame["MonthStart"] = frame["Date"].dt.is_month_start.astype(int)
    frame["MonthEnd"] = frame["Date"].dt.is_month_end.astype(int)

    holiday_dates = frame.loc[frame["StateHoliday"].ne(0), "Date"]
    since_holiday, until_holiday = _nearest_distances(frame["Date"], holiday_dates)
    frame["DaysToNearestHoliday"] = np.minimum(since_holiday, until_holiday)
    frame["DaysToNearestHoliday"] = frame["DaysToNearestHoliday"].fillna(365)

    frame["CompetitionOpenDays"] = (
        (frame["Date"].dt.year - frame["CompetitionOpenSinceYear"]) * 365
        + (frame["Date"].dt.month - frame["CompetitionOpenSinceMonth"]) * 30
    ).clip(lower=0).fillna(0)
    frame["Promo2Active"] = frame.apply(_promo2_active, axis=1)
    _add_event_distances(frame, "Promo2Active", "DaysSincePromo2", "DaysUntilPromo2")
    frame[["DaysSincePromo2", "DaysUntilPromo2"]] = frame[
        ["DaysSincePromo2", "DaysUntilPromo2"]
    ].fillna(0)

    return frame


def model_features(data: pd.DataFrame, require_sales: bool = True) -> pd.DataFrame:
    """Return the stable feature matrix used by all model scripts."""
    features = build_features(data, require_sales=require_sales)
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise KeyError(f"Feature construction omitted columns: {missing}")
    return features[FEATURE_COLUMNS]
