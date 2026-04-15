from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.config import DATA_PROCESSED_DIR, LiveDataConfig


class LiveDataError(Exception):
    pass


@dataclass
class FREDClient:
    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(
        self,
        series_id: str,
        observation_start: str = "2020-01-01",
        observation_end: str | None = None,
    ) -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
        }

        if observation_end:
            params["observation_end"] = observation_end

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        observations = payload.get("observations", [])
        if not observations:
            raise LiveDataError(f"No FRED observations returned for {series_id}")

        df = pd.DataFrame(observations)[["date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["series_id"] = series_id
        return df.dropna(subset=["value"])


@dataclass
class NWSClient:
    base_url: str = "https://api.weather.gov"
    user_agent: str = "supply-chain-ai-research/1.0 (contact: research@example.com)"

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/geo+json",
        }

    def fetch_point_forecast(self, latitude: float, longitude: float) -> dict[str, Any]:
        point_url = f"{self.base_url}/points/{latitude},{longitude}"
        point_response = requests.get(point_url, headers=self._headers(), timeout=30)
        point_response.raise_for_status()
        point_payload = point_response.json()

        forecast_url = point_payload["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=self._headers(), timeout=30)
        forecast_response.raise_for_status()
        forecast_payload = forecast_response.json()

        periods = forecast_payload.get("properties", {}).get("periods", [])
        if not periods:
            raise LiveDataError(f"No forecast periods returned for {latitude}, {longitude}")

        first = periods[0]
        return {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_name": first.get("name"),
            "temperature": first.get("temperature"),
            "temperature_unit": first.get("temperatureUnit"),
            "wind_speed": first.get("windSpeed"),
            "wind_direction": first.get("windDirection"),
            "short_forecast": first.get("shortForecast"),
            "is_daytime": first.get("isDaytime"),
            "start_time": first.get("startTime"),
            "end_time": first.get("endTime"),
        }


class LiveSupplyChainDataPipeline:
    def __init__(self, fred_api_key: str, config: LiveDataConfig | None = None) -> None:
        self.config = config or LiveDataConfig()
        self.fred = FREDClient(api_key=fred_api_key)
        self.nws = NWSClient()

    def fetch_fred_data(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for series_id in self.config.fred_series:
            df = self.fred.fetch_series(
                series_id=series_id,
                observation_start=self.config.fred_start_date,
                observation_end=self.config.fred_end_date,
            )
            frames.append(df)

        long_df = pd.concat(frames, ignore_index=True)
        pivot_df = (
            long_df.pivot(index="date", columns="series_id", values="value")
            .sort_index()
            .reset_index()
        )
        pivot_df.columns.name = None
        return pivot_df

    #def fetch_fred_data(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for series_id in self.config.fred_series:
            df = self.fred.fetch_series(
                series_id=series_id,
                observation_start=self.config.fred_start_date,
                observation_end=self.config.fred_end_date,
            )
            frames.append(df)

        long_df = pd.concat(frames, ignore_index=True)
        pivot_df = (
            long_df.pivot(index="date", columns="series_id", values="value")
            .sort_index()
            .reset_index()
        )
        pivot_df.columns.name = None
        return pivot_df

    def fetch_weather_data(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for lat, lon in self.config.nws_points:
            rows.append(self.nws.fetch_point_forecast(lat, lon))
        return pd.DataFrame(rows)

    def build_feature_store(self) -> dict[str, pd.DataFrame]:
        fred_df = self.fetch_fred_data()
        weather_df = self.fetch_weather_data()

        if not fred_df.empty:
            fred_df["truck_cost_index_3m_avg"] = fred_df["PCU484484"].rolling(3).mean()
            fred_df["truckload_index_3m_avg"] = fred_df["PCU484121484121"].rolling(3).mean()
            fred_df["local_freight_index_3m_avg"] = fred_df["PCU484110484110P"].rolling(3).mean()

        return {
            "fred_features": fred_df,
            "weather_features": weather_df,
        }

    def save_feature_store(self, feature_store: dict[str, pd.DataFrame]) -> None:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        for name, df in feature_store.items():
            out_path = DATA_PROCESSED_DIR / f"{name}.csv"
            df.to_csv(out_path, index=False)