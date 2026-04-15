from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
FRED_PATH = BASE_DIR / "data" / "processed" / "fred_features.csv"
WEATHER_PATH = BASE_DIR / "data" / "processed" / "weather_features.csv"

st.set_page_config(page_title="Supply Chain AI Dashboard", layout="wide")
st.title("Supply Chain AI - Live Public Data Dashboard")
st.caption("BTS/FRED-style freight indicators + NOAA weather disruption signals")

if not FRED_PATH.exists() or not WEATHER_PATH.exists():
    st.warning("Processed live data files not found. Run `python main.py` first.")
    st.stop()

fred_df = pd.read_csv(FRED_PATH, parse_dates=["date"])
weather_df = pd.read_csv(WEATHER_PATH)

col1, col2, col3 = st.columns(3)
col1.metric("Latest Truck Transportation PPI", f"{fred_df['PCU484484'].dropna().iloc[-1]:.2f}")
col2.metric("Latest Truckload Index", f"{fred_df['PCU484121484121'].dropna().iloc[-1]:.2f}")
col3.metric("Weather Locations", f"{len(weather_df)}")

fig1 = px.line(
    fred_df,
    x="date",
    y=["PCU484484", "PCU484121484121", "PCU484110484110P"],
    title="Freight and Trucking Economic Indicators",
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Live Weather Risk Snapshot")
st.dataframe(weather_df, use_container_width=True)

forecast_counts = (
    weather_df["short_forecast"]
    .fillna("Unknown")
    .value_counts()
    .reset_index()
)
forecast_counts.columns = ["forecast", "count"]

fig2 = px.bar(
    forecast_counts,
    x="forecast",
    y="count",
    title="Current Forecast Categories Across Selected Logistics Hubs",
)
st.plotly_chart(fig2, use_container_width=True)