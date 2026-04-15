from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
FRED_PATH = BASE_DIR / "data" / "processed" / "fred_features.csv"
WEATHER_PATH = BASE_DIR / "data" / "processed" / "weather_features.csv"
RESULTS_PATH = BASE_DIR / "results" / "comparison.csv"

st.set_page_config(
    page_title="Supply Chain AI Dashboard",
    layout="wide",
)

st.title("Supply Chain AI - Adaptive Routing Dashboard")
st.caption(
    "Live freight indicators, weather disruption signals, and reinforcement learning policy comparison"
)

# -------------------------------------------------------------------
# Helper: fetch live weather from weather.gov using latitude/longitude
# -------------------------------------------------------------------
# def fetch_live_weather(latitude: float, longitude: float) -> dict:
#     headers = {
#         "User-Agent": "supply-chain-ai-dashboard/1.0 (research use)",
#         "Accept": "application/geo+json",
#     }

#     point_url = f"https://api.weather.gov/points/{latitude},{longitude}"
#     point_response = requests.get(point_url, headers=headers, timeout=30)
#     point_response.raise_for_status()
#     point_payload = point_response.json()

#     forecast_url = point_payload["properties"]["forecast"]
#     forecast_response = requests.get(forecast_url, headers=headers, timeout=30)
#     forecast_response.raise_for_status()
#     forecast_payload = forecast_response.json()

#     periods = forecast_payload.get("properties", {}).get("periods", [])
#     if not periods:
#         raise ValueError("No forecast periods returned for the selected coordinates.")

#     first = periods[0]

#     return {
#         "latitude": latitude,
#         "longitude": longitude,
#         "forecast_name": first.get("name"),
#         "temperature": first.get("temperature"),
#         "temperature_unit": first.get("temperatureUnit"),
#         "wind_speed": first.get("windSpeed"),
#         "wind_direction": first.get("windDirection"),
#         "short_forecast": first.get("shortForecast"),
#         "is_daytime": first.get("isDaytime"),
#         "start_time": first.get("startTime"),
#         "end_time": first.get("endTime"),
#     }
def fetch_live_weather(latitude: float, longitude: float) -> dict:
    headers = {
        "User-Agent": "supply-chain-ai-dashboard/1.0 (research use)",
        "Accept": "application/geo+json",
    }

    point_url = f"https://api.weather.gov/points/{latitude},{longitude}"
    point_response = requests.get(point_url, headers=headers, timeout=30)
    point_response.raise_for_status()
    point_payload = point_response.json()

    props = point_payload.get("properties", {})
    rel_props = props.get("relativeLocation", {}).get("properties", {})

    city = rel_props.get("city", "Unknown")
    state = rel_props.get("state", "")
    forecast_url = props["forecast"]

    forecast_response = requests.get(forecast_url, headers=headers, timeout=30)
    forecast_response.raise_for_status()
    forecast_payload = forecast_response.json()

    periods = forecast_payload.get("properties", {}).get("periods", [])
    if not periods:
        raise ValueError("No forecast periods returned for the selected coordinates.")

    first = periods[0]

    return {
        "city": city,
        "state": state,
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


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Live Weather Input")
user_lat = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=41.8781,
    step=0.0001,
    format="%.4f",
)
user_lon = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-87.6298,
    step=0.0001,
    format="%.4f",
)

fetch_weather_btn = st.sidebar.button("Fetch Live Weather")

# -------------------------------------------------------------------
# Data availability checks
# -------------------------------------------------------------------
missing_files = []
if not FRED_PATH.exists():
    missing_files.append(str(FRED_PATH))
if not WEATHER_PATH.exists():
    missing_files.append(str(WEATHER_PATH))

if missing_files:
    st.warning("Required processed data files are missing. Run `python main.py` first.")
    with st.expander("Missing files"):
        for file in missing_files:
            st.write(file)
    st.stop()

# -------------------------------------------------------------------
# Load local processed data
# -------------------------------------------------------------------
fred_df = pd.read_csv(FRED_PATH, parse_dates=["date"])
weather_df = pd.read_csv(WEATHER_PATH)

results_df = None
if RESULTS_PATH.exists():
    results_df = pd.read_csv(RESULTS_PATH)

# -------------------------------------------------------------------
# Header metrics
# -------------------------------------------------------------------
st.subheader("Live Economic and Weather Signals")

latest_truck_ppi = fred_df["PCU484484"].dropna().iloc[-1] if "PCU484484" in fred_df.columns else None
latest_truckload = (
    fred_df["PCU484121484121"].dropna().iloc[-1]
    if "PCU484121484121" in fred_df.columns
    else None
)
latest_local = (
    fred_df["PCU484110484110P"].dropna().iloc[-1]
    if "PCU484110484110P" in fred_df.columns
    else None
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Truck Transportation PPI",
    f"{latest_truck_ppi:.2f}" if latest_truck_ppi is not None else "N/A",
)
col2.metric(
    "Truckload Index",
    f"{latest_truckload:.2f}" if latest_truckload is not None else "N/A",
)
col3.metric(
    "Local Freight Index",
    f"{latest_local:.2f}" if latest_local is not None else "N/A",
)
col4.metric("Stored Weather Locations", f"{len(weather_df)}")

# -------------------------------------------------------------------
# Live user-selected weather
# -------------------------------------------------------------------
# st.subheader("User-Selected Live Weather Forecast")

# if fetch_weather_btn:
#     try:
#         live_weather = fetch_live_weather(user_lat, user_lon)
#         live_weather_df = pd.DataFrame([live_weather])

#         live_col1, live_col2, live_col3, live_col4 = st.columns(4)
#         live_col1.metric("Forecast", str(live_weather.get("short_forecast", "N/A")))
#         live_col2.metric(
#             "Temperature",
#             f"{live_weather.get('temperature', 'N/A')} {live_weather.get('temperature_unit', '')}",
#         )
#         live_col3.metric("Wind Speed", str(live_weather.get("wind_speed", "N/A")))
#         live_col4.metric("Wind Direction", str(live_weather.get("wind_direction", "N/A")))

#         st.dataframe(live_weather_df, width='stretch')

#     except Exception as e:
#         st.error(f"Could not fetch live weather for the selected coordinates: {e}")
# else:
#     st.info("Enter latitude and longitude in the sidebar and click 'Fetch Live Weather'.")
st.subheader("User-Selected Live Weather Forecast")

if fetch_weather_btn:
    try:
        live_weather = fetch_live_weather(user_lat, user_lon)
        live_weather_df = pd.DataFrame([live_weather])

        city_label = live_weather.get("city", "Unknown")
        state_label = live_weather.get("state", "")
        place_label = f"{city_label}, {state_label}" if state_label else city_label

        st.success(f"Nearest location from coordinates: {place_label}")

        live_col1, live_col2, live_col3, live_col4 = st.columns(4)
        live_col1.metric("Location", place_label)
        live_col2.metric(
            "Temperature",
            f"{live_weather.get('temperature', 'N/A')} {live_weather.get('temperature_unit', '')}",
        )
        live_col3.metric("Forecast", str(live_weather.get("short_forecast", "N/A")))
        live_col4.metric("Wind", f"{live_weather.get('wind_speed', 'N/A')} {live_weather.get('wind_direction', '')}")

        st.dataframe(live_weather_df, width='stretch')

    except Exception as e:
        st.error(f"Could not fetch live weather for the selected coordinates: {e}")
else:
    st.info("Enter latitude and longitude in the sidebar and click 'Fetch Live Weather'.")
# -------------------------------------------------------------------
# Freight indicators chart
# -------------------------------------------------------------------
st.subheader("Freight and Trucking Economic Indicators")

fred_plot_cols = [
    col for col in ["PCU484484", "PCU484121484121", "PCU484110484110P"] if col in fred_df.columns
]

if fred_plot_cols:
    fred_long = fred_df.melt(
        id_vars="date",
        value_vars=fred_plot_cols,
        var_name="indicator",
        value_name="value",
    )

    indicator_name_map = {
        "PCU484484": "Truck Transportation PPI",
        "PCU484121484121": "Long-Distance Truckload Index",
        "PCU484110484110P": "Local Freight Services Index",
    }
    fred_long["indicator"] = fred_long["indicator"].map(indicator_name_map).fillna(fred_long["indicator"])

    fig1 = px.line(
        fred_long,
        x="date",
        y="value",
        color="indicator",
        title="Freight and Trucking Economic Indicators Over Time",
    )
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Value",
        legend_title="Indicator",
        height=550,
    )
    st.plotly_chart(fig1, width='stretch')
else:
    st.info("No freight indicator columns available for plotting.")

# -------------------------------------------------------------------
# Weather snapshot from stored weather data
# -------------------------------------------------------------------
st.subheader("Stored Weather Risk Snapshot")

display_weather_cols = [
    col
    for col in [
        "latitude",
        "longitude",
        "forecast_name",
        "temperature",
        "temperature_unit",
        "wind_speed",
        "wind_direction",
        "short_forecast",
        "is_daytime",
        "start_time",
        "end_time",
    ]
    if col in weather_df.columns
]

st.dataframe(
    weather_df[display_weather_cols] if display_weather_cols else weather_df,
    width='stretch',
)

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
    title="Current Forecast Categories Across Stored Logistics Hubs",
)
fig2.update_layout(
    xaxis_title="Forecast Category",
    yaxis_title="Count",
    height=450,
)
st.plotly_chart(fig2, width='stretch')

# -------------------------------------------------------------------
# Model comparison section
# -------------------------------------------------------------------
st.subheader("Routing Policy Performance")

if results_df is None:
    st.info("Model comparison file not found yet. Run `python main.py` to generate `results/comparison.csv`.")
else:
    st.dataframe(results_df, width='stretch')

    chart_col1, chart_col2 = st.columns(2)

    if {"model", "avg_reward"}.issubset(results_df.columns):
        fig3 = px.bar(
            results_df,
            x="model",
            y="avg_reward",
            title="Reward Comparison Across Routing Policies",
            text="avg_reward",
        )
        fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig3.update_layout(
            xaxis_title="Policy",
            yaxis_title="Average Reward",
            height=420,
        )
        chart_col1.plotly_chart(fig3, width='stretch')

    if {"model", "avg_on_time_rate"}.issubset(results_df.columns):
        fig4 = px.bar(
            results_df,
            x="model",
            y="avg_on_time_rate",
            title="On-Time Delivery Rate Comparison",
            text="avg_on_time_rate",
        )
        fig4.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig4.update_layout(
            xaxis_title="Policy",
            yaxis_title="On-Time Rate",
            yaxis_tickformat=".0%",
            height=420,
        )
        chart_col2.plotly_chart(fig4, width='stretch')

    chart_col3, chart_col4 = st.columns(2)

    if {"model", "avg_cost"}.issubset(results_df.columns):
        fig5 = px.bar(
            results_df,
            x="model",
            y="avg_cost",
            title="Operational Cost Comparison",
            text="avg_cost",
        )
        fig5.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig5.update_layout(
            xaxis_title="Policy",
            yaxis_title="Average Normalized Cost",
            height=420,
        )
        chart_col3.plotly_chart(fig5, width='stretch')

    if {"model", "avg_disruption_score"}.issubset(results_df.columns):
        fig6 = px.bar(
            results_df,
            x="model",
            y="avg_disruption_score",
            title="Disruption Exposure by Policy",
            text="avg_disruption_score",
        )
        fig6.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig6.update_layout(
            xaxis_title="Policy",
            yaxis_title="Average Disruption Score",
            height=420,
        )
        chart_col4.plotly_chart(fig6, width='stretch')

    st.subheader("Key Insight")

    best_reward_model = None
    lowest_cost_model = None
    best_ontime_model = None
    lowest_disruption_model = None

    if {"model", "avg_reward"}.issubset(results_df.columns):
        best_reward_model = results_df.loc[results_df["avg_reward"].idxmax(), "model"]

    if {"model", "avg_cost"}.issubset(results_df.columns):
        lowest_cost_model = results_df.loc[results_df["avg_cost"].idxmin(), "model"]

    if {"model", "avg_on_time_rate"}.issubset(results_df.columns):
        best_ontime_model = results_df.loc[results_df["avg_on_time_rate"].idxmax(), "model"]

    if {"model", "avg_disruption_score"}.issubset(results_df.columns):
        lowest_disruption_model = results_df.loc[
            results_df["avg_disruption_score"].idxmin(), "model"
        ]

    insight_lines = []
    if best_reward_model is not None:
        insight_lines.append(f"- **Best overall reward:** {best_reward_model}")
    if lowest_cost_model is not None:
        insight_lines.append(f"- **Lowest operational cost:** {lowest_cost_model}")
    if best_ontime_model is not None:
        insight_lines.append(f"- **Highest on-time rate:** {best_ontime_model}")
    if lowest_disruption_model is not None:
        insight_lines.append(f"- **Lowest disruption exposure:** {lowest_disruption_model}")

    if insight_lines:
        st.markdown("\n".join(insight_lines))

    st.info(
        "This dashboard shows that routing policies behave differently under uncertainty. "
        "Heuristic routing can preserve strong service levels, while reinforcement learning can adapt "
        "routing choices to optimize broader reward trade-offs and manage disruption exposure."
    )

# -------------------------------------------------------------------
# Research summary
# -------------------------------------------------------------------
st.subheader("Research Summary")
st.markdown(
    """
This dashboard supports a research framework for adaptive supply chain routing under uncertainty.
The system integrates:
- live public freight-related economic indicators,
- live weather disruption signals,
- stored research datasets for reproducible experiments,
- and routing policy evaluation using baseline, heuristic, and reinforcement learning approaches.

The goal is to study how AI-based policies balance cost, delivery reliability, and disruption resilience
under dynamic operating conditions.
"""
)