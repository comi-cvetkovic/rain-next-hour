import streamlit as st
import pandas as pd

from src.data import geocode_city
from src.predict import predict_rain_next_hour

st.set_page_config(page_title="Will it rain in the next hour?", layout="centered")

st.title("🌧️ Will it rain in the next hour?")
st.write("Dynamic data from Open-Meteo. Model predicts probability of rain in the next hour based on latest hourly conditions.")

with st.sidebar:
    st.header("Location")
    city = st.text_input("City", value="Stockholm")
    country_code = st.text_input("Country code (optional, e.g. SE)", value="SE")
    st.caption("Tip: If geocoding fails, try a more specific city name.")

if st.button("Predict"):
    try:
        loc = geocode_city(city, country_code=country_code.strip() or None)
        st.subheader(f"📍 {loc.name}")
        res = predict_rain_next_hour(loc.latitude, loc.longitude)

        prob = res["prob_rain_next_hour"]
        pred = res["pred_rain_next_hour"]

        st.metric(
            label="Probability of rain in the next hour",
            value=f"{prob*100:.1f}%",
            delta="Rain likely" if pred == 1 else "Rain unlikely",
        )

        st.caption(f"Latest hour used for prediction: {res['time_used']}")

        st.subheader("Inputs used (latest hour)")
        st.json(res["features_used"])

        st.subheader("Recent weather (last 24 hours)")
        recent_df: pd.DataFrame = res["recent_df"][["time", "precipitation", "cloud_cover", "relative_humidity_2m", "temperature_2m", "wind_speed_10m"]]
        st.dataframe(recent_df, use_container_width=True)

        st.subheader("Precipitation history (last 24 hours)")
        chart_df = recent_df.set_index("time")[["precipitation"]]
        st.line_chart(chart_df)

    except Exception as e:
        st.error(str(e))

st.divider()
st.caption("Disclaimer: This is a baseline ML model trained on recent historical data (past days) and intended for demonstration of an end-to-end prediction service.")
