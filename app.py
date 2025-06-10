import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Time Series Forecasting App")

st.write("Upload a CSV file and forecast future values using Exponential Smoothing or SARIMA.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Select columns
    with st.form("forecast_form"):
        date_col = st.selectbox("Select Date Column", df.columns)
        value_col = st.selectbox("Select Value Column (to forecast)", df.columns)
        forecast_period = st.number_input("Number of periods to forecast", min_value=1, value=12)
        model_choice = st.radio("Choose Forecasting Model", ("Exponential Smoothing", "SARIMA"))
        submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)

        ts_data = df[value_col]

        fig, ax = plt.subplots(figsize=(10, 5))

        if model_choice == "Exponential Smoothing":
            model = ExponentialSmoothing(ts_data, seasonal_periods=12, trend='add', seasonal='add')
            model_fit = model.fit()
            forecast = model_fit.forecast(forecast_period)

            ax.plot(ts_data, label="Original Data")
            ax.plot(model_fit.fittedvalues, label="Fitted")
            ax.plot(forecast, label="Forecast")
            ax.set_title("Triple Exponential Smoothing Forecast")
        
        else:  # SARIMA
            model = SARIMAX(ts_data, order=(0, 1, 1), seasonal_order=(2, 1, 1, 12))
            model_fit = model.fit()
            forecast = model_fit.predict(start=len(ts_data), end=(len(ts_data) + forecast_period - 1))

            ax.plot(ts_data, label="Original Data")
            ax.plot(forecast, label="SARIMA Forecast")
            ax.set_title("SARIMA Forecast")

        ax.legend()
        st.pyplot(fig)
