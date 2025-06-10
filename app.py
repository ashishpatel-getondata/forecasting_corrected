import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Time Series Forecasting App")

st.markdown(
    """
    Upload your time series data in CSV format and explore basic time series techniques:

    - **Exponential Smoothing**: Forecast future values using triple exponential smoothing.
    - **SARIMA**: Generate forecasts with Seasonal ARIMA modeling.
    - **Seasonal Decomposition**: Visualize the underlying trend, seasonal, and residual components of the data.

    Select the appropriate columns and method, then specify the forecast horizon if applicable.
    """
)

# File upload
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Select columns
    with st.form("forecast_form"):
        date_col = st.selectbox("Select Date Column", df.columns)
        value_col = st.selectbox("Select Value Column (to forecast)", df.columns)
        model_choice = st.radio("Choose Forecasting Model", ("Exponential Smoothing", "SARIMA", "Seasonal Decomposition"))
        forecast_period = st.number_input("Number of periods to forecast", min_value=1, value=12)

        submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)

        ts_data = df[value_col]

        if model_choice == "Exponential Smoothing":
            fig, ax = plt.subplots(figsize=(10, 5))
            model = ExponentialSmoothing(ts_data, seasonal_periods=12, trend='add', seasonal='add')
            model_fit = model.fit()
            forecast = model_fit.forecast(forecast_period)

            ax.plot(ts_data, label="Original Data")
            ax.plot(model_fit.fittedvalues, label="Fitted")
            ax.plot(forecast, label="Forecast")
            ax.set_title("Triple Exponential Smoothing Forecast")
            ax.legend()
            st.pyplot(fig)

            forecast_df = forecast.reset_index()
            forecast_df.columns = ['date', 'forecast']

            forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
            forecast_df['forecast'] = forecast_df['forecast'].round(2)
            
            st.subheader("Forecasted Data")
            st.dataframe(forecast_df)

        elif model_choice == "SARIMA":
            fig, ax = plt.subplots(figsize=(10, 5))
            model = SARIMAX(ts_data, order=(0, 1, 1), seasonal_order=(2, 1, 1, 12))
            model_fit = model.fit()
            forecast = model_fit.predict(start=len(ts_data), end=(len(ts_data) + forecast_period - 1))

            ax.plot(ts_data, label="Original Data")
            ax.plot(forecast, label="SARIMA Forecast")
            ax.set_title("SARIMA Forecast")
            ax.legend()
            st.pyplot(fig)

            forecast_df = forecast.reset_index()
            forecast_df.columns = ['date', 'forecast']

            forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
            forecast_df['forecast'] = forecast_df['forecast'].round(2)
            
            st.subheader("Forecasted Data")
            st.dataframe(forecast_df)

        elif model_choice == "Seasonal Decomposition":
            result = seasonal_decompose(ts_data, model='multiplicative', period=12)

            fig2, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            result.observed.plot(ax=axes[0], legend=False)
            axes[0].set_ylabel('Observed')
            result.trend.plot(ax=axes[1], legend=False)
            axes[1].set_ylabel('Trend')
            result.seasonal.plot(ax=axes[2], legend=False)
            axes[2].set_ylabel('Seasonal')
            result.resid.plot(ax=axes[3], legend=False)
            axes[3].set_ylabel('Residual')
            plt.tight_layout()

            st.subheader("Seasonal Decomposition (Multiplicative Model)")
            st.pyplot(fig2)
