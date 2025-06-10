from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Function to run Exponential Smoothing model
def run_exponential_smoothing(sales_data, forecast_period, seasonal_period):
    model = ExponentialSmoothing(sales_data, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
    forecast = model.forecast(steps=forecast_period)
    return forecast

# Function to run Moving Average model
def run_moving_average(sales_data, forecast_period):
    window_size = 12  # 12-month moving average
    moving_avg = sales_data.rolling(window=window_size).mean()
    forecast = moving_avg[-1]  # Use last moving average value
    return [forecast] * forecast_period

# Function to run Last 12 Months model
def run_last_12_months(sales_data, forecast_period):
    last_value = sales_data[-1]
    return [last_value] * forecast_period

# Shared logic for data filtering and resampling
def prepare_sales_data(data, date_column, sales_column, product_column, selected_product, frequency):
    if selected_product:
        data = data[data[product_column] == selected_product]

    freq_map = {"D": "D", "W": "W", "M": "M"}
    data_resampled = data.resample(freq_map[frequency]).sum()
    sales_data = data_resampled[sales_column]
    return sales_data

# Streamlit application
st.title("Sales Forecasting Application")
st.sidebar.header("Model Selection")

models = [
    "Exponential Smoothing (EST)",
    "Moving Average",
    "Last 12 Months",
]
selected_model = st.sidebar.selectbox("Select a Forecasting Model", models)

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        st.warning("File encoding is not UTF-8. Trying 'latin1'.")
        data = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
    else:
        st.write("Uploaded Dataset Preview:")
        st.dataframe(data.head())

        columns = data.columns.tolist()
        date_column = st.selectbox("Select Date Column", options=columns)
        sales_column = st.selectbox("Select Sales Column", options=columns)
        product_column = st.selectbox("Select Product Column (Optional)", options=columns + ["None"])

        selected_product = None
        if product_column != "None":
            unique_products = data[product_column].unique()
            selected_product = st.selectbox("Select Product to Forecast", options=unique_products)

        # Date parsing
        try:
            if data[date_column].astype(str).str.isdigit().all():
                data[date_column] = pd.to_datetime(data[date_column], format="%Y%m%d", errors="coerce")
            else:
                data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

            if data[date_column].isna().any():
                st.warning("Some rows have invalid dates. These rows will be dropped.")
                data = data.dropna(subset=[date_column])

            data.set_index(date_column, inplace=True)
        except Exception as e:
            st.error(f"Error parsing dates: {str(e)}")
            st.stop()

        forecast_period = st.number_input("Enter Forecast Period", min_value=1, step=1)
        frequency = st.selectbox("Frequency of Forecast", ["D", "W", "M"], index=0)

        freq_map = {"D": "D", "W": "W-SUN", "M": "MS"}  # For date generation
        seasonal_map = {"D": 7, "W": 52, "M": 12}       # For seasonal_periods

        # Run selected model
        if st.button("Run Forecast"):
            try:
                sales_data = prepare_sales_data(data, date_column, sales_column, product_column, selected_product, frequency)

                if sales_data.empty:
                    st.error("Selected data is empty. Check filters or dataset.")
                    st.stop()

                if selected_model == "Exponential Smoothing (EST)":
                    forecast = run_exponential_smoothing(sales_data, forecast_period, seasonal_map[frequency])
                elif selected_model == "Moving Average":
                    forecast = run_moving_average(sales_data, forecast_period)
                else:  # Last 12 Months
                    forecast = run_last_12_months(sales_data, forecast_period)

                last_date = sales_data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_period,
                    freq=freq_map[frequency]
                )

                forecast_df = pd.DataFrame({
                    "Forecast Date": forecast_dates,
                    "Forecast Value": forecast,
                })
                st.write(f"{selected_model} Forecast for next {forecast_period} {frequency}:")
                st.dataframe(forecast_df)

                # Plot
                fig, ax = plt.subplots()
                sales_data.plot(ax=ax, label="Historical Data")
                pd.Series(forecast, index=forecast_dates).plot(ax=ax, label="Forecast", linestyle="--")
                ax.set_title(f"{selected_model} Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
