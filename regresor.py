import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the sales data
file_path = 'sales_data.csv'
sales_data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%Y-%m-%d')

# Group by Date to ensure there are no duplicates, summing the revenue
daily_revenue = sales_data.groupby('Date').agg({'Revenue': 'sum'}).reset_index()

# Ensure the 'Date' column is set as the index and the frequency is set to daily
daily_revenue = daily_revenue.set_index('Date').asfreq('D', method='pad')

# Aggregate the data by month to calculate total monthly revenue
monthly_revenue = daily_revenue.resample('M').sum().reset_index()

# Set the date as the index for the time series model
monthly_revenue.set_index('Date', inplace=True)

# Split the data into training and testing sets
train = monthly_revenue.iloc[:-12]  # Use all but the last 12 months for training
test = monthly_revenue.iloc[-12:]    # Use the last 12 months for testing

# Define and fit the ARIMA model on the training set
model = ARIMA(train['Revenue'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast on the test set period
forecast = model_fit.forecast(steps=len(test))

# Calculate the accuracy metrics
mse = mean_squared_error(test['Revenue'], forecast)
rmse = np.sqrt(mse)

# Print accuracy metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

def forecast_revenue(input_year, input_month):
    # Define the input date
    input_date = pd.Timestamp(f"{input_year}-{input_month:02d}-01")

    # Calculate the number of months to forecast
    last_date = monthly_revenue.index[-1]
    months_ahead = (input_date.year - last_date.year) * 12 + (input_date.month - last_date.month)

    # Forecast the revenue
    forecast = model_fit.forecast(steps=months_ahead)

    # Get the predicted value for the requested month
    predicted_revenue = forecast.iloc[-1]

    # Print the result
    print(f"Predicted Revenue for {input_date.strftime('%B %Y')}: {predicted_revenue:.2f}")

    # Plot the forecasted revenue
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_revenue.index, monthly_revenue['Revenue'], label='Actual Revenue')
    future_dates = [last_date + DateOffset(months=i) for i in range(1, months_ahead + 1)]
    plt.plot(future_dates, forecast, label='Forecasted Revenue', color='red')
    plt.axvline(x=input_date, color='green', linestyle='--', label=f'Forecast for {input_date.strftime("%B %Y")}')
    plt.title('Monthly Revenue Prediction')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.show()

# Example: Forecast for August 2024
while True:
    input_month = int(input("Nhap thang: "))
    if 1 <= input_month <= 12:
        forecast_revenue(2024, input_month)
    else:
        print("Please input again!")

    # The following block of code seems to be misplaced and is likely meant for some kind of GUI or event loop handling.
    # If you have such a requirement, you might want to handle it differently.
    if 0xFF == ord('q'):
        break
