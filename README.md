# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
Date: 

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error

#### Set seed for reproducibility
np.random.seed(0)

##### Load and preprocess data
data = pd.read_csv('/content/apple_stock.csv')

##### Assuming your data has 'Date' and 'Close' columns, adjust if necessary
data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime
data = data.sort_values(by='Date')  # Sort by date
data.set_index('Date', inplace=True)
data.dropna(inplace=True)

##### Plot the stock price data
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Stock Price')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
plt.title('Apple Stock Price Over Time')
plt.show()

##### Split into train and test data (80% train, 20% test)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
y_train = train_data['Close']
y_test = test_data['Close']

##### Compute and plot ACF for the first 3 lags
plt.figure(figsize=(12, 6))
plot_acf(data['Close'], lags=35)
plt.title('ACF of Apple Stock Price (First 35 Lags)')
plt.show()

##### Fit an autoregressive model (AR)
lag_order = 35  # You can adjust this based on ACF plot insights
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()

##### Predictions
y_pred = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

##### Compute MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
variance = np.var(y_test)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Variance of Testing Data: {variance:.2f}')


### OUTPUT:

![image](https://github.com/user-attachments/assets/f6c1642e-a9d0-431d-a5b8-097c77f0b2d0)


### RESULT:
        Thus we have successfully implemented the auto correlation function in python.
