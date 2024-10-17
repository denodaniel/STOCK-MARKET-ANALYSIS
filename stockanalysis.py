import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM # type: ignore


# COMPANY NAME FOR DATA LOADING
company = 'GOOG'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = yf.download(company, start=start, end=end)

# Debug print to inspect data
print(data)

# PREPARE THE DATA
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])  # Corrected slicing
    y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Debug print to inspect shapes
print(x_train.shape)
print(y_train.shape)

# MODEL BUILDING PROCESS
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # This line predicts the next price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# TEST THE MODEL ACCURACY
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(company, start=test_start, end=test_end)

# Debug print to inspect test data
print(test_data)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_input)

# Debug print to inspect model input
print(model_input)

# MAKING PREDICTIONS
x_test = []
for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x - prediction_days:x, 0])  #Corrected slicing

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Debug print to inspect shapes
print(x_test.shape)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Debug print to inspect predicted prices
print(predicted_prices)

# PLOT THE GRAPH
plt.plot(actual_prices, color="magenta", label=f"Actual {company} price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()
