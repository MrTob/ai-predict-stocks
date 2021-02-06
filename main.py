import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


company = 'BNTX'
data = pdr.DataReader(company, 'yahoo', dt.datetime(2010,1,1), dt.datetime(2020,12,31))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
pred_days = 70
x_train=[]
y_train =[]
for x in range(pred_days, len(scaled_data)):
    x_train.append(scaled_data[x - pred_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

# load the data
model = Sequential()
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train, epochs=25, batch_size=32)

# test model

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()
test_data = pdr.DataReader(company,'yahoo',test_start,test_end)
actual_prices = test_data['Close'].values

total_test = pd.concat((data['Close'],test_data['Close']),axis=0)
model_inputs = total_test[len(total_test)- len(test_data)-pred_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# do pred on test data

x_test =[]

for x in range(pred_days, len(model_inputs)):
    x_test.append(model_inputs[x-pred_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


pred_prices = model.predict(x_test)
pred_prices = scaler.inverse_transform(pred_prices)

# plot test
plt.plot(actual_prices, color="red", label=f"Actual {company} price")
plt.plot(pred_prices, color="green", label=f"Predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel('Time')
plt.ylabel(f"{company} share price")
plt.legend()
plt.show()
real_data = [model_inputs[len(model_inputs)+1 -pred_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Pred:{prediction}")