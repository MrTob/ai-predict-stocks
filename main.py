import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
#from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, LSTM

# load the data
company = 'FB'
start = dt.datetime(2010,1,1)
end = dt.datetime(2020,1,1)
data = pdr.DataReader(company,'yahoo',start,end)


