# Import libraries

import math
import pandas_datareader.data as pdr
import yfinance as yfin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Override pdr due to lack of connection between pdr and yahoo
yfin.pdr_override() 

# Store program vars
STOCK_NAME = 'GOOG'
START_DATE = '2000-01-01'
END_DATE = '2023-07-01'
PREDICTION_LEN = 60

# Load in selected stock over given start and end dates
dataframe = pdr.get_data_yahoo(STOCK_NAME, start= START_DATE, end= END_DATE)

# Store filtered closing price in dataset as a np array
dataset = np.array( dataframe.filter(['Close']) )

# Train on 80% of dataset
training_len = math.ceil(len(dataset) * 0.8)

# Normalize dataset for use in NN
scale = MinMaxScaler(feature_range= (0, 1)) # Set range between 0 and 1
scaled_data = scale.fit_transform(dataset)

# Create data set to make training dataset for NN
training_data = scaled_data[ :training_len, : ]

prev_train = []
next_train = []

# Loop is training_len - PREDICTION_LEN long
for day in range(PREDICTION_LEN, training_len):
    
    # hold a range of days PREDICTION_LEN long
    prev_train.append( training_data[ day - PREDICTION_LEN : day, 0 ] )

    # get next days closing price
    next_train.append( training_data[ day, 0 ] )

# LSTM model expects 3 dimensions
# Need to reshape our training data to fit this model

prev_train, next_train = np.array(prev_train), np.array(next_train)

prev_train = np.reshape(prev_train, (prev_train.shape[0],prev_train.shape[1], 1))