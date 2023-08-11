# Import libraries

import math
import pandas_datareader.data as pdr
import yfinance as yfin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from df import df
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
from keras.losses import Huber

SLIDING_WINDOW_LEN = 365

# Load in selected stock over given start and end dates
DF = df()
dataframe = DF.get_data()

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

# Loop is training_len - SLIDING_WINDOW_LEN long
for day in range(SLIDING_WINDOW_LEN, training_len):
    
    # hold a range of days SLIDING_WINDOW_LEN long
    prev_train.append( training_data[ day - SLIDING_WINDOW_LEN : day, 0 ] )

    # get next days closing price
    next_train.append( training_data[ day, 0 ] )

# LSTM model expects 3 dimensions
# Need to reshape our training data to fit this model
prev_train, next_train = np.array(prev_train), np.array(next_train)
prev_train = np.reshape(prev_train, (prev_train.shape[0], prev_train.shape[1], 1))

# Now that we reshaped our data, we can build our LSTM model
LSTM_model = Sequential()

# Add LSTM layers of NN
LSTM_model.add( LSTM(100, return_sequences= True, input_shape= (prev_train.shape[1], 1)) )
LSTM_model.add( LSTM(200, return_sequences= True) )
LSTM_model.add( LSTM(400, return_sequences= False, ) )

# Add regular densly connected layers of NN
LSTM_model.add( Dense(400) )
LSTM_model.add( Dense(200) )
LSTM_model.add( Dense(100) )
LSTM_model.add( Dense(1) )

# Compile the model with the adam optimizer
# Measure loss using MSQE
LSTM_model.compile(optimizer= 'adam', loss=Huber(delta=1.0))

# Train the model
LSTM_model.fit(prev_train, next_train, batch_size= 10, epochs= 5)

LSTM_model.save('Stock-LSTM.keras')