# Import libraries

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from df import dataframe
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
from keras.losses import Huber


SLIDING_WINDOW_LEN = 45

# Load in selected stock over given start and end dates
DF = dataframe()
df = DF.get_data()

# Store filtered closing price in dataset as a np array
dataset = np.array( data_frame.filter(['Close']) )

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
LSTM_model.fit(prev_train, next_train, batch_size= 3, epochs= 2)

# Make dataset for testing
test_data = scaled_data[ training_len - SLIDING_WINDOW_LEN : , : ]

# Make arrays for holding data to test model on (prev_test) 
# and actual values the model is attempting to predict (actual_vals)
prev_test = []
actual_vals = dataset[ training_len : , : ]

# Fill test dataset with last SLIDING_WINDOW_LEN vals
for day in range(SLIDING_WINDOW_LEN, len(test_data)):
    prev_test.append( test_data[day - SLIDING_WINDOW_LEN : day, 0] )

# Convert to np array to be used in LSTM model
prev_test = np.array(prev_test)

# Reshape test dataset to fit into LSTM model
prev_test = np.reshape(prev_test, (prev_test.shape[0], prev_test.shape[1], 1))

# Run test dataset on model to evaluate performance
predicted = LSTM_model.predict(prev_test)
predicted = scale.inverse_transform(predicted)

# Tests to determine effectiveness of model
mean_squared_error = np.sqrt( np.mean( (predicted - actual_vals) ** 2 ) )
print( "Root Mean Squared Error: " + str(mean_squared_error) )

future_days = 365
new_input = scale.transform(actual_vals)
new_input = new_input[-SLIDING_WINDOW_LEN:]
print("new_input:" + str(new_input))
predicted_future = []

for day in range(future_days):

    # Reshape and normalize input
    normalized_input = np.reshape(new_input, (1, SLIDING_WINDOW_LEN, 1))

    future_price = LSTM_model.predict(normalized_input)
    future_price_denorm = scale.inverse_transform(future_price)

    # Append to list of future prices
    predicted_future.append(future_price_denorm[0, 0])
    
    # Update new_input to include prediction
    new_input = np.append(new_input[1:], future_price)

print(predicted_future)
# Data to be plotted
data = dataframe.filter(['Close'])
trained_on = data[ : training_len ]
actual = data[ training_len : ]

# Plot data on graph
plt.figure( figsize= (16, 8))
plt.title("Model Predictions")
plt.xlabel("Date")
plt.ylabel("Closing Price [USD]")

plt.plot(dataframe.index[:training_len], trained_on, label='Trained On')
plt.plot(dataframe.index[training_len:], actual, label='Actual')
plt.plot(dataframe.index[training_len:], predicted, label='Predicted')

future_dates = pd.date_range(start=dataframe.index[-1], periods=future_days + 1)[1:]
plt.plot(future_dates, predicted_future, label='Predicted Future')

plt.legend()
plt.show()

LSTM_model.save('my_model.h5')