# LSTM model expects 3 dimensions
# Need to reshape our training data to fit this model
prev_train, next_train = np.array(prev_train), np.array(next_train)
prev_train = np.reshape(prev_train, (prev_train.shape[0], prev_train.shape[1], 1))

# Now that we reshaped our data, we can build our LSTM model
LSTM_model = Sequential()

# Add LSTM layers of NN
LSTM_model.add( LSTM(100, return_sequences= True, input_shape= (prev_train.shape[1], 1)) )
LSTM_model.add( LSTM(100, return_sequences= False, ) )

# Add regular densly connected layers of NN
LSTM_model.add( Dense(50) )
LSTM_model.add( Dense(1) )