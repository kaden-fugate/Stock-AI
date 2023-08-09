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

dataframe = pdr.get_data_yahoo(STOCK_NAME, start= '2023-02-01', end= '2023-02-02')


