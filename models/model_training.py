import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_and_save_model(crypto):
    # Load raw data
    data = pd.read_csv(f'./data/processed/{crypto}_processed.csv')
    
    # Process data
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    final_data = data.values
    scaled_data = scaler.fit_transform(final_data)

    # Split data into training and validation sets
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    # valid_data = scaled_data[train_size:]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create and train LSTM model
    model = create_lstm_model(x_train, y_train)
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Save trained model
    model.save(f'models/saved_models/{crypto}_model.h5')

if __name__ == "__main__":
    cryptos = ['ADA', 'BTC', 'ETH']
    for crypto in cryptos:
        train_and_save_model(crypto)
