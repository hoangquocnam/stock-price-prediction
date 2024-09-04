import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def evaluate_model(crypto):
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
    # train_data = scaled_data[:train_size]
    valid_data = scaled_data[train_size:]


    x_valid, y_valid = [], []
    for i in range(60, len(valid_data)):
        x_valid.append(valid_data[i-60:i, 0])
        y_valid.append(final_data[train_size + i, 0])  

    x_valid, y_valid = np.array(x_valid), np.array(y_valid)
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    # Load model
    model = load_model(f'models/saved_models/{crypto}_model.h5')

    # Make predictions
    predictions = model.predict(x_valid)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    rms = np.sqrt(np.mean(np.power((predictions - y_valid.reshape(-1, 1)), 2)))
    print(f'Root Mean Square Error for {crypto}: {rms}')

if __name__ == "__main__":
    for crypto in ['ADA', 'BTC', 'ETH']:
        evaluate_model(crypto)
