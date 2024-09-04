import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

# Load datasets for different cryptocurrencies
btc_df = pd.read_csv("./datasets/BTC-USD.csv")
eth_df = pd.read_csv("./datasets/ETH-USD.csv")
ada_df = pd.read_csv("./datasets/ADA-USD.csv")

# Function to process and prepare data
def prepare_data(df):
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]
    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)
    return new_data

# Prepare data for each cryptocurrency
btc_data = prepare_data(btc_df)
eth_data = prepare_data(eth_df)
ada_data = prepare_data(ada_df)

# Load your pre-trained model
model = load_model("saved_model.h5")

def get_predictions(data, model, scaler):
    dataset = data.values
    train = dataset[0:987, :]
    valid = dataset[987:, :]

    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    inputs = data[len(data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    # Check the shapes of valid and closing_price
    print(f"Shape of valid: {valid.shape}")
    print(f"Shape of closing_price: {closing_price.shape}")
    
    # Make sure valid and closing_price have matching shapes
    valid = pd.DataFrame(valid, columns=['Close'])  # Ensure valid is a DataFrame
    valid['Predictions'] = closing_price[:len(valid)]  # Truncate to match the valid length

    return valid

# Dash Layout
app.layout = html.Div([
    html.H1("Cryptocurrency Price Prediction Dashboard", style={"textAlign": "center"}),

    # Dropdown for selecting currency pair
    dcc.Dropdown(
        id='crypto-dropdown',
        options=[
            {'label': 'BTC-USD', 'value': 'BTC'},
            {'label': 'ETH-USD', 'value': 'ETH'},
            {'label': 'ADA-USD', 'value': 'ADA'}
        ],
        value='BTC',  # default value
        style={"width": "50%", "margin": "auto"}
    ),

    html.Div(id='crypto-graphs')
])


# Callback to update graphs based on selected cryptocurrency
@app.callback(
    Output('crypto-graphs', 'children'),
    [Input('crypto-dropdown', 'value')]
)
def update_crypto_graphs(selected_crypto):
    if selected_crypto == 'BTC':
        data = btc_data
    elif selected_crypto == 'ETH':
        data = eth_data
    else:
        data = ada_data

    valid = get_predictions(data, model, scaler)

    return html.Div([
        html.H2(f"{selected_crypto}-USD Actual vs Predicted Closing Price", style={"textAlign": "center"}),
        
        dcc.Graph(
            id=f'{selected_crypto}-actual',
            figure={
                "data": [
                    go.Scatter(
                        x=data.index,
                        y=data["Close"],
                        mode='lines',
                        name='Actual'
                    ),
                    go.Scatter(
                        x=valid.index,
                        y=valid["Predictions"],
                        mode='lines',
                        name='Predicted'
                    )
                ],
                "layout": go.Layout(
                    title=f"{selected_crypto}-USD Closing Price Prediction",
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Closing Price (USD)'}
                )
            }
        )
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
