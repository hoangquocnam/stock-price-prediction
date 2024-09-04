import dash
from dash import html
from dash import dcc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

cryptos = ['ADA', 'BTC', 'ETH']

# Function to load and prepare data
def load_data(crypto):
    data = pd.read_csv(f'./data/processed/{crypto}_processed.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Function to predict prices
def predict_prices(crypto):
    data = load_data(crypto)
    scaler = MinMaxScaler(feature_range=(0, 1))
    final_data = data[['Close']].values
    scaled_data = scaler.fit_transform(final_data)

    train_size = int(len(scaled_data) * 0.7)
    valid_data = scaled_data[train_size:]

    x_valid, y_valid = [], []
    for i in range(60, len(valid_data)):
        x_valid.append(valid_data[i-60:i, 0])
        y_valid.append(final_data[train_size + i, 0])

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    model = load_model(f'./models/saved_models/{crypto}_model.h5')
    predictions = model.predict(x_valid)
    predictions = scaler.inverse_transform(predictions).flatten()
    return data.index[train_size+60:], y_valid, predictions

app.layout = html.Div([
    html.H1("Stock Price Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='Stock Price Prediction', children=[
            html.Div([
                # html.H2("ADA Price Prediction", style={"textAlign": "center"}),
                dcc.Graph(id='ada-prediction'),
                # html.H2("BTC Price Prediction", style={"textAlign": "center"}),
                dcc.Graph(id='btc-prediction'),
                # html.H2("ETH Price Prediction", style={"textAlign": "center"}),
                dcc.Graph(id='eth-prediction')
            ])
        ]),

        dcc.Tab(label='Stock Price Analytics', children=[
            html.Div([
                html.H1("Stock Price Analytics", style={'textAlign': 'center'}),
                dcc.Dropdown(id='crypto-dropdown',
                             options=[{'label': crypto, 'value': crypto} for crypto in cryptos],
                             value='BTC', style={"width": "50%", "margin": "auto"}),
                dcc.Graph(id='high-low-chart'),
                dcc.Graph(id='open-close-chart')
            ])
        ])
    ])
])

@app.callback(
    [Output('ada-prediction', 'figure'),
     Output('btc-prediction', 'figure'),
     Output('eth-prediction', 'figure')],
    [Input('tabs', 'value')]
)
def update_predictions(_):
    figures = []
    for crypto in cryptos:
        dates, actual, predicted = predict_prices(crypto)
        fig = {
            "data": [
                go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(color='orange')),
                go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted', line=dict(color='blue'))
            ],
            "layout": go.Layout(title=f'{crypto} Price Prediction', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
        }
        figures.append(fig)
    return figures

@app.callback(
    [Output('high-low-chart', 'figure'),
     Output('open-close-chart', 'figure')],
    [Input('crypto-dropdown', 'value')]
)
def update_analytics(selected_crypto):
    data = load_data(selected_crypto)

    high_low_fig = {
        "data": [
            go.Scatter(x=data.index, y=data['High'], mode='lines', name='High', line=dict(color='green')),
            go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low', line=dict(color='red'))
        ],
        "layout": go.Layout(title=f'{selected_crypto} High vs Low Prices', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
    }

    open_close_fig = {
        "data": [
            go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open', line=dict(color='purple')),
            go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='black'))
        ],
        "layout": go.Layout(title=f'{selected_crypto} Open vs Close Prices', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
    }

    return high_low_fig, open_close_fig

if __name__ == '__main__':
    app.run_server(debug=True)