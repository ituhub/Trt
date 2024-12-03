# trading_bot.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Automated Trading Bot Dashboard", layout="wide")

# Retrieve API key
api_key = os.getenv('FMP_API_KEY')
if not api_key:
    st.error("FMP API key not found. Please set it as an environment variable 'FMP_API_KEY'.")
    st.stop()

# Title and description
st.title("Automated Trading Bot Dashboard")
st.markdown("""
Welcome to the trading bot dashboard. This tool analyzes top-performing commodities, forex pairs, and indices to provide trading signals based on advanced strategies.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", ["Commodities", "Forex", "Indices"])

# Define tickers based on selection
if section == "Commodities":
    st.header("Top 5 Commodities")
    tickers = ['GOLD', 'SILVER', 'WTI', 'NATURAL_GAS', 'COPPER']
    asset_class = 'Commodities'
elif section == "Forex":
    st.header("Top 5 Forex Pairs")
    tickers = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    asset_class = 'Forex'
elif section == "Indices":
    st.header("Global Indices Overview")
    tickers = ['^GSPC', '^DJI', '^IXIC', '^N225', '^FTSE']  # S&P 500, Dow Jones, Nasdaq, Nikkei 225, FTSE 100
    asset_class = 'Indices'

# Function to fetch data from FMP
def fetch_data(tickers, asset_class):
    data = {}
    for ticker in tickers:
        try:
            if asset_class == 'Forex':
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{ticker}?apikey={api_key}'
            else:
                url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}&serietype=line'
            response = requests.get(url)
            response.raise_for_status()
            data_json = response.json()
            if asset_class == 'Forex':
                df = pd.DataFrame(data_json)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.rename(columns={'close': 'Close'})
            else:
                df = pd.DataFrame(data_json['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.rename(columns={'close': 'Close'})
            df = df.sort_index()
            data[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
    return data

# Functions to compute indicators
def compute_indicators(df):
    df = df.copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    df['Stochastic'] = compute_stochastic(df)
    return df

def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

def compute_stochastic(df, period=14):
    lowest_low = df['Close'].rolling(window=period).min()
    highest_high = df['Close'].rolling(window=period).max()
    stochastic = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    return stochastic

# Function to generate combined signal
def generate_signal(df):
    df['Signal'] = 0
    df['Signal'] = np.where(df['MA50'] > df['MA200'], 1, -1)
    return df

# Function to calculate model accuracy
def calculate_accuracy(df):
    df = df.dropna()
    X = df[['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal', 'Stochastic']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Fetch data
data = fetch_data(tickers, asset_class)

if not data:
    st.error("No data fetched. Please check the tickers and API availability.")
    st.stop()

# Process data and display
for ticker in tickers:
    if ticker in data:
        st.subheader(f"{ticker} Analysis")
        df = compute_indicators(data[ticker])
        df = df.dropna()
        if df.empty:
            st.warning(f"Not enough data for {ticker} to perform analysis.")
            continue
        df = generate_signal(df)
        accuracy = calculate_accuracy(df)
        st.write(f"Model Accuracy for {ticker}: {accuracy:.2%}")
        st.write(f"Latest Combined Signal for {ticker}: {'Buy' if df['Signal'].iloc[-1] == 1 else 'Sell'}")
        st.line_chart(df['Close'])
    else:
        st.warning(f"No data available for {ticker}.")
