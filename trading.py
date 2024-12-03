# trading_bot.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    st.header("Top Commodities")
    tickers = ['GOLD', 'WTI']  # Gold and Crude Oil
    asset_class = 'Commodities'
elif section == "Forex":
    st.header("Top Forex Pairs")
    tickers = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    asset_class = 'Forex'
elif section == "Indices":
    st.header("Global Indices Overview")
    tickers = ['^GSPC', '^GDAXI', '^N225']  # S&P 500, DAX 40, Nikkei 225
    asset_class = 'Indices'

# Function to fetch data from FMP
def fetch_data(tickers, asset_class):
    data = {}
    for ticker in tickers:
        try:
            if asset_class == 'Forex':
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{ticker}?apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()
                df = pd.DataFrame(data_json)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.rename(columns={
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low'
                })
            else:
                url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()
                df = pd.DataFrame(data_json['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.rename(columns={
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low'
                })
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
    lowest_low = df['Low'].rolling(window=period).min()
    highest_high = df['High'].rolling(window=period).max()
    stochastic = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    return stochastic

# Function to generate combined signal
def generate_signals(df):
    df['Signal'] = 0
    df['Signal'] = np.where(df['MA50'] > df['MA200'], 1, -1)
    df['Buy_Price'] = np.nan
    df['Sell_Price'] = np.nan

    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and df['Signal'].iloc[i - 1] == -1:
            df['Buy_Price'].iloc[i] = df['Close'].iloc[i]
        elif df['Signal'].iloc[i] == -1 and df['Signal'].iloc[i - 1] == 1:
            df['Sell_Price'].iloc[i] = df['Close'].iloc[i]
    return df

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
        df = generate_signals(df)

        # Display latest signals and prices
        latest_signal = 'Buy' if df['Signal'].iloc[-1] == 1 else 'Sell'
        latest_price = df['Close'].iloc[-1]
        st.write(f"Latest Signal for {ticker}: **{latest_signal}**")
        st.write(f"Latest Close Price for {ticker}: **{latest_price:.2f}**")

        # Display Buy and Sell Prices
        recent_trades = df[['Buy_Price', 'Sell_Price']].dropna(how='all').tail(5)
        st.write("Recent Trade Signals:")
        st.write(recent_trades)

        # Plotting the Closing Price with Buy/Sell Signals
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax.scatter(df.index, df['Buy_Price'], label='Buy Signal', marker='^', color='green')
        ax.scatter(df.index, df['Sell_Price'], label='Sell Signal', marker='v', color='red')
        ax.set_title(f"{ticker} Price Chart with Buy/Sell Signals")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {ticker}.")
