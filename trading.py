# trading_bot.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Automated Trading Bot Dashboard", layout="wide")

# Retrieve API key
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
if not api_key:
    st.error("Alpha Vantage API key not found. Please set it as an environment variable 'ALPHA_VANTAGE_API_KEY'.")
    st.stop()

# Title and description
st.title("Automated Trading Bot Dashboard")
st.markdown("""
Welcome to the trading bot dashboard. This tool analyzes top-performing commodities, forex pairs, cryptocurrencies, and indices to provide trading signals based on advanced strategies.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", ["Commodities", "Forex", "Cryptocurrencies", "Indices"])

# Define tickers based on selection
if section == "Commodities":
    st.header("Top 5 Profit-Making Commodities")
    tickers = ['XAUUSD', 'XAGUSD', 'WTI', 'NG', 'HG']  # Gold, Silver, Crude Oil, Natural Gas, Copper
elif section == "Forex":
    st.header("Top 5 Profit-Making Forex Pairs")
    tickers = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
elif section == "Cryptocurrencies":
    st.header("Top 5 Profit-Making Cryptocurrencies")
    tickers = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA']
elif section == "Indices":
    st.header("Global Indices Overview")
    tickers = ['SPX', 'DAX', 'N225', '000001.SS', 'STI', 'AXJO']  # S&P 500, DAX, Nikkei 225, SSE Composite, Straits Times, ASX 200

# Function to fetch data
def fetch_data(tickers, asset_class):
    data = {}
    if asset_class == 'Forex':
        fx = ForeignExchange(key=api_key, output_format='pandas')
        for ticker in tickers:
            from_symbol, to_symbol = ticker[:3], ticker[3:]
            try:
                df, _ = fx.get_currency_exchange_daily(
                    from_symbol=from_symbol,
                    to_symbol=to_symbol,
                    outputsize='full'
                )
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                data[ticker] = df
            except Exception as e:
                st.warning(f"Failed to fetch data for {ticker}: {e}")
    elif asset_class == 'Cryptocurrencies':
        cc = CryptoCurrencies(key=api_key, output_format='pandas')
        for ticker in tickers:
            try:
                df, _ = cc.get_digital_currency_daily(
                    symbol=ticker,
                    market='USD'
                )
                df = df.rename(columns={
                    '1a. open (USD)': 'Open',
                    '2a. high (USD)': 'High',
                    '3a. low (USD)': 'Low',
                    '4a. close (USD)': 'Close',
                    '5. volume': 'Volume',
                    '6. market cap (USD)': 'Market Cap'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                data[ticker] = df
            except Exception as e:
                st.warning(f"Failed to fetch data for {ticker}: {e}")
    else:
        ts = TimeSeries(key=api_key, output_format='pandas')
        for ticker in tickers:
            try:
                df, _ = ts.get_daily(
                    symbol=ticker,
                    outputsize='full'
                )
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                data[ticker] = df
            except Exception as e:
                st.warning(f"Failed to fetch data for {ticker}: {e}")
    return data

# Functions to compute indicators
def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
    avg_loss = loss.ewm(com=(period - 1), min_periods=period).mean()
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

# Function to apply strategies
def apply_strategies(data):
    strategies_applied = {}
    for ticker, df in data.items():
        df = df.copy()
        # Ensure there are enough data points
        if df.shape[0] < 200:
            st.warning(f"Not enough data for {ticker} to perform analysis.")
            continue
        # Moving Average Crossover
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        # Relative Strength Index (RSI)
        df['RSI'] = compute_RSI(df['Close'])
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        # MACD
        df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
        # Stochastic Oscillator
        df['Stochastic'] = compute_stochastic(df)
        strategies_applied[ticker] = df
    return strategies_applied

# Function to generate signals
def generate_signals(strategies):
    signals = {}
    for ticker, df in strategies.items():
        signal = pd.DataFrame(index=df.index)
        signal['Price'] = df['Close']
        signal['Signal'] = 0

        # Strategy 1: Moving Average Crossover
        signal['MA_Signal'] = 0
        signal['MA_Signal'][df['MA50'] > df['MA200']] = 1
        signal['MA_Signal'][df['MA50'] < df['MA200']] = -1

        # Strategy 2: RSI
        signal['RSI_Signal'] = 0
        signal['RSI_Signal'][df['RSI'] < 30] = 1
        signal['RSI_Signal'][df['RSI'] > 70] = -1

        # Strategy 3: Bollinger Bands
        signal['BB_Signal'] = 0
        signal['BB_Signal'][df['Close'] < df['BB_Lower']] = 1
        signal['BB_Signal'][df['Close'] > df['BB_Upper']] = -1

        # Strategy 4: MACD
        signal['MACD_Signal'] = 0
        signal['MACD_Signal'][df['MACD'] > df['MACD_Signal']] = 1
        signal['MACD_Signal'][df['MACD'] < df['MACD_Signal']] = -1

        # Strategy 5: Stochastic Oscillator
        signal['Stochastic_Signal'] = 0
        signal['Stochastic_Signal'][df['Stochastic'] < 20] = 1
        signal['Stochastic_Signal'][df['Stochastic'] > 80] = -1

        # Combine signals
        signal['Combined_Signal'] = signal[['MA_Signal', 'RSI_Signal', 'BB_Signal', 'MACD_Signal', 'Stochastic_Signal']].sum(axis=1)
        signal['Final_Signal'] = 0
        signal['Final_Signal'][signal['Combined_Signal'] > 0] = 1
        signal['Final_Signal'][signal['Combined_Signal'] < 0] = -1

        signals[ticker] = signal
    return signals

# Functions for visualization
def plot_data(df, ticker):
    st.line_chart(df[['Close', 'MA50', 'MA200']], height=250)

def plot_signals(signal_df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal_df.index, signal_df['Price'], label='Price', color='blue')
    buy_signals = signal_df[signal_df['Final_Signal'] == 1]
    sell_signals = signal_df[signal_df['Final_Signal'] == -1]
    ax.scatter(buy_signals.index, buy_signals['Price'], label='Buy Signal', marker='^', color='green')
    ax.scatter(sell_signals.index, sell_signals['Price'], label='Sell Signal', marker='v', color='red')
    ax.legend()
    st.pyplot(fig)

# Fetch data
data = fetch_data(tickers, section)

if not data:
    st.error("No data fetched. Please check the tickers and API availability.")
    st.stop()

# Apply strategies
strategies_applied = apply_strategies(data)

if not strategies_applied:
    st.error("Not enough data to apply strategies. Please select a different asset class or wait for more data.")
    st.stop()

# Generate signals
signals = generate_signals(strategies_applied)

# Display data and signals
for ticker in tickers:
    if ticker in strategies_applied:
        st.subheader(f"{ticker} Analysis")
        plot_data(strategies_applied[ticker], ticker)
        plot_signals(signals[ticker], ticker)
        st.write(signals[ticker].tail())
    else:
        st.warning(f"Not enough data for {ticker} to perform analysis.")
