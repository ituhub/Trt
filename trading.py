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

# Initialize Demo Account
initial_balance = 10000  # $10,000
balance = initial_balance
allocated_capital = {}  # Capital allocated per ticker
open_positions = {}     # Track open positions per ticker
trade_history = []      # List to store trade details

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

# Allocate capital per ticker
num_tickers = len(tickers)
capital_per_ticker = balance / num_tickers

# Initialize allocated capital and open positions
for ticker in tickers:
    allocated_capital[ticker] = capital_per_ticker
    open_positions[ticker] = None  # None indicates no open position

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
    return df

# Function to simulate trades
def simulate_trades(data):
    global balance
    for ticker in tickers:
        if ticker in data:
            df = compute_indicators(data[ticker])
            df = df.dropna()
            df = generate_signals(df)
            
            allocated = allocated_capital[ticker]
            position = open_positions[ticker]
            
            for current_time, row in df.iterrows():
                signal = row['Signal']
                price = row['Close']
                
                if position is None:
                    if signal == 1:
                        # Execute Buy
                        quantity = allocated / price
                        buy_price = price
                        position = {
                            'Buy_Time': current_time,
                            'Buy_Price': buy_price,
                            'Quantity': quantity
                        }
                        open_positions[ticker] = position
                        st.write(f"Bought {ticker} at {buy_price:.2f} on {current_time.date()}")
                else:
                    # Check if 10% profit achieved
                    if price >= position['Buy_Price'] * 1.10:
                        sell_price = price
                        profit = (sell_price - position['Buy_Price']) * position['Quantity']
                        balance += profit
                        trade_history.append({
                            'Ticker': ticker,
                            'Buy_Time': position['Buy_Time'],
                            'Buy_Price': position['Buy_Price'],
                            'Sell_Time': current_time,
                            'Sell_Price': sell_price,
                            'Close_Time': current_time,
                            'Close_Price': sell_price,
                            'Profit/Loss': profit
                        })
                        st.write(f"Sold {ticker} at {sell_price:.2f} on {current_time.date()} | Profit: ${profit:.2f}")
                        open_positions[ticker] = None
    return trade_history

# Fetch data
data = fetch_data(tickers, asset_class)

if not data:
    st.error("No data fetched. Please check the tickers and API availability.")
    st.stop()

# Simulate trades
trade_history = simulate_trades(data)

# Display Account Balance
st.sidebar.header("Account Overview")
st.sidebar.write(f"**Initial Balance:** ${initial_balance:,.2f}")
st.sidebar.write(f"**Current Balance:** ${balance:,.2f}")

# Display Trade History
if trade_history:
    st.header("Trade History")
    trades_df = pd.DataFrame(trade_history)
    trades_df['Profit/Loss'] = trades_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(trades_df)
    
    # Calculate Total Profit/Loss
    total_profit = sum([trade['Profit/Loss'] for trade in trade_history])
    st.write(f"**Total Profit/Loss:** ${total_profit:,.2f}")
else:
    st.info("No trades executed yet.")

# Visualization: Account Balance Over Time
# For simplicity, we can plot the final balance
st.header("Account Performance")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(['Initial Balance', 'Current Balance'], [initial_balance, balance], color=['blue', 'green'])
ax.set_ylabel('Balance ($)')
ax.set_title('Account Balance Overview')
st.pyplot(fig)

# Display Buy/Sell Signals and Prices per Ticker
st.header("Trade Signals and Prices")

for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker])
        df = df.dropna()
        df = generate_signals(df)
        
        # Find buy and sell points from trade_history
        trades = [trade for trade in trade_history if trade['Ticker'] == ticker]
        
        if trades:
            for trade in trades:
                st.write(f"**{ticker} Trade:**")
                st.write(f"Buy at **${trade['Buy_Price']:.2f}** on {trade['Buy_Time'].date()}")
                st.write(f"Sell at **${trade['Sell_Price']:.2f}** on {trade['Sell_Time'].date()}")
                st.write(f"Closed at **${trade['Close_Price']:.2f}** on {trade['Close_Time'].date()} | Profit/Loss: {trade['Profit/Loss']}")
        else:
            st.write(f"No trades executed for {ticker}.")
    else:
        st.warning(f"No data available for {ticker}.")

# Display Combined Signals and Final Signals (if needed)
# Since we've simplified signals to 'Buy' or 'Sell', they are already displayed above.

# Optional: Display Data Charts with Trade Signals
st.header("Price Charts with Trade Signals")

for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker])
        df = df.dropna()
        df = generate_signals(df)
        
        trades = [trade for trade in trade_history if trade['Ticker'] == ticker]
        
        if trades:
            buy_times = [trade['Buy_Time'] for trade in trades]
            buy_prices = [trade['Buy_Price'] for trade in trades]
            sell_times = [trade['Sell_Time'] for trade in trades]
            sell_prices = [trade['Sell_Price'] for trade in trades]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df['Close'], label='Close Price', color='blue')
            ax.scatter(buy_times, buy_prices, label='Buy Signal', marker='^', color='green', s=100)
            ax.scatter(sell_times, sell_prices, label='Sell Signal', marker='v', color='red', s=100)
            ax.set_title(f"{ticker} Price Chart with Buy/Sell Signals")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning(f"No data available for {ticker}.")

