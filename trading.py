# trading_bot.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.graph_objects as go
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
balance_history = []    # Track balance over time

# Title and description
st.title("📈 Automated Trading Bot Dashboard")
st.markdown("""
Welcome to the trading bot dashboard. This tool analyzes top-performing commodities, forex pairs, and indices to provide trading signals based on advanced strategies.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Asset Class", ["Commodities", "Forex", "Indices"])

# Define tickers based on selection
if section == "Commodities":
    st.header("🌐 Top Commodities")
    tickers = ['GOLD', 'WTI']  # Gold and Crude Oil
    asset_class = 'Commodities'
elif section == "Forex":
    st.header("💱 Top Forex Pairs")
    tickers = ['EURUSD', 'GBPUSD', 'USDJPY']
    asset_class = 'Forex'
elif section == "Indices":
    st.header("📊 Global Indices Overview")
    tickers = ['^GSPC', '^GDAXI', '^N225']  # S&P 500, DAX 40, Nikkei 225
    asset_class = 'Indices'

# Allocate capital per ticker
num_tickers = len(tickers)
capital_per_ticker = balance / num_tickers

# Initialize allocated capital and open positions
for ticker in tickers:
    allocated_capital[ticker] = capital_per_ticker
    open_positions[ticker] = None  # None indicates no open position

# Function to fetch live data from FMP
def fetch_live_data(tickers, asset_class):
    data = {}
    for ticker in tickers:
        try:
            if asset_class == 'Forex':
                # Fetch the latest intraday data (5-minute intervals)
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?apikey={api_key}'
            elif asset_class == 'Commodities':
                # Fetch the latest intraday data for commodities
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?apikey={api_key}'
            elif asset_class == 'Indices':
                # Fetch the latest intraday data for indices
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?apikey={api_key}'
            else:
                # Default to daily data
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?apikey={api_key}'

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
            df = df.sort_index()
            # Filter data for the last 2 days to reduce data size
            df = df[df.index >= (datetime.now() - timedelta(days=2))]
            data[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
    return data

# Functions to compute indicators
def compute_indicators(df):
    df = df.copy()
    df['MA50'] = df['Close'].rolling(window=10).mean()
    df['MA200'] = df['Close'].rolling(window=30).mean()
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
def simulate_trades_live(data):
    global balance
    for ticker in tickers:
        if ticker in data:
            df = compute_indicators(data[ticker])
            df = df.dropna()
            df = generate_signals(df)
            allocated = allocated_capital[ticker]
            position = open_positions[ticker]

            # Only process the most recent data point
            current_time = df.index[-1]
            row = df.iloc[-1]
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
                    # Update balance
                    balance -= allocated
                    balance_history.append({'Time': current_time, 'Balance': balance})
                    st.write(f"Bought {ticker} at {buy_price:.2f} on {current_time}")
            else:
                # Check if 10% profit achieved or sell signal
                if price >= position['Buy_Price'] * 1.10 or signal == -1:
                    sell_price = price
                    profit = (sell_price - position['Buy_Price']) * position['Quantity']
                    balance += allocated + profit
                    balance_history.append({'Time': current_time, 'Balance': balance})
                    trade_history.append({
                        'Ticker': ticker,
                        'Buy_Time': position['Buy_Time'],
                        'Buy_Price': position['Buy_Price'],
                        'Sell_Time': current_time,
                        'Sell_Price': sell_price,
                        'Profit/Loss': profit
                    })
                    st.write(f"Sold {ticker} at {sell_price:.2f} on {current_time} | Profit: ${profit:.2f}")
                    open_positions[ticker] = None
        else:
            st.warning(f"No data available for {ticker}.")
    return trade_history

# Fetch live data
data = fetch_live_data(tickers, asset_class)

if not data:
    st.error("No data fetched. Please check the tickers and API availability.")
    st.stop()

# Simulate trades on live data
trade_history = simulate_trades_live(data)

# Main layout
st.markdown("---")
col1, col2 = st.columns(2)

# Display Account Overview
with col1:
    st.header("💰 Account Overview")
    st.metric("Initial Balance", f"${initial_balance:,.2f}")
    st.metric("Current Balance", f"${balance:,.2f}")
    if trade_history:
        total_profit = sum([trade['Profit/Loss'] for trade in trade_history])
        st.metric("Total Profit/Loss", f"${total_profit:,.2f}")
        num_trades = len(trade_history)
        winning_trades = sum(1 for trade in trade_history if trade['Profit/Loss'] > 0)
        win_rate = (winning_trades / num_trades) * 100
        st.metric("Total Trades", f"{num_trades}")
        st.metric("Winning Percentage", f"{win_rate:.2f}%")
    else:
        st.metric("Total Profit/Loss", "$0.00")
        st.metric("Total Trades", "0")
        st.metric("Winning Percentage", "0.00%")

# Display Account Balance Over Time
with col2:
    st.header("📈 Account Balance Over Time")
    if balance_history:
        balance_df = pd.DataFrame(balance_history)
        balance_df = balance_df.drop_duplicates(subset=['Time'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=balance_df['Time'], y=balance_df['Balance'], mode='lines', name='Balance'))
        fig.update_layout(xaxis_title='Time', yaxis_title='Balance ($)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No account activity to display.")

st.markdown("---")

# Display Trade History
st.header("📝 Trade History")
if trade_history:
    trades_df = pd.DataFrame(trade_history)
    trades_df['Buy_Time'] = trades_df['Buy_Time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_df['Sell_Time'] = trades_df['Sell_Time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_df['Profit/Loss'] = trades_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
    trades_df_display = trades_df[['Ticker', 'Buy_Time', 'Buy_Price', 'Sell_Time', 'Sell_Price', 'Profit/Loss']]
    st.dataframe(trades_df_display.style.format({'Buy_Price': '${:,.2f}', 'Sell_Price': '${:,.2f}'}))
else:
    st.info("No trades executed yet.")

st.markdown("---")

# Display Open Positions
st.header("📌 Current Open Positions")
if any(position is not None for position in open_positions.values()):
    open_positions_list = []
    for ticker, position in open_positions.items():
        if position:
            current_price = data[ticker]['Close'][-1]
            profit_loss = (current_price - position['Buy_Price']) * position['Quantity']
            open_positions_list.append({
                'Ticker': ticker,
                'Buy_Time': position['Buy_Time'].strftime('%Y-%m-%d %H:%M'),
                'Buy_Price': position['Buy_Price'],
                'Current_Price': current_price,
                'Profit/Loss': profit_loss
            })
    open_positions_df = pd.DataFrame(open_positions_list)
    open_positions_df['Profit/Loss'] = open_positions_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(open_positions_df.style.format({'Buy_Price': '${:,.2f}', 'Current_Price': '${:,.2f}'}))
else:
    st.info("No open positions.")

st.markdown("---")

# Display Trade Signals and Prices per Ticker
st.header("🔍 Trade Signals and Price Charts")

for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker])
        df = df.dropna()
        df = generate_signals(df)

        # Find buy and sell points from trade_history
        trades = [trade for trade in trade_history if trade['Ticker'] == ticker]
        position = open_positions[ticker]

        st.subheader(f"{ticker} Price Chart with Trade Signals")

        # Prepare data for plotting
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA50'], line=dict(color='blue', width=1), name='MA50'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA200'], line=dict(color='orange', width=1), name='MA200'
        ))

        # Add Buy/Sell markers
        if trades:
            buy_times = [trade['Buy_Time'] for trade in trades]
            buy_prices = [trade['Buy_Price'] for trade in trades]
            sell_times = [trade['Sell_Time'] for trade in trades]
            sell_prices = [trade['Sell_Price'] for trade in trades]

            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode='markers', marker_symbol='triangle-up', marker_color='green',
                marker_size=12, name='Buy Signal'
            ))
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode='markers', marker_symbol='triangle-down', marker_color='red',
                marker_size=12, name='Sell Signal'
            ))

        # Add current open position marker
        if position:
            fig.add_trace(go.Scatter(
                x=[position['Buy_Time']], y=[position['Buy_Price']], mode='markers',
                marker_symbol='star', marker_color='gold', marker_size=15, name='Open Position'
            ))

        fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)', height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {ticker}.")

# Footer
st.markdown("---")
st.markdown("<center>© 2023 Trading Bot Dashboard | Powered by Streamlit</center>", unsafe_allow_html=True)
