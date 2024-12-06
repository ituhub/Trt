import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
from datetime import datetime, timedelta

# Constants for symbols (for use elsewhere in your program)
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "LTC-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI", "000300.SS"]

# List of API endpoints with placeholders removed
api_endpoints = {
    "cryptocurrencies": "https://financialmodelingprep.com/api/v3/symbol/available-cryptocurrencies",
    "forex": "https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs",
    "indexes": "https://financialmodelingprep.com/api/v3/symbol/available-indexes",
    "commodities": "https://financialmodelingprep.com/api/v3/symbol/available-commodities"
}

# Function to fetch data from each endpoint
def fetch_data(endpoints):
    data = {}
    api_key = os.getenv("FMP_API_KEY")  # Fetch the API key from environment variables
    
    if not api_key:
        raise ValueError("API key not found in environment variables. Set 'FMP_API_KEY'.")

    for key, url in endpoints.items():
        try:
            # Construct the full URL with the API key
            full_url = f"{url}?apikey={api_key}"
            
            # Send the HTTP request
            response = requests.get(full_url, headers={"User-Agent": "MyApp/1.0"})
            response.raise_for_status()  # Raise an error for bad HTTP responses
            
            # Parse JSON and store in the dictionary
            data[key] = response.json()
        
        except requests.RequestException as e:
            print(f"Error fetching data from {key}: {e}")
            data[key] = None  # Store None for failed requests

    return data

# Fetch data from all endpoints
if __name__ == "__main__":
    fetched_data = fetch_data(api_endpoints)
    for key, value in fetched_data.items():
        print(f"\n{key.upper()}:\n{value}")

# Set page configuration
st.set_page_config(
    page_title="Automated Trading Bot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for balance and trades
if 'initial_balance' not in st.session_state:
    st.session_state.initial_balance = 10000  # $10,000
if 'balance' not in st.session_state:
    st.session_state.balance = st.session_state.initial_balance
if 'allocated_capital' not in st.session_state:
    st.session_state.allocated_capital = {}
if 'open_positions' not in st.session_state:
    st.session_state.open_positions = {}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'balance_history' not in st.session_state:
    st.session_state.balance_history = []

# Title and description
st.title("🚀 Automated Trading Bot Dashboard")
st.markdown("""
Welcome to the professional trading bot dashboard. This tool analyzes top-performing commodities, forex pairs, and indices to provide trading signals and execute trades based on advanced strategies.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Asset Class", ["Forex", "Commodities", "Indices", "Crypto"])

# Define tickers based on selection
if section == "Forex":
    st.header("💱 Top Forex Pairs")
    tickers = FOREX_SYMBOLS  # Use the predefined constant
    asset_class = 'Forex'
elif section == "Commodities":
    st.header("🌐 Top Commodities")
    tickers = COMMODITIES  # Use the predefined constant
    asset_class = 'Commodities'
elif section == "Indices":
    st.header("📊 Global Indices Overview")
    tickers = INDICES_SYMBOLS  # Use the predefined constant
    asset_class = 'Indices'
elif section == "Cryptocurrency":
    st.header("💎 Top Cryptocurrencies")
    tickers = CRYPTO_SYMBOLS  # Use the predefined constant
    asset_class = 'Cryptocurrency'

# Allocate capital per ticker
num_tickers = len(tickers)
capital_per_ticker = st.session_state.balance / num_tickers

# Initialize allocated capital and open positions
for ticker in tickers:
    if ticker not in st.session_state.allocated_capital:
        st.session_state.allocated_capital[ticker] = capital_per_ticker
    if ticker not in st.session_state.open_positions:
        st.session_state.open_positions[ticker] = None  # None indicates no open position

# Function to fetch live data from FMP
def fetch_live_data(tickers, asset_class):
    data = {}
    api_key = os.getenv("FMP_API_KEY")

    for ticker in tickers:
        try:
            ticker_api = ticker.replace('/', '')
            if asset_class in ['Forex', 'Commodities']:
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker_api}?apikey={api_key}'
            else:
                url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_api}?timeseries=30&apikey={api_key}'

            response = requests.get(url)
            response.raise_for_status()
            data_json = response.json()

            if not data_json:
                st.warning(f"No data returned for {ticker}.")
                continue

            if asset_class == 'Indices':
                df = pd.DataFrame(data_json.get('historical', []))
            else:
                df = pd.DataFrame(data_json)

            if df.empty:
                st.warning(f"No data available for {ticker}.")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'})
            df = df.sort_index()

            # Filter data for the last 2 days for intraday or last 30 days for daily
            if asset_class == 'Indices':
                df = df[df.index >= (datetime.utcnow() - timedelta(days=30))]
            else:
                df = df[df.index >= (datetime.utcnow() - timedelta(days=2))]

            if df.empty:
                st.warning(f"No recent data available for {ticker}.")
                continue

            data[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
    return data

# Functions to compute indicators
def compute_indicators(df, asset_class):
    df = df.copy()
    if asset_class == 'Indices':
        df['MA_Short'] = df['Close'].rolling(window=5).mean()
        df['MA_Long'] = df['Close'].rolling(window=20).mean()
    else:
        df['MA_Short'] = df['Close'].rolling(window=10).mean()
        df['MA_Long'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    return df

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

# Function to generate combined signal
def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
    df.loc[df['MA_Short'] < df['MA_Long'], 'Signal'] = -1
    return df

# Function to simulate trades live
def simulate_trades_live(data):
    for ticker in tickers:
        if ticker in data:
            df = compute_indicators(data[ticker], asset_class)
            df = df.dropna()

            if df.empty:
                st.warning(f"No data to process for {ticker}.")
                continue
            
            df = generate_signals(df)
            allocated = st.session_state.allocated_capital[ticker]
            position = st.session_state.open_positions[ticker]

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
                    st.session_state.open_positions[ticker] = position
                    # Update balance
                    st.session_state.balance -= allocated
                    st.session_state.balance_history.append({'Time': current_time, 'Balance': st.session_state.balance})
                    st.success(f"✅ Bought {ticker} at ${buy_price:.2f} on {current_time}")
            else:
                # Check if 10% profit achieved or sell signal
                if price >= position['Buy_Price'] * 1.10 or signal == -1:
                    sell_price = price
                    profit = (sell_price - position['Buy_Price']) * position['Quantity']
                    st.session_state.balance += allocated + profit
                    st.session_state.balance_history.append({'Time': current_time, 'Balance': st.session_state.balance})
                    st.session_state.trade_history.append({
                        'Ticker': ticker,
                        'Buy_Time': position['Buy_Time'],
                        'Buy_Price': position['Buy_Price'],
                        'Sell_Time': current_time,
                        'Sell_Price': sell_price,
                        'Profit/Loss': profit
                    })
                    st.success(f"✅ Sold {ticker} at ${sell_price:.2f} on {current_time} | Profit: ${profit:.2f}")
                    st.session_state.open_positions[ticker] = None
        else:
            st.warning(f"No data available for {ticker}.")
    return

# Fetch live data
data = fetch_live_data(tickers, asset_class)

if not data:
    st.error("No data fetched. Please check the tickers and API availability.")
    st.stop()

# Simulate trades on live data
simulate_trades_live(data)

# Main layout
st.markdown("---")
col1, col2 = st.columns(2)

# Display Account Overview
with col1:
    st.header("💰 Account Overview")
    st.metric("Initial Balance", f"${st.session_state.initial_balance:,.2f}")
    st.metric("Current Balance", f"${st.session_state.balance:,.2f}")
    if st.session_state.trade_history:
        total_profit = sum([trade['Profit/Loss'] for trade in st.session_state.trade_history])
        st.metric("Total Profit/Loss", f"${total_profit:,.2f}")
        num_trades = len(st.session_state.trade_history)
        winning_trades = sum(1 for trade in st.session_state.trade_history if trade['Profit/Loss'] > 0)
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
    if st.session_state.balance_history:
        balance_df = pd.DataFrame(st.session_state.balance_history)
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
if st.session_state.trade_history:
    trades_df = pd.DataFrame(st.session_state.trade_history)
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
if any(position is not None for position in st.session_state.open_positions.values()):
    open_positions_list = []
    for ticker, position in st.session_state.open_positions.items():
        if position:
            if ticker in data and not data[ticker].empty:
                current_price = data[ticker]['Close'][-1]
                profit_loss = (current_price - position['Buy_Price']) * position['Quantity']
                open_positions_list.append({
                    'Ticker': ticker,
                    'Buy_Time': position['Buy_Time'].strftime('%Y-%m-%d %H:%M'),
                    'Buy_Price': position['Buy_Price'],
                    'Current_Price': current_price,
                    'Profit/Loss': profit_loss
                })
            else:
                st.warning(f"No current price data available for {ticker}.")
    if open_positions_list:
        open_positions_df = pd.DataFrame(open_positions_list)
        open_positions_df['Profit/Loss'] = open_positions_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(open_positions_df.style.format({'Buy_Price': '${:,.2f}', 'Current_Price': '${:,.2f}'}))
    else:
        st.info("No open positions to display.")
else:
    st.info("No open positions.")

st.markdown("---")

# Display Trade Signals and Prices per Ticker
st.header("🔍 Trade Signals and Price Charts")

for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker], asset_class)
        df = df.dropna()
        if df.empty:
            st.warning(f"No data to display for {ticker}.")
            continue
        df = generate_signals(df)

        # Find buy and sell points from trade_history
        trades = [trade for trade in st.session_state.trade_history if trade['Ticker'] == ticker]
        position = st.session_state.open_positions[ticker]

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
            x=df.index, y=df['MA_Short'], line=dict(color='blue', width=1), name='MA Short'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_Long'], line=dict(color='orange', width=1), name='MA Long'
        ))

        # Add Buy/Sell markers
        if trades:
            buy_times = [pd.to_datetime(trade['Buy_Time']) for trade in trades]
            buy_prices = [trade['Buy_Price'] for trade in trades]
            sell_times = [pd.to_datetime(trade['Sell_Time']) for trade in trades]
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

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {ticker}.")

# Footer
st.markdown("---")
st.markdown("<center>© 2023 Trading Bot Dashboard | Powered by Streamlit</center>", unsafe_allow_html=True)
