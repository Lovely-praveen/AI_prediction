import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="BDX AI Predictor", layout="wide", page_icon="📈")
SYMBOL = "BDX-USDT"
INTERVAL = "4hour"

# --- FUNCTIONS (Same logic, adapted for Web) ---
def get_live_inr_rate():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        data = requests.get(url, timeout=2).json()
        return data['rates']['INR']
    except:
        return 87.50

def get_crypto_data():
    url = f"https://api.kucoin.com/api/v1/market/candles?type={INTERVAL}&symbol={SYMBOL}"
    try:
        data = requests.get(url).json()
        if 'data' in data:
            df = pd.DataFrame(data['data'], columns=['time', 'open', 'close', 'high', 'low', 'vol', 'turnover'])
            df['close'] = pd.to_numeric(df['close'])
            df['time'] = pd.to_numeric(df['time'])
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            return df.iloc[::-1].reset_index(drop=True)
    except:
        return None

def calculate_rsi(df, periods=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_ai_prediction(df):
    # Linear Regression for Trend
    df['step'] = np.arange(len(df))
    X = df['step'].values.reshape(-1, 1)
    y = df['close'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next candle
    future_step = np.array([[len(df) + 1]])
    predicted_price = model.predict(future_step)[0][0]
    slope = model.coef_[0][0]
    
    return predicted_price, slope, model

# --- APP LAYOUT ---
st.title(f"🤖 {SYMBOL} AI Forecaster")
st.caption("Live AI Analysis running on Python")

if st.button('🔄 Refresh Data'):
    st.rerun()

# 1. Fetch Data
df = get_crypto_data()
inr_rate = get_live_inr_rate()

if df is not None:
    # Calculations
    current_price = df['close'].iloc[-1]
    df['rsi'] = calculate_rsi(df)
    current_rsi = df['rsi'].iloc[-1]
    pred_price, slope, model = get_ai_prediction(df)
    
    # Determine Status
    trend_color = "normal"
    if slope > 0: trend_str = "UP 📈"; trend_color="normal"
    else: trend_str = "DOWN 📉"; trend_color="inverse"
    
    rsi_state = "Neutral"
    if current_rsi < 30: rsi_state = "OVERSOLD (Buy Zone)"
    elif current_rsi > 70: rsi_state = "OVERBOUGHT (Sell Zone)"

    # 2. Mobile Friendly Metrics (Top Row)
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price (USD)", f"${current_price:.5f}", f"{slope*100:.4f}%")
    col2.metric("Current Price (INR)", f"₹{(current_price*inr_rate):.2f}", f"Rate: ₹{inr_rate}")
    col3.metric("RSI Strength", f"{current_rsi:.1f}", rsi_state)

    # 3. AI Verdict Box
    st.divider()
    if current_rsi < 35 and slope > 0:
        st.success("## 🟢 AI Verdict: STRONG BUY")
        st.write("Reason: Price is oversold and trend is turning up.")
    elif current_rsi > 65 and slope < 0:
        st.error("## 🔴 AI Verdict: STRONG SELL")
        st.write("Reason: Price is overbought and trend is falling.")
    else:
        st.warning(f"## ✋ AI Verdict: WAIT / HOLD")
        st.write(f"Reason: Market is undecided. Trend is {trend_str} but no strong entry signal.")

    # 4. Interactive Chart (Visual Suggestion)
    st.subheader("Price Chart + AI Prediction Line")
    
    # Create the base chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['datetime'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='Market Data'))

    # Add the "AI Trend Line" (Regression Line)
    # We calculate the line points based on the model we trained
    trend_line = model.predict(df['step'].values.reshape(-1,1))
    fig.add_trace(go.Scatter(x=df['datetime'], y=trend_line.flatten(), 
                             mode='lines', name='AI Trend Line', line=dict(color='blue', width=2, dash='dot')))

    # Add the Future Point
    future_time = df['datetime'].iloc[-1] + pd.Timedelta(hours=4)
    fig.add_trace(go.Scatter(x=[future_time], y=[pred_price],
                             mode='markers+text', name='Prediction',
                             text=[f"Target: ${pred_price:.4f}"], textposition="top center",
                             marker=dict(color='purple', size=12, symbol='star')))

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 5. Raw Data (Optional)
    with st.expander("See Raw Data"):
        st.dataframe(df.tail(5))

else:
    st.error("Could not fetch data from KuCoin. Try again later.")
