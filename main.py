import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="BDX Pro AI Predictor", layout="wide", page_icon="🚀")
SYMBOL = "BDX-USDT"
TIMEFRAMES = {"Short Term (4H)": "4hour", "Mid Term (1D)": "1day", "Long Term (1W)": "1week"}

# --- HELPER FUNCTIONS ---
def get_live_inr_rate():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        data = requests.get(url, timeout=2).json()
        return data['rates']['INR']
    except:
        return 87.50

def get_crypto_data(interval):
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={SYMBOL}"
    try:
        data = requests.get(url).json()
        if 'data' in data:
            df = pd.DataFrame(data['data'], columns=['time', 'open', 'close', 'high', 'low', 'vol', 'turnover'])
            cols = ['open', 'close', 'high', 'low', 'vol']
            df[cols] = df[cols].apply(pd.to_numeric)
            df['time'] = pd.to_numeric(df['time'])
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            return df.iloc[::-1].reset_index(drop=True)
    except:
        return None

def calculate_technical_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Support & Resistance (Simple Rolling Min/Max)
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    return df

def get_ai_prediction(df):
    # Linear Regression for Trend
    df['step'] = np.arange(len(df))
    X = df['step'].values.reshape(-1, 1)
    y = df['close'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next step
    future_step = np.array([[len(df) + 1]])
    predicted_price = model.predict(future_step)[0][0]
    slope = model.coef_[0][0]
    
    # Accuracy/Confidence Score Calculation
    # We check if RSI and Trend align. 
    # If RSI < 30 (Buy) AND Slope > 0 (Up), Confidence is High.
    current_rsi = df['rsi'].iloc[-1]
    confidence = 0
    signal = "NEUTRAL"
    
    if slope > 0:
        if current_rsi < 40: confidence = 95; signal = "STRONG BUY 🟢"
        elif current_rsi < 60: confidence = 75; signal = "BUY 🟢"
        else: confidence = 50; signal = "WEAK BUY (Risky) ⚠️"
    else:
        if current_rsi > 70: confidence = 95; signal = "STRONG SELL 🔴"
        elif current_rsi > 40: confidence = 75; signal = "SELL 🔴"
        else: confidence = 50; signal = "WEAK SELL (Risky) ⚠️"

    return predicted_price, slope, signal, confidence, model

# --- MAIN APP UI ---
st.title(f"🤖 {SYMBOL} AI Master Predictor")
st.markdown("### 99% Data-Driven Multi-Timeframe Analysis")

# Fetch INR Rate once
inr_rate = get_live_inr_rate()
st.metric("🇺🇸 USD to 🇮🇳 INR Rate", f"₹{inr_rate:.2f}")

# --- SECTION 1: MULTI-TIMEFRAME CARDS ---
st.markdown("---")
st.subheader("🔮 Timeframe Predictions")
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

results = {} # Store results for simulator

for i, (label, interval) in enumerate(TIMEFRAMES.items()):
    df = get_crypto_data(interval)
    if df is not None:
        df = calculate_technical_indicators(df)
        pred_price, slope, signal, conf, model = get_ai_prediction(df)
        
        curr_price = df['close'].iloc[-1]
        change_pct = ((pred_price - curr_price) / curr_price) * 100
        
        # Store for later
        results[label] = {
            "current": curr_price,
            "target": pred_price,
            "signal": signal,
            "df": df,
            "model": model
        }

        # Display Card
        with cols[i]:
            st.info(f"**{label}**")
            st.metric("Target Price", f"${pred_price:.4f}", f"{change_pct:.2f}%")
            st.write(f"🚦 Signal: **{signal}**")
            st.write(f"🎯 Confidence: **{conf}%**")
            st.write(f"📉 Buy Zone: **${df['support'].iloc[-1]:.4f}**")
            st.write(f"📈 Sell Zone: **${df['resistance'].iloc[-1]:.4f}**")

# --- SECTION 2: PROFIT CALCULATOR (SIDEBAR) ---
st.sidebar.header("💰 Trade Simulator")
st.sidebar.write("Input your plan to see potential gains.")

sim_amount = st.sidebar.number_input("Investment Amount (INR)", value=10000, step=500)
sim_action = st.sidebar.radio("I want to...", ["BUY", "SELL"])
sim_timeframe = st.sidebar.selectbox("Timeframe Goal", list(TIMEFRAMES.keys()))

if sim_timeframe in results:
    data = results[sim_timeframe]
    curr_inr = data['current'] * inr_rate
    target_inr = data['target'] * inr_rate
    
    st.sidebar.divider()
    st.sidebar.write(f"**Current Price:** ₹{curr_inr:.2f}")
    st.sidebar.write(f"**AI Target:** ₹{target_inr:.2f}")
    
    qty = sim_amount / curr_inr
    projected_value = qty * target_inr
    profit = projected_value - sim_amount
    
    if sim_action == "BUY":
        if profit > 0:
            st.sidebar.success(f"🎉 Projected Profit: ₹{profit:.2f}")
        else:
            st.sidebar.error(f"⚠️ Projected Loss: ₹{profit:.2f}")
    else: # Short Selling logic
        profit = -profit # Profit if price goes down
        if profit > 0:
            st.sidebar.success(f"🎉 Projected Profit: ₹{profit:.2f}")
        else:
            st.sidebar.error(f"⚠️ Projected Loss: ₹{profit:.2f}")

# --- SECTION 3: INTERACTIVE CHART ---
st.markdown("---")
st.subheader("📊 Live Technical Chart")
selected_tf = st.selectbox("Select Chart Timeframe", list(TIMEFRAMES.keys()))

if selected_tf in results:
    res = results[selected_tf]
    df = res['df']
    model = res['model']
    
    # Create Plotly Chart
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df['datetime'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='Price'))
    
    # AI Trend Line
    trend_line = model.predict(df['step'].values.reshape(-1,1))
    fig.add_trace(go.Scatter(x=df['datetime'], y=trend_line.flatten(), 
                             mode='lines', name='AI Trend', line=dict(color='blue', width=2)))
    
    # Buy/Sell Zones
    fig.add_hline(y=df['support'].iloc[-1], line_dash="dash", line_color="green", annotation_text="Buy Zone")
    fig.add_hline(y=df['resistance'].iloc[-1], line_dash="dash", line_color="red", annotation_text="Sell Zone")
    
    # Prediction Point
    last_time = df['datetime'].iloc[-1]
    # Estimate next candle time roughly
    if "4H" in selected_tf: offset = pd.Timedelta(hours=4)
    elif "1D" in selected_tf: offset = pd.Timedelta(days=1)
    else: offset = pd.Timedelta(weeks=1)
    
    fig.add_trace(go.Scatter(x=[last_time + offset], y=[res['target']],
                             mode='markers+text', name='AI Target',
                             text=[f"Target: ${res['target']:.4f}"], textposition="top center",
                             marker=dict(color='purple', size=15, symbol='star')))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
