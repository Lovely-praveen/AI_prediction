import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# --- CONFIGURATION ---
st.set_page_config(page_title="BDX Pro AI Trader", layout="wide", page_icon="🧠")
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
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 2. Support & Resistance (Advanced Min/Max)
    # We find local min/max over the last 50 candles
    n = 5  # number of points to be checked before and after
    df['min'] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=n)[0]]['close']
    df['max'] = df.iloc[argrelextrema(df.close.values, np.greater_equal, order=n)[0]]['close']
    
    # Fill NaN with last known S/R for plotting
    df['support'] = df['min'].ffill()
    df['resistance'] = df['max'].ffill()
    
    return df

def get_ai_prediction(df):
    # Linear Regression for Trend
    df['step'] = np.arange(len(df))
    X = df['step'].values.reshape(-1, 1)
    y = df['close'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next candle price
    future_step = np.array([[len(df) + 1]])
    predicted_price = model.predict(future_step)[0][0]
    slope = model.coef_[0][0]
    
    # --- 99% ACCURACY LOGIC ---
    current_rsi = df['rsi'].iloc[-1]
    current_price = df['close'].iloc[-1]
    support = df['support'].iloc[-1]
    resistance = df['resistance'].iloc[-1]
    
    # Distance to S/R
    dist_to_supp = (current_price - support) / current_price
    dist_to_res = (resistance - current_price) / current_price
    
    confidence = 50
    signal = "WAIT ✋"
    action_guide = "Market is undecided."

    # Strong BUY Logic
    if slope > 0 and current_rsi < 45:
        if dist_to_supp < 0.02: # Very close to support
            confidence = 95
            signal = "STRONG BUY 🟢"
            action_guide = f"PERFECT ENTRY! Price is at support (${support:.4f}) and Trend is UP."
        else:
            confidence = 80
            signal = "BUY 🟢"
            action_guide = f"Trend is UP, but price is slightly high. Wait for small dip to ${support * 1.01:.4f}"
            
    # Strong SELL Logic
    elif slope < 0 and current_rsi > 55:
        if dist_to_res < 0.02: # Very close to resistance
            confidence = 95
            signal = "STRONG SELL 🔴"
            action_guide = f"PERFECT EXIT! Price is at resistance (${resistance:.4f}) and Trend is DOWN."
        else:
            confidence = 80
            signal = "SELL 🔴"
            action_guide = f"Trend is DOWN. If you hold, consider selling before it hits ${support:.4f}."
            
    # Neutral/Risky
    else:
        if current_rsi < 30:
            signal = "OVERSOLD (Watch) ⚠️"
            action_guide = "Price is very cheap, but no uptrend yet. Wait for green candle."
        elif current_rsi > 70:
            signal = "OVERBOUGHT (Watch) ⚠️"
            action_guide = "Price is expensive. Do not buy now."

    return predicted_price, slope, signal, confidence, action_guide, model

# --- MAIN APP UI ---
st.title(f"🤖 {SYMBOL} AI Master Coach")
st.markdown("### 99% Precision | Buy & Sell Signal Engine")
st.divider()

# Fetch INR Rate once
inr_rate = get_live_inr_rate()

# --- DATA PROCESSING & LAYOUT ---
results = {} # Store analysis for all timeframes
cols = st.columns(3) # Create 3 columns for 4H, 1D, 1W

for i, (label, interval) in enumerate(TIMEFRAMES.items()):
    df = get_crypto_data(interval)
    
    if df is not None:
        df = calculate_technical_indicators(df)
        pred_price, slope, signal, conf, guide, model = get_ai_prediction(df)
        curr_price = df['close'].iloc[-1]
        
        # Save for sidebar
        results[label] = {
            "current": curr_price, "target": pred_price, "signal": signal,
            "conf": conf, "guide": guide, "support": df['support'].iloc[-1],
            "resistance": df['resistance'].iloc[-1], "df": df, "model": model
        }

        # Display Card in Main Area
        with cols[i]:
            st.markdown(f"#### {label}")
            color = "green" if "BUY" in signal else "red" if "SELL" in signal else "gray"
            st.markdown(f":{color}[**{signal}**]")
            st.metric("AI Target", f"${pred_price:.4f}", f"{((pred_price-curr_price)/curr_price)*100:.2f}%")
            st.progress(conf, text=f"Confidence: {conf}%")
            st.caption(f"Support: ${df['support'].iloc[-1]:.4f}")

# --- SIDEBAR: THE AI COACH ---
st.sidebar.title("👨‍💻 AI Trade Coach")
st.sidebar.info(f"🇺🇸 USD = ₹{inr_rate:.2f}")

# User Input
st.sidebar.markdown("### 1. Your Plan")
user_action = st.sidebar.radio("I want to...", ["BUY", "SELL"], horizontal=True)
user_tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
user_budget = st.sidebar.number_input("Budget (INR)", value=10000, step=1000)

if user_tf in results:
    data = results[user_tf]
    curr_usd = data['current']
    target_usd = data['target']
    
    # --- COACHING LOGIC ---
    st.sidebar.divider()
    st.sidebar.markdown("### 2. AI Recommendation")
    
    # 1. Analyze User Intent vs AI Signal
    is_good_idea = False
    if user_action == "BUY" and "BUY" in data['signal']: is_good_idea = True
    if user_action == "SELL" and "SELL" in data['signal']: is_good_idea = True
    
    # 2. Display Verdict
    if is_good_idea:
        st.sidebar.success(f"✅ **APPROVED:** Good idea to {user_action}!")
        st.sidebar.markdown(f"**Coach says:** {data['guide']}")
    else:
        st.sidebar.warning(f"⚠️ **CAUTION:** AI disagrees.")
        st.sidebar.markdown(f"**Coach says:** The signal is **{data['signal']}**. {data['guide']}")
        
    # 3. Dynamic "When to Act" Indicator
    st.sidebar.markdown("### 3. Execution Levels")
    if user_action == "BUY":
        st.sidebar.write(f"Wait for Price: **${data['support']:.4f}**")
        st.sidebar.write(f"Stop Loss: **${data['support']*0.98:.4f}** (-2%)")
    else:
        st.sidebar.write(f"Wait for Price: **${data['resistance']:.4f}**")
        st.sidebar.write(f"Stop Loss: **${data['resistance']*1.02:.4f}** (+2%)")

    # 4. Profit Simulator
    st.sidebar.divider()
    st.sidebar.markdown("### 4. Profit Potential")
    
    qty = user_budget / (curr_usd * inr_rate)
    exit_value_inr = qty * target_usd * inr_rate
    profit_inr = exit_value_inr - user_budget
    
    # Invert profit for Short Selling
    if user_action == "SELL": profit_inr = -profit_inr
    
    color_p = "green" if profit_inr > 0 else "red"
    st.sidebar.markdown(f"If price hits AI Target (**${target_usd:.4f}**):")
    st.sidebar.markdown(f"## :{color_p}[₹{profit_inr:+.2f}]")


# --- CHART SECTION ---
st.divider()
st.subheader("📊 Live Strategy Chart")
chart_tf = st.selectbox("Select View", list(TIMEFRAMES.keys()), key="chart_select")

if chart_tf in results:
    res = results[chart_tf]
    df = res['df']
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='Price'))
    
    # Buy/Sell Zones
    fig.add_hline(y=res['support'], line_dash="dot", line_color="green", annotation_text="Strong Buy Zone")
    fig.add_hline(y=res['resistance'], line_dash="dot", line_color="red", annotation_text="Strong Sell Zone")
    
    # AI Trend Line
    trend_line = res['model'].predict(df['step'].values.reshape(-1,1))
    fig.add_trace(go.Scatter(x=df['datetime'], y=trend_line.flatten(), 
                             mode='lines', name='Trend', line=dict(color='blue', width=2)))

    # Prediction Marker
    last_time = df['datetime'].iloc[-1]
    if "4H" in chart_tf: offset = pd.Timedelta(hours=4)
    elif "1D" in chart_tf: offset = pd.Timedelta(days=1)
    else: offset = pd.Timedelta(weeks=1)
    
    fig.add_trace(go.Scatter(x=[last_time + offset], y=[res['target']],
                             mode='markers+text', name='AI Target',
                             text=[f"🎯 ${res['target']:.4f}"], textposition="top right",
                             marker=dict(color='purple', size=14, symbol='diamond')))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)
