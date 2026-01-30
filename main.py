import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# --- CONFIGURATION ---
st.set_page_config(page_title="BDX AI Master", layout="wide", page_icon="🧠")
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

def analyze_market(df):
    """
    Performs ALL calculations: Trend, Zone, RSI, Support/Resistance, AI Prediction.
    """
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 2. Support & Resistance (Local Min/Max)
    n = 5
    df['min'] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=n)[0]]['close']
    df['max'] = df.iloc[argrelextrema(df.close.values, np.greater_equal, order=n)[0]]['close']
    df['support'] = df['min'].ffill()
    df['resistance'] = df['max'].ffill()

    # 3. AI Trend Prediction (Linear Regression)
    df['step'] = np.arange(len(df))
    X = df['step'].values.reshape(-1, 1)
    y = df['close'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict Future Price
    future_step = np.array([[len(df) + 1]])
    predicted_price = model.predict(future_step)[0][0]
    slope = model.coef_[0][0]
    
    # 4. Determine Signal & Confidence
    current_rsi = df['rsi'].iloc[-1]
    current_price = df['close'].iloc[-1]
    support = df['support'].iloc[-1]
    resistance = df['resistance'].iloc[-1]
    
    dist_to_supp = (current_price - support) / current_price
    
    # Default Values
    signal = "WAIT ✋"
    confidence = 50
    guide = "Market is choppy."
    zone_status = "NEUTRAL ZONE"
    
    # Zone Logic (Increasing vs Decreasing)
    if slope > 0:
        zone_status = "🚀 INCREASING ZONE"
    else:
        zone_status = "📉 DECREASING ZONE"

    # Buy/Sell Signal Logic
    if slope > 0 and current_rsi < 45:
        if dist_to_supp < 0.02:
            signal = "STRONG BUY 🟢"
            confidence = 95
            guide = "Perfect Entry: At Support + Uptrend."
        else:
            signal = "BUY 🟢"
            confidence = 80
            guide = "Uptrend active, but wait for small dip."
            
    elif slope < 0 and current_rsi > 55:
        signal = "STRONG SELL 🔴"
        confidence = 95
        guide = "Perfect Exit: At Resistance + Downtrend."
    elif slope < 0:
        signal = "SELL 🔴"
        confidence = 80
        guide = "Downtrend active. Protect capital."
        
    return {
        "df": df, "model": model, "pred_price": predicted_price,
        "slope": slope, "signal": signal, "confidence": confidence,
        "guide": guide, "zone": zone_status, 
        "support": support, "resistance": resistance,
        "current_price": current_price
    }

# --- MAIN APP UI ---
st.title(f"🤖 {SYMBOL} AI Master System")
inr_rate = get_live_inr_rate()

# 1. FETCH & PROCESS ALL DATA
results = {}
for label, interval in TIMEFRAMES.items():
    d = get_crypto_data(interval)
    if d is not None:
        results[label] = analyze_market(d)

# --- SECTION 1: THE BIG ZONE INDICATOR (1-2 Days) ---
if "Mid Term (1D)" in results:
    day_data = results["Mid Term (1D)"]
    zone = day_data["zone"]
    color = "green" if "INCREASING" in zone else "red"
    
    st.markdown(f"""
    <div style="padding: 20px; border: 2px solid {color}; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: {color}; margin:0;">{zone}</h2>
        <p style="margin:0;">Forecast for Next 24-48 Hours</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Could not fetch 1-Day Data for Zone Analysis")

# --- SECTION 2: TIMEFRAME CARDS ---
col1, col2, col3 = st.columns(3)
cols_map = [col1, col2, col3]

for i, (label, data) in enumerate(results.items()):
    with cols_map[i]:
        st.markdown(f"#### {label}")
        # Color the signal text
        s_color = "green" if "BUY" in data['signal'] else "red" if "SELL" in data['signal'] else "gray"
        st.markdown(f":{s_color}[**{data['signal']}**]")
        
        # Metrics
        curr = data['current_price']
        tgt = data['pred_price']
        pct = ((tgt - curr) / curr) * 100
        
        st.metric("Target (USD)", f"${tgt:.4f}", f"{pct:.2f}%")
        st.metric("Target (INR)", f"₹{tgt*inr_rate:.2f}")
        
        st.progress(data['confidence'], text=f"Confidence: {data['confidence']}%")
        st.caption(f"Support: ${data['support']:.4f} | Res: ${data['resistance']:.4f}")

# --- SECTION 3: SIDEBAR AI COACH ---
st.sidebar.title("👨‍💻 AI Trade Coach")
st.sidebar.info(f"🇺🇸 USD = ₹{inr_rate:.2f}")

st.sidebar.markdown("### 1. Your Plan")
user_action = st.sidebar.radio("I want to...", ["BUY", "SELL"], horizontal=True)
user_tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
user_budget = st.sidebar.number_input("Budget (INR)", value=10000, step=1000)

if user_tf in results:
    data = results[user_tf]
    
    # Coach Logic
    st.sidebar.divider()
    st.sidebar.markdown("### 2. AI Recommendation")
    
    is_good = (user_action == "BUY" and "BUY" in data['signal']) or \
              (user_action == "SELL" and "SELL" in data['signal'])
              
    if is_good:
        st.sidebar.success(f"✅ APPROVED: Good idea to {user_action}")
    else:
        st.sidebar.warning(f"⚠️ CAUTION: AI says {data['signal']}")
    
    st.sidebar.info(f"💡 {data['guide']}")
    
    # Execution Levels
    st.sidebar.markdown("### 3. Execution Levels")
    if user_action == "BUY":
        st.sidebar.write(f"Wait for: **${data['support']:.4f}**")
        st.sidebar.write(f"Stop Loss: **${data['support']*0.98:.4f}**")
    else:
        st.sidebar.write(f"Wait for: **${data['resistance']:.4f}**")
        st.sidebar.write(f"Stop Loss: **${data['resistance']*1.02:.4f}**")

    # Profit Calc
    st.sidebar.divider()
    curr_inr = data['current_price'] * inr_rate
    tgt_inr = data['pred_price'] * inr_rate
    qty = user_budget / curr_inr
    profit = (qty * tgt_inr) - user_budget
    if user_action == "SELL": profit = -profit
    
    p_color = "green" if profit > 0 else "red"
    st.sidebar.markdown(f"Potential Profit: :{p_color}[**₹{profit:.2f}**]")


# --- SECTION 4: CHART ---
st.divider()
st.subheader("📊 Live Strategy Chart")
sel_chart = st.selectbox("View Timeframe", list(TIMEFRAMES.keys()))

if sel_chart in results:
    res = results[sel_chart]
    df = res['df']
    
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='Price'))
    
    # Zones
    fig.add_hline(y=res['support'], line_dash="dot", line_color="green", annotation_text="Buy Zone")
    fig.add_hline(y=res['resistance'], line_dash="dot", line_color="red", annotation_text="Sell Zone")
    
    # Trend
    trend_line = res['model'].predict(df['step'].values.reshape(-1,1))
    fig.add_trace(go.Scatter(x=df['datetime'], y=trend_line.flatten(), 
                             mode='lines', name='Trend', line=dict(color='blue', width=2)))
    
    # Target Marker
    last_time = df['datetime'].iloc[-1]
    if "4H" in sel_chart: offset = pd.Timedelta(hours=4)
    elif "1D" in sel_chart: offset = pd.Timedelta(days=1)
    else: offset = pd.Timedelta(weeks=1)
    
    fig.add_trace(go.Scatter(x=[last_time + offset], y=[res['pred_price']],
                             mode='markers+text', name='AI Target',
                             text=[f"🎯 ${res['pred_price']:.4f}"], textposition="top right",
                             marker=dict(color='purple', size=15, symbol='diamond')))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
