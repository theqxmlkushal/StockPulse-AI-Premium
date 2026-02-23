import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
import os
from groq import Groq
from datetime import datetime, timedelta

# --- CONFIG & STYLING ---
st.set_page_config(page_title="StockPulse AI v3", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stRadio [data-baseweb="radio"] { padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚀 StockPulse AI v3")
    st.markdown("---")
    ticker_input = st.text_input("Ticker Symbol", value="AAPL").upper()
    page = st.radio("Navigation", ["📊 Dashboard", "🧠 AI Conclusion"])
    st.markdown("---")
    st.subheader("🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", value=st.session_state.groq_key, type="password")
    if groq_api_key:
        st.session_state.groq_key = groq_api_key
    st.markdown("---")
    st.info("✨ Systems Online | AI Model: Llama-3.1-8b")

# --- HELPERS ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    try:
        df = yf.download(ticker, period="10y")
        if df.empty: return None
        # Handle MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return None

def get_model_path(ticker, model_type):
    if not os.path.exists("models"): os.makedirs("models")
    return f"models/{ticker}_{model_type}.keras"

def load_model_safe(path):
    try:
        if os.path.exists(path): return load_model(path)
    except:
        if os.path.exists(path): os.remove(path)
    return None

def get_sentiment(ticker_symbol, api_key):
    try:
        t = yf.Ticker(ticker_symbol)
        news = t.news[:5]
        if not news: return "Neutral"
        
        headlines = [n['title'] for n in news]
        client = Groq(api_key=api_key)
        prompt = f"Analyze the following news headlines for {ticker_symbol} and return ONLY one word: Bullish, Bearish, or Neutral.\n\n" + "\n".join(headlines)
        
        comp = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant")
        return comp.choices[0].message.content.strip()
    except: return "Neutral"

# --- CORE LOGIC ---
if ticker_input:
    df = load_data(ticker_input)
    if df is not None:
        # Strict scalar extraction for basic price metrics
        df_clean = df[['Date', 'Close']].dropna()
        today_price = float(df_clean['Close'].iloc[-1])
        
        if page == "📊 Dashboard":
            st.title(f"📈 {ticker_input} Market Pulse")
            
            # Graph Controls
            period_options = {"1M": 30, "6M": 180, "1Y": 365, "5Y": 1825, "MAX": len(df_clean)}
            selected_period = st.select_slider("Select Graph Period", options=list(period_options.keys()), value="1Y")
            
            # Filter Data based on period
            days = period_options[selected_period]
            plot_df = df_clean.tail(days)
            
            fig = go.Figure(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], line=dict(color='#00d1ff')))
            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            # Use .item() and float() to guarantee scalars
            c1.metric("Current", f"₹{float(today_price):.2f}")
            c2.metric("1Y High", f"₹{float(df_clean['Close'].tail(252).max()):.2f}")
            c3.metric("1Y Low", f"₹{float(df_clean['Close'].tail(252).min()):.2f}")

        elif page == "🧠 AI Conclusion":
            st.title("🧠 Intelligence Hub")
            
            with st.spinner("Processing deep analysis..."):
                # Prediction Logic
                scaler = MinMaxScaler()
                close_data = df_clean[['Close']].values
                scaled = scaler.fit_transform(close_data)
                
                X_train = []
                for i in range(60, len(scaled)): X_train.append(scaled[i-60:i, 0])
                X_train = np.array(X_train).reshape(-1, 60, 1)
                y_train = scaled[60:, 0]
                last_60 = scaled[-60:].reshape(1, 60, 1)

                lstm_path = get_model_path(ticker_input, "lstm")
                gru_path = get_model_path(ticker_input, "gru")
                
                # Self-healing Load
                m_lstm = load_model_safe(lstm_path)
                if not m_lstm:
                    m_lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)), Dropout(0.2), LSTM(50), Dense(1)])
                    m_lstm.compile(optimizer='adam', loss='mse')
                    m_lstm.fit(X_train, y_train, epochs=2, verbose=0)
                    m_lstm.save(lstm_path)
                
                m_gru = load_model_safe(gru_path)
                if not m_gru:
                    m_gru = Sequential([GRU(50, return_sequences=True, input_shape=(60,1)), Dropout(0.2), GRU(50), Dense(1)])
                    m_gru.compile(optimizer='adam', loss='mse')
                    m_gru.fit(X_train, y_train, epochs=2, verbose=0)
                    m_gru.save(gru_path)

                p_lstm = float(scaler.inverse_transform(m_lstm.predict(last_60))[0][0])
                p_gru = float(scaler.inverse_transform(m_gru.predict(last_60))[0][0])
                sentiment = get_sentiment(ticker_input, st.session_state.groq_key) if st.session_state.groq_key else "N/A"

            col1, col2, col3 = st.columns(3)
            col1.metric("LSTM Goal", f"₹{p_lstm:.2f}", f"{((p_lstm/today_price)-1)*100:.2f}%")
            col2.metric("GRU Goal", f"₹{p_gru:.2f}", f"{((p_gru/today_price)-1)*100:.2f}%")
            col3.metric("Market Sentiment", sentiment)

            st.markdown("---")
            if st.session_state.groq_key:
                if st.button("Generate Final Investment Thesis"):
                    client = Groq(api_key=st.session_state.groq_key)
                    prompt = f"Stock: {ticker_input}. Context: Price={today_price}, Predictions(LSTM={p_lstm}, GRU={p_gru}), Sentiment={sentiment}. Give a final Buy/Sell/Hold with 3 data points."
                    res = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant")
                    st.info(res.choices[0].message.content)
            else:
                st.warning("Please configure Groq Key in sidebar.")

    else: st.error("Invalid Ticker.")
else:
    st.info("👈 Enter a ticker symbol in the sidebar to begin.")
