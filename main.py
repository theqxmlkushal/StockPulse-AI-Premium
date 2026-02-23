import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from datetime import datetime, timedelta
import os
import json
from groq import Groq

# --- CONFIG & STYLING ---
st.set_page_config(page_title="StockPulse AI | Premium Trader", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-bottom: 2px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚀 StockPulse AI")
    st.markdown("---")
    ticker_input = st.text_input("Ticker Symbol", value="AAPL", help="e.g. AAPL, TSLA, INFY.NS")
    st.markdown("---")
    st.subheader("🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", value=st.session_state.groq_key, type="password")
    
    st.markdown("---")
    st.info("💡 Tip: Use `.NS` suffix for National Stock Exchange (India) tickers.")

# --- HELPERS ---
@st.cache_data(ttl=3600)
def load_data(ticker):
    try:
        df = yf.download(ticker, period="15y")
        if df.empty: return None
        df.reset_index(inplace=True)
        return df
    except Exception:
        return None

def get_model_path(ticker, model_type):
    if not os.path.exists("models"):
        os.makedirs("models")
    return f"models/{ticker}_{model_type}.keras"

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(len(X), look_back, 1), np.array(y)

# --- CORE LOGIC ---
if ticker_input:
    df = load_data(ticker_input)
    
    if df is not None and not df.empty:
        # Preprocessing
        df_clean = df[['Date', 'Close']].dropna()
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        today_price = float(df_clean['Close'].iloc[-1])
        
        # Tabs Layout
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predictions", "🤖 AI Insights"])
        
        with tab1:
            st.title(f"📈 {ticker_input.upper()} Market Pulse")
            
            # Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_clean['Date'], y=df_clean['Close'], name="Close Price", line=dict(color='#00d1ff')))
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{float(today_price):.2f}")
            col2.metric("1Y High", f"₹{float(df_clean['Close'].tail(252).max()):.2f}")
            col3.metric("1Y Low", f"₹{float(df_clean['Close'].tail(252).min()):.2f}")

        with tab2:
            st.subheader("Deep Learning Forecasts")
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_clean[['Close']].values)
            look_back = 60
            
            # Prediction Models (LSTM/GRU)
            with st.spinner("Analyzing market patterns..."):
                lstm_path = get_model_path(ticker_input, "lstm")
                gru_path = get_model_path(ticker_input, "gru")
                
                X_train_seq, y_train_seq = create_sequences(scaled_data, look_back)
                last_60 = np.array([scaled_data[-look_back:]]).reshape(1, look_back, 1)

                # LSTM
                if os.path.exists(lstm_path):
                    lstm_model = load_model(lstm_path)
                else:
                    lstm_model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                        Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25), Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                    lstm_model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=32, verbose=0)
                    lstm_model.save(lstm_path)
                
                # GRU
                if os.path.exists(gru_path):
                    gru_model = load_model(gru_path)
                else:
                    gru_model = Sequential([
                        GRU(50, return_sequences=True, input_shape=(look_back, 1)),
                        Dropout(0.2), GRU(50), Dropout(0.2), Dense(25), Dense(1)
                    ])
                    gru_model.compile(optimizer='adam', loss='mean_squared_error')
                    gru_model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=32, verbose=0)
                    gru_model.save(gru_path)

                pred_lstm = float(scaler.inverse_transform(lstm_model.predict(last_60))[0][0])
                pred_gru = float(scaler.inverse_transform(gru_model.predict(last_60))[0][0])

            p_col1, p_col2 = st.columns(2)
            p_col1.metric("LSTM Projection", f"₹{pred_lstm:.2f}", f"{((pred_lstm/today_price)-1)*100:.2f}%")
            p_col2.metric("GRU Projection", f"₹{pred_gru:.2f}", f"{((pred_gru/today_price)-1)*100:.2f}%")

        with tab3:
            st.subheader("Groq AI Financial Advisor")
            if groq_api_key:
                if st.button("Generate AI Investment Thesis"):
                    try:
                        client = Groq(api_key=groq_api_key)
                        prompt = f"""
                        Analyze the stock {ticker_input} with the following data:
                        - Current Price: {today_price}
                        - LSTM Predicted Price: {pred_lstm}
                        - GRU Predicted Price: {pred_gru}
                        - Historical Trend: Past 1 year high was {df_clean['Close'].tail(252).max()} and low was {df_clean['Close'].tail(252).min()}
                        
                        Give a concise investment suggestion (Buy/Sell/Hold) with 3 key reasons. 
                        Tone: Professional, data-driven.
                        """
                        
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama3-8b-8192",
                        )
                        st.markdown(chat_completion.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")
            else:
                st.warning("Please enter your Groq API Key in the sidebar.")


    else:
        st.error("Invalid Ticker or No Data Found. Please try another symbol.")
else:
    st.title("Welcome to StockPulse AI")
    st.image("https://images.unsplash.com/photo-1611974717482-58-f996d924c13a?auto=format&fit=crop&q=80&w=1000", use_container_width=True)
    st.markdown("### Enter a ticker in the sidebar to begin your premium analysis.")


