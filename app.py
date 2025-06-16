import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.data_utils import load_data, make_dataset
from src.model_utils import build_svr_model, build_ridge_model

FEATURES = [
    "Close",
    "Log Return",
    "MA_5", "EMA_5",
    "Vol_5", "BollingerWidth_5",
    "Volume", "VolRatio_10",
    "VWAP",
    "RSI_14",
]

# ─── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)  # cache 60 seconds so repeated clicks don’t re-download
def _load_data(ticker: str, period: str = "5d", interval: str = "1m"):
    return load_data(ticker, period, interval)

# ─── Sidebar controls ───────────────────────────────────────────────────────
st.sidebar.header("Parameters")
model_choice = st.sidebar.pills("Regression Model", options=["SVR", "Ridge"], default="SVR", selection_mode="single")
ticker = st.sidebar.text_input("Stock Ticker", value="^IXIC")
features = st.sidebar.multiselect("Feature Selection", options=FEATURES, default=["Close"])

model_parameters = st.sidebar.container()
with model_parameters:
    if model_choice == "SVR":
        kernel = st.sidebar.selectbox("SVR kernel", options=["linear", "rbf", "poly"])
        if kernel == "linear":
            C = st.slider("C (regularization)", 0.01, 100.0, 1.0)
            eps = st.slider("Epsilon (sensitivity)",0.01, 1.0, 0.1 )
            hyperparams = {"kernel": "linear", "C": C, "epsilon": eps}

        elif kernel == "rbf":
            C     = st.slider("C (regularization)", 0.01, 100.0, 1.0)
            eps = st.slider("Epsilon (sensitivity)",0.01, 1.0, 0.1 )
            # a mix of “scale”, “auto” and some numeric γ’s
            gamma = st.select_slider(
                "Gamma (kernel width)",
                options=["scale", "auto"] + list(10 ** np.linspace(-4, 0, 9))
            )
            hyperparams = {
                "kernel": "rbf",
                "C": C,
                "gamma": gamma,
                "epsilon": eps
            }

        elif kernel == "poly":
            C      = st.slider("C (regularization)", 0.01, 100.0, 1.0)
            eps = st.slider("Epsilon (sensitivity)",0.01, 1.0, 0.1 )
            degree = st.slider("degree", 2, 5, 3)
            gamma  = st.select_slider(
                "γ",
                options=["scale", "auto"] + list(10 ** np.linspace(-4, 0, 9))
            )
            coef0  = st.slider("coef0", 0.0, 1.0, 0.0)
            hyperparams = {
                "kernel": "poly",
                "C": C, "degree": degree,
                "gamma": gamma, "coef0": coef0,
                "epsilon": eps
            }
    elif model_choice == "Ridge":
        alpha = st.slider("Alpha (regularization)", 0.0, 100.0, 1.0)
        hyperparams = {"alpha": alpha}

k = st.sidebar.slider("Window size (k)", min_value=1, max_value=30, value=5)
horizon = st.sidebar.slider("Prediction Horizon in Minutes", 1, 15, 1)
test_pct = st.sidebar.slider("Test split %", 5, 50, 20)
run_it = st.sidebar.button("Run & Train")
# ─── Main ───────────────────────────────────────────────────────────────────
st.title("Why Regression on Chart Data Fundamentally Fails")

if run_it:
    with st.spinner("Downloading data…"):
        df = _load_data(ticker)

    st.write(f"Loaded {len(df)} closing prices for `{ticker}`.")
    # Build supervised set
    X, y = make_dataset(df, features, k, horizon)
    split = int(len(X) * (1 - test_pct / 100))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_choice == "SVR":
        model = build_svr_model(hyperparams)
    elif model_choice == "Ridge":
        model = build_ridge_model(hyperparams)

    model.fit(X_train, y_train)
    #---------------------------------------------
    timestamps = df.index[k+horizon-1:]        # the timestamp for each row in X/y
    train_ts, test_ts = timestamps[:split], timestamps[split:]
    print("Train spans:", train_ts.min(), "→", train_ts.max())
    print("Test  spans:", test_ts.min(),  "→", test_ts.max())
    #---------------------------------------------

    # Predict & inverse‐scale
    y_pred = model.predict(X_test)

    # Metrics
    correct = []
    # start at i = h, since y_test[0] is the price at t = k+h-1,
    # and we need a price h steps earlier in the test-series
    for i in range(horizon, len(y_test)):
    # actual change over horizon minutes
        actual_change = y_test[i] - y_test[i - horizon]
        # predicted change relative to the same baseline
        pred_change   = y_pred[i] - y_test[i - horizon]
        correct.append(1 if actual_change * pred_change > 0 else 0)

    directional_accuracy = sum(correct) / len(correct)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    st.metric("Test MSE", f"{mse:.2f}")
    st.metric("Test R²",  f"{r2:.3f}")
    st.metric("Directional Accuracy", f"{directional_accuracy:.2f}")

    X_plot = df.index[-len(y_test):]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_plot, y=y_test,
        mode='lines',
        name='Actual Price',
        line=dict(color='lightblue')
    ))

    fig.add_trace(go.Scatter(
        x=X_plot, y=y_pred,
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', dash='dot')
    ))

    fig.update_layout(
        title=f'{model_choice} Point Prediction on Closing Prices with {horizon} Minute Horizon',
        xaxis_title='Time (UTC)',
        yaxis_title='Price',
        xaxis=dict(
            tickformat='%H:%M',
            rangeslider=dict(visible=True),
            tickangle=45,
            showgrid=True
        ),
        yaxis=dict(showgrid=True),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 13.5], pattern="hour")
        ],
        tickformat='%H:%M\n%b %d',
        tickangle=45
        )
    
    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(r"""
## 1. Sliding-Window Forecasting

We turn a time-series $\{P_t\}$ into a **supervised** dataset by taking:

- **Inputs** $X^{(i)} = \bigl[P_{t-k}, P_{t-k+1}, \dots, P_{t-1}\bigr]$
- **Target** $y^{(i)} = P_{t+h}$

with two parameters:

- \($k$\): window size (how many past points)  
- \($h$\): forecast horizon (how many steps ahead)

This lets us apply any regression model to predict \($h$\)-step‐ahead prices.

---

## 2. Model Overviews & Assumptions

### 2.1. Linear Regression

We assume a model of the form
$$
  y = X\beta + \varepsilon,
  \quad
  \varepsilon \sim \mathcal{N}(0, \sigma^2 I).
$$
Equivalently,
$$
  \mathbb{E}[\,y \mid X\,] = X\beta,
  \quad
  \mathrm{Var}(y\mid X) = \sigma^2.
$$

- **Key assumption:** Conditional on the features \($X$\), the target \($y$\) is ​Gaussian with constant variance.  
- **Fitting** is done by Ordinary Least Squares → $\left(\hat\beta = \arg\min_\beta \sum (y - X\beta)^2\right)$.

### 2.2. Support-Vector Regression (SVR)

SVR solves
$$
  \min_{w,b}\;\tfrac12\lVert w\rVert^2 
  + C\,\sum_{i=1}^n \bigl[\lvert y_i - (w^\top\phi(x_i)+b)\rvert - \varepsilon\bigr]_+,
$$
where:

- $\phi(\cdot)$ maps inputs into a (possibly high-dim) feature space via a kernel \($K(x,x')$\).  
- \($\varepsilon$\) defines an “insensitive tube”: errors smaller than \($\varepsilon$\) incur no penalty.  
- **No explicit probabilistic assumption** on \($y\mid X$\); it’s a margin-based loss.

---

## 3. Why They Automatically Fail on Raw Prices

1. **Martingale structure**  
   $$
     P_t = P_{t-1} + \epsilon_t,\quad \epsilon_t\perp\!\!\!\perp \mathcal{F}_{t-1},\;\mathbb{E}[\epsilon_t]=0.
   $$
   Neither Linear Regression nor SVR can learn anything beyond "$\hat P_{t+h}\approx P_t$", so they simply **extrapolate the last trend**.

2. **Violation of Gaussian/Stationarity**  
   - The true **conditional distribution** of \($P_{t+h}\mid X$\) is not homoskedastic Gaussian around a linear mean.  
   - **Heteroskedasticity** (volatility clustering) and **jumps** break the \($\mathcal{N}(0,\sigma^2)$\) error assumption.

3. **Deterministic inputs**  
   All “technical” features \($\phi(X)$\) are deterministic functions of past prices.  Under the martingale, past contains **no information** on the innovation \($\epsilon_{t+h}$\).  

4. **High \($R^2$\) ≠ predictive power**  
   Because minute‐bar prices move little over short \($h$\), a naive “\($\hat P_{t+h}=P_t$\)” yields $R^2\approx0.8–0.9$.  But **directional accuracy** $\approx 50\%$ confirms there is no genuine forecast skill.

---

**Bottom line:**  
Regression on raw price‐based sliding windows *necessarily* reduces to lagging the last observed value, so it cannot beat a naïve forecast or coin-flip on direction.  
""")

st.markdown(r"""
## Outlook: Possible Improvements

- **Forecast Returns or Direction**  
  Instead of raw prices, predict $r_{t+h} = (P_{t+h}-P_t)/P_t$ or classify “up” vs. “down.”

- **Add Volatility & Volume Signals**  
  Use realized intraday volatility, high–low ranges, or volume spikes to capture market activity.

- **Introduce Exogenous Data**  
  Bring in order-flow metrics or sentiment/news indicators for fresh information beyond prices.

- **Explore Non-Linear Models**  
  Try tree-based learners (RandomForest, XGBoost) to uncover complex feature interactions.

""")