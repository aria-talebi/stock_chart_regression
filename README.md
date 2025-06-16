# Why Regression Doesn’t Work on Chart Data

A Streamlit demo app illustrating why sliding-window regression (linear or SVR) on raw intraday price data merely **lags** the last observed price and fails to deliver genuine predictive power.

## 🔗 Live Demo

Try the interactive [Demo](https://stockchartregression-lt4xorybppvvfj3x65rrjz.streamlit.app/)

## 📖 Overview

This project shows:

1. **Sliding-Window Setup**  
   - Inputs: last *k* closing prices  
   - Target: price *h* minutes ahead  

2. **Baseline Models**  
   - **Ridge Regression**  
   - **Support Vector Regression** (RBF & linear kernels)
   - Inputs and Outputs are normalized using sklearn.preprocessing.RobustScaler

3. **Core Finding**  
   - Both regressors simply **extrapolate** the last price  
   - High \(R^2\) (> 0.8) but **directional accuracy ≈ 50%**  
   - Demonstrates the “random-walk” nature of intraday prices

4. **Theory Section**  
   - Martingale property of prices  
   - Model assumptions vs. reality  
   - Why technical features add no new information

5. **Interactive Controls**  
   - Choose ticker, window size (*k*), forecast horizon (*h*)  
   - Select regression method & kernel hyperparameters  
   - Toggle engineered features