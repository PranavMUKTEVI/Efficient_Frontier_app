import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions
from fredapi import Fred

# API Key for FRED
fred_api_key = '4b780f2d5edc3b1810ec07fa199cc3ad'
fred = Fred(api_key=fred_api_key)

# Cache data to optimize performance
@st.cache_data
def load_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.isnull().values.any():
            st.warning("Data contains NaN values. They will be dropped.")
            data = data.dropna()
        if data.empty:
            st.error("No valid data retrieved for the given tickers and date range.")
            st.stop()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# Retrieve risk-free rate
def get_risk_free_rate():
    try:
        series_id = 'DTB3'
        data = fred.get_series(series_id)
        return data.iloc[-1] / 100
    except Exception as e:
        st.error(f"Error fetching risk-free rate: {e}")
        return 0.02  # Default to 2% if FRED fails

# Plot Efficient Frontier
def plot_efficient_frontier(tickers, start_date, end_date, risk_free_rate, bounds, allow_short_positions, monte_carlo):
    data = load_data(tickers, start_date, end_date)
    
    try:
        mu = mean_historical_return(data)
        S = CovarianceShrinkage(data).ledoit_wolf()
    except ValueError as e:
        st.error(f"Error in covariance calculation: {e}")
        st.stop()

    ef = EfficientFrontier(mu, S, weight_bounds=bounds)

    try:
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef_max_sharpe = ef.max_sharpe(risk_free_rate=risk_free_rate)
        ret_sharpe, vol_sharpe, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        weights_sharpe = ef.clean_weights()
    except Exception as e:
        st.error(f"Error optimizing Max Sharpe Portfolio: {e}")
        st.stop()

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=bounds)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef_min_vol = ef.min_volatility()
        ret_vol, vol_vol, _ = ef.portfolio_performance()
        weights_vol = ef.clean_weights()
    except Exception as e:
        st.error(f"Error optimizing Min Volatility Portfolio: {e}")
        st.stop()

    fig = go.Figure()

    if monte_carlo:
        n_portfolios = 5000
        results = np.zeros((3, n_portfolios))
        for i in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(len(tickers)))
            portfolio_return = np.dot(weights, mu)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
            results[0, i] = portfolio_return
            results[1, i] = portfolio_stddev
            results[2, i] = sharpe_ratio

        fig.add_trace(go.Scatter(
            x=results[1, :], y=results[0, :],
            mode='markers',
            marker=dict(color=results[2, :], colorscale='Viridis', showscale=True),
            name='Monte Carlo Portfolios'
        ))

    fig.add_trace(go.Scatter(x=[vol_sharpe], y=[ret_sharpe], mode='markers', marker=dict(color='red', size=10), name='Max Sharpe Ratio'))
    fig.add_trace(go.Scatter(x=[vol_vol], y=[ret_vol], mode='markers', marker=dict(color='blue', size=10), name='Min Volatility'))

    fig.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Return')

    st.plotly_chart(fig)

    st.write(f"Max Sharpe Portfolio: Return = {ret_sharpe:.2%}, Volatility = {vol_sharpe:.2%}, Sharpe Ratio = {sharpe_ratio:.2f}")
    st.write(f"Min Volatility Portfolio: Return = {ret_vol:.2%}, Volatility = {vol_vol:.2%}")

    st.write("### Portfolio Allocations")
    st.write(pd.DataFrame.from_dict(weights_sharpe, orient='index', columns=['Max Sharpe Weights']))
    st.write(pd.DataFrame.from_dict(weights_vol, orient='index', columns=['Min Volatility Weights']))

st.title('Efficient Frontier App')
st.sidebar.header('Configuration')

tickers = st.sidebar.text_input('Enter Tickers (comma separated)', '').strip()
time_period = st.sidebar.selectbox('Time Period', ['Year-to-Year', 'Month-to-Month'])
risk_free_rate_option = st.sidebar.selectbox('Risk-Free Rate', ['3-Month Treasury Bill', 'Custom Rate'])
risk_free_rate = get_risk_free_rate() if risk_free_rate_option == '3-Month Treasury Bill' else st.sidebar.number_input('Custom Risk-Free Rate (%)', 0.0, 10.0, 2.0) / 100

if time_period == 'Year-to-Year':
    start_year = st.sidebar.number_input('Start Year', 1978, datetime.now().year, 1978)
    end_year = st.sidebar.number_input('End Year', 1978, datetime.now().year, 2024)
    start_date, end_date = f'{start_year}-01-01', f'{end_year}-12-31'
else:
    start_month = st.sidebar.date_input('Start Date', min_value=datetime(1978, 1, 1))
    end_month = st.sidebar.date_input('End Date', min_value=start_month)
    start_date, end_date = start_month.strftime('%Y-%m-%d'), end_month.strftime('%Y-%m-%d')

tickers_list = [t.strip() for t in tickers.split(',') if t.strip()]

asset_constraints = st.sidebar.radio('Add Asset Constraints?', ['No', 'Yes']) == 'Yes'
allow_short_positions = st.sidebar.radio('Allow Short Positions?', ['No', 'Yes']) == 'Yes'
monte_carlo = st.sidebar.radio('Use Monte Carlo Simulation?', ['No', 'Yes']) == 'Yes'

bounds = (-1, 1) if allow_short_positions else (0, 1)

if st.sidebar.button('Optimize'):
    if not tickers_list:
        st.error('Please enter at least one ticker.')
    else:
        plot_efficient_frontier(tickers_list, start_date, end_date, risk_free_rate, bounds, allow_short_positions, monte_carlo)
def load_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        print(f"Retrieved Data:\n{data.head()}")  # Debugging output
        if data.isnull().values.all():  
            st.warning("Data contains only NaN values. Please check ticker symbols.")
            st.stop()
        if data.empty:
            st.error("No valid data retrieved for the given tickers and date range.")
            st.stop()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    

  

  
