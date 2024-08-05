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


fred_api_key = '4b780f2d5edc3b1810ec07fa199cc3ad'
fred = Fred(api_key=fred_api_key)

@st.cache_data
def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def get_risk_free_rate():
    series_id = 'DTB3'  
    data = fred.get_series(series_id)
    return data.iloc[-1] / 100

def plot_efficient_frontier(tickers, start_date, end_date, risk_free_rate, bounds, allow_short_positions, monte_carlo):
    data = load_data(tickers, start_date, end_date)
    mu = mean_historical_return(data)
    S = CovarianceShrinkage(data).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=bounds)

    # Maximum Sharpe Ratio Portfolio
    try:
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)  
        ef_max_sharpe = ef.max_sharpe(risk_free_rate=risk_free_rate)
        ret_sharpe, vol_sharpe, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        weights_sharpe = ef.clean_weights()
    except KeyError as e:
        st.error(f"KeyError: {e}")
        st.stop()

    # Minimum Volatility Portfolio
    try:
        ef = EfficientFrontier(mu, S, weight_bounds=bounds)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Regularization
        ef_min_vol = ef.min_volatility()
        ret_vol, vol_vol, _ = ef.portfolio_performance()
        weights_vol = ef.clean_weights()
    except KeyError as e:
        st.error(f"KeyError: {e}")
        st.stop()

    # Plot Efficient Frontier
    fig = go.Figure()

    if monte_carlo:
        n_portfolios = 5000
        results = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(len(tickers)))
            portfolio_return = np.dot(weights, mu)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
            results[0,i] = portfolio_return
            results[1,i] = portfolio_stddev
            results[2,i] = sharpe_ratio

        fig.add_trace(go.Scatter(
            x=results[1,:], y=results[0,:],
            mode='markers',
            marker=dict(color=results[2,:], colorscale='Viridis', showscale=True),
            name='Monte Carlo Portfolios'
        ))

    target_returns = np.linspace(ret_vol, ret_sharpe, 100)
    efficient_portfolios = []
    for r in target_returns:
        ef = EfficientFrontier(mu, S, weight_bounds=bounds)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Regularization
        ef.efficient_return(target_return=r)
        efficient_portfolios.append(ef.portfolio_performance(risk_free_rate=risk_free_rate))

    returns = [portfolio[0] for portfolio in efficient_portfolios]
    volatilities = [portfolio[1] for portfolio in efficient_portfolios]

    fig.add_trace(go.Scatter(x=volatilities, y=returns, mode='lines', name='Efficient Frontier'))
    fig.add_trace(go.Scatter(x=[vol_sharpe], y=[ret_sharpe], mode='markers', marker=dict(color='red', size=10), name='Max Sharpe Ratio'))
    fig.add_trace(go.Scatter(x=[vol_vol], y=[ret_vol], mode='markers', marker=dict(color='blue', size=10), name='Min Volatility'))

    fig.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Return')

    st.plotly_chart(fig)

    st.write(f"Maximum Sharpe Ratio Portfolio: Return = {ret_sharpe:.2%}, Volatility = {vol_sharpe:.2%}, Sharpe Ratio = {sharpe_ratio:.2f}")
    st.write(f"Minimum Volatility Portfolio: Return = {ret_vol:.2%}, Volatility = {vol_vol:.2%}")

    st.write("### Allocation for Maximum Sharpe Ratio Portfolio")
    st.write(pd.DataFrame.from_dict(weights_sharpe, orient='index', columns=['Weight']))

    st.write("### Allocation for Minimum Volatility Portfolio")
    st.write(pd.DataFrame.from_dict(weights_vol, orient='index', columns=['Weight']))

    st.write("### Asset Correlations")
    correlation_matrix = data.pct_change().corr()
    st.write(correlation_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))


    def format_weights(weights):
        return {k: (v if v >= 0 else v) for k, v in weights.items()}

    st.write("### Maximum Sharpe Ratio Portfolio Allocation")
    fig_sharpe = go.Figure(data=[go.Pie(labels=list(weights_sharpe.keys()), values=list(format_weights(weights_sharpe).values()))])
    st.plotly_chart(fig_sharpe)

    st.write("### Minimum Volatility Portfolio Allocation")
    fig_vol = go.Figure(data=[go.Pie(labels=list(weights_vol.keys()), values=list(format_weights(weights_vol).values()))])
    st.plotly_chart(fig_vol)

    st.write("### Individual Asset Statistics")
    asset_stats = []
    for i, ticker in enumerate(tickers):
        try:
            exp_return = mu[ticker]
            std_dev = np.sqrt(S.loc[ticker, ticker])
            sharpe_ratio = exp_return / std_dev
            if asset_constraints:
                min_weight, max_weight = bounds[i]
            else:
                min_weight, max_weight = 0, 1
            asset_stats.append([ticker, exp_return, std_dev, sharpe_ratio, min_weight, max_weight])
        except KeyError as e:
            st.error(f"KeyError: {e}")
            st.stop()
    
    asset_stats_df = pd.DataFrame(asset_stats, columns=['Ticker', 'Expected Return', 'Standard Deviation', 'Sharpe Ratio', 'Min Weight', 'Max Weight'])
    st.write(asset_stats_df)

st.title('Efficient Frontier App')
st.sidebar.header('Efficient Frontier Configuration')

tickers = st.sidebar.text_input('Tickers (comma separated)', '')
time_period = st.sidebar.selectbox('Time Period', ['Year-to-Year', 'Month-to-Month'])
risk_free_rate_option = st.sidebar.selectbox('Risk-Free Rate', ['Use 3-Month Treasury Bill Rate', 'Custom Rate'])
if risk_free_rate_option == 'Use 3-Month Treasury Bill Rate':
    risk_free_rate = get_risk_free_rate()
else:
    risk_free_rate = st.sidebar.number_input('Custom Risk-Free Rate (in %)', min_value=0.0, max_value=100.0, value=0.0) / 100

st.sidebar.markdown("Note: Enter the risk-free rate as a percentage (e.g., 5.12 for 5.12%).")
asset_constraints = st.sidebar.radio('Add Asset Constraints?', ['No', 'Yes']) == 'Yes'
allow_short_positions = st.sidebar.radio('Allow Short Positions?', ['No', 'Yes']) == 'Yes'
monte_carlo = st.sidebar.radio('Use Monte Carlo Simulation?', ['No', 'Yes']) == 'Yes'

if time_period == 'Year-to-Year':
    start_year = st.sidebar.number_input('Start Year', min_value=1978, max_value=datetime.now().year, value=1978)
    end_year = st.sidebar.number_input('End Year', min_value=1978, max_value=datetime.now().year, value=2024)
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
else:
    start_month = st.sidebar.date_input('Start Month', min_value=datetime(1978, 1, 1), max_value=datetime.now(), value=datetime(1978, 1, 1))
    end_month = st.sidebar.date_input('End Month', min_value=datetime(1978, 1, 1), max_value=datetime.now(), value=datetime(2024, 1, 1))
    start_date = start_month.strftime('%Y-%m-%d')
    end_date = end_month.strftime('%Y-%m-%d')
tickers_list = tickers.split(',')

if asset_constraints:
    bounds = []
    for ticker in tickers_list:
        min_weight = st.sidebar.number_input(f"Min Weight for {ticker}", -1.0, 1.0, 0.0)
        max_weight = st.sidebar.number_input(f"Max Weight for {ticker}", -1.0, 1.0, 1.0)
        bounds.append((min_weight, max_weight))
    
    total_min_weight = sum([b[0] for b in bounds])
    total_max_weight = sum([b[1] for b in bounds])
    
    if not np.isclose(total_min_weight, 1.0, atol=0.01) or not np.isclose(total_max_weight, 1.0, atol=0.01):
        st.sidebar.error("Please ensure the sum of the constraints for minimum and maximum weights are approximately equal to 100%")
else:
    bounds = (-1, 1) if allow_short_positions else (0, 1)
    

if st.sidebar.button('Optimize'):
    if not tickers_list:
        st.error('Please enter at least one ticker symbol.')
    elif asset_constraints and (not np.isclose(total_min_weight, 1.0, atol=0.01) or not np.isclose(total_max_weight, 1.0, atol=0.01)):
        st.error("Please ensure the sum of the constraints for minimum and maximum weights are approximately equal to 100%")
    else:
        plot_efficient_frontier(tickers_list, start_date, end_date, risk_free_rate, bounds, allow_short_positions, monte_carlo)

