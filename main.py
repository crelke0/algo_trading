import numpy as np
import yfinance as yf
import strategies

def calculate_sharpe_ratio(returns):
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)

    if std_return == 0:
        return 0

    sharpe_ratio = mean_return / std_return * np.sqrt(252)
    return sharpe_ratio

def engine(strategy, market_data, ticker_to_idx, feature_to_idx):
    capital_allocation = [0.0]*len(ticker_to_idx)
    returns = []
    equity_curve = [1.0]
    for d in range(len(market_data) - 2):
        capital_allocation = strategy(capital_allocation, market_data[:d + 1], ticker_to_idx, feature_to_idx)
        uninvested = 1 - sum(capital_allocation)
        if uninvested < -1e-6:
            print(uninvested)
            raise ValueError("Capital allocation values must sum to less than 1.")

        day_return = 0.0
        for i in range(len(capital_allocation)):
            open_price1 = market_data[d + 1, i, feature_to_idx["Open"]]
            open_price2 = market_data[d + 2, i, feature_to_idx["Open"]]
            day_return += capital_allocation[i]*(open_price2 / open_price1 - 1)
        returns.append(day_return)
        equity_curve.append(equity_curve[-1]*(1 + day_return))

    return returns, equity_curve

def monte_carlo_simulation(make_strategy, market_data, ticker_to_idx, feature_to_idx, training_fraction=0.8, num_simulations=1000):
    test_idx = int(len(market_data)*training_fraction)
    
    sharpe_ratios = []
    strategy = make_strategy(market_data, ticker_to_idx, feature_to_idx, training_fraction=training_fraction)
    actual_returns, _ = engine(strategy, market_data[test_idx:], ticker_to_idx, feature_to_idx)
    actual_sharpe = calculate_sharpe_ratio(actual_returns)
    print(f"Actual Sharpe Ratio: {actual_sharpe:.4f}")
    random_wins = 0
    for _ in range(num_simulations):
        permuted_data = permute_data(market_data, feature_to_idx)
        strategy = make_strategy(permuted_data, ticker_to_idx, feature_to_idx, training_fraction=training_fraction)
        returns, _ = engine(strategy, permuted_data[test_idx:], ticker_to_idx, feature_to_idx)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        if sharpe_ratio >= actual_sharpe:
            random_wins += 1
        sharpe_ratios.append(sharpe_ratio)
    p_value = random_wins / num_simulations
    mean = np.mean(sharpe_ratios)
    std = np.std(sharpe_ratios)
    return p_value, mean, std

def permute_data(market_data, feature_to_idx):
    log_data = np.log(market_data)
    r_o = log_data[1:, :, feature_to_idx["Open"]] - log_data[:-1, :, feature_to_idx["Close"]]

    open_prices = log_data[:, :, feature_to_idx["Open"]]
    r_h = log_data[:, :, feature_to_idx["High"]] - open_prices
    r_l = log_data[:, :, feature_to_idx["Low"]] - open_prices
    r_c = log_data[:, :, feature_to_idx["Close"]] - open_prices

    np.random.shuffle(r_o)

    permutation = np.random.permutation(market_data.shape[0])
    r_h = r_h[permutation]
    r_l = r_l[permutation]
    r_c = r_c[permutation]

    permuted_data = np.zeros_like(market_data)
    permuted_data[0, :, feature_to_idx["Open"]] = open_prices[0, :]
    for i in range(market_data.shape[0]):
        if i != 0:
            permuted_data[i, :, feature_to_idx["Open"]] = permuted_data[i - 1, :, feature_to_idx["Close"]] + r_o[i - 1]
        
        todays_open = permuted_data[i, :, feature_to_idx["Open"]]
        permuted_data[i, :, feature_to_idx["High"]] = todays_open + r_h[i]
        permuted_data[i, :, feature_to_idx["Low"]] = todays_open + r_l[i]
        permuted_data[i, :, feature_to_idx["Close"]] = todays_open + r_c[i]

    return np.exp(permuted_data)

def format_yfinance_data(tickers, period):
    market_data = yf.download(tickers, period=period, auto_adjust=True).ffill().bfill()
    market_data = market_data.swaplevel(axis=1).sort_index(axis=1)

    tickers = market_data.columns.levels[0].tolist()
    features = market_data.columns.levels[1].tolist()

    num_days = market_data.shape[0]
    num_tickers = len(tickers)
    num_features = len(features)

    X_flat = market_data.to_numpy()

    X = X_flat.reshape(num_days, num_tickers, num_features)

    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    feature_to_idx = {feature: i for i, feature in enumerate(features)}
    return X, ticker_to_idx, feature_to_idx

tickers = ["AAPL", "MSFT", "GOOG", "NVDA", "META", "AMZN", "TSLA"]

years_held_out = 4

market_data, ticker_to_idx, feature_to_idx = format_yfinance_data(tickers, "8y")[-252*years_held_out:]

training_fraction = 0.8

strategies.make_ml_statarb_strategy(market_data, ticker_to_idx, feature_to_idx, training_fraction=training_fraction)