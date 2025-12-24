import numpy as np

def diversify(capital_allocation, market_data, ticker_to_idx, feature_to_idx):
    capital_allocation = np.full_like(capital_allocation, 1.0 / len(capital_allocation))
    return capital_allocation