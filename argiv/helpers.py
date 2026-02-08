import math
import numpy as np
from scipy.stats import norm
import pyarrow as pa

def _bs_call_price(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, r, q, sigma):
    call = _bs_call_price(S, K, T, r, q, sigma)
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)

def generate_dataset(n):
    """Generate a realistic mixed options dataset with known-valid IV solutions."""
    rng = np.random.default_rng(42)

    option_type = rng.choice([1, -1], size=n).astype(np.int32)
    spot = np.full(n, 100.0)
    strike = rng.uniform(80.0, 120.0, size=n)
    expiry = rng.uniform(0.1, 2.0, size=n)
    rate = np.full(n, 0.05)
    dividend_yield = np.full(n, 0.01)
    true_sigma = rng.uniform(0.1, 0.5, size=n)

    # Compute market prices from known sigmas via BS formula
    market_price = np.empty(n)
    for i in range(n):
        if option_type[i] == 1:
            market_price[i] = _bs_call_price(
                spot[i], strike[i], expiry[i], rate[i], dividend_yield[i], true_sigma[i]
            )
        else:
            market_price[i] = _bs_put_price(
                spot[i], strike[i], expiry[i], rate[i], dividend_yield[i], true_sigma[i]
            )

    return pa.table({
        "option_type": pa.array(option_type, type=pa.int32()),
        "spot": pa.array(spot, type=pa.float64()),
        "strike": pa.array(strike, type=pa.float64()),
        "expiry": pa.array(expiry, type=pa.float64()),
        "rate": pa.array(rate, type=pa.float64()),
        "dividend_yield": pa.array(dividend_yield, type=pa.float64()),
        "market_price": pa.array(market_price, type=pa.float64()),
    })
