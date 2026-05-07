import pyarrow as pa

def compute_greeks(
    table: pa.Table, iv_solver: str = "numerical"
) -> pa.Table:
    """
    Compute Greeks (and optionally IV) for a pyarrow Table of options.

    If the input has an ``iv`` column, Greeks are computed directly from it
    and ``iv_solver`` is ignored. Otherwise IV is solved from ``market_price``.

    Args:
        table (pa.Table): Input Arrow Table with columns:
            - option_type (int32): 1 for Call, -1 for Put
            - spot (float64): Current price of the underlying asset
            - strike (float64): Strike price of the option
            - expiry (float64): Time to expiry in years
            - rate (float64): Risk-free interest rate
            - dividend_yield (float64): Dividend yield of the underlying asset
            - market_price (float64): Option price (price path) — OR
            - iv (float64): Pre-computed implied volatility (skip solve)
        iv_solver (str): "numerical", "schadner", or "lookup". Ignored when
            ``iv`` is supplied.

    Returns:
        pa.Table: Input columns plus:
            - delta, gamma, vega, theta, rho (float64)
            - iv (float64): only on the price path
            - iv_bid, iv_ask (float64): only when bid_price/ask_price provided
    """
    ...

def fit_vol_surface(
    table: pa.Table,
    delta_pillars: list[float] | None = None,
) -> pa.Table:
    """
    Fit a vol surface via SVI on OTM options.

    Args:
        table (pa.Table): Input Arrow Table with columns:
            - iv (float64): Implied volatility
            - option_type (int32): 1 for Call, -1 for Put
            - timestamp (timestamp): Observation time
            - expiration (date32): Option expiration date
            - spot, strike, expiry (float64)
            Optional: rate, dividend_yield, iv_bid, iv_ask (float64).
        delta_pillars (list[float] | None): Absolute delta percentages for
            pillar points (default: [5, 10, 15, 20, 25, 30, 35, 40, 45]).

    Returns:
        pa.Table: One row per (timestamp, expiration, delta) with columns:
            - timestamp: Observation time
            - expiration: Option expiration date
            - delta (float64): Signed delta (negative for puts, positive for calls, 0.50 for ATM)
            - iv (float64): Interpolated implied volatility
            - log_moneyness (float64): log(K/S)
    """
    ...
