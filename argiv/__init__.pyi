import pyarrow as pa

def compute_greeks(table: pa.Table) -> pa.Table:
    """
    Computes the IV and first order Greeks plus Gamma for a pyarrow Table.

    Args:
        table (pa.Table): Input Arrow Table with columns:
            - option_type (int32): 1 for Call, -1 for Put
            - spot (float64): Current price of the underlying asset
            - strike (float64): Strike price of the option
            - expiry (float64): Time to expiry in years
            - rate (float64): Risk-free interest rate
            - dividend_yield (float64): Dividend yield of the underlying asset
            - market_price (float64): Market price of the option
    Returns:
        pa.Table: Output Arrow Table with additional columns:
            - iv (float64): Implied volatility
            - delta (float64): Option delta
            - gamma (float64): Option gamma
            - vega (float64): Option vega
            - theta (float64): Option theta
            - rho (float64): Option rho

    """
    ...

def fit_vol_surface(
    table: pa.Table,
    delta_pillars: list[float] | None = None,
) -> pa.Table:
    """
    Fit a vol surface via delta-space cubic spline interpolation.

    Args:
        table (pa.Table): Input Arrow Table with columns:
            - iv (float64): Implied volatility
            - delta (float64): Option delta
            - option_type (int32): 1 for Call, -1 for Put
            - timestamp (timestamp): Observation time
            - expiration (date32): Option expiration date
            Optional: spot (float64), strike (float64) for log_moneyness.
        delta_pillars (list[float] | None): Absolute delta percentages for
            pillar points (default: [5, 10, 15, 20, 25, 30, 35, 40, 45]).

    Returns:
        pa.Table: One row per (timestamp, expiration, delta) with columns:
            - timestamp: Observation time
            - expiration: Option expiration date
            - delta (float64): Signed delta (negative for puts, positive for calls, 0.50 for ATM)
            - iv (float64): Interpolated implied volatility
            - log_moneyness (float64): log(K/S), null if spot/strike not in input
    """
    ...

def compute_fit_vol_surface(
    table: pa.Table,
    delta_pillars: list[float] | None = None,
) -> pa.Table:
    """
    Compute Greeks and fit a vol surface in one step.

    Args:
        table (pa.Table): Input Arrow Table with columns:
            - option_type (int32): 1 for Call, -1 for Put
            - spot (float64): Current price of the underlying asset
            - strike (float64): Strike price of the option
            - expiry (float64): Time to expiry in years
            - rate (float64): Risk-free interest rate
            - dividend_yield (float64): Dividend yield of the underlying asset
            - market_price (float64): Market price of the option
            - timestamp (timestamp): Observation time
            - expiration (date32): Option expiration date
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
