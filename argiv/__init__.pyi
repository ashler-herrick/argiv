import pyarrow as pa

def compute_greeks(
    table: pa.Table, iv_solver: str = "numerical"
) -> pa.Table:
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

def compute_greeks_from_iv(table: pa.Table) -> pa.Table:
    """
    Compute Greeks from pre-computed implied volatility (no IV solve).

    Args:
        table (pa.Table): Input Arrow Table with columns:
            - option_type (int32): 1 for Call, -1 for Put
            - spot (float64): Current price of the underlying asset
            - strike (float64): Strike price of the option
            - expiry (float64): Time to expiry in years
            - rate (float64): Risk-free interest rate
            - dividend_yield (float64): Dividend yield of the underlying asset
            - iv (float64): Implied volatility
    Returns:
        pa.Table: Input columns plus:
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
    iv_solver: str = "numerical",
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

def constant_maturity_surface(
    surface: pa.Table,
    tenors: list[float] | None = None,
    decomposition: pa.Table | None = None,
) -> pa.Table:
    """
    Interpolate a per-expiration surface to fixed tenors via variance interpolation.

    Args:
        surface (pa.Table): Output of fit_vol_surface with columns:
            timestamp, expiry, delta, iv.
            Optional: log_moneyness, iv_bid, iv_ask.
        tenors (list[float] | None): Target tenors in years
            (default: 7d, 14d, 30d, 60d, 90d, 180d, 1y).
        decomposition (pa.Table | None): Output of decompose_variance.
            If provided, event-aware interpolation is used.

    Returns:
        pa.Table: One row per (timestamp, tenor, delta) with columns:
            - timestamp: Observation time
            - tenor (float64): Target tenor in years
            - delta (float64): Signed delta
            - iv (float64): Interpolated implied volatility
            - log_moneyness (float64): if present in input
            - iv_bid, iv_ask (float64): if present in input
    """
    ...

def decompose_variance(
    surface: pa.Table,
    events: pa.Table,
) -> pa.Table:
    """
    Decompose implied variance into diffusive and event components via NNLS.

    Args:
        surface (pa.Table): Output of fit_vol_surface with columns:
            timestamp, expiry, delta, iv.
        events (pa.Table): Event calendar with columns:
            - event_date (date32 or timestamp): Event dates
            Optional: event_type (utf8, e.g. "earnings", "fomc").

    Returns:
        pa.Table: One row per (timestamp, delta, event) with columns:
            - timestamp: Observation time
            - delta (float64): Signed delta
            - sigma_d_sq (float64): Diffusive variance rate
            - event_expiry (float64): Time to event in years (null if no events)
            - event_var (float64): Event variance contribution
            - event_type (utf8): Event type label
    """
    ...

def greeks_at_pillars(enriched: pa.Table, surface: pa.Table) -> pa.Table:
    """
    Compute Greeks at delta pillars by joining enriched and surface tables.

    Args:
        enriched (pa.Table): Output of compute_greeks with columns:
            timestamp, expiration, spot, expiry, rate, dividend_yield.
        surface (pa.Table): Output of fit_vol_surface with columns:
            timestamp, expiration, delta, iv, log_moneyness.

    Returns:
        pa.Table: Surface columns plus:
            - strike (float64): Strike at each delta pillar
            - gamma (float64): Option gamma
            - vega (float64): Option vega
            - theta (float64): Option theta
            - rho (float64): Option rho
    """
    ...
