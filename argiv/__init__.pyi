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