import pyarrow as pa
from argiv._core import (
    compute_greeks as _compute_greeks_impl,
    fit_vol_surface as _fit_vol_surface_impl,
)

__all__ = [
    "compute_greeks",
    "fit_vol_surface",
]


def _ensure_column_types(
    table: pa.Table, type_map: dict[str, pa.DataType]
) -> pa.Table:
    """Cast columns to expected Arrow types where needed."""
    for col_name, expected_type in type_map.items():
        idx = table.schema.get_field_index(col_name)
        if idx == -1:
            continue
        if not table.schema.field(idx).type.equals(expected_type):
            table = table.set_column(
                idx,
                pa.field(col_name, expected_type),
                table.column(idx).cast(expected_type),
            )
    return table


def _check_nulls(table: pa.Table, columns: list[str]) -> None:
    """Raise if any of the specified columns contain null values."""
    for col_name in columns:
        idx = table.schema.get_field_index(col_name)
        if idx == -1:
            continue
        null_count = table.column(idx).null_count
        if null_count > 0:
            raise ValueError(
                f"Column '{col_name}' has {null_count} null values "
                f"(of {len(table)} total rows). "
                f"Fill or drop nulls before processing."
            )


def _check_required(table: pa.Table, columns: list[str]) -> None:
    """Raise if any required columns are missing."""
    missing = [c for c in columns if c not in table.schema.names]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


# -- compute_greeks -----------------------------------------------------------

_GREEKS_CORE = ["option_type", "spot", "strike", "expiry", "rate", "dividend_yield"]

_GREEKS_CORE_TYPES: dict[str, pa.DataType] = {
    "option_type": pa.int32(),
    "spot": pa.float64(),
    "strike": pa.float64(),
    "expiry": pa.float64(),
    "rate": pa.float64(),
    "dividend_yield": pa.float64(),
}

_GREEKS_OPTIONAL_DOUBLE = ["bid_price", "ask_price"]


def compute_greeks(
    table: pa.Table, iv_solver: str = "numerical"
) -> pa.Table:
    """Compute Greeks (and optionally IV) for a table of options.

    If the input has an ``iv`` column, Greeks are computed directly from it
    and ``iv_solver`` is ignored. Otherwise, IV is solved from
    ``market_price`` using the chosen solver.

    Parameters
    ----------
    table : pyarrow.Table
        Required columns: option_type (int32, 1=call/-1=put),
        spot, strike, expiry, rate, dividend_yield (all float64).
        Plus EITHER market_price (to solve for IV) OR iv (to skip the solve).
        If both are present, iv wins.
        Optional (price path only): bid_price, ask_price (float64) for
        bid/ask IV bounds.
    iv_solver : {"numerical", "schadner", "lookup"}, default "numerical"
        IV solver to use when solving from price. Ignored when ``iv`` is
        provided. "numerical" runs Brent root-finding on the Black-Scholes
        price; "schadner" uses the closed-form inverse Gaussian quantile from
        Schadner (arXiv:2604.24480); "lookup" bicubically interpolates a 2D
        Catmull-Rom table indexed by |log(K/F)| and logit(OTM-normalized
        price). Lookup hits ~4-decimal accuracy on σ for inputs in the
        tabulated domain (|k| ≤ 1.5, σ-range covering typical markets) and
        returns NaN outside; the table is built once at module load.

    Returns
    -------
    pyarrow.Table
        Input columns plus delta, gamma, vega, theta, rho. On the price path,
        also iv (and iv_bid, iv_ask if bid/ask provided).
    """
    _check_required(table, _GREEKS_CORE)
    has_iv = "iv" in table.schema.names
    has_price = "market_price" in table.schema.names
    if not has_iv and not has_price:
        raise ValueError(
            "compute_greeks requires either 'iv' or 'market_price' column."
        )

    type_map = dict(_GREEKS_CORE_TYPES)
    if has_iv:
        type_map["iv"] = pa.float64()
    else:
        type_map["market_price"] = pa.float64()
        for col in _GREEKS_OPTIONAL_DOUBLE:
            if col in table.schema.names:
                type_map[col] = pa.float64()

    table = _ensure_column_types(table, type_map)
    _check_nulls(table, list(type_map.keys()))

    return _compute_greeks_impl(table, iv_solver)


# -- fit_vol_surface -----------------------------------------------------------

_SURFACE_REQUIRED = [
    "iv", "option_type", "timestamp", "expiration",
    "spot", "strike", "expiry",
]

_SURFACE_TYPES: dict[str, pa.DataType] = {
    "iv": pa.float64(),
    "option_type": pa.int32(),
    "spot": pa.float64(),
    "strike": pa.float64(),
    "expiry": pa.float64(),
}

_SURFACE_OPTIONAL_DOUBLE = ["rate", "dividend_yield", "iv_bid", "iv_ask"]


def fit_vol_surface(table: pa.Table, delta_pillars=None) -> pa.Table:
    """Fit a vol surface using SVI model on OTM options.

    Parameters
    ----------
    table : pyarrow.Table
        Must contain columns: iv (float64), option_type (int32, 1=call/-1=put),
        timestamp (timestamp), expiration (date32),
        spot, strike, expiry (all float64).
        Optional: rate, dividend_yield (float64, default 0).
        Optional: iv_bid, iv_ask (float64) for bid/ask IV surface bounds.
    delta_pillars : list of float, optional
        Wing delta percentages, must be < 50 (default: [5,10,...,45]).

    Returns
    -------
    pyarrow.Table
        One row per (timestamp, expiration, delta) with columns:
        timestamp, expiration, delta (signed), iv, log_moneyness.
    """
    _check_required(table, _SURFACE_REQUIRED)

    type_map = dict(_SURFACE_TYPES)
    for col in _SURFACE_OPTIONAL_DOUBLE:
        if col in table.schema.names:
            type_map[col] = pa.float64()

    table = _ensure_column_types(table, type_map)
    _check_nulls(table, list(type_map.keys()))

    return _fit_vol_surface_impl(table, delta_pillars)
