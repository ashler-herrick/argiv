import pyarrow as pa
from argiv._core import (
    compute_greeks as _compute_greeks_impl,
    fit_vol_surface as _fit_vol_surface_impl,
    compute_fit_vol_surface as _compute_fit_vol_surface_impl,
)

__all__ = ["compute_greeks", "fit_vol_surface", "compute_fit_vol_surface"]


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

_GREEKS_REQUIRED = [
    "option_type", "spot", "strike", "expiry",
    "rate", "dividend_yield", "market_price",
]

_GREEKS_TYPES: dict[str, pa.DataType] = {
    "option_type": pa.int32(),
    "spot": pa.float64(),
    "strike": pa.float64(),
    "expiry": pa.float64(),
    "rate": pa.float64(),
    "dividend_yield": pa.float64(),
    "market_price": pa.float64(),
}

_GREEKS_OPTIONAL_DOUBLE = ["bid_price", "ask_price"]


def compute_greeks(table: pa.Table) -> pa.Table:
    """Compute implied volatility and Greeks for a table of options.

    Parameters
    ----------
    table : pyarrow.Table
        Must contain columns: option_type (int32, 1=call/-1=put),
        spot, strike, expiry, rate, dividend_yield, market_price (all float64).
        Optional: bid_price, ask_price (float64) for bid/ask IV bounds.

    Returns
    -------
    pyarrow.Table
        Input columns plus: iv, delta, gamma, vega, theta, rho.
        If bid_price and ask_price are present: also iv_bid, iv_ask.
    """
    _check_required(table, _GREEKS_REQUIRED)

    type_map = dict(_GREEKS_TYPES)
    for col in _GREEKS_OPTIONAL_DOUBLE:
        if col in table.schema.names:
            type_map[col] = pa.float64()

    table = _ensure_column_types(table, type_map)
    _check_nulls(table, list(type_map.keys()))

    return _compute_greeks_impl(table)


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


# -- compute_fit_vol_surface ---------------------------------------------------

_COMBO_REQUIRED = _GREEKS_REQUIRED + ["timestamp", "expiration"]


def compute_fit_vol_surface(table: pa.Table, delta_pillars=None) -> pa.Table:
    """Compute Greeks and fit a vol surface in one step.

    Parameters
    ----------
    table : pyarrow.Table
        Must contain columns: option_type (int32, 1=call/-1=put),
        spot, strike, expiry, rate, dividend_yield, market_price (all float64),
        timestamp (timestamp), expiration (date32).
        Optional: bid_price, ask_price (float64) for bid/ask IV bounds.
    delta_pillars : list of float, optional
        Wing delta percentages, must be < 50 (default: [5,10,...,45]).

    Returns
    -------
    pyarrow.Table
        One row per (timestamp, expiration, delta) with columns:
        timestamp, expiration, delta (signed), iv, log_moneyness.
    """
    _check_required(table, _COMBO_REQUIRED)

    type_map = dict(_GREEKS_TYPES)
    for col in _GREEKS_OPTIONAL_DOUBLE:
        if col in table.schema.names:
            type_map[col] = pa.float64()

    table = _ensure_column_types(table, type_map)
    _check_nulls(table, list(type_map.keys()))

    return _compute_fit_vol_surface_impl(table, delta_pillars)
