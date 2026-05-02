import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from argiv._core import (
    compute_greeks as _compute_greeks_impl,
    compute_greeks_from_iv as _compute_greeks_from_iv_impl,
    fit_vol_surface as _fit_vol_surface_impl,
    compute_fit_vol_surface as _compute_fit_vol_surface_impl,
)

__all__ = [
    "compute_greeks",
    "compute_greeks_from_iv",
    "fit_vol_surface",
    "compute_fit_vol_surface",
    "greeks_at_pillars",
    "constant_maturity_surface",
    "decompose_variance",
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


def compute_greeks(
    table: pa.Table, iv_solver: str = "numerical"
) -> pa.Table:
    """Compute implied volatility and Greeks for a table of options.

    Parameters
    ----------
    table : pyarrow.Table
        Must contain columns: option_type (int32, 1=call/-1=put),
        spot, strike, expiry, rate, dividend_yield, market_price (all float64).
        Optional: bid_price, ask_price (float64) for bid/ask IV bounds.
    iv_solver : {"numerical", "schadner", "lookup"}, default "numerical"
        IV solver to use. "numerical" runs Brent root-finding on the
        Black-Scholes price; "schadner" uses the closed-form inverse
        Gaussian quantile from Schadner (arXiv:2604.24480); "lookup"
        bicubically interpolates a 2D Catmull-Rom table indexed by
        |log(K/F)| and logit(OTM-normalized price). Lookup hits
        ~4-decimal accuracy on σ for inputs in the tabulated domain
        (|k| ≤ 1.5, σ-range covering typical markets) and returns NaN
        outside; the table is built once at module load.

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

    return _compute_greeks_impl(table, iv_solver)


# -- compute_greeks_from_iv ----------------------------------------------------

_GREEKS_IV_REQUIRED = [
    "option_type", "spot", "strike", "expiry",
    "rate", "dividend_yield", "iv",
]

_GREEKS_IV_TYPES: dict[str, pa.DataType] = {
    "option_type": pa.int32(),
    "spot": pa.float64(),
    "strike": pa.float64(),
    "expiry": pa.float64(),
    "rate": pa.float64(),
    "dividend_yield": pa.float64(),
    "iv": pa.float64(),
}


def compute_greeks_from_iv(table: pa.Table) -> pa.Table:
    """Compute Greeks from pre-computed implied volatility (no IV solve).

    Parameters
    ----------
    table : pyarrow.Table
        Must contain columns: option_type (int32, 1=call/-1=put),
        spot, strike, expiry, rate, dividend_yield, iv (all float64).

    Returns
    -------
    pyarrow.Table
        Input columns plus: delta, gamma, vega, theta, rho.
    """
    _check_required(table, _GREEKS_IV_REQUIRED)
    table = _ensure_column_types(table, _GREEKS_IV_TYPES)
    _check_nulls(table, list(_GREEKS_IV_TYPES.keys()))
    return _compute_greeks_from_iv_impl(table)


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


def compute_fit_vol_surface(
    table: pa.Table, delta_pillars=None, iv_solver: str = "numerical"
) -> pa.Table:
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

    return _compute_fit_vol_surface_impl(table, delta_pillars, iv_solver)


# -- greeks_at_pillars ---------------------------------------------------------


def greeks_at_pillars(enriched: pa.Table, surface: pa.Table) -> pa.Table:
    """Compute Greeks at delta pillars by joining enriched and surface tables.

    Parameters
    ----------
    enriched : pyarrow.Table
        Output of ``compute_greeks`` — must contain: timestamp, expiration,
        spot, expiry, rate, dividend_yield.
    surface : pyarrow.Table
        Output of ``fit_vol_surface`` — must contain: timestamp, expiration,
        delta, iv, log_moneyness.

    Returns
    -------
    pyarrow.Table
        Surface columns plus: strike, gamma, vega, theta, rho.
    """
    # --- Extract one row per (timestamp, expiration) from enriched ---
    _check_required(enriched, ["timestamp", "expiration", "spot", "expiry",
                               "rate", "dividend_yield"])
    _check_required(surface, ["timestamp", "expiration", "delta", "iv",
                              "log_moneyness"])

    # Only join columns that aren't already in the surface
    join_cols = ["spot", "rate", "dividend_yield"]
    if "expiry" not in surface.schema.names:
        join_cols.append("expiry")
    keep_cols = ["timestamp", "expiration"] + join_cols
    params_df = enriched.select(
        [c for c in keep_cols if c in enriched.schema.names]
    ).to_pandas()
    params_df = params_df.drop_duplicates(subset=["timestamp", "expiration"])

    # --- Join surface onto params ---
    surf_df = surface.to_pandas()
    merged = surf_df.merge(
        params_df,
        on=["timestamp", "expiration"],
        how="left",
    )

    # --- Drop rows with null iv or log_moneyness (failed SVI fits) ---
    valid = merged["iv"].notna() & merged["log_moneyness"].notna()
    merged = merged[valid].reset_index(drop=True)

    # --- Derive strike and option_type ---
    merged["strike"] = merged["spot"] * np.exp(merged["log_moneyness"])
    merged["option_type"] = np.where(merged["delta"] < 0, -1, 1).astype(np.int32)

    # --- Build input for compute_greeks_from_iv ---
    greeks_input = pa.table({
        "option_type": pa.array(merged["option_type"], type=pa.int32()),
        "spot": pa.array(merged["spot"], type=pa.float64()),
        "strike": pa.array(merged["strike"], type=pa.float64()),
        "expiry": pa.array(merged["expiry"], type=pa.float64()),
        "rate": pa.array(merged["rate"], type=pa.float64()),
        "dividend_yield": pa.array(merged["dividend_yield"], type=pa.float64()),
        "iv": pa.array(merged["iv"], type=pa.float64()),
    })

    greeks_result = compute_greeks_from_iv(greeks_input)

    # --- Assemble output: surface columns + strike + greeks ---
    out_arrays = {}
    for col in surface.schema.names:
        out_arrays[col] = pa.array(merged[col])
    out_arrays["strike"] = greeks_result.column("strike")
    for col in ["gamma", "vega", "theta", "rho"]:
        out_arrays[col] = greeks_result.column(col)

    return pa.table(out_arrays)


# -- constant_maturity_surface -------------------------------------------------

_DEFAULT_TENORS = [7/365, 14/365, 30/365, 60/365, 90/365, 180/365, 1.0]


def constant_maturity_surface(
    surface: pa.Table,
    tenors: list[float] | None = None,
    decomposition: pa.Table | None = None,
) -> pa.Table:
    """Interpolate a per-expiration surface to fixed tenors.

    Interpolation is performed in total variance space (w = iv^2 * T)
    to preserve calendar arbitrage constraints.

    Parameters
    ----------
    surface : pyarrow.Table
        Output of ``fit_vol_surface`` — must contain: timestamp, expiry,
        delta, iv. Optional: log_moneyness, iv_bid, iv_ask.
    tenors : list of float, optional
        Target tenors in years (default: 7d, 14d, 30d, 60d, 90d, 180d, 1y).
    decomposition : pyarrow.Table, optional
        Output of ``decompose_variance``. If provided, only diffusive variance
        is interpolated and event jumps are placed at their correct positions.

    Returns
    -------
    pyarrow.Table
        One row per (timestamp, tenor, delta) with columns:
        timestamp, tenor, delta, iv. Plus log_moneyness, iv_bid, iv_ask
        if present in the input.
    """
    from scipy.interpolate import interp1d

    _check_required(surface, ["timestamp", "expiry", "delta", "iv"])

    if tenors is None:
        tenors = list(_DEFAULT_TENORS)

    df = surface.to_pandas()
    has_lm = "log_moneyness" in df.columns
    has_bid_ask = "iv_bid" in df.columns and "iv_ask" in df.columns

    # Load decomposition if provided
    decomp_map = None
    if decomposition is not None:
        decomp_df = decomposition.to_pandas()
        decomp_map = {}
        for (ts_val, delta_val), grp in decomp_df.groupby(
            ["timestamp", "delta"]
        ):
            sigma_d_sq = grp["sigma_d_sq"].iloc[0]
            events = list(zip(grp["event_expiry"], grp["event_var"]))
            decomp_map[(ts_val, delta_val)] = (sigma_d_sq, events)

    rows = []
    for (ts_val, delta_val), grp in df.groupby(["timestamp", "delta"]):
        grp = grp.dropna(subset=["iv"]).sort_values("expiry")
        if len(grp) < 2:
            continue

        T_arr = grp["expiry"].values
        iv_arr = grp["iv"].values
        w_arr = iv_arr**2 * T_arr  # total variance

        if decomp_map is not None and (ts_val, delta_val) in decomp_map:
            sigma_d_sq, events = decomp_map[(ts_val, delta_val)]
            # Build diffusive variance: w_diff(T) = w(T) - sum(J_i^2 for t_i <= T)
            w_diff = w_arr.copy()
            for evt_t, evt_var in events:
                w_diff[T_arr >= evt_t] -= evt_var
            w_diff = np.maximum(w_diff, 0.0)
            interp_w = interp1d(T_arr, w_diff, kind="linear",
                                bounds_error=True)
        else:
            sigma_d_sq = None
            events = []
            interp_w = interp1d(T_arr, w_arr, kind="linear",
                                bounds_error=True)

        # Optional interpolators
        if has_lm:
            lm_valid = grp.dropna(subset=["log_moneyness"])
            interp_lm = (
                interp1d(lm_valid["expiry"].values,
                         lm_valid["log_moneyness"].values,
                         kind="linear", bounds_error=True)
                if len(lm_valid) >= 2 else None
            )
        if has_bid_ask:
            ba_valid = grp.dropna(subset=["iv_bid", "iv_ask"])
            w_bid = ba_valid["iv_bid"].values**2 * ba_valid["expiry"].values
            w_ask = ba_valid["iv_ask"].values**2 * ba_valid["expiry"].values
            interp_bid = (
                interp1d(ba_valid["expiry"].values, w_bid,
                         kind="linear", bounds_error=True)
                if len(ba_valid) >= 2 else None
            )
            interp_ask = (
                interp1d(ba_valid["expiry"].values, w_ask,
                         kind="linear", bounds_error=True)
                if len(ba_valid) >= 2 else None
            )

        T_min, T_max = T_arr[0], T_arr[-1]

        for tenor in tenors:
            if tenor < T_min or tenor > T_max:
                continue

            try:
                w_star = float(interp_w(tenor))
            except ValueError:
                continue

            # Add back event variance if using decomposition
            if decomp_map is not None and (ts_val, delta_val) in decomp_map:
                for evt_t, evt_var in events:
                    if evt_t <= tenor:
                        w_star += evt_var

            iv_star = np.sqrt(max(w_star, 0.0) / tenor)

            row = {
                "timestamp": ts_val,
                "tenor": tenor,
                "delta": delta_val,
                "iv": iv_star,
            }

            if has_lm and interp_lm is not None:
                try:
                    row["log_moneyness"] = float(interp_lm(tenor))
                except ValueError:
                    row["log_moneyness"] = None
            elif has_lm:
                row["log_moneyness"] = None

            if has_bid_ask:
                if interp_bid is not None:
                    try:
                        wb = float(interp_bid(tenor))
                        row["iv_bid"] = np.sqrt(max(wb, 0.0) / tenor)
                    except ValueError:
                        row["iv_bid"] = None
                else:
                    row["iv_bid"] = None
                if interp_ask is not None:
                    try:
                        wa = float(interp_ask(tenor))
                        row["iv_ask"] = np.sqrt(max(wa, 0.0) / tenor)
                    except ValueError:
                        row["iv_ask"] = None
                else:
                    row["iv_ask"] = None

            rows.append(row)

    if not rows:
        ts_type = surface.schema.field("timestamp").type
        fields = {
            "timestamp": pa.array([], type=ts_type),
            "tenor": pa.array([], type=pa.float64()),
            "delta": pa.array([], type=pa.float64()),
            "iv": pa.array([], type=pa.float64()),
        }
        if has_lm:
            fields["log_moneyness"] = pa.array([], type=pa.float64())
        if has_bid_ask:
            fields["iv_bid"] = pa.array([], type=pa.float64())
            fields["iv_ask"] = pa.array([], type=pa.float64())
        return pa.table(fields)

    import pandas as pd
    out_df = pd.DataFrame(rows)
    return pa.Table.from_pandas(out_df, preserve_index=False)


# -- decompose_variance --------------------------------------------------------


def decompose_variance(
    surface: pa.Table,
    events: pa.Table,
) -> pa.Table:
    """Decompose implied variance into diffusive and event components.

    Uses non-negative least squares to solve:
        w(T) = sigma_d^2 * T + sum(J_i^2 for t_i <= T)

    Parameters
    ----------
    surface : pyarrow.Table
        Output of ``fit_vol_surface`` — must contain: timestamp, expiry,
        delta, iv.
    events : pyarrow.Table
        Event calendar — must contain: event_date (date32 or timestamp).
        Optional: event_type (utf8, e.g. "earnings", "fomc").
        Events are matched against surface timestamps: an event's expiry
        is computed relative to each timestamp.

    Returns
    -------
    pyarrow.Table
        One row per (timestamp, delta, event) with columns:
        timestamp, delta, sigma_d_sq, event_expiry, event_var, event_type.
    """
    from scipy.optimize import nnls
    import pandas as pd

    _check_required(surface, ["timestamp", "expiry", "delta", "iv"])
    _check_required(events, ["event_date"])

    surf_df = surface.to_pandas()
    events_df = events.to_pandas()

    has_event_type = "event_type" in events_df.columns

    # Normalize event_date to date objects
    if hasattr(events_df["event_date"].dtype, "tz") or np.issubdtype(
        events_df["event_date"].dtype, np.datetime64
    ):
        events_df["event_date"] = pd.to_datetime(
            events_df["event_date"]
        ).dt.date
    event_dates = sorted(events_df["event_date"].unique())
    event_types = {}
    if has_event_type:
        for _, row in events_df.iterrows():
            d = row["event_date"] if not hasattr(row["event_date"], "date") else row["event_date"]
            event_types[d] = row["event_type"]

    rows = []
    for (ts_val, delta_val), grp in surf_df.groupby(["timestamp", "delta"]):
        grp = grp.dropna(subset=["iv"]).sort_values("expiry")
        if len(grp) < 2:
            continue

        T_arr = grp["expiry"].values
        iv_arr = grp["iv"].values
        w_arr = iv_arr**2 * T_arr

        # Determine timestamp as a date for event comparison
        if hasattr(ts_val, "date"):
            ts_date = ts_val.date()
        elif hasattr(ts_val, "timetuple"):
            ts_date = ts_val
        else:
            ts_date = pd.Timestamp(ts_val).date()

        # Find events that fall within the expiry range
        # Compute event expiry (time from timestamp to event) in years
        relevant_events = []
        for evt_date in event_dates:
            days_to_event = (evt_date - ts_date).days
            if days_to_event <= 0:
                continue
            evt_T = days_to_event / 365.0
            if evt_T <= T_arr[-1]:
                evt_type = event_types.get(evt_date, "unknown") if has_event_type else "unknown"
                relevant_events.append((evt_T, evt_date, evt_type))

        N_events = len(relevant_events)
        M = len(T_arr)

        # Build A matrix: [T | indicator columns for each event]
        A = np.zeros((M, 1 + N_events))
        A[:, 0] = T_arr
        for j, (evt_T, _, _) in enumerate(relevant_events):
            A[:, 1 + j] = (T_arr >= evt_T).astype(float)

        # Solve w = A x with x >= 0
        x, _ = nnls(A, w_arr)
        sigma_d_sq = x[0]

        # Always output the diffusive component
        for j, (evt_T, evt_date, evt_type) in enumerate(relevant_events):
            rows.append({
                "timestamp": ts_val,
                "delta": delta_val,
                "sigma_d_sq": sigma_d_sq,
                "event_expiry": evt_T,
                "event_var": x[1 + j],
                "event_type": evt_type,
            })

        # If no events, still output one row with the diffusive component
        if N_events == 0:
            rows.append({
                "timestamp": ts_val,
                "delta": delta_val,
                "sigma_d_sq": sigma_d_sq,
                "event_expiry": None,
                "event_var": 0.0,
                "event_type": None,
            })

    if not rows:
        return pa.table({
            "timestamp": pa.array([], type=surf_df["timestamp"].dtype),
            "delta": pa.array([], type=pa.float64()),
            "sigma_d_sq": pa.array([], type=pa.float64()),
            "event_expiry": pa.array([], type=pa.float64()),
            "event_var": pa.array([], type=pa.float64()),
            "event_type": pa.array([], type=pa.utf8()),
        })

    out_df = pd.DataFrame(rows)
    return pa.Table.from_pandas(out_df, preserve_index=False)
