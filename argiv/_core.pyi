import pyarrow as pa

def compute_greeks(
    table: pa.Table, iv_solver: str = "numerical"
) -> pa.Table: ...
def fit_vol_surface(
    table: pa.Table,
    delta_pillars: list[float] | None = None,
) -> pa.Table: ...
