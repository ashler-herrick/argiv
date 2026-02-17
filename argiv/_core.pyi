import pyarrow as pa

def compute_greeks(table: pa.Table) -> pa.Table: ...
def fit_vol_surface(
    table: pa.Table,
    delta_pillars: list[float] | None = None,
) -> pa.Table: ...
def compute_fit_vol_surface(
    table: pa.Table,
    delta_pillars: list[float] | None = None,
) -> pa.Table: ...
