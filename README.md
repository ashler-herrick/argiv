# Arrow Greeks and Implied Volatility (argiv)

Small python wrapper around arrow dataframes and Quantlib with one sole purpose: Calculating implied volatility and greeks in parallel with Python.

## Benchmarks versus other Python options

### 1,000 rows

| Library        |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|----------------|------------|------------|------------|------------|------------|----------|
| argiv          |     0.0007 |     0.0002 |     0.0007 |     0.0005 |     0.0013 |        - |
| Python (scipy) |     0.7766 |     0.0207 |     0.7682 |     0.7594 |     0.8624 |   1174x  |
| QuantLib       |     0.0487 |     0.0002 |     0.0487 |     0.0484 |     0.0491 |     74x  |

### 10,000 rows

| Library        |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|----------------|------------|------------|------------|------------|------------|----------|
| argiv          |     0.0004 |     0.0006 |     0.0003 |     0.0003 |     0.0046 |        - |
| QuantLib       |     0.4885 |     0.0063 |     0.4869 |     0.4828 |     0.5163 |   1465x  |

### 100,000 rows

| Library        |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|----------------|------------|------------|------------|------------|------------|----------|
| argiv          |     0.0044 |     0.0023 |     0.0032 |     0.0023 |     0.0105 |        - |