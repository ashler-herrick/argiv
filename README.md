# Arrow Greeks and Implied Volatility (argiv)

Small python wrapper around arrow dataframes and Quantlib with one sole purpose: Calculating implied volatility and greeks in parallel with a Python interface.

## Benchmarks versus other Python options

### 1,000 rows

| Library                   |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|---------------------------|------------|------------|------------|------------|------------|----------|
| argiv                     |     0.0011 |     0.0010 |     0.0007 |     0.0006 |     0.0042 |        - |
| Python (scipy, parallel)  |     0.1751 |     0.0207 |     0.1668 |     0.1546 |     0.2362 |    245x  |
| PyQuantLib (parallel)     |     0.0637 |     0.0041 |     0.0624 |     0.0593 |     0.0813 |     92x  |

### 10,000 rows

| Library                   |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|---------------------------|------------|------------|------------|------------|------------|----------|
| argiv                     |     0.0035 |     0.0016 |     0.0035 |     0.0019 |     0.0076 |        - |
| PyQuantLib (parallel)     |     0.1640 |     0.0272 |     0.1566 |     0.1240 |     0.2259 |     45x  |

### 100,000 rows

| Library                   |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|---------------------------|------------|------------|------------|------------|------------|----------|
| argiv                     |     0.0170 |     0.0011 |     0.0166 |     0.0157 |     0.0198 |        - |

### 1,000,000 rows

| Library                   |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |
|---------------------------|------------|------------|------------|------------|------------|----------|
| argiv                     |     0.1928 |     0.0138 |     0.1874 |     0.1760 |     0.2430 |        - |