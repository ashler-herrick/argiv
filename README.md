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

## Installation

### Prerequisites

`argiv` requires C++ dependencies (QuantLib and Apache Arrow). You can install them via:

#### Option 1: vcpkg (Recommended)

[vcpkg](https://github.com/microsoft/vcpkg) provides portable, versioned C++ dependencies:

```bash
# Install vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh

# Set VCPKG_ROOT environment variable
export VCPKG_ROOT=/path/to/vcpkg  # Add to ~/.bashrc or ~/.zshrc

# Install dependencies
vcpkg install quantlib arrow
```

#### Option 2: System packages

Install QuantLib and Apache Arrow via your system package manager:

```bash
# Ubuntu/Debian
sudo apt-get install libquantlib0-dev libarrow-dev

# macOS (Homebrew)
brew install quantlib apache-arrow
```

### Build and Install

Once dependencies are installed:

```bash
# Clone the repository
git clone https://github.com/yourusername/argiv.git
cd argiv

# Install with uv (recommended) or pip
uv pip install -e .

# Or with pip
pip install -e .
```

### Troubleshooting

**CMake can't find QuantLib or Arrow:**
- If using vcpkg, ensure `VCPKG_ROOT` is set correctly
- If using system packages, verify they're installed: `pkg-config --modversion arrow` or `dpkg -l | grep quantlib`

**Build fails with "CMAKE_TOOLCHAIN_FILE not found":**
- The project uses CMakePresets.json to automatically configure vcpkg
- Ensure `VCPKG_ROOT` environment variable points to your vcpkg installation

**Force rebuild:**
```bash
uv pip install -e . --force-reinstall --no-build-isolation
```