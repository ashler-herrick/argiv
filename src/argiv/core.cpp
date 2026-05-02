#include "argiv/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <ql/math/solvers1d/brent.hpp>
#include <ql/pricingengines/blackcalculator.hpp>

#include <boost/math/distributions/normal.hpp>

namespace argiv {

namespace {

class IVObjective {
  public:
    IVObjective(QuantLib::Option::Type type, double strike, double forward,
                double discount, double T, double market_price)
        : type_(type), strike_(strike), forward_(forward),
          discount_(discount), sqrtT_(std::sqrt(T)),
          market_price_(market_price) {}

    double operator()(double sigma) const {
        double stdDev = sigma * sqrtT_;
        QuantLib::BlackCalculator calc(type_, strike_, forward_, stdDev,
                                       discount_);
        return calc.value() - market_price_;
    }

  private:
    QuantLib::Option::Type type_;
    double strike_;
    double forward_;
    double discount_;
    double sqrtT_;
    double market_price_;
};

// IG(μ, λ=1) CDF: F(x) = Φ(z1) + e^{2/μ} Φ(-z2),
// with z1 = (x/μ - 1)/√x, z2 = (x/μ + 1)/√x.
inline double ig_cdf_l1(double x, double mu) {
    constexpr double inv_sqrt2 = 0.7071067811865475;
    double sqrtx = std::sqrt(x);
    double t = x / mu;
    double z1 = (t - 1.0) / sqrtx;
    double z2 = (t + 1.0) / sqrtx;
    double Phi_z1 = 0.5 * std::erfc(-z1 * inv_sqrt2);
    double Phi_neg_z2 = 0.5 * std::erfc(z2 * inv_sqrt2);
    return Phi_z1 + std::exp(2.0 / mu) * Phi_neg_z2;
}

// IG(μ, λ=1) PDF: f(x) = (2π x³)^{-1/2} exp(-(x-μ)² / (2 μ² x)).
inline double ig_pdf_l1(double x, double mu) {
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    double diff = x - mu;
    return inv_sqrt_2pi / std::sqrt(x * x * x)
           * std::exp(-(diff * diff) / (2.0 * mu * mu * x));
}

// Hand-rolled IG(μ, λ=1) quantile.
// Initial guess: min of two complementary asymptotic approximations:
//   - Levy(0,1) limit (IG → Levy as μ → ∞):  x ≈ 1/(Φ⁻¹(q/2))²
//   - Mean-anchored cap 4μ (avoids CDF saturation when q is near 1 and μ small)
// Then Newton-Raphson in log-x:  y_{n+1} = y_n − (F(eʸ) − q) / (eʸ f(eʸ)),
// which keeps step magnitudes O(1) regardless of x scale and is well-conditioned
// in the slow-decaying right tail.
double ig_quantile_l1(double q, double mu) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(q > 0.0 && q < 1.0) || !(mu > 0.0)) return nan;

    boost::math::normal_distribution<double> N(0.0, 1.0);
    double phi_inv_half = boost::math::quantile(N, 0.5 * q);
    double x_levy = 1.0 / (phi_inv_half * phi_inv_half);
    double x = std::min(x_levy, 4.0 * mu);
    if (!(x > 0.0)) x = mu;

    for (int i = 0; i < 60; ++i) {
        double F = ig_cdf_l1(x, mu);
        double err = F - q;
        double f = ig_pdf_l1(x, mu);
        if (!(f > 0.0)) {
            // PDF underflow: shrink x toward the mode region.
            x *= 0.5;
            continue;
        }
        // Newton step in log-x:  Δy = (F − q) / (x · f).
        double dy = err / (x * f);
        // Clamp to keep step length sane (factor ≤ e^2 per iter).
        if (dy > 2.0) dy = 2.0;
        if (dy < -2.0) dy = -2.0;
        x *= std::exp(-dy);
        if (std::abs(dy) < 1e-13) break;
    }
    return x;
}

// Compute total IV w = σ√T from (|k|, OTM-normalized price o).
// This is the canonical Schadner inversion expressed in OTM-symmetric coordinates:
// for a call OTM at k>0 we have o = c, q = 1 − o; for a put OTM at k<0 we have
// o = p · e^{−k}, q = 1 − o. Both regimes share the same w(x, o) function.
double w_from_xo(double x, double o) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(o > 0.0 && o < 1.0)) return nan;
    if (x < 1e-14) {
        // Exact ATM closed form
        boost::math::normal_distribution<double> N(0.0, 1.0);
        return 2.0 * boost::math::quantile(N, 0.5 * (o + 1.0));
    }
    double mu = 2.0 / x;
    double xq = ig_quantile_l1(1.0 - o, mu);
    if (!(xq > 0.0) || !std::isfinite(xq)) return nan;
    return 2.0 / std::sqrt(xq);
}

// ---------------------------------------------------------------------------
// 2D lookup table for w(x, o), x = |k|, o = OTM-normalized price.
// Bilinear interpolation gives ~4-decimal accuracy on σ for typical inputs.
// ---------------------------------------------------------------------------

namespace lut {
constexpr int N_X = 257;
constexpr int N_Z = 257;
constexpr double X_MIN = 0.0;
constexpr double X_MAX = 1.5;       // covers moneyness e^{±1.5} ≈ [0.22, 4.48]
// z = logit(o) = log(o / (1 − o)). Range ±9 covers o ∈ [1.2e-4, 0.99988].
constexpr double Z_MAX = 9.0;
constexpr double H_X = (X_MAX - X_MIN) / (N_X - 1);
constexpr double H_Z = (2.0 * Z_MAX) / (N_Z - 1);

inline double logit(double o) { return std::log(o / (1.0 - o)); }
inline double sigmoid(double z) {
    if (z >= 0.0) {
        double e = std::exp(-z);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(z);
    return e / (1.0 + e);
}

struct Table {
    std::array<std::array<double, N_Z>, N_X> w;
    Table() {
        for (int i = 0; i < N_X; ++i) {
            double x = X_MIN + H_X * i;
            for (int j = 0; j < N_Z; ++j) {
                double z = -Z_MAX + H_Z * j;
                double o = sigmoid(z);
                w[i][j] = w_from_xo(x, o);
            }
        }
    }
};
const Table& table() {
    static const Table T;
    return T;
}

// Catmull-Rom 1D — cubic interpolant through (p1, p2) using p0..p3 as tangent
// support. Coefficients hard-coded for h = 1 (uniform grid).
inline double cr1d(double t, double p0, double p1, double p2, double p3) {
    double t2 = t * t;
    double t3 = t2 * t;
    return 0.5 * ((-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + p2) * t
                + 2.0 * p1);
}

// Bicubic Catmull-Rom lookup over (x, z) where z = logit(o). NaN if outside
// the tabulated domain.
inline double lookup(double x, double o) {
    if (x < X_MIN || x > X_MAX || !(o > 0.0 && o < 1.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double z = logit(o);
    if (z < -Z_MAX || z > Z_MAX) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double if_x = (x - X_MIN) / H_X;
    double if_z = (z + Z_MAX) / H_Z;
    int i = std::clamp(static_cast<int>(if_x), 1, N_X - 3);
    int j = std::clamp(static_cast<int>(if_z), 1, N_Z - 3);
    double a = if_x - i;
    double b = if_z - j;
    const auto& T = table();
    double r0 = cr1d(b, T.w[i - 1][j - 1], T.w[i - 1][j],
                        T.w[i - 1][j + 1], T.w[i - 1][j + 2]);
    double r1 = cr1d(b, T.w[i    ][j - 1], T.w[i    ][j],
                        T.w[i    ][j + 1], T.w[i    ][j + 2]);
    double r2 = cr1d(b, T.w[i + 1][j - 1], T.w[i + 1][j],
                        T.w[i + 1][j + 1], T.w[i + 1][j + 2]);
    double r3 = cr1d(b, T.w[i + 2][j - 1], T.w[i + 2][j],
                        T.w[i + 2][j + 1], T.w[i + 2][j + 2]);
    return cr1d(a, r0, r1, r2, r3);
}
}  // namespace lut

// IV via bilinear LUT lookup on (|k|, OTM-normalized price).
// ITM observations are reflected to the OTM side via put-call parity.
double lookup_iv(bool is_call, double normalized_price, double K, double F,
                 double T) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    double k = std::log(K / F);
    double abs_k = std::abs(k);
    double ek = std::exp(k);

    // Normalize observation to OTM-side price o ∈ (0, 1).
    // Parity (in (D·F)-units): c_norm − p_norm = 1 − e^k.
    double c_norm = is_call ? normalized_price
                            : (normalized_price + 1.0 - ek);
    double p_norm = is_call ? (normalized_price - 1.0 + ek)
                            : normalized_price;
    double o;
    if (k > 0.0) {
        o = c_norm;            // OTM call
    } else if (k < 0.0) {
        o = p_norm / ek;       // OTM put: o = p · e^{-k}
    } else {
        o = c_norm;            // ATM
    }
    if (!(o > 0.0 && o < 1.0)) return nan;

    double w = lut::lookup(abs_k, o);
    if (!std::isfinite(w) || !(w > 0.0)) return nan;
    return w / std::sqrt(T);
}

// Schadner (arXiv:2604.24480) closed-form IV.
// Returns NaN if the inputs are out of the formula's domain.
double schadner_iv(bool is_call, double normalized_price, double K, double F,
                   double T) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    double k = std::log(K / F);
    double sqrtT = std::sqrt(T);

    if (std::abs(k) < 1e-14) {
        double c = is_call ? normalized_price
                           : normalized_price + 1.0 - K / F;
        if (c <= 0.0 || c >= 1.0) return nan;
        boost::math::normal_distribution<double> N(0.0, 1.0);
        return (2.0 / sqrtT) * boost::math::quantile(N, (c + 1.0) / 2.0);
    }

    double m = (K > F) ? 1.0 : (K / F);
    double q_arg = is_call ? (1.0 - normalized_price) / m
                           : (std::exp(k) - normalized_price) / m;
    if (!(q_arg > 0.0 && q_arg < 1.0)) return nan;

    double mu = 2.0 / std::abs(k);
    double x = ig_quantile_l1(q_arg, mu);
    if (!(x > 0.0) || !std::isfinite(x)) return nan;
    return (2.0 / sqrtT) / std::sqrt(x);
}

}  // namespace

OptionResult compute_single(int option_type, double spot, double strike,
                            double T, double r, double q,
                            double market_price, IVSolver solver) {
    OptionResult result;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    if (T <= 0.0 || spot <= 0.0 || strike <= 0.0 || market_price <= 0.0) {
        result = {nan, nan, nan, nan, nan, nan};
        return result;
    }

    auto ql_type = (option_type == 1) ? QuantLib::Option::Call
                                      : QuantLib::Option::Put;

    double discount = std::exp(-r * T);
    double forward = spot * std::exp((r - q) * T);

    double iv = nan;
    if (solver == IVSolver::Schadner) {
        bool is_call = (option_type == 1);
        double normalized_price = market_price / (discount * forward);
        iv = schadner_iv(is_call, normalized_price, strike, forward, T);
    } else if (solver == IVSolver::Lookup) {
        bool is_call = (option_type == 1);
        double normalized_price = market_price / (discount * forward);
        iv = lookup_iv(is_call, normalized_price, strike, forward, T);
    }

    if (!std::isfinite(iv)) {
        // Numerical fallback (also the path for solver == Numerical).
        try {
            IVObjective objective(ql_type, strike, forward, discount, T,
                                  market_price);
            QuantLib::Brent solver_brent;
            solver_brent.setMaxEvaluations(100);
            iv = solver_brent.solve(objective, 1e-8, 0.2, 1e-6, 5.0);
        } catch (...) {
            result = {nan, nan, nan, nan, nan, nan};
            return result;
        }
    }

    double stdDev = iv * std::sqrt(T);
    QuantLib::BlackCalculator calc(ql_type, strike, forward, stdDev, discount);

    result.iv = iv;
    result.delta = calc.delta(spot);
    result.gamma = calc.gamma(spot);
    result.vega = calc.vega(T);
    result.theta = calc.theta(spot, T);
    result.rho = calc.rho(T);

    return result;
}

}  // namespace argiv
