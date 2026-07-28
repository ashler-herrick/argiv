#pragma once

#include <cmath>

namespace argiv {

enum class IVSolver {
    Numerical,  // QuantLib BlackCalculator + Brent
    Schadner,   // Closed-form via inverse Gaussian quantile (arXiv:2604.24480)
    Lookup,     // Precomputed 2D table on (|k|, OTM-normalized price), bilinear interp
};

// False unless x is finite and strictly positive. Written as a positive test
// on purpose: every comparison with NaN is false, so the natural `x <= 0.0`
// reject form lets NaN through, and QuantLib then throws inside an OpenMP
// loop where the exception cannot propagate -- std::terminate kills the whole
// process. A bad input must produce a NaN row, never an abort.
inline bool pos_finite(double x) { return x > 0.0 && std::isfinite(x); }

struct OptionResult {
    double iv;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

struct HigherOrderGreeks {
    double vanna;  // d(vega)/dS   = d2V/dS dsigma
    double volga;  // d(vega)/dsigma
    double charm;  // d(delta)/dt
    double speed;  // d(gamma)/dS
    double zomma;  // d(gamma)/dsigma
    double color;  // d(gamma)/dt
};

// Closed-form second/third-order Greeks in the same (forward, stdDev, discount)
// European-Black parameterisation used for the first-order Greeks, expressed in
// terms of the already-computed delta/gamma/vega so the two stay consistent.
//
// charm and color differentiate w.r.t. calendar time t (= -d/dT), matching the
// sign convention of QuantLib's theta. All are per unit of the natural argument:
// per 1.0 of spot, per 1.0 of vol (decimal), per year.
//
// NaN in any first-order Greek propagates to every output.
inline HigherOrderGreeks higher_order_greeks(double spot, double strike,
                                             double forward, double T,
                                             double r, double q, double sigma,
                                             double delta, double gamma,
                                             double vega) {
    double stdDev = sigma * std::sqrt(T);
    double d1 = (std::log(forward / strike) + 0.5 * stdDev * stdDev) / stdDev;
    double d2 = d1 - stdDev;
    // A = 2(r-q)T - d2*stdDev, so that dd1/dT = A / (2 * T * stdDev). Both
    // time derivatives below reduce to this one term.
    double A = 2.0 * (r - q) * T - d2 * stdDev;

    HigherOrderGreeks g;
    g.vanna = vega * (1.0 - d1 / stdDev) / spot;
    g.volga = vega * d1 * d2 / sigma;
    g.charm = q * delta - gamma * spot * A / (2.0 * T);
    g.speed = -gamma * (1.0 + d1 / stdDev) / spot;
    g.zomma = gamma * (d1 * d2 - 1.0) / sigma;
    g.color = (gamma / (2.0 * T)) * (2.0 * q * T + 1.0 + d1 * A / stdDev);
    return g;
}

// Compute IV and Greeks for a single option.
// option_type: 1 for call, -1 for put
// spot, strike, T (years), r (risk-free rate), q (dividend yield),
// market_price (observed option price)
OptionResult compute_single(int option_type, double spot, double strike,
                            double T, double r, double q,
                            double market_price,
                            IVSolver solver = IVSolver::Numerical);

}  // namespace argiv
