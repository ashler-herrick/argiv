#pragma once

#include <cmath>

namespace argiv {

struct OptionResult {
    double iv;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// Compute IV and Greeks for a single option.
// option_type: 1 for call, -1 for put
// spot, strike, T (years), r (risk-free rate), q (dividend yield),
// market_price (observed option price)
OptionResult compute_single(int option_type, double spot, double strike,
                            double T, double r, double q,
                            double market_price);

}  // namespace argiv
