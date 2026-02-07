#include "argiv/core.hpp"

#include <cmath>
#include <limits>

#include <ql/math/solvers1d/brent.hpp>
#include <ql/pricingengines/blackcalculator.hpp>

namespace argiv {

namespace {

// Functor for the Brent solver: returns calc.value() - market_price
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

}  // namespace

OptionResult compute_single(int option_type, double spot, double strike,
                            double T, double r, double q,
                            double market_price) {
    OptionResult result;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    if (T <= 0.0 || spot <= 0.0 || strike <= 0.0 || market_price <= 0.0) {
        result = {nan, nan, nan, nan, nan, nan};
        return result;
    }

    auto ql_type = (option_type == 1) ? QuantLib::Option::Call
                                      : QuantLib::Option::Put;

    double discount = std::exp(-r * T);
    double growth = std::exp(-q * T);
    double forward = spot * std::exp((r - q) * T);

    // Solve for IV using Brent
    double iv;
    try {
        IVObjective objective(ql_type, strike, forward, discount, T,
                              market_price);
        QuantLib::Brent solver;
        solver.setMaxEvaluations(100);
        iv = solver.solve(objective, 1e-8, 0.2, 1e-6, 5.0);
    } catch (...) {
        result = {nan, nan, nan, nan, nan, nan};
        return result;
    }

    // Compute Greeks at the solved IV
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
