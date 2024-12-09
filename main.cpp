#include <iostream>
#include <numeric>
#include <random>
#include <chrono>

auto simulate_path(double initial_stock_price, double time_to_maturity_years,
                   double risk_free_rate, double volatility,
                   double number_of_time_steps_per_path, std::mt19937& rng,
                   std::normal_distribution<double>& distribution) -> double {
    auto delta_t = time_to_maturity_years / number_of_time_steps_per_path;
    auto St = initial_stock_price;

    for (int i = 0; i < number_of_time_steps_per_path; i++) {
        auto Z = distribution(rng);
        St *= std::exp((risk_free_rate - 0.5 * volatility * volatility) *
                           delta_t +
                       volatility * std::sqrt(delta_t) * Z);
    }

    return St;
}

auto monte_carlo_call_price(double initial_stock_price, double strike_price,
                            double time_to_maturity_years,
                            double risk_free_rate, double volatility,
                            int number_of_simulations,
                            int number_of_time_steps_per_path) -> double {
    auto random_device = std::random_device{};
    auto rng = std::mt19937{random_device()};
    auto distribution = std::normal_distribution<double>{0.0, 1.0};

    auto payoffs = std::vector<double>{};

    for (int i = 0; i < number_of_simulations; i++) {
        auto ST = simulate_path(
            initial_stock_price, time_to_maturity_years, risk_free_rate,
            volatility, number_of_time_steps_per_path, rng, distribution);
        auto payoff = std::max(ST - strike_price, 0.0);
        payoffs.push_back(payoff);
    }

    auto average_payoff =
        std::accumulate(payoffs.cbegin(), payoffs.cend(), 0.0) /
        number_of_simulations;
    auto discounted_present_value =
        std::exp(-risk_free_rate * time_to_maturity_years) * average_payoff;

    return discounted_present_value;
}

auto main() -> int {
    auto initial_stock_price = 100.0;
    auto strike_price = 100.0;
    auto time_to_maturity_years = 1.0;
    auto risk_free_rate = 0.05;
    auto volatility = 0.2;
    auto number_of_simulations = 1'000'000;
    auto number_of_time_steps_per_path = 100;

    auto start = std::chrono::high_resolution_clock::now();
    auto call_price = monte_carlo_call_price(
        initial_stock_price, strike_price, time_to_maturity_years,
        risk_free_rate, volatility, number_of_simulations,
        number_of_time_steps_per_path);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;

    std::cout << "Monte Carlo European Call Option Price: " << call_price << std::endl;
    std::cout << "Execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << '\n';
}
