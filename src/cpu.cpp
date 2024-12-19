#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

auto simulate_path(double initial_stock_price, double time_to_maturity_years,
                   double risk_free_rate, double volatility,
                   double number_of_time_steps_per_path, std::mt19937& rng,
                   std::normal_distribution<double>& distribution) -> double {
    auto delta_t =
        time_to_maturity_years / number_of_time_steps_per_path; // Time step
    auto simulated_price_at_t =
        initial_stock_price; // Start with initial stock price

    for (int i = 0; i < number_of_time_steps_per_path; i++) {
        // Generate a random normal variable
        // introduces randomness to simulate the effects of Brownian Motion in
        // the stock price a positive value is a positive shock, and a negative
        // value is a negative shock to the price the magnitude of the value
        // determines the size of the shock
        auto random_normal_variable = distribution(rng);

        // Use the stochastic differential equation for Geometric Brownian
        // Motion to determine the stock price for each step at each time step,
        // the price is updated based on deterministic drift, and random
        // fluctuation
        simulated_price_at_t *= std::exp(
            (risk_free_rate - 0.5 * volatility * volatility) * delta_t +
            volatility * std::sqrt(delta_t) * random_normal_variable);
    }

    return simulated_price_at_t;
}

auto monte_carlo_call_price(double initial_stock_price, double strike_price,
                            double time_to_maturity_years,
                            double risk_free_rate, double volatility,
                            int number_of_simulations,
                            int number_of_time_steps_per_path) -> double {
    auto random_device = std::random_device{};
    auto rng = std::mt19937{random_device()};
    auto distribution = std::normal_distribution<double>{0.0, 1.0};

    auto payoffs = std::vector<double>(number_of_simulations);

    for (int i = 0; i < number_of_simulations; i++) {
        auto simulated_stock_price_at_maturity = simulate_path(
            initial_stock_price, time_to_maturity_years, risk_free_rate,
            volatility, number_of_time_steps_per_path, rng, distribution);
        auto payoff =
            std::max(simulated_stock_price_at_maturity - strike_price, 0.0);
        payoffs.push_back(payoff);
    }

    auto average_payoff =
        std::accumulate(payoffs.cbegin(), payoffs.cend(), 0.0) /
        number_of_simulations;

    // Above we calculated the average payoff for an option at maturity, which
    // we treat as the expected payoff under the simulated scenarios now, we
    // need to express their value today, not at maturity we use a discount
    // factor of e^(-rT) to account of the time value of money, where r is the
    // risk free interest rate, and T is the time to maturity this brings the
    // expected payoff back to its present value
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
    // auto number_of_simulations = 1 << 17;
    auto number_of_simulations = 1'000'000;
    auto number_of_time_steps_per_path = 100;

    auto start = std::chrono::high_resolution_clock::now();
    auto call_price = monte_carlo_call_price(
        initial_stock_price, strike_price, time_to_maturity_years,
        risk_free_rate, volatility, number_of_simulations,
        number_of_time_steps_per_path);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;

    std::cout << "Monte Carlo European Call Option Price: " << call_price
              << std::endl;
    std::cout << "CPU Execution time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                     .count()
              << '\n';
}
