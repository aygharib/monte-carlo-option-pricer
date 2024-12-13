#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <fstream>

__global__ void
monte_carlo_kernel(float* A_d, double strike_price, double initial_stock_price,
                   double time_to_maturity_years, double risk_free_rate,
                   double volatility, double number_of_simulations,
                   double number_of_time_steps_per_path,
                   double simulations_per_thread, // (1000000/22528)
                   unsigned long seed
                //    , curandState* states
                ) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= number_of_simulations)
        return;

    // auto local_state = states[thread_id];
    curandState local_state;
    curand_init(seed, thread_id, 0, &local_state);

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
        // auto random_normal_variable = distribution(rng);
        // auto random_normal_variable = curand_uniform(&local_state);
        auto random_normal_variable = curand_normal(&local_state);

        // Use the stochastic differential equation for Geometric Brownian
        // Motion to determine the stock price for each step at each time step,
        // the price is updated based on deterministic drift, and random
        // fluctuation
        simulated_price_at_t *= std::exp(
            (risk_free_rate - 0.5 * volatility * volatility) * delta_t +
            volatility * std::sqrt(delta_t) * random_normal_variable);

        // if (thread_id == 0) { // Debugging only thread 0, first 5 steps
        //     printf("Step %d: random_normal_variable = %f, simulated_price_at_t "
        //        "= %f\n",
        //        i, random_normal_variable, simulated_price_at_t);
        // }
    }

    auto payoff = max(simulated_price_at_t - strike_price, 0.0);

    A_d[thread_id] = payoff;
}

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__,
               __LINE__);
        exit(EXIT_FAILURE);
    }
}

auto main() -> int {
    auto initial_stock_price = 100.0;
    auto strike_price = 100.0;
    auto time_to_maturity_years = 1.0;
    auto risk_free_rate = 0.05;
    auto volatility = 0.2;
    
    // auto initial_stock_price = 100000.0;
    // auto strike_price = 100000.0;
    // auto time_to_maturity_years = 5.0;
    // auto risk_free_rate = 0.3;
    // auto volatility = 0.5;
    // auto const number_of_simulations = 1 << 30; // approx 1 billion
    // int64_t const number_of_simulations = 1 << 20;
    auto const number_of_simulations = 10'000'000;
    auto number_of_time_steps_per_path = 100;

    auto start = std::chrono::high_resolution_clock::now();

    auto threads_per_block = 1024;
    auto blocks = static_cast<int>(std::ceil(
        static_cast<double>(number_of_simulations) / threads_per_block));

    float* A_d;
    int64_t size = number_of_simulations * sizeof(float);
    auto err_a = cudaMalloc((void**) &A_d, size);
    CUDA_CHECK(err_a);

    std::cout << sizeof(curandState) << '\n';
    // curandState* d_states; // curandState is 48 bytes
    // auto err_b = cudaMalloc((void**) &d_states,
    //                         number_of_simulations * sizeof(curandState));
    // CUDA_CHECK(err_b);

    // initCurandStates<<<blocks, threads_per_block>>>(d_states, time(NULL));
    // cudaDeviceSynchronize();

    monte_carlo_kernel<<<blocks, threads_per_block>>>(
        A_d, strike_price, initial_stock_price, time_to_maturity_years,
        risk_free_rate, volatility, number_of_simulations,
        number_of_time_steps_per_path, 1, time(NULL));
    cudaDeviceSynchronize();

    float* outputs = new float[number_of_simulations];
    // std::cout << "look:\n";
    auto err_c = cudaMemcpy(outputs, A_d, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_c);
    // std::cout << "output[0]=" << outputs[0] << '\n';
    // for (int i = 0; i < 20; i++) {
    //     std::cout << "output[i]=" << outputs[i] << '\n';
    // }

    // std::ofstream myFile("foo.csv");
    // for (int i = 0; i < number_of_simulations; i++) {
    //     myFile << outputs[i] << '\n';
    // }
    // myFile.close();

    // 2 + 4 + 6 + 8
    // (2 + 4) / 2 + (6 + 8) / 2

    // bug is here
    auto average_payoff =
        std::accumulate(outputs, outputs + number_of_simulations, 0.0F) /
        number_of_simulations;

    // auto discounted_present_value =
    //     std::exp(-risk_free_rate * time_to_maturity_years) * average_payoff;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    // std::cout << "Monte Carlo European Call Option Price: "
    //           << discounted_present_value << std::endl;
    std::cout << "GPU Execution time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                     .count()
              << '\n';
}
