#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <numeric>

__global__ void initCurandStates(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void
monte_carlo_kernel(float* A_d, double strike_price, double initial_stock_price,
                   double time_to_maturity_years, double risk_free_rate,
                   double volatility, double number_of_time_steps_per_path,
                   double simulations_per_thread, // (1000000/22528)
                   unsigned long seed, curandState* states) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= 1'000'000)
        return;

    auto local_state = states[thread_id];

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

        if (thread_id == 0 && i < 5) { // Debugging only thread 0, first 5 steps
            printf("Step %d: random_normal_variable = %f, simulated_price_at_t "
                   "= %f\n",
                   i, random_normal_variable, simulated_price_at_t);
        }
    }

    auto payoff = max(simulated_price_at_t - strike_price, 0.0);

    A_d[thread_id] = payoff;
    // A_d[thread_id] = 100.0F;
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
    // auto const number_of_simulations = 1'000'000;
    auto const number_of_simulations = 64;
    auto number_of_time_steps_per_path = 100;
    std::cout << "hello world\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor
              << std::endl;


    auto start = std::chrono::high_resolution_clock::now();

    // threads per block, choose a multiple of the warp size (warp size = 32
    // threads) for efficiency

    auto threads_per_block = prop.maxThreadsPerBlock;
    // auto total_threads = 1'000'000;
    auto blocks = static_cast<int>(std::ceil(
        static_cast<double>(number_of_simulations) / threads_per_block));

    std::cout << blocks << '\n';

    float* A_d;
    int size = number_of_simulations * sizeof(float);
    auto error_a = cudaMalloc((void**) &A_d, size);
    std::cout << "a\n";
    CUDA_CHECK(error_a);

    curandState* d_states;
    auto err_b = cudaMalloc((void**) &d_states,
                            number_of_simulations * sizeof(curandState));
                            std::cout << "b\n";
    CUDA_CHECK(err_b);

    initCurandStates<<<blocks, threads_per_block>>>(d_states, time(NULL));
    // initCurandStates<<<1, 5>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    monte_carlo_kernel<<<blocks, threads_per_block>>>(
    // monte_carlo_kernel<<<1, 5>>>(
    A_d, strike_price, initial_stock_price,
                                 time_to_maturity_years, risk_free_rate,
                                 volatility, number_of_time_steps_per_path, 1,
                                 time(NULL), d_states);
    cudaDeviceSynchronize();

    float outputs[number_of_simulations];
    // float outputs[5];
    // cudaMemcpy(&outputs, A_d, sizeof(float), cudaMemcpyDeviceToHost);
    // auto err_c = 
    cudaMemcpy(&outputs, A_d, size, cudaMemcpyDeviceToHost);
    // std::cout << "c\n";
    // CUDA_CHECK(err_c);

    // std::cout << "outputs[0] = " << outputs[0] << '\n';
    for (int i = 0; i < 25; i++) {
        std::cout << "outputs[" << i << "] = " << outputs[i] << '\n';
    }

    // auto average_payoff =
    //     std::accumulate(outputs, outputs + number_of_simulations, 0.0F) /
    //     number_of_simulations;
    // // std::cout << "average payoff: " << average_payoff << '\n';

    // auto discounted_present_value =
    //     std::exp(-risk_free_rate * time_to_maturity_years) * average_payoff;
    // // 1024 threads / SM * 22 SM's = 22,528 threads can be in flight at once

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = end - start;
    // std::cout << "Monte Carlo European Call Option Price: "
    //           << discounted_present_value << std::endl;
    // std::cout << "Execution time (ms): "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
    //                  .count()
    //           << '\n';
}
