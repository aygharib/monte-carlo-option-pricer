#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <math.h>

#include <fmt/core.h>
#include <fmt/color.h>

#include <cuda/api.hpp>
#include <cuda/api/device_properties.hpp>

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

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

void global_memory()
{
	auto device = cuda::device::current::get();

	auto device_global_mem = device.memory();
	auto total_memory = device_global_mem.amount_total();
	auto free_memory = device_global_mem.amount_total();

	std::cout
		<< "Device " << std::to_string(device.id()) << " reports it has "
		<< free_memory << " bytes free out of " << total_memory << " bytes total global memory "
		<< "(" << (total_memory - free_memory) << " bytes used).\n";

    if (device != device.memory().associated_device()) {
        // f"The device's reported ID and the device's memory object's reported devices differ: "
        // + std::to_string(device.id()) + " !=" +  std::to_string(device.memory().associated_device().id()));
    }

	assert(free_memory <= total_memory);
}

auto main() -> int {

    // Initialize financial constants
    auto constexpr initial_stock_price = float{100.0};
    auto constexpr strike_price = float{100.0};
    auto constexpr time_to_maturity_years = float{1.0};
    auto constexpr risk_free_rate = float{0.05};
    auto constexpr volatility = float{0.2};

    // auto const number_of_simulations = 1 << 30; // approx 1 billion
    // int64_t const number_of_simulations = 1 << 20;

    // Determine the capabilities of GPU
    if (cuda::device::count() == 0) {
        fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold, "No CUDA-supporting GPUs found. Exiting.\n");
        return EXIT_FAILURE;
    }
    auto device = cuda::device::current::get();
    fmt::println("CUDA device [{}]", device.name());
    fmt::println("Architecture: Name: {}, Is Valid: {}", device.architecture().name(), device.architecture().is_valid());
    fmt::println("Max in flight threads per processor: {}", device.compute_capability().max_in_flight_threads_per_processor());
    fmt::println("Max shared memory per block: {}", device.compute_capability().max_shared_memory_per_block());
    fmt::println("Max warp schedulings per processor cycle: {}", device.compute_capability().max_warp_schedulings_per_processor_cycle());
    fmt::println("Max warp schedulings per processor: {}", device.compute_capability().max_resident_warps_per_processor());
    // device.compute_capability().architecture;
    // fmt::print("typeof: {}", typeof(device));
    // std::cout << "CUDA device [" << device.name() << "]\n";

    // 1 warp = 32 threads
    // Max warps per block = X
    // 32 in-flight threads/processor
    // 22 processors/GPU
    // 1 GPU * 22 processors / 1 GPU * 64 in-flight threads / 1 processor = 1408 in-flight threads at a given point in time

    // 1660 SUPER
    auto block_count = static_cast<int>(std::ceil(
        static_cast<double>(simulation_count) / threads_per_block_count));
    
    auto A_h = std::make_unique<float>(simulation_count);

    // Allocate memory for a consecutive sequence of typed elements in device-global memory
    auto A_d = cuda::memory::make_unique_span<float>(device, 10);
    

    // auto start = std::chrono::high_resolution_clock::now();
    //
    // float* A_d;
    // int64_t size = number_of_simulations * sizeof(float);
    // auto err_a = cudaMalloc((void**) &A_d, size);
    // CUDA_CHECK(err_a);
    //
    // std::cout << sizeof(curandState) << '\n';
    // // curandState* d_states; // curandState is 48 bytes
    // // auto err_b = cudaMalloc((void**) &d_states,
    // //                         number_of_simulations * sizeof(curandState));
    // // CUDA_CHECK(err_b);
    //
    // // initCurandStates<<<blocks, threads_per_block>>>(d_states, time(NULL));
    // // cudaDeviceSynchronize();
    //
    // monte_carlo_kernel<<<blocks, threads_per_block>>>(
    //     A_d, strike_price, initial_stock_price, time_to_maturity_years,
    //     risk_free_rate, volatility, number_of_simulations,
    //     number_of_time_steps_per_path, 1, time(NULL));
    // cudaDeviceSynchronize();
    //
    // float* outputs = new float[number_of_simulations];
    // // std::cout << "look:\n";
    // auto err_c = cudaMemcpy(outputs, A_d, size, cudaMemcpyDeviceToHost);
    // CUDA_CHECK(err_c);
    // // std::cout << "output[0]=" << outputs[0] << '\n';
    // // for (int i = 0; i < 20; i++) {
    // //     std::cout << "output[i]=" << outputs[i] << '\n';
    // // }
    //
    // // std::ofstream myFile("foo.csv");
    // // for (int i = 0; i < number_of_simulations; i++) {
    // //     myFile << outputs[i] << '\n';
    // // }
    // // myFile.close();
    //
    // // 2 + 4 + 6 + 8
    // // (2 + 4) / 2 + (6 + 8) / 2
    //
    // // bug is here
    // auto average_payoff =
    //     std::accumulate(outputs, outputs + number_of_simulations, 0.0F) /
    //     number_of_simulations;
    //
    // // auto discounted_present_value =
    // //     std::exp(-risk_free_rate * time_to_maturity_years) * average_payoff;
    //
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = end - start;
    // // std::cout << "Monte Carlo European Call Option Price: "
    // //           << discounted_present_value << std::endl;
    // std::cout << "GPU Execution time (ms): "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
    //                  .count()
    //           << '\n';
}
