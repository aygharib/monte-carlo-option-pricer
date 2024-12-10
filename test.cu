#include <iostream>
#include <cuda_runtime.h>

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void test(float* A_d, int size) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure the thread is within bounds
    if (thread_id < size) {
        A_d[thread_id] = 1.0F;
    }
}

auto main() -> int {
    float* A_d;
    // float A_d[] = {0.0F, 0.0F, 0.0F};
    int const num_elements = 3; // 3 elements to fill
    int size = num_elements * sizeof(float);

    auto* A_d_ptr = &A_d;
    // Allocate memory on the device
    auto err_a = cudaMalloc((void**) &A_d, size);
    printf("a");
    CUDA_CHECK(err_a);

    // Launch the kernel with 3 threads
    test<<<1, num_elements>>>(A_d, num_elements);

    // Synchronize to ensure the kernel has completed
    cudaDeviceSynchronize();

    // Copy data from device to host
    float outputs[num_elements];
    auto err_b = cudaMemcpy(outputs, A_d, size, cudaMemcpyDeviceToHost);
    printf("b");
    CUDA_CHECK(err_b);

    std::cout << "outputs[0] = " << outputs[0] << '\n';
    std::cout << "outputs[1] = " << outputs[1] << '\n';
    std::cout << "outputs[2] = " << outputs[2] << '\n';

    // Clean up device memory
    cudaFree(A_d);
}
