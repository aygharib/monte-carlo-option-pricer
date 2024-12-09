#include <iostream>

// Compute vector sum C = A + B
// Each thread performs one pair-wise addition
__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    // Performs addition on the parameters passed in
    // These are the device memory arrays since that's what was passed when the host called this function
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n)
{
    // Host code allocates A_d, B_d, C_d of appropriate size on the device global memory
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // Host code copies from host memory to device global memory
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Invoke device code to operate on array copies so computation is done in parallel
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Host code copies from device global memory to host memory
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // Host code frees the allocated memory from the device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    float A[] = {1.0, 2.0, 3.0};
    float B[] = {4.0, 5.0, 6.0};
    float C[] = {0.0, 0.0, 0.0};
    int n = 3;

    vecAdd(A, B, C, n);

    std::cout << C[0] << '\n';
    std::cout << C[1] << '\n';
    std::cout << C[2] << '\n';
}
