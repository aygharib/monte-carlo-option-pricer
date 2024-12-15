#include <iostream>

#include <cuda/api.hpp>

auto main() -> int {
    auto device_count = cuda::device::count();
	std::cout << device_count << '\n';
    std::cout << "hello world\n";
}
