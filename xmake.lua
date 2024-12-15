add_rules("mode.debug", "mode.release")

set_languages("c++23")

add_requires("conan::cuda-api-wrappers/0.7.0-b2", { alias = "cuda-api-wrappers", system = false })

target("monte-carlo-option-pricer")
set_kind("binary")
-- add_files("src/*.cpp")
add_files("src/*.cu")
-- add_includedirs("headers")

-- on_load(function (target)
--     -- Run `nvcc --version` and capture the output
--     local version_output = os.iorun("nvcc --version")
--     if version_output then
--         -- Extract major and minor version from the output
--         local major, minor = version_output:match("release (%d+)%.(%d+)")
--         if major and minor then
--             local cuda_version = tonumber(major) * 1000 + tonumber(minor) * 10
--             print("Detected CUDA Version: " .. cuda_version)
--             -- Add a definition for the CUDA version
--             target:add("defines", "CUDA_VERSION=" .. cuda_version)
--         else
--             print("Failed to parse CUDA version.")
--         end
--     else
--         print("nvcc not found. Ensure CUDA toolkit is installed and in PATH.")
--     end
-- end)

-- print("hello1")
-- if os.getenv("CUDA_VERSION") then
--     print("hello2")
--     local cuda_version = tonumber(os.getenv("CUDA_VERSION"))
--     local version_output = io.readfile("ahmad-version.txt")
--     if version_output then
--         -- Extract major and minor version
--         local major, minor = version_output:match("release (%d+)%.(%d+)")
--         if major and minor then
--             local cuda_version = tonumber(major) * 1000 + tonumber(minor) * 10
--             target:add("defines", "CUDA_VERSION=" .. cuda_version)
--         end
--     end
-- else
--     print("hello3")
-- end

-- generate SASS code for SM architecture of current host
add_cugencodes("native")
add_links("cuda")
-- -- generate PTX code for the virtual architecture to guarantee compatibility
-- add_cugencodes("compute_30")
add_packages("cuda-api-wrappers")
