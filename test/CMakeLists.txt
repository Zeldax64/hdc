add_executable(test_vector
    test_vector.cpp
)

target_link_libraries(test_vector PRIVATE Catch2::Catch2WithMain hdc_hpp)

add_executable(benchmark
    benchmark.cpp
)
target_link_libraries(benchmark PRIVATE Catch2::Catch2WithMain hdc_hpp)
