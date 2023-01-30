add_executable(tests
    ${CMAKE_CURRENT_LIST_DIR}/test_hdc.cpp
)
add_executable(test_vector
    ${CMAKE_CURRENT_LIST_DIR}/test_vector.cpp
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain hdc_o)
target_link_libraries(test_vector PRIVATE Catch2::Catch2WithMain hdc_o)

