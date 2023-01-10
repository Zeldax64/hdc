add_executable(tests
    ${CMAKE_CURRENT_LIST_DIR}/test_hdc.cpp
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain hdc_o)

