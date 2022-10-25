add_library(hdv_o OBJECT ${CMAKE_CURRENT_LIST_DIR}/HDV.cpp)
target_include_directories(hdv_o PUBLIC src)

add_executable(language
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/language.cpp
)
target_compile_definitions(language PUBLIC LANGUAGE)
target_link_libraries(language hdv_o)

add_executable(voicehd
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/voicehd.cpp
)
target_compile_definitions(voicehd PUBLIC VOICEHD)
target_link_libraries(voicehd hdv_o)

add_executable(mnist
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mnist.cpp
)
target_compile_definitions(mnist PUBLIC MNIST)
target_link_libraries(mnist hdv_o)

add_executable(emg
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/emg.cpp
)
target_compile_definitions(emg PUBLIC EMG)
target_link_libraries(emg hdv_o)
