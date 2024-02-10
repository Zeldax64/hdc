add_library(hdc_hpp INTERFACE)
target_include_directories(hdc_hpp INTERFACE .)

add_executable(language
    language.cpp
)
target_compile_definitions(language PUBLIC LANGUAGE)
target_link_libraries(language argparse)

add_executable(voicehd
    voicehd.cpp
)
target_compile_definitions(voicehd PUBLIC VOICEHD)
target_link_libraries(voicehd argparse)

add_executable(mnist
    mnist.cpp
)
target_compile_definitions(mnist PUBLIC MNIST)
target_link_libraries(mnist argparse)

add_executable(emg
    emg.cpp
    common.cpp
)
target_compile_definitions(emg PUBLIC EMG)
target_link_libraries(emg argparse)
