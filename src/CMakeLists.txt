add_library(hdc_o OBJECT
    HDV.cpp
    )
target_include_directories(hdc_o PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(hdc_o PUBLIC cxx_std_17)

add_executable(language
    language.cpp
)
target_compile_definitions(language PUBLIC LANGUAGE)
target_link_libraries(language hdc_o argparse)

add_executable(voicehd
    voicehd.cpp
)
target_compile_definitions(voicehd PUBLIC VOICEHD)
target_link_libraries(voicehd hdc_o argparse)

add_executable(mnist
    main.cpp
    mnist.cpp
)
target_compile_definitions(mnist PUBLIC MNIST)
target_link_libraries(mnist hdc_o)

add_executable(emg
    main.cpp
    emg.cpp
    common.cpp
)
target_compile_definitions(emg PUBLIC EMG)
target_link_libraries(emg hdc_o)
