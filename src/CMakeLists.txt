add_subdirectory(libbin)

add_library(libhdc STATIC Vector.cpp)
target_include_directories(libhdc INTERFACE .)
target_link_libraries(libhdc INTERFACE libbin)

add_executable(language
    language.cpp
)
target_compile_definitions(language PUBLIC LANGUAGE)
target_link_libraries(language argparse libhdc)

add_executable(voicehd
    voicehd.cpp
)
target_compile_definitions(voicehd PUBLIC VOICEHD)
target_link_libraries(voicehd argparse libhdc)

add_executable(mnist
    mnist.cpp
)
target_compile_definitions(mnist PUBLIC MNIST)
target_link_libraries(mnist argparse libhdc)

add_executable(emg
    emg.cpp
    common.cpp
)
target_link_libraries(emg argparse libhdc)
