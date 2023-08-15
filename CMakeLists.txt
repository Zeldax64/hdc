cmake_minimum_required(VERSION 3.11)
project(libhdc VERSION 0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Create compile_commands.json

add_subdirectory(src)
add_subdirectory(lib)
add_subdirectory(test)
