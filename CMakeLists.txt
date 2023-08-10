cmake_minimum_required(VERSION 3.11)
project(libhdc VERSION 0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Create compile_commands.json

include(src/CMakeLists.txt)
add_subdirectory(lib)
include(test/CMakeLists.txt)
