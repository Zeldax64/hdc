cmake_minimum_required(VERSION 3.11)
project(libhdc VERSION 0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Create compile_commands.json

#add_executable(${CMAKE_PROJECT_NAME} "")
add_library(language_o OBJECT "")
#target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE src)

include(src/CMakeLists.txt)
include(lib/CMakeLists.txt)
include(test/CMakeLists.txt)
