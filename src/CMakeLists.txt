add_library(hdv_o OBJECT ${CMAKE_CURRENT_LIST_DIR}/HDV.cpp)
target_include_directories(hdv_o PUBLIC src)

add_executable(language
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/language.cpp
)
target_link_libraries(language hdv_o)
