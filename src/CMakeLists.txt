target_sources(language_o
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/HDV.cpp
    ${CMAKE_CURRENT_LIST_DIR}/language.cpp
)
target_include_directories(language_o PUBLIC src)

#target_sources(${CMAKE_PROJECT_NAME}
#    PRIVATE
#    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
#)
add_executable(language
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)
target_link_libraries(language language_o)
