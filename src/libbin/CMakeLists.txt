add_library(
    libbin
    STATIC
    bitmanip.cpp
    )

target_compile_options(
    libbin
    PRIVATE "-mbmi"
    PRIVATE "-mbmi2"
    PRIVATE "-mavx2")

target_include_directories(libbin PUBLIC .)

add_custom_command(TARGET libbin POST_BUILD
        COMMAND objdump ARGS --section=.text --visualize-jumps --source -M intel -CD "$<TARGET_FILE:libbin>" > "$<TARGET_FILE:libbin>.dump"
        COMMENT "Invoking: objdump $<TARGET_FILE_BASE_NAME:libbin>")