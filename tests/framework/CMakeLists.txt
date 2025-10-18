# Test Framework CMake Configuration

cmake_minimum_required(VERSION 3.16)
project(DupFinderTestFramework)

# Find required Qt components
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Test Concurrent)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Test Framework Library
set(TEST_FRAMEWORK_SOURCES
    test_harness.cpp
    test_environment.cpp
    test_reporting.cpp
    test_utilities.cpp
)

set(TEST_FRAMEWORK_HEADERS
    test_harness.h
    test_environment.h
    test_reporting.h
    test_utilities.h
)

# Create test framework library
add_library(test_framework STATIC
    ${TEST_FRAMEWORK_SOURCES}
    ${TEST_FRAMEWORK_HEADERS}
)

target_link_libraries(test_framework
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_framework PROPERTIES
    AUTOMOC ON
)

# Example test executable
add_executable(example_test_suite
    example_test_suite.cpp
)

target_link_libraries(example_test_suite
    test_framework
    Qt6::Core
    Qt6::Test
)

set_target_properties(example_test_suite PROPERTIES
    AUTOMOC ON
)

# Test framework unit tests (optional - create if needed)
# add_executable(test_framework_tests
#     test_framework_tests.cpp
# )
# 
# target_link_libraries(test_framework_tests
#     test_framework
#     Qt6::Core
#     Qt6::Test
# )
# 
# set_target_properties(test_framework_tests PROPERTIES
#     AUTOMOC ON
# )

# Add tests to CTest
add_test(NAME ExampleTestSuite COMMAND example_test_suite)

set_tests_properties(ExampleTestSuite PROPERTIES
    TIMEOUT 60
    LABELS "example;framework"
)

# Installation
install(TARGETS test_framework
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${TEST_FRAMEWORK_HEADERS}
    DESTINATION include/test_framework
)

# Documentation generation (if Doxygen is available)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(test_framework_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating test framework documentation with Doxygen"
        VERBATIM
    )
endif()

# Code coverage (if gcov/lcov is available)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(test_framework PRIVATE --coverage)
        target_link_libraries(test_framework PRIVATE --coverage)
        
        find_program(LCOV_PATH lcov)
        find_program(GENHTML_PATH genhtml)
        
        if(LCOV_PATH AND GENHTML_PATH)
            add_custom_target(coverage
                COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${LCOV_PATH} --list coverage.info
                COMMAND ${GENHTML_PATH} -o coverage coverage.info
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating code coverage report"
            )
        endif()
    endif()
endif()

# Performance profiling (if available)
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(ENABLE_PROFILING)
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(test_framework PRIVATE -pg)
        target_link_libraries(test_framework PRIVATE -pg)
    endif()
endif()

# Static analysis integration (disabled for now to reduce noise)
# find_program(CLANG_TIDY_PATH clang-tidy)
# if(CLANG_TIDY_PATH)
#     set_target_properties(test_framework PROPERTIES
#         CXX_CLANG_TIDY "${CLANG_TIDY_PATH};-checks=-*,readability-*,performance-*,modernize-*"
#     )
# endif()

# Sanitizers (for debug builds)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(ENABLE_SANITIZERS "Enable sanitizers" OFF)
    if(ENABLE_SANITIZERS)
        if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(test_framework PRIVATE 
                -fsanitize=address 
                -fsanitize=undefined
                -fno-omit-frame-pointer
            )
            target_link_libraries(test_framework PRIVATE 
                -fsanitize=address 
                -fsanitize=undefined
            )
        endif()
    endif()
endif()