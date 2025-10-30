# Theme Compliance Checker - Static Analysis Tool
cmake_minimum_required(VERSION 3.16)

project(theme_compliance_checker VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt
find_package(Qt6 COMPONENTS Core REQUIRED)

# Build the tool
add_executable(theme_compliance_checker
    theme_compliance_checker.cpp
)

target_link_libraries(theme_compliance_checker
    Qt6::Core
)

# Install to tools directory
install(TARGETS theme_compliance_checker
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/tools
)

# Add a custom target to run the checker
add_custom_target(check-theme-compliance
    COMMAND theme_compliance_checker ${CMAKE_SOURCE_DIR}/src
    DEPENDS theme_compliance_checker
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running theme compliance checker on source code"
)
