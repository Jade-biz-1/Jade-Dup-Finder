# Test configuration for DupFinder
find_package(Qt6 REQUIRED COMPONENTS Test)

# Common test settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for tests
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Common source files that tests might need
set(TEST_COMMON_SOURCES
    ../src/core/duplicate_detector.cpp
    ../src/core/file_scanner.cpp
    ../src/core/hash_calculator.cpp
    ../src/core/file_manager.cpp
    ../src/core/safety_manager.cpp
)

# Unit Tests
set(UNIT_TEST_SOURCES
    unit/test_duplicate_detector.cpp
    unit/test_file_scanner.cpp
    unit/test_hash_calculator.cpp
    unit/test_file_manager.cpp
    unit/test_safety_manager.cpp
)

# Integration Tests
set(INTEGRATION_TEST_SOURCES
    integration/test_full_workflow.cpp
    integration/test_cross_platform.cpp
)

# Create unit test executable
add_executable(unit_tests
    unit/main_test.cpp
    ${UNIT_TEST_SOURCES}
    ${TEST_COMMON_SOURCES}
)

target_link_libraries(unit_tests
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

# Create integration test executable
add_executable(integration_tests
    integration/main_test.cpp
    ${INTEGRATION_TEST_SOURCES}
    ${TEST_COMMON_SOURCES}
)

target_link_libraries(integration_tests
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

# Add tests to CTest
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)

# Set test properties
set_tests_properties(UnitTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

set_tests_properties(IntegrationTests PROPERTIES
    TIMEOUT 120
    LABELS "integration"
)

# Enable Qt's test framework features
set_target_properties(unit_tests PROPERTIES
    AUTOMOC ON
)

set_target_properties(integration_tests PROPERTIES
    AUTOMOC ON
)