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
    ../src/core/app_config.cpp
)

# GUI source files for integration tests
set(TEST_GUI_SOURCES
    ../src/gui/main_window.cpp
    ../src/gui/main_window_widgets.cpp
    ../src/gui/results_window.cpp
)

# Header files needed for MOC processing
set(TEST_COMMON_HEADERS
    ../include/duplicate_detector.h
    ../include/file_scanner.h
    ../include/hash_calculator.h
    ../include/file_manager.h
    ../include/safety_manager.h
    ../include/app_config.h
    ../include/main_window.h
    ../src/gui/results_window.h
)

# Unit Tests
set(UNIT_TEST_SOURCES
    # unit/test_duplicate_detector.cpp  # Included directly in main_test.cpp
    # unit/test_file_scanner.cpp  # Included directly in main_test.cpp
    unit/test_hash_calculator.cpp
    unit/test_file_manager.cpp
    unit/test_safety_manager.cpp
)

# Integration Tests
set(INTEGRATION_TEST_SOURCES
    integration/test_full_workflow.cpp
    integration/test_cross_platform.cpp
    integration/test_integration_workflow.cpp
    integration/test_filescanner_hashcalculator.cpp
)

# Create unit test executable
add_executable(unit_tests
    unit/main_test.cpp
    ${UNIT_TEST_SOURCES}
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
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
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(integration_tests
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

# Add tests to CTest
# Performance test sources
set(PERFORMANCE_TEST_SOURCES
    performance/test_thread_pool_performance.cpp
    performance/test_hc002b_batch_processing.cpp
    performance/test_hc002c_io_optimization.cpp
    # TODO: Add remaining performance tests
    # performance/test_progress_reporting_overhead.cpp
    # performance/test_comprehensive_integration.cpp
)

# Create performance test executable
add_executable(performance_tests
    performance/main_performance_test.cpp
    performance/performance_test_framework.cpp
    ${PERFORMANCE_TEST_SOURCES}
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(performance_tests
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(performance_tests PROPERTIES
    AUTOMOC ON
)

add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)
add_test(NAME PerformanceTests COMMAND performance_tests)

# Set test properties
set_tests_properties(UnitTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

set_tests_properties(IntegrationTests PROPERTIES
    TIMEOUT 120
    LABELS "integration"
)

set_tests_properties(PerformanceTests PROPERTIES
    TIMEOUT 600
    LABELS "performance"
)

# Enable Qt's test framework features
set_target_properties(unit_tests PROPERTIES
    AUTOMOC ON
)

set_target_properties(integration_tests PROPERTIES
    AUTOMOC ON
)

# Standalone Integration Test - Workflow
add_executable(test_integration_workflow
    integration/test_integration_workflow.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_integration_workflow
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_integration_workflow PROPERTIES
    AUTOMOC ON
)

# Standalone Integration Test - FileScanner <-> HashCalculator
add_executable(test_filescanner_hashcalculator
    integration/test_filescanner_hashcalculator.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_filescanner_hashcalculator
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_filescanner_hashcalculator PROPERTIES
    AUTOMOC ON
)

# Standalone Integration Test - FileScanner <-> DuplicateDetector
add_executable(test_filescanner_duplicatedetector
    integration/test_filescanner_duplicatedetector.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_filescanner_duplicatedetector
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_filescanner_duplicatedetector PROPERTIES
    AUTOMOC ON
)

# Standalone Integration Test - End-to-End Workflow
add_executable(test_end_to_end_workflow
    integration/test_end_to_end_workflow.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_end_to_end_workflow
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_end_to_end_workflow PROPERTIES
    AUTOMOC ON
)

# Standalone Performance Tests
add_executable(test_file_scanner_performance
    performance/test_file_scanner_performance_simple.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_file_scanner_performance
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_file_scanner_performance PROPERTIES
    AUTOMOC ON
)

add_executable(test_hc002b_batch_processing
    performance/test_hc002b_batch_processing.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_hc002b_batch_processing
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_hc002b_batch_processing PROPERTIES
    AUTOMOC ON
)

add_executable(test_hc002c_io_optimization
    performance/test_hc002c_io_optimization.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_hc002c_io_optimization
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_hc002c_io_optimization PROPERTIES
    AUTOMOC ON
)

# Add standalone tests to CTest
add_test(NAME FileScannerPerformanceTest COMMAND test_file_scanner_performance)
add_test(NAME IntegrationWorkflowTest COMMAND test_integration_workflow)
add_test(NAME FileScannerHashCalculatorTest COMMAND test_filescanner_hashcalculator)
add_test(NAME FileScannerDuplicateDetectorTest COMMAND test_filescanner_duplicatedetector)
add_test(NAME EndToEndWorkflowTest COMMAND test_end_to_end_workflow)
add_test(NAME BatchProcessingTest COMMAND test_hc002b_batch_processing)
add_test(NAME IOOptimizationTest COMMAND test_hc002c_io_optimization)

# Set standalone test properties
set_tests_properties(FileScannerPerformanceTest PROPERTIES
    TIMEOUT 600
    LABELS "performance;standalone;file_scanner"
)

set_tests_properties(IntegrationWorkflowTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone"
)

set_tests_properties(FileScannerHashCalculatorTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone;filescanner;hashcalculator"
)

set_tests_properties(FileScannerDuplicateDetectorTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone;filescanner;duplicatedetector"
)

set_tests_properties(EndToEndWorkflowTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone;end-to-end;workflow"
)

set_tests_properties(BatchProcessingTest PROPERTIES
    TIMEOUT 600
    LABELS "performance;standalone"
)

set_tests_properties(IOOptimizationTest PROPERTIES
    TIMEOUT 600
    LABELS "performance;standalone"
)

# Standalone Coverage Test - FileScanner Coverage
add_executable(test_file_scanner_coverage
    unit/test_file_scanner_coverage.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_file_scanner_coverage
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_file_scanner_coverage PROPERTIES
    AUTOMOC ON
)

add_test(NAME FileScannerCoverageTest COMMAND test_file_scanner_coverage)

set_tests_properties(FileScannerCoverageTest PROPERTIES
    TIMEOUT 300
    LABELS "unit;coverage;filescanner"
)

# Standalone Integration Test - Scan-to-Delete Workflow
add_executable(test_scan_to_delete_workflow
    integration/test_scan_to_delete_workflow.cpp
    ${TEST_COMMON_SOURCES}
    ../src/gui/results_window.cpp
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_scan_to_delete_workflow
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_scan_to_delete_workflow PROPERTIES
    AUTOMOC ON
)

add_test(NAME ScanToDeleteWorkflowTest COMMAND test_scan_to_delete_workflow)

set_tests_properties(ScanToDeleteWorkflowTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone;scan-to-delete;workflow"
)

# Standalone Integration Test - Restore Functionality
add_executable(test_restore_functionality
    integration/test_restore_functionality.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_restore_functionality
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_restore_functionality PROPERTIES
    AUTOMOC ON
)

add_test(NAME RestoreFunctionalityTest COMMAND test_restore_functionality)

set_tests_properties(RestoreFunctionalityTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone;restore;file-operations"
)

# Standalone Integration Test - Error Scenarios
add_executable(test_error_scenarios
    integration/test_error_scenarios.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(test_error_scenarios
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_error_scenarios PROPERTIES
    AUTOMOC ON
)

add_test(NAME ErrorScenariosTest COMMAND test_error_scenarios)

set_tests_properties(ErrorScenariosTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;standalone;error-handling;robustness"
)
