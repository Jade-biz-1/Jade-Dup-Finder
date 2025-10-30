# Test configuration for DupFinder
cmake_minimum_required(VERSION 3.16)
project(DupFinderTests)

find_package(Qt6 REQUIRED COMPONENTS Test Core Widgets Concurrent Sql)

# Common test settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enhanced Test Framework
set(ENHANCED_TEST_FRAMEWORK_SOURCES
    test_config.cpp
    test_base.cpp
    enhanced_test_runner.cpp
)

set(ENHANCED_TEST_FRAMEWORK_HEADERS
    test_config.h
    test_base.h
    enhanced_test_runner.h
)

# Create enhanced test framework library
add_library(enhanced_test_framework STATIC
    ${ENHANCED_TEST_FRAMEWORK_SOURCES}
    ${ENHANCED_TEST_FRAMEWORK_HEADERS}
)

target_link_libraries(enhanced_test_framework
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(enhanced_test_framework PROPERTIES
    AUTOMOC ON
)

# Test Data Management System
set(TEST_DATA_MANAGEMENT_SOURCES
    test_data_generator.cpp
    test_environment_isolator.cpp
    test_database_manager.cpp
)

set(TEST_DATA_MANAGEMENT_HEADERS
    test_data_generator.h
    test_database_manager.h
)

# Create test data management library
add_library(test_data_management STATIC
    ${TEST_DATA_MANAGEMENT_SOURCES}
    ${TEST_DATA_MANAGEMENT_HEADERS}
)

target_link_libraries(test_data_management
    Qt6::Core
    Qt6::Sql
    Qt6::Gui
)

set_target_properties(test_data_management PROPERTIES
    AUTOMOC ON
)

# Enhanced Test Framework Example
add_executable(example_enhanced_test
    example_enhanced_test.cpp
)

target_link_libraries(example_enhanced_test
    enhanced_test_framework
    Qt6::Core
    Qt6::Test
)

set_target_properties(example_enhanced_test PROPERTIES
    AUTOMOC ON
)

# Test Data Management Example
add_executable(example_test_data_management
    example_test_data_management.cpp
)

target_link_libraries(example_test_data_management
    enhanced_test_framework
    test_data_management
    Qt6::Core
    Qt6::Test
    Qt6::Sql
)

set_target_properties(example_test_data_management PROPERTIES
    AUTOMOC ON
)

# UI Automation Framework
set(UI_AUTOMATION_SOURCES
    ui_automation.cpp
    widget_selector.cpp
)

set(UI_AUTOMATION_HEADERS
    ui_automation.h
    widget_selector.h
)

# Create UI automation library
add_library(ui_automation STATIC
    ${UI_AUTOMATION_SOURCES}
    ${UI_AUTOMATION_HEADERS}
)

target_include_directories(ui_automation PRIVATE
    ${Qt6Test_INCLUDE_DIRS}
)

target_link_libraries(ui_automation
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Gui
)

set_target_properties(ui_automation PROPERTIES
    AUTOMOC ON
)

# UI Automation Example
add_executable(example_ui_automation
    example_ui_automation.cpp
)

target_link_libraries(example_ui_automation
    enhanced_test_framework
    ui_automation
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(example_ui_automation PROPERTIES
    AUTOMOC ON
)

# Visual Testing Framework
set(VISUAL_TESTING_SOURCES
    visual_testing.cpp
    theme_accessibility_testing.cpp
)

set(VISUAL_TESTING_HEADERS
    visual_testing.h
    theme_accessibility_testing.h
)

# Create visual testing library
add_library(visual_testing STATIC
    ${VISUAL_TESTING_SOURCES}
    ${VISUAL_TESTING_HEADERS}
)

target_link_libraries(visual_testing
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
)

set_target_properties(visual_testing PROPERTIES
    AUTOMOC ON
)

# Visual and Theme Testing Example
add_executable(example_visual_theme_testing
    example_visual_theme_testing.cpp
)

target_link_libraries(example_visual_theme_testing
    enhanced_test_framework
    ui_automation
    visual_testing
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(example_visual_theme_testing PROPERTIES
    AUTOMOC ON
)

# Performance Benchmarking Framework
set(PERFORMANCE_BENCHMARKS_SOURCES
    performance_benchmarks.cpp
)

set(PERFORMANCE_BENCHMARKS_HEADERS
    performance_benchmarks.h
)

# Create performance benchmarks library
add_library(performance_benchmarks STATIC
    ${PERFORMANCE_BENCHMARKS_SOURCES}
    ${PERFORMANCE_BENCHMARKS_HEADERS}
)

target_link_libraries(performance_benchmarks
    Qt6::Core
    Qt6::Concurrent
)

set_target_properties(performance_benchmarks PROPERTIES
    AUTOMOC ON
)

# Performance Testing Example
add_executable(example_performance_testing
    example_performance_testing.cpp
)

target_link_libraries(example_performance_testing
    enhanced_test_framework
    performance_benchmarks
    test_data_management
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(example_performance_testing PROPERTIES
    AUTOMOC ON
)

# Add enhanced framework tests to CTest
add_test(NAME EnhancedFrameworkExample COMMAND example_enhanced_test)
add_test(NAME TestDataManagementExample COMMAND example_test_data_management)
add_test(NAME UIAutomationExample COMMAND example_ui_automation)
add_test(NAME VisualThemeTestingExample COMMAND example_visual_theme_testing)

set_tests_properties(EnhancedFrameworkExample PROPERTIES
    TIMEOUT 60
    LABELS "enhanced;framework;example"
)

set_tests_properties(TestDataManagementExample PROPERTIES
    TIMEOUT 120
    LABELS "enhanced;data-management;example"
)

set_tests_properties(UIAutomationExample PROPERTIES
    TIMEOUT 180
    LABELS "enhanced;ui-automation;example"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

set_tests_properties(VisualThemeTestingExample PROPERTIES
    TIMEOUT 240
    LABELS "enhanced;visual;theme;accessibility;example"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

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
    ../src/core/logger.cpp
    ../src/core/file_operation_queue.cpp
)

# GUI source files for integration tests
set(TEST_GUI_SOURCES
    ../src/gui/main_window.cpp
    ../src/gui/main_window_widgets.cpp
    ../src/gui/results_window.cpp
    ../src/gui/thumbnail_cache.cpp
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
    ../include/thumbnail_cache.h
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
    ${TEST_GUI_SOURCES}
    ${TEST_COMMON_HEADERS}
)

target_link_libraries(unit_tests
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
    Qt6::Widgets  # Add Widgets for GUI components
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
    Qt6::Widgets  # Add Widgets for GUI components
)

# Add tests to CTest
# Performance test sources
set(PERFORMANCE_TEST_SOURCES
    # Individual test files with main() functions are built as separate executables
    # performance/test_thread_pool_performance.cpp      # Has its own main()
    # performance/test_hc002b_batch_processing.cpp      # Has its own main()
    # performance/test_hc002c_io_optimization.cpp       # Has its own main()
    # Additional performance tests to be implemented:
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
    Qt6::Widgets
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
    Qt6::Widgets
    Qt6::Concurrent
)

set_target_properties(test_integration_workflow PROPERTIES
    AUTOMOC ON
)

# Standalone Integration Test - FileScanner <-> HashCalculator
add_executable(test_filescanner_hashcalculator
    integration/test_filescanner_hashcalculator.cpp
    ../src/core/duplicate_detector.cpp
    ../src/core/file_scanner.cpp
    ../src/core/hash_calculator.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    ../include/duplicate_detector.h
    ../include/file_scanner.h
    ../include/hash_calculator.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(test_filescanner_hashcalculator
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
    Qt6::Widgets  # Add Widgets for GUI components
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
    Qt6::Widgets
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
    Qt6::Widgets
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
    ${TEST_GUI_SOURCES}
    ../src/core/selection_history_manager.cpp
    ../src/core/scan_history_manager.cpp
    ../src/core/theme_manager.cpp
    ../src/core/theme_error_handler.cpp
    ../src/core/theme_performance_optimizer.cpp
    ../src/core/component_registry.cpp
    ../src/core/style_validator.cpp
    ../src/core/theme_persistence.cpp
    ../src/gui/scan_dialog.cpp
    ../src/gui/settings_dialog.cpp
    ../src/gui/scan_history_dialog.cpp
    ../src/gui/restore_dialog.cpp
    ../src/gui/thumbnail_delegate.cpp
    ../src/gui/exclude_pattern_widget.cpp
    ../src/gui/preset_manager_dialog.cpp
    ../src/gui/scan_scope_preview_widget.cpp
    ../src/gui/scan_progress_dialog.cpp
    ../src/gui/scan_error_dialog.cpp
    ../src/gui/advanced_filter_dialog.cpp
    ../src/gui/file_operation_progress_dialog.cpp
    ../src/gui/grouping_options_dialog.cpp
    ../src/gui/safety_features_dialog.cpp
    ../src/gui/about_dialog.cpp
    ../src/gui/theme_editor.cpp
    ../src/gui/theme_recovery_dialog.cpp
    ../src/gui/theme_notification_widget.cpp
    ${TEST_COMMON_HEADERS}
    ../include/selection_history_manager.h
    ../include/scan_history_manager.h
    ../include/theme_manager.h
    ../include/theme_error_handler.h
    ../include/theme_performance_optimizer.h
    ../include/component_registry.h
    ../include/style_validator.h
    ../include/theme_persistence.h
    ../include/scan_dialog.h
    ../include/settings_dialog.h
    ../include/scan_history_dialog.h
    ../include/restore_dialog.h
    ../include/thumbnail_delegate.h
    ../include/exclude_pattern_widget.h
    ../include/preset_manager_dialog.h
    ../include/scan_scope_preview_widget.h
    ../include/scan_progress_dialog.h
    ../include/scan_error_dialog.h
    ../include/advanced_filter_dialog.h
    ../include/file_operation_progress_dialog.h
    ../include/grouping_options_dialog.h
    ../include/safety_features_dialog.h
    ../include/about_dialog.h
    ../include/theme_editor.h
    ../include/theme_recovery_dialog.h
    ../include/theme_notification_widget.h
    ../include/file_operation_queue.h
)

target_link_libraries(test_scan_to_delete_workflow
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Concurrent
    Qt6::Network
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
    Qt6::Widgets
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
    ${TEST_GUI_SOURCES}
    ${TEST_COMMON_HEADERS}
    ../include/file_operation_queue.h
)

target_link_libraries(test_error_scenarios
    Qt6::Core
    Qt6::Widgets
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

# Standalone Unit Test - ThumbnailCache
add_executable(test_thumbnail_cache
    unit/test_thumbnail_cache.cpp
    ../src/gui/thumbnail_cache.cpp
    ../include/thumbnail_cache.h
)

target_link_libraries(test_thumbnail_cache
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_thumbnail_cache PROPERTIES
    AUTOMOC ON
)

add_test(NAME ThumbnailCacheTest COMMAND test_thumbnail_cache)

set_tests_properties(ThumbnailCacheTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;thumbnail;cache"
)

# Standalone Unit Test - ThumbnailDelegate
add_executable(test_thumbnail_delegate
    unit/test_thumbnail_delegate.cpp
    ../src/gui/thumbnail_cache.cpp
    ../src/gui/thumbnail_delegate.cpp
    ../include/thumbnail_cache.h
    ../include/thumbnail_delegate.h
)

target_link_libraries(test_thumbnail_delegate
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_thumbnail_delegate PROPERTIES
    AUTOMOC ON
)

add_test(NAME ThumbnailDelegateTest COMMAND test_thumbnail_delegate)

set_tests_properties(ThumbnailDelegateTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;thumbnail;delegate;ui"
)

# Standalone Unit Test - ExcludePatternWidget
add_executable(test_exclude_pattern_widget
    unit/test_exclude_pattern_widget.cpp
    ../src/gui/exclude_pattern_widget.cpp
    ../include/exclude_pattern_widget.h
)

target_link_libraries(test_exclude_pattern_widget
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_exclude_pattern_widget PROPERTIES
    AUTOMOC ON
)

add_test(NAME ExcludePatternWidgetTest COMMAND test_exclude_pattern_widget)

set_tests_properties(ExcludePatternWidgetTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;exclude;pattern;widget;ui"
)

# Standalone Unit Test - PresetManager
add_executable(test_preset_manager
    unit/test_preset_manager.cpp
    ../src/gui/preset_manager_dialog.cpp
    ../src/gui/scan_dialog.cpp
    ../src/gui/exclude_pattern_widget.cpp
    ../src/core/logger.cpp
    ../include/preset_manager_dialog.h
    ../include/scan_dialog.h
    ../include/exclude_pattern_widget.h
    ../src/core/logger.h
)

target_link_libraries(test_preset_manager
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_preset_manager PROPERTIES
    AUTOMOC ON
)

add_test(NAME PresetManagerTest COMMAND test_preset_manager)

set_tests_properties(PresetManagerTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;preset;manager;ui"
)

# Standalone Unit Test - ScanConfigurationValidation
add_executable(test_scan_configuration_validation
    unit/test_scan_configuration_validation.cpp
    ../src/gui/scan_dialog.cpp
    ../src/gui/exclude_pattern_widget.cpp
    ../src/core/logger.cpp
    ../include/scan_dialog.h
    ../include/exclude_pattern_widget.h
    ../src/core/logger.h
)

target_link_libraries(test_scan_configuration_validation
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_scan_configuration_validation PROPERTIES
    AUTOMOC ON
)

add_test(NAME ScanConfigurationValidationTest COMMAND test_scan_configuration_validation)

set_tests_properties(ScanConfigurationValidationTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;scan;validation;configuration"
)

# Standalone Unit Test - ScanScopePreviewWidget
add_executable(test_scan_scope_preview_widget
    unit/test_scan_scope_preview_widget.cpp
    ../src/gui/scan_scope_preview_widget.cpp
    ../src/core/logger.cpp
    ../include/scan_scope_preview_widget.h
    ../src/core/logger.h
)

target_link_libraries(test_scan_scope_preview_widget
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_scan_scope_preview_widget PROPERTIES
    AUTOMOC ON
)

add_test(NAME ScanScopePreviewWidgetTest COMMAND test_scan_scope_preview_widget)

set_tests_properties(ScanScopePreviewWidgetTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;scan;scope;preview;widget;ui"
)

# Standalone Unit Test - ScanProgressTracking (Task 7)
add_executable(test_scan_progress_tracking
    unit/test_scan_progress_tracking.cpp
    ../src/core/file_scanner.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    ../include/file_scanner.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(test_scan_progress_tracking
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_scan_progress_tracking PROPERTIES
    AUTOMOC ON
)

add_test(NAME ScanProgressTrackingTest COMMAND test_scan_progress_tracking)

set_tests_properties(ScanProgressTrackingTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;scan;progress;tracking;task7"
)

# Standalone Unit Test - ScanProgressDialog (Task 8)
add_executable(test_scan_progress_dialog
    unit/test_scan_progress_dialog.cpp
    ../src/gui/scan_progress_dialog.cpp
    ../include/scan_progress_dialog.h
)

target_link_libraries(test_scan_progress_dialog
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Test
)

set_target_properties(test_scan_progress_dialog PROPERTIES
    AUTOMOC ON
)

add_test(NAME ScanProgressDialogTest COMMAND test_scan_progress_dialog)

set_tests_properties(ScanProgressDialogTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;scan;progress;dialog;task8"
)
# UI Theme Integration Test
add_executable(test_ui_theme_integration
    test_ui_theme_integration.cpp
    ../src/core/ui_theme_test_integration.cpp
    ../src/core/theme_manager.cpp
    ../src/core/theme_error_handler.cpp
    ../src/core/component_registry.cpp
    ../src/core/style_validator.cpp
    ../src/core/theme_persistence.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    ../include/ui_theme_test_integration.h
    ../include/theme_manager.h
    ../include/theme_error_handler.h
    ../include/component_registry.h
    ../include/style_validator.h
    ../include/theme_persistence.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(test_ui_theme_integration
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Gui
)

set_target_properties(test_ui_theme_integration PROPERTIES
    AUTOMOC ON
)

add_test(NAME UIThemeIntegrationTest COMMAND test_ui_theme_integration)

set_tests_properties(UIThemeIntegrationTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;theme;ui;automation"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# Comprehensive Theme UI Workflow Tests (Task 11.1 & 11.2)
add_executable(theme_ui_workflow_tests
    theme_ui_workflow_tests.cpp
    workflow_testing.cpp
    user_scenario_testing.cpp
    ui_automation.cpp
    visual_testing.cpp
    theme_accessibility_testing.cpp
    ../src/core/ui_theme_test_integration.cpp
    ../src/core/theme_manager.cpp
    ../src/core/theme_error_handler.cpp
    ../src/core/component_registry.cpp
    ../src/core/style_validator.cpp
    ../src/core/theme_persistence.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    theme_ui_workflow_tests.h
    workflow_testing.h
    user_scenario_testing.h
    ui_automation.h
    visual_testing.h
    theme_accessibility_testing.h
    ../include/ui_theme_test_integration.h
    ../include/theme_manager.h
    ../include/theme_error_handler.h
    ../include/component_registry.h
    ../include/style_validator.h
    ../include/theme_persistence.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(theme_ui_workflow_tests
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Gui
    Qt6::Concurrent
    enhanced_test_framework
    ui_automation
    visual_testing
)

set_target_properties(theme_ui_workflow_tests PROPERTIES
    AUTOMOC ON
)

add_test(NAME ThemeUIWorkflowTests COMMAND theme_ui_workflow_tests)

set_tests_properties(ThemeUIWorkflowTests PROPERTIES
    TIMEOUT 600
    LABELS "integration;theme;ui;workflow;comprehensive;task11"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# Cross-Theme Interaction Validation Tests (Task 11.2)
add_executable(cross_theme_interaction_tests
    cross_theme_interaction_tests.cpp
    ui_automation.cpp
    visual_testing.cpp
    theme_accessibility_testing.cpp
    ../src/core/ui_theme_test_integration.cpp
    ../src/core/theme_manager.cpp
    ../src/core/theme_error_handler.cpp
    ../src/core/component_registry.cpp
    ../src/core/style_validator.cpp
    ../src/core/theme_persistence.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    ui_automation.h
    visual_testing.h
    theme_accessibility_testing.h
    ../include/ui_theme_test_integration.h
    ../include/theme_manager.h
    ../include/theme_error_handler.h
    ../include/component_registry.h
    ../include/style_validator.h
    ../include/theme_persistence.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(cross_theme_interaction_tests
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Gui
    enhanced_test_framework
    ui_automation
    visual_testing
)

set_target_properties(cross_theme_interaction_tests PROPERTIES
    AUTOMOC ON
)

add_test(NAME CrossThemeInteractionTests COMMAND cross_theme_interaction_tests)

set_tests_properties(CrossThemeInteractionTests PROPERTIES
    TIMEOUT 400
    LABELS "integration;theme;ui;interaction;cross-theme;task11.2"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# Add validation subdirectory
add_subdirectory(validation)

# Add framework subdirectory if not already added
if(NOT TARGET test_harness)
    add_subdirectory(framework)
endif()

# Create comprehensive test target that runs all validation
add_custom_target(validate_all
    COMMAND ${CMAKE_COMMAND} -E echo "Running comprehensive test suite validation..."
    COMMAND $<TARGET_FILE:test_suite_validator> --output ${CMAKE_BINARY_DIR}/validation_reports
    COMMAND ${CMAKE_COMMAND} -E echo "Running performance scalability validation..."
    COMMAND $<TARGET_FILE:performance_scalability_validator> --output ${CMAKE_BINARY_DIR}/validation_reports
    DEPENDS test_suite_validator performance_scalability_validator
    COMMENT "Running all validation tests"
)

# Create deployment target
add_custom_target(deploy_testing_infrastructure
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/deployment/deploy_testing_infrastructure.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Deploying testing infrastructure"
)

# Comprehensive Theme Validation Test (Task 12.2)
add_executable(comprehensive_theme_validation
    comprehensive_theme_validation.cpp
    ../src/core/theme_manager.cpp
    ../src/core/theme_error_handler.cpp
    ../src/core/theme_performance_optimizer.cpp
    ../src/core/component_registry.cpp
    ../src/core/style_validator.cpp
    ../src/core/theme_persistence.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    ../include/theme_manager.h
    ../include/theme_error_handler.h
    ../include/theme_performance_optimizer.h
    ../include/component_registry.h
    ../include/style_validator.h
    ../include/theme_persistence.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(comprehensive_theme_validation
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Gui
)

set_target_properties(comprehensive_theme_validation PROPERTIES
    AUTOMOC ON
)

add_test(NAME ComprehensiveThemeValidation COMMAND comprehensive_theme_validation)

set_tests_properties(ComprehensiveThemeValidation PROPERTIES
    TIMEOUT 600
    LABELS "comprehensive;theme;validation;performance;accessibility"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# Final Theme Validation Test (Task 12.3)
add_executable(final_theme_validation_test
    final_theme_validation_test.cpp
    ../src/core/final_theme_validator.cpp
    ../src/core/theme_manager.cpp
    ../src/core/theme_error_handler.cpp
    ../src/core/theme_performance_optimizer.cpp
    ../src/core/component_registry.cpp
    ../src/core/style_validator.cpp
    ../src/core/theme_persistence.cpp
    ../src/core/app_config.cpp
    ../src/core/logger.cpp
    ../include/final_theme_validator.h
    ../include/theme_manager.h
    ../include/theme_error_handler.h
    ../include/theme_performance_optimizer.h
    ../include/component_registry.h
    ../include/style_validator.h
    ../include/theme_persistence.h
    ../include/app_config.h
    ../src/core/logger.h
)

target_link_libraries(final_theme_validation_test
    Qt6::Core
    Qt6::Widgets
    Qt6::Test
    Qt6::Gui
)

set_target_properties(final_theme_validation_test PROPERTIES
    AUTOMOC ON
)

add_test(NAME FinalThemeValidationTest COMMAND final_theme_validation_test)

set_tests_properties(FinalThemeValidationTest PROPERTIES
    TIMEOUT 300
    LABELS "final;validation;documentation;certification"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)