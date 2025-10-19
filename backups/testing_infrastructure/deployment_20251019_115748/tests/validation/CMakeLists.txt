# Validation Tools CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Find required Qt components
find_package(Qt6 REQUIRED COMPONENTS Core Test Concurrent)

# Test Suite Validator
set(TEST_SUITE_VALIDATOR_SOURCES
    test_suite_validator.cpp
    main_validation.cpp
)

set(TEST_SUITE_VALIDATOR_HEADERS
    test_suite_validator.h
)

add_executable(test_suite_validator
    ${TEST_SUITE_VALIDATOR_SOURCES}
    ${TEST_SUITE_VALIDATOR_HEADERS}
)

target_link_libraries(test_suite_validator
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(test_suite_validator PROPERTIES
    AUTOMOC ON
)

# Performance Scalability Validator
set(PERFORMANCE_SCALABILITY_VALIDATOR_SOURCES
    performance_scalability_validator.cpp
    main_performance_scalability.cpp
)

set(PERFORMANCE_SCALABILITY_VALIDATOR_HEADERS
    performance_scalability_validator.h
)

add_executable(performance_scalability_validator
    ${PERFORMANCE_SCALABILITY_VALIDATOR_SOURCES}
    ${PERFORMANCE_SCALABILITY_VALIDATOR_HEADERS}
)

target_link_libraries(performance_scalability_validator
    Qt6::Core
    Qt6::Test
    Qt6::Concurrent
)

set_target_properties(performance_scalability_validator PROPERTIES
    AUTOMOC ON
)

# Add validation tests to CTest
add_test(NAME ComprehensiveTestSuiteValidation 
         COMMAND test_suite_validator --output ${CMAKE_BINARY_DIR}/validation_reports)

add_test(NAME PerformanceScalabilityValidation 
         COMMAND performance_scalability_validator --output ${CMAKE_BINARY_DIR}/validation_reports)

# Set test properties
set_tests_properties(ComprehensiveTestSuiteValidation PROPERTIES
    TIMEOUT 1800  # 30 minutes
    LABELS "validation;comprehensive;suite"
)

set_tests_properties(PerformanceScalabilityValidation PROPERTIES
    TIMEOUT 3600  # 60 minutes
    LABELS "validation;performance;scalability"
)

# Create validation reports directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/validation_reports)

# Install validation tools
install(TARGETS test_suite_validator performance_scalability_validator
        RUNTIME DESTINATION bin/validation
)

# Integration Deployment Test
add_executable(integration_deployment_test
    integration_deployment_test.cpp
)

target_link_libraries(integration_deployment_test
    Qt6::Core
    Qt6::Test
)

set_target_properties(integration_deployment_test PROPERTIES
    AUTOMOC ON
)

# Add integration deployment test to CTest
add_test(NAME IntegrationDeploymentTest 
         COMMAND integration_deployment_test)

set_tests_properties(IntegrationDeploymentTest PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;integration;deployment"
)

# Install validation tools
install(TARGETS test_suite_validator performance_scalability_validator integration_deployment_test
        RUNTIME DESTINATION bin/validation
)

# Install validation scripts
install(FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/../deployment/deploy_testing_infrastructure.sh
        DESTINATION bin/deployment
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)