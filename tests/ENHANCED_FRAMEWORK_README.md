# Enhanced Test Framework

This document describes the enhanced test framework for CloneClean, which provides standardized test infrastructure, configuration management, and improved test organization.

## Overview

The enhanced test framework builds upon Qt Test Framework to provide:

- **Standardized test structure** with consistent naming conventions
- **Configuration-based test management** for flexible test execution
- **Test categorization and tagging** for better organization
- **Enhanced reporting and statistics** for comprehensive test analysis
- **Parallel execution support** for improved performance
- **Platform and environment awareness** for conditional testing

## Key Components

### 1. TestConfig (`test_config.h/cpp`)

Provides centralized configuration management for all tests:

- **Categories**: Unit, Integration, Performance, UI, EndToEnd, Security, Regression, Smoke
- **Priorities**: Critical, High, Medium, Low
- **Execution Modes**: Sequential, Parallel, Isolated
- **Filtering**: By category, tags, or specific test names
- **Environment Detection**: CI/CD, platform-specific settings

### 2. TestBase (`test_base.h/cpp`)

Base class for all test classes providing:

- **Standardized lifecycle hooks** (setup/teardown)
- **Enhanced assertion macros** with detailed logging
- **Test data management** (automatic cleanup)
- **Performance measurement** utilities
- **Platform detection** and conditional execution
- **Logging and reporting** integration

### 3. EnhancedTestRunner (`enhanced_test_runner.h/cpp`)

Advanced test execution engine supporting:

- **Configuration-based filtering** and selection
- **Parallel execution** with resource management
- **Comprehensive reporting** (JSON, JUnit, HTML)
- **Progress tracking** and statistics
- **CI/CD integration** features

## Usage

### Creating a Test Class

Use the standardized macros to create test classes:

```cpp
#include "test_base.h"

DECLARE_TEST_CLASS(MyComponentTest, Unit, High, "component", "core")

private slots:
    // Test method using naming convention
    TEST_METHOD(test_component_validInput_returnsExpectedResult) {
        logTestStep("Testing component with valid input");
        
        // Enhanced assertions with logging
        TEST_VERIFY_WITH_MSG(condition, "Condition should be true");
        TEST_COMPARE_WITH_MSG(actual, expected, "Values should match");
        
        logTestStep("Test completed successfully");
    }

    TEST_METHOD(test_component_invalidInput_throwsException) {
        // Test implementation
    }

END_TEST_CLASS()
```

### Test Naming Convention

Follow the standardized naming pattern:

- **Test Classes**: `<Component>Test` (e.g., `FileManagerTest`, `HashCalculatorTest`)
- **Test Methods**: `test_<component>_<scenario>_<expectedResult>`
  - `test_fileManager_emptyDirectory_returnsEmptyList`
  - `test_hashCalculator_largeFile_completesWithinTimeout`
  - `test_safetyManager_invalidPath_throwsException`

### Configuration

Create or modify `test_config.json` to control test execution:

```json
{
  "global": {
    "defaultExecutionMode": "Sequential",
    "verboseOutput": true,
    "enabledCategories": ["Unit", "Integration"],
    "disabledTests": ["SlowPerformanceTest"]
  },
  "testSuites": {
    "MyTest": {
      "category": "Unit",
      "priority": "High",
      "tags": ["core", "critical"],
      "timeoutSeconds": 120
    }
  }
}
```

### Running Tests

#### Command Line

```bash
# Run all tests
./enhanced_test_runner

# Run specific categories
./enhanced_test_runner --categories Unit,Integration

# Run tests with specific tags
./enhanced_test_runner --tags core,critical

# Run with verbose output
./enhanced_test_runner --verbose

# Generate reports
./enhanced_test_runner --report-path ./reports/
```

#### Programmatic

```cpp
#include "enhanced_test_runner.h"

EnhancedTestRunner runner;
runner.loadConfiguration("test_config.json");
runner.setEnabledCategories({"Unit", "Integration"});
runner.runAllTests();

// Generate reports
runner.generateReport("test_results.json");
runner.generateJUnitReport("junit_results.xml");
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Execution**: Fast, parallel execution
- **Examples**: HashCalculator, FileManager, SafetyManager

### Integration Tests
- **Purpose**: Test component interactions and workflows
- **Execution**: Sequential, may require setup/teardown
- **Examples**: FileScanner-HashCalculator integration, workflow tests

### Performance Tests
- **Purpose**: Benchmark and performance validation
- **Execution**: Isolated, longer timeouts
- **Examples**: Large file processing, memory usage tests

### UI Tests
- **Purpose**: User interface and visual testing
- **Execution**: Sequential, requires display
- **Examples**: Widget behavior, theme testing

### End-to-End Tests
- **Purpose**: Complete user workflows
- **Execution**: Sequential, comprehensive setup
- **Examples**: Full scan-to-delete workflow

### Security Tests
- **Purpose**: Safety and security validation
- **Execution**: Sequential, careful isolation
- **Examples**: File operation safety, input validation

## Best Practices

### 1. Test Organization

- Group related tests in the same test class
- Use descriptive test method names following the convention
- Add appropriate tags for filtering and organization

### 2. Test Data Management

- Use `createTestDirectory()` and `createTestFile()` for test data
- Automatic cleanup is handled by the framework
- Avoid hardcoded paths or external dependencies

### 3. Performance Testing

- Use `startPerformanceMeasurement()` and `stopPerformanceMeasurement()`
- Record metrics with `recordPerformanceMetric()`
- Set appropriate timeouts for performance tests

### 4. Conditional Testing

- Use platform detection for platform-specific tests
- Use CI detection for environment-specific behavior
- Skip tests gracefully when conditions aren't met

### 5. Error Handling

- Use enhanced assertion macros for better error reporting
- Provide meaningful error messages
- Log test steps for debugging

## Configuration Reference

### Global Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `defaultExecutionMode` | String | Sequential/Parallel/Isolated | "Sequential" |
| `defaultTimeoutSeconds` | Integer | Default test timeout | 300 |
| `verboseOutput` | Boolean | Enable detailed logging | false |
| `generateReports` | Boolean | Generate test reports | true |
| `reportOutputDirectory` | String | Report output path | "test_reports" |
| `maxParallelTests` | Integer | Max concurrent tests | 4 |
| `stopOnFirstFailure` | Boolean | Stop on first failure | false |
| `enabledCategories` | Array | Enabled test categories | All |
| `enabledTags` | Array | Enabled test tags | All |
| `disabledTests` | Array | Disabled test names | None |

### Test Suite Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `name` | String | Test suite name |
| `category` | String | Test category |
| `priority` | String | Test priority |
| `timeoutSeconds` | Integer | Test timeout |
| `enabledByDefault` | Boolean | Default enabled state |
| `executionMode` | String | Execution mode |
| `tags` | Array | Test tags |
| `customProperties` | Object | Custom configuration |

## Integration with CI/CD

The enhanced test framework provides built-in CI/CD integration:

- **Environment Detection**: Automatically detects CI environments
- **Report Generation**: JUnit XML for CI integration
- **Exit Codes**: Proper exit codes for build systems
- **Parallel Execution**: Optimized for CI environments
- **Timeout Management**: Prevents hanging builds

### GitHub Actions Example

```yaml
- name: Run Tests
  run: |
    cd tests
    cmake --build build
    ./build/enhanced_test_runner --categories Unit,Integration --report-path ./reports/
    
- name: Upload Test Results
  uses: actions/upload-artifact@v2
  with:
    name: test-results
    path: tests/reports/
```

## Migration Guide

### From Existing Tests

1. **Include the new headers**:
   ```cpp
   #include "test_base.h"
   ```

2. **Update class declaration**:
   ```cpp
   // Old
   class MyTest : public QObject {
   
   // New
   DECLARE_TEST_CLASS(MyTest, Unit, High, "tag1", "tag2")
   ```

3. **Update test methods**:
   ```cpp
   // Old
   void testSomething() {
       QVERIFY(condition);
   }
   
   // New
   TEST_METHOD(test_component_scenario_expectedResult) {
       TEST_VERIFY_WITH_MSG(condition, "Meaningful message");
   }
   ```

4. **Add configuration**:
   - Create or update `test_config.json`
   - Register test suite with appropriate category and tags

### Gradual Migration

The enhanced framework is designed to coexist with existing tests:

1. Start with new tests using the enhanced framework
2. Gradually migrate existing tests
3. Update build configuration to use enhanced runner
4. Retire old test infrastructure when migration is complete

## Troubleshooting

### Common Issues

1. **Test not running**: Check configuration filters and enabled categories
2. **Timeout errors**: Increase timeout in test configuration
3. **Parallel execution issues**: Use Sequential mode for problematic tests
4. **Missing reports**: Ensure report directory exists and is writable

### Debug Mode

Enable verbose output for detailed debugging:

```json
{
  "global": {
    "verboseOutput": true
  }
}
```

### Log Analysis

The framework provides detailed logging:

- `[INFO]`: General information
- `[STEP]`: Test execution steps
- `[PASS]`: Successful assertions
- `[FAIL]`: Failed assertions
- `[WARN]`: Warnings and issues

## Future Enhancements

Planned improvements for the enhanced test framework:

1. **Visual Testing**: Screenshot comparison and UI regression testing
2. **Code Coverage**: Integrated coverage reporting
3. **Test Generation**: Automated test template generation
4. **Advanced Reporting**: Interactive HTML reports with charts
5. **Test Orchestration**: Advanced dependency management and scheduling