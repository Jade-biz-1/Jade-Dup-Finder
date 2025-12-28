# Testing Framework API Reference

## Overview

This directory contains comprehensive API documentation for all components of the CloneClean testing framework. The documentation is organized by component and includes detailed class references, method signatures, and usage examples.

## API Documentation Structure

### Core Framework APIs
- [TestHarness](test-harness.md) - Central test execution and coordination
- [TestEnvironment](test-environment.md) - Test data and environment management
- [TestReporting](test-reporting.md) - Test result reporting and metrics
- [TestUtilities](test-utilities.md) - Common testing utilities and helpers

### UI Testing APIs
- [UIAutomation](ui-automation.md) - Widget interaction and simulation
- [VisualTesting](visual-testing.md) - Screenshot capture and comparison
- [AccessibilityTesting](accessibility-testing.md) - Accessibility validation
- [ThemeTesting](theme-testing.md) - Theme consistency validation

### Performance Testing APIs
- [PerformanceBenchmarks](performance-benchmarks.md) - Performance measurement and baselines
- [LoadTesting](load-testing.md) - Stress and load testing capabilities
- [MemoryTesting](memory-testing.md) - Memory leak detection and profiling

### End-to-End Testing APIs
- [WorkflowTesting](workflow-testing.md) - Complete user journey validation
- [ScenarioTesting](scenario-testing.md) - Real-world usage scenarios
- [CrossPlatformTesting](cross-platform-testing.md) - Platform-specific validation

### Data Management APIs
- [TestDataGenerator](test-data-generator.md) - Test data creation and management
- [TestDatabaseManager](test-database-manager.md) - Database testing utilities
- [TestEnvironmentIsolator](test-environment-isolator.md) - Environment isolation

## Quick Reference

### Common Classes

| Class | Purpose | Header |
|-------|---------|--------|
| `TestHarness` | Test execution coordination | `test_harness.h` |
| `UIAutomation` | UI interaction automation | `ui_automation.h` |
| `VisualTesting` | Visual regression testing | `visual_testing.h` |
| `PerformanceBenchmarks` | Performance measurement | `performance_benchmarks.h` |
| `TestDataGenerator` | Test data creation | `test_data_generator.h` |

### Common Patterns

#### Basic Test Setup
```cpp
#include "test_harness.h"
#include "test_environment.h"

class MyTest : public QObject {
    Q_OBJECT
private slots:
    void initTestCase() {
        TestEnvironment::setup();
    }
    void testSomething() {
        // Test implementation
    }
};
```

#### UI Test Pattern
```cpp
#include "ui_automation.h"

void testUIInteraction() {
    MainWindow window;
    window.show();
    
    QVERIFY(UIAutomation::clickWidget(&window, "startButton"));
    QVERIFY(UIAutomation::waitForDialog("Progress", 5000));
}
```

#### Performance Test Pattern
```cpp
#include "performance_benchmarks.h"

void testPerformance() {
    PerformanceBenchmarks bench;
    auto result = bench.runBenchmark("file_scan_performance");
    
    QVERIFY(bench.compareWithBaseline(result, "file_scan_performance"));
}
```

## API Conventions

### Naming Conventions
- **Classes**: PascalCase (e.g., `TestHarness`, `UIAutomation`)
- **Methods**: camelCase (e.g., `runTest`, `clickWidget`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT`)
- **Enums**: PascalCase with prefixed values (e.g., `TestResult::Passed`)

### Return Values
- **Boolean methods**: Return `true` on success, `false` on failure
- **Object methods**: Return valid object or nullptr on failure
- **Collection methods**: Return empty collection on failure (not nullptr)

### Error Handling
- Use Qt's assertion macros in tests (`QVERIFY`, `QCOMPARE`)
- Provide detailed error messages with `QVERIFY2`
- Log debug information for complex operations

### Memory Management
- Follow Qt's parent-child ownership model
- Use smart pointers for complex ownership scenarios
- Clean up resources in test cleanup methods

## Usage Examples

### Running Tests Programmatically
```cpp
#include "test_harness.h"

int main() {
    TestHarness harness;
    TestSuiteConfig config;
    config.enabledCategories = {"unit", "integration"};
    config.parallelExecution = true;
    
    bool success = harness.runTestSuite(config);
    return success ? 0 : 1;
}
```

### Creating Custom Test Data
```cpp
#include "test_data_generator.h"

void setupTestData() {
    TestDataGenerator generator;
    
    QString testDir = generator.createTestDirectory("my_test");
    generator.createTestFiles(testDir, {
        {"file1.txt", 1024},  // 1KB file
        {"file2.txt", 2048},  // 2KB file
        {"subdir/file3.txt", 512}  // 512B file in subdirectory
    });
}
```

### Visual Regression Testing
```cpp
#include "visual_testing.h"

void testVisualRegression() {
    MyWidget widget;
    widget.show();
    
    QPixmap screenshot = VisualTesting::captureWidget(&widget);
    bool matches = VisualTesting::compareWithBaseline(screenshot, "my_widget_baseline");
    
    QVERIFY2(matches, "Widget appearance has changed");
}
```

## Version Compatibility

### API Stability
- **Stable APIs**: Core framework classes (TestHarness, TestEnvironment)
- **Evolving APIs**: UI automation and visual testing (may change with Qt updates)
- **Experimental APIs**: Advanced performance testing features

### Deprecation Policy
- Deprecated methods are marked with `[[deprecated]]`
- Deprecated APIs are supported for at least 2 major versions
- Migration guides are provided for breaking changes

## Contributing to API Documentation

### Documentation Standards
- Use Doxygen-style comments for all public APIs
- Include usage examples for complex methods
- Document parameter constraints and return value meanings
- Provide links to related methods and classes

### Example Documentation
```cpp
/**
 * @brief Clicks on a widget identified by selector
 * @param parent The parent widget to search within
 * @param selector Widget selector (object name, class, or text)
 * @param timeout Maximum time to wait for widget (default: 5000ms)
 * @return true if click was successful, false otherwise
 * 
 * @code
 * MainWindow window;
 * bool success = UIAutomation::clickWidget(&window, "startButton");
 * @endcode
 * 
 * @see waitForWidget(), verifyWidgetExists()
 */
static bool clickWidget(QWidget* parent, const QString& selector, int timeout = 5000);
```

## Support and Feedback

For API-related questions or suggestions:
1. Check the specific API documentation pages
2. Review the examples in `tests/examples/`
3. Create an issue in the repository for bugs or feature requests
4. Contribute improvements via pull requests

## Index

- [Core Framework APIs](#core-framework-apis)
- [UI Testing APIs](#ui-testing-apis)
- [Performance Testing APIs](#performance-testing-apis)
- [End-to-End Testing APIs](#end-to-end-testing-apis)
- [Data Management APIs](#data-management-apis)