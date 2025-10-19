# DupFinder Test Framework

A comprehensive automated testing framework for the DupFinder application, providing multi-layered testing capabilities including unit tests, integration tests, UI automation, end-to-end workflows, performance testing, and accessibility validation.

## Features

### Core Framework Components

- **TestHarness**: Central test execution and coordination system
- **TestEnvironment**: Test data and environment management
- **TestReporting**: Comprehensive reporting in multiple formats (HTML, JSON, JUnit XML, Console)
- **TestUtilities**: Common testing utilities and helper functions

### Testing Capabilities

- **Unit Testing**: Fast, isolated component testing with enhanced Qt Test integration
- **Integration Testing**: Component interaction validation and data flow verification
- **UI Testing**: Automated user interface testing with widget interaction simulation
- **End-to-End Testing**: Complete user workflow validation
- **Performance Testing**: Benchmarking, memory leak detection, and performance regression testing
- **Visual Regression Testing**: Screenshot-based UI consistency validation
- **Accessibility Testing**: Keyboard navigation and screen reader compatibility validation
- **Cross-Platform Testing**: Platform-specific behavior validation

## Quick Start

### Basic Usage

```cpp
#include "test_harness.h"
#include "test_environment.h"
#include "test_reporting.h"

// Create a simple test suite
class MyTestSuite : public TestSuite {
    Q_OBJECT
public:
    MyTestSuite() : TestSuite("MyTests", TestCategory::Unit) {
        REGISTER_TEST(testExample);
    }
    
    TEST_METHOD(testExample);
};

void MyTestSuite::testExample() {
    TEST_COMPARE(2 + 2, 4);
    TEST_VERIFY(true);
    recordTestResult("testExample", true);
}

// Run the tests
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    TestHarness harness;
    auto suite = std::make_shared<MyTestSuite>();
    harness.registerTestSuite(suite);
    
    return harness.runAllTests() ? 0 : 1;
}
```

### Advanced Usage with Environment and Reporting

```cpp
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    // Create test harness
    TestHarness harness;
    
    // Set up test environment
    auto testEnv = std::make_shared<TestEnvironment>();
    harness.setTestEnvironment(testEnv);
    
    // Configure reporting
    auto reporter = std::make_shared<TestReporting>();
    ReportConfig config;
    config.formats = {ReportFormat::HTML, ReportFormat::JUnit, ReportFormat::Console};
    config.outputDirectory = "test_results";
    reporter->setReportConfig(config);
    harness.setReportGenerator(reporter);
    
    // Register test suites
    harness.registerTestSuite(std::make_shared<UnitTestSuite>());
    harness.registerTestSuite(std::make_shared<IntegrationTestSuite>());
    harness.registerTestSuite(std::make_shared<UITestSuite>());
    
    // Configure execution
    TestSuiteConfig testConfig;
    testConfig.parallelExecution = true;
    testConfig.maxParallelThreads = 4;
    testConfig.generateHtmlReport = true;
    harness.setConfiguration(testConfig);
    
    // Run tests
    bool success = harness.runAllTests();
    
    // Generate reports
    harness.generateReport();
    
    return success ? 0 : 1;
}
```

## Test Categories

### Unit Tests
Fast, isolated tests for individual components:
```cpp
class ComponentTestSuite : public TestSuite {
public:
    ComponentTestSuite() : TestSuite("ComponentTests", TestCategory::Unit) {}
    
    TEST_METHOD(testCalculation) {
        Calculator calc;
        TEST_COMPARE(calc.add(2, 3), 5);
    }
};
```

### Integration Tests
Tests for component interactions:
```cpp
class IntegrationTestSuite : public TestSuite {
public:
    IntegrationTestSuite() : TestSuite("IntegrationTests", TestCategory::Integration) {}
    
    TEST_METHOD(testFileProcessing) {
        FileScanner scanner;
        DuplicateDetector detector;
        
        // Test scanner -> detector workflow
        auto files = scanner.scanDirectory("/test/path");
        auto duplicates = detector.findDuplicates(files);
        
        TEST_VERIFY(!duplicates.isEmpty());
    }
};
```

### UI Tests
Automated user interface testing:
```cpp
class UITestSuite : public TestSuite {
public:
    UITestSuite() : TestSuite("UITests", TestCategory::UI) {}
    
    TEST_METHOD(testButtonClick) {
        MainWindow window;
        window.show();
        
        QPushButton* button = FIND_WIDGET(&window, "startButton");
        TEST_ASSERT_WIDGET_EXISTS(&window, "startButton");
        TEST_ASSERT_WIDGET_ENABLED(button);
        
        CLICK_WIDGET(button);
        
        WAIT_FOR_CONDITION(window.isScanRunning());
        TEST_VERIFY(window.isScanRunning());
    }
};
```

### Performance Tests
Benchmarking and performance validation:
```cpp
class PerformanceTestSuite : public TestSuite {
public:
    PerformanceTestSuite() : TestSuite("PerformanceTests", TestCategory::Performance) {}
    
    TEST_METHOD(testScanPerformance) {
        PERFORMANCE_MEASURE(scanTest);
        
        FileScanner scanner;
        auto files = scanner.scanDirectory("/large/test/dataset");
        
        // Automatically measured when scope ends
        TEST_ASSERT_PERFORMANCE(scanTest, 5000); // Max 5 seconds
    }
};
```

## Test Environment Management

### Creating Test Data
```cpp
void setupTestData() {
    TestEnvironment env;
    
    // Create test directory structure
    TestDirectorySpec spec;
    spec.name = "test_photos";
    spec.files = {
        {"photo1.jpg", generateBinaryContent(100000)},
        {"photo2.jpg", generateBinaryContent(100000)},
        {"photo1_copy.jpg", generateBinaryContent(100000)} // Duplicate
    };
    
    env.createTestDirectoryStructure("/tmp/test_data", spec);
    
    // Or use predefined datasets
    env.createPhotoLibraryDataset("/tmp/photos", 1000, 20); // 1000 photos, 20% duplicates
    env.createDocumentDataset("/tmp/docs", 500, {"txt", "pdf", "doc"});
}
```

### Application Testing
```cpp
void testApplicationWorkflow() {
    TestEnvironment env;
    
    // Launch application
    AppLaunchConfig config;
    config.executablePath = "./dupfinder";
    config.arguments = {"--test-mode"};
    
    TEST_VERIFY(env.launchApplication(config));
    
    // Perform UI interactions
    // ...
    
    // Application automatically closed when env goes out of scope
}
```

## Reporting and Analysis

### HTML Reports
Comprehensive HTML reports with:
- Test summary and statistics
- Detailed failure information with stack traces
- Performance metrics and trends
- Screenshot galleries for visual tests
- Interactive charts and graphs

### JUnit XML
Standard JUnit XML format for CI/CD integration:
```xml
<testsuites name="DupFinder Tests" tests="150" failures="2" time="45.2">
    <testsuite name="UnitTests" tests="50" failures="0" time="12.1">
        <testcase name="testCalculation" time="0.001"/>
        <!-- ... -->
    </testsuite>
</testsuites>
```

### JSON Reports
Machine-readable JSON format for programmatic analysis:
```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "overallResults": {
        "totalTests": 150,
        "passedTests": 148,
        "failedTests": 2,
        "executionTimeMs": 45200,
        "successRate": 98.7
    },
    "suiteResults": { /* ... */ }
}
```

## Utilities and Helpers

### Widget Interaction
```cpp
// Find and interact with widgets
QWidget* button = FIND_WIDGET(parent, "buttonName");
CLICK_WIDGET(button);
TYPE_TEXT(textEdit, "Hello World");

// Wait for conditions
WAIT_FOR_WIDGET(parent, "dialogWidget");
WAIT_FOR_CONDITION(someCondition());
```

### File Operations
```cpp
// Temporary file management
TEMP_FILE_GUARD("file content");
TEST_VERIFY(tempFile.isValid());

// File comparison
TEST_ASSERT_FILES_EQUAL("expected.txt", "actual.txt");
```

### Performance Measurement
```cpp
// Automatic measurement
{
    PERFORMANCE_MEASURE(operationName);
    performExpensiveOperation();
    // Measurement stops automatically
}

// Manual measurement
TestUtilities::startPerformanceMeasurement("test");
doWork();
qint64 elapsed = TestUtilities::stopPerformanceMeasurement("test");
```

### Visual Testing
```cpp
// Screenshot comparison
QPixmap current = CAPTURE_SCREENSHOT(widget);
TEST_VERIFY(TestUtilities::compareImages(current, baseline, 0.95));

// Save screenshots on failure
CAPTURE_SCREENSHOT(widget, "failure_screenshot.png");
```

## Configuration

### Test Configuration File (JSON)
```json
{
    "enabledCategories": ["Unit", "Integration", "UI"],
    "disabledTests": ["flakyTest1", "slowTest2"],
    "timeoutSeconds": 300,
    "parallelExecution": true,
    "maxParallelThreads": 4,
    "outputDirectory": "test_results",
    "generateHtmlReport": true,
    "generateJunitXml": true,
    "captureScreenshots": true,
    "visualThreshold": 0.95,
    "verboseOutput": false
}
```

### Programmatic Configuration
```cpp
TestSuiteConfig config;
config.enabledCategories = {"Unit", "Integration"};
config.parallelExecution = true;
config.maxParallelThreads = 8;
config.generateHtmlReport = true;
config.captureScreenshots = true;

harness.setConfiguration(config);
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Install Qt
      uses: jurplel/install-qt-action@v3
    - name: Build and Test
      run: |
        cmake -B build -DENABLE_COVERAGE=ON
        cmake --build build
        cd build && ctest --output-on-failure
    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: build/test_results/
```

## Best Practices

### Test Organization
- Group related tests into logical test suites
- Use descriptive test names that explain what is being tested
- Keep tests independent and isolated
- Use appropriate test categories for different types of tests

### Performance Testing
- Establish performance baselines for critical operations
- Use realistic test data sizes
- Measure both execution time and memory usage
- Set reasonable performance thresholds

### UI Testing
- Use object names for reliable widget identification
- Wait for UI state changes rather than using fixed delays
- Capture screenshots on test failures for debugging
- Test keyboard navigation and accessibility

### Error Handling
- Test both success and failure scenarios
- Verify proper error messages and user feedback
- Test recovery from error conditions
- Validate input sanitization and validation

## Building and Running

### Build Requirements
- Qt 6.4 or later
- CMake 3.16 or later
- C++17 compatible compiler

### Build Commands
```bash
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build framework and tests
cmake --build build

# Run example test
./build/tests/framework/example_test_suite

# Run all tests
cd build && ctest --output-on-failure
```

### Optional Features
```bash
# Enable code coverage
cmake -B build -DENABLE_COVERAGE=ON

# Enable sanitizers (debug builds)
cmake -B build -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug

# Enable profiling
cmake -B build -DENABLE_PROFILING=ON
```

## Extending the Framework

### Creating Custom Test Suites
```cpp
class CustomTestSuite : public TestSuite {
    Q_OBJECT
public:
    CustomTestSuite() : TestSuite("CustomTests", TestCategory::Unit) {
        setupTests();
    }

private:
    void setupTests() {
        REGISTER_TEST(testCustomFeature);
        REGISTER_TEST(testAnotherFeature);
    }
    
    bool runAllTests() override {
        // Custom test execution logic
        return TestSuite::runAllTests();
    }
    
    TEST_METHOD(testCustomFeature);
    TEST_METHOD(testAnotherFeature);
};
```

### Adding Custom Utilities
```cpp
class CustomTestUtilities {
public:
    static bool validateCustomCondition(const CustomObject& obj) {
        // Custom validation logic
        return obj.isValid();
    }
    
    static CustomObject createTestObject() {
        // Custom test object creation
        return CustomObject("test_data");
    }
};
```

### Custom Report Formats
```cpp
class CustomReporter : public TestReporting {
public:
    bool generateCustomReport(const TestResults& results, const QString& filePath) {
        // Custom report generation logic
        return true;
    }
};
```

## Troubleshooting

### Common Issues

**Tests not found**: Ensure test methods are registered with `REGISTER_TEST()` and properly declared with `TEST_METHOD()`.

**Widget not found**: Verify widget object names are set correctly and widgets are visible when accessed.

**Timing issues**: Use `WAIT_FOR_CONDITION()` instead of fixed delays for UI state changes.

**Memory leaks**: Use RAII helpers like `TempFileGuard` and `SettingsGuard` for automatic cleanup.

**Performance test failures**: Establish realistic baselines and account for system load variations.

### Debug Output
Enable verbose output for detailed test execution information:
```cpp
TestSuiteConfig config;
config.verboseOutput = true;
config.logLevel = "DEBUG";
harness.setConfiguration(config);
```

### Test Isolation
Ensure tests are properly isolated:
```cpp
void MyTestSuite::setUp() override {
    // Reset state before each test
    TestUtilities::resetApplicationState();
}

void MyTestSuite::tearDown() override {
    // Clean up after each test
    TestUtilities::clearApplicationCache();
}
```

## License

This test framework is part of the DupFinder project and follows the same licensing terms.