# TestHarness API Reference

## Overview

The `TestHarness` class is the central component for test execution and coordination in the CloneClean testing framework. It provides a unified interface for running tests, managing test suites, and generating comprehensive reports.

## Class Declaration

```cpp
#include "test_harness.h"

class TestHarness : public QObject {
    Q_OBJECT
    
public:
    explicit TestHarness(QObject* parent = nullptr);
    ~TestHarness();
    
    // Test execution
    bool runTestSuite(const TestSuiteConfig& config);
    bool runTestCategory(TestCategory category);
    bool runSpecificTest(const QString& testName);
    
    // Test management
    void registerTestSuite(std::shared_ptr<TestSuite> suite);
    void setTestEnvironment(std::shared_ptr<TestEnvironment> env);
    void setReportGenerator(std::shared_ptr<TestReporting> reporter);
    
    // Configuration
    void loadConfiguration(const QString& configFile);
    void setParallelExecution(bool enabled, int maxThreads = 0);
    void setTimeout(int seconds);
    
    // Results
    TestResults getResults() const;
    bool hasFailures() const;
    void generateReport(const QString& outputPath);
    
signals:
    void testStarted(const QString& testName);
    void testCompleted(const QString& testName, TestResult result);
    void suiteCompleted(const TestResults& results);
    
private slots:
    void onTestFinished();
    void onTestTimeout();
};
```

## Public Methods

### Test Execution

#### `runTestSuite(const TestSuiteConfig& config)`
Executes a complete test suite based on the provided configuration.

**Parameters:**
- `config`: Configuration object specifying which tests to run and how

**Returns:**
- `true` if all tests passed, `false` if any test failed

**Example:**
```cpp
TestHarness harness;
TestSuiteConfig config;
config.enabledCategories = {"unit", "integration"};
config.parallelExecution = true;
config.timeoutSeconds = 300;

bool success = harness.runTestSuite(config);
if (!success) {
    qDebug() << "Test suite failed";
    harness.generateReport("test_results.html");
}
```

#### `runTestCategory(TestCategory category)`
Runs all tests in a specific category.

**Parameters:**
- `category`: The test category to execute (Unit, Integration, UI, etc.)

**Returns:**
- `true` if all tests in the category passed

**Example:**
```cpp
TestHarness harness;
bool unitTestsPass = harness.runTestCategory(TestCategory::Unit);
bool integrationTestsPass = harness.runTestCategory(TestCategory::Integration);
```

#### `runSpecificTest(const QString& testName)`
Executes a single named test.

**Parameters:**
- `testName`: Name of the specific test to run

**Returns:**
- `true` if the test passed

**Example:**
```cpp
TestHarness harness;
bool passed = harness.runSpecificTest("TestFileScanner::testBasicScan");
```

### Test Management

#### `registerTestSuite(std::shared_ptr<TestSuite> suite)`
Registers a test suite with the harness for execution.

**Parameters:**
- `suite`: Shared pointer to the test suite to register

**Example:**
```cpp
auto unitTestSuite = std::make_shared<UnitTestSuite>();
auto integrationTestSuite = std::make_shared<IntegrationTestSuite>();

TestHarness harness;
harness.registerTestSuite(unitTestSuite);
harness.registerTestSuite(integrationTestSuite);
```

#### `setTestEnvironment(std::shared_ptr<TestEnvironment> env)`
Sets the test environment manager for the harness.

**Parameters:**
- `env`: Shared pointer to the test environment manager

**Example:**
```cpp
auto environment = std::make_shared<TestEnvironment>();
environment->setTestDataDirectory("/tmp/test_data");

TestHarness harness;
harness.setTestEnvironment(environment);
```

#### `setReportGenerator(std::shared_ptr<TestReporting> reporter)`
Sets the report generator for test results.

**Parameters:**
- `reporter`: Shared pointer to the test reporting component

**Example:**
```cpp
auto reporter = std::make_shared<TestReporting>();
reporter->setOutputFormat(TestReporting::HTML | TestReporting::JUnit);

TestHarness harness;
harness.setReportGenerator(reporter);
```

### Configuration

#### `loadConfiguration(const QString& configFile)`
Loads test configuration from a JSON file.

**Parameters:**
- `configFile`: Path to the configuration file

**Example:**
```cpp
TestHarness harness;
harness.loadConfiguration("test_config.json");
```

**Configuration File Format:**
```json
{
  "execution": {
    "parallel": true,
    "maxThreads": 4,
    "timeout": 300
  },
  "categories": {
    "unit": { "enabled": true, "timeout": 30 },
    "integration": { "enabled": true, "timeout": 120 },
    "ui": { "enabled": false, "timeout": 180 }
  },
  "reporting": {
    "html": true,
    "junit": true,
    "coverage": true
  }
}
```

#### `setParallelExecution(bool enabled, int maxThreads = 0)`
Configures parallel test execution.

**Parameters:**
- `enabled`: Whether to enable parallel execution
- `maxThreads`: Maximum number of threads (0 = auto-detect)

**Example:**
```cpp
TestHarness harness;
harness.setParallelExecution(true, 4);  // Use 4 threads
```

#### `setTimeout(int seconds)`
Sets the global timeout for test execution.

**Parameters:**
- `seconds`: Timeout in seconds (0 = no timeout)

**Example:**
```cpp
TestHarness harness;
harness.setTimeout(600);  // 10 minute timeout
```

### Results

#### `getResults() const`
Returns the results of the last test execution.

**Returns:**
- `TestResults` object containing detailed test results

**Example:**
```cpp
TestHarness harness;
harness.runTestSuite(config);

TestResults results = harness.getResults();
qDebug() << "Total tests:" << results.totalTests;
qDebug() << "Passed:" << results.passedTests;
qDebug() << "Failed:" << results.failedTests;
qDebug() << "Coverage:" << results.codeCoverage << "%";
```

#### `hasFailures() const`
Checks if the last test execution had any failures.

**Returns:**
- `true` if there were test failures

**Example:**
```cpp
TestHarness harness;
harness.runTestSuite(config);

if (harness.hasFailures()) {
    qDebug() << "Some tests failed";
    harness.generateReport("failure_report.html");
}
```

#### `generateReport(const QString& outputPath)`
Generates a comprehensive test report.

**Parameters:**
- `outputPath`: Path where the report should be saved

**Example:**
```cpp
TestHarness harness;
harness.runTestSuite(config);
harness.generateReport("test_results.html");
```

## Signals

### `testStarted(const QString& testName)`
Emitted when a test begins execution.

**Parameters:**
- `testName`: Name of the test that started

### `testCompleted(const QString& testName, TestResult result)`
Emitted when a test completes execution.

**Parameters:**
- `testName`: Name of the completed test
- `result`: Result of the test execution

### `suiteCompleted(const TestResults& results)`
Emitted when the entire test suite completes.

**Parameters:**
- `results`: Complete results of the test suite execution

## Usage Patterns

### Basic Test Execution
```cpp
#include "test_harness.h"

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    TestHarness harness;
    
    // Load configuration
    harness.loadConfiguration("test_config.json");
    
    // Set up environment
    auto env = std::make_shared<TestEnvironment>();
    harness.setTestEnvironment(env);
    
    // Run tests
    TestSuiteConfig config;
    config.enabledCategories = {"unit", "integration"};
    
    bool success = harness.runTestSuite(config);
    
    // Generate report
    harness.generateReport("test_results.html");
    
    return success ? 0 : 1;
}
```

### Custom Test Suite Registration
```cpp
class CustomTestSuite : public TestSuite {
public:
    QStringList getTestNames() const override {
        return {"CustomTest1", "CustomTest2"};
    }
    
    bool runTest(const QString& testName) override {
        if (testName == "CustomTest1") {
            return runCustomTest1();
        } else if (testName == "CustomTest2") {
            return runCustomTest2();
        }
        return false;
    }
};

// Usage
auto customSuite = std::make_shared<CustomTestSuite>();
TestHarness harness;
harness.registerTestSuite(customSuite);
```

### Progress Monitoring
```cpp
TestHarness harness;

// Connect to progress signals
QObject::connect(&harness, &TestHarness::testStarted,
                [](const QString& testName) {
                    qDebug() << "Starting test:" << testName;
                });

QObject::connect(&harness, &TestHarness::testCompleted,
                [](const QString& testName, TestResult result) {
                    qDebug() << "Test" << testName << "completed with result:" 
                             << (result == TestResult::Passed ? "PASS" : "FAIL");
                });

QObject::connect(&harness, &TestHarness::suiteCompleted,
                [](const TestResults& results) {
                    qDebug() << "Suite completed:" << results.passedTests 
                             << "passed," << results.failedTests << "failed";
                });

harness.runTestSuite(config);
```

## Error Handling

### Common Error Scenarios
1. **Configuration file not found**: Check file path and permissions
2. **Test timeout**: Increase timeout or optimize slow tests
3. **Memory issues**: Monitor memory usage during test execution
4. **Parallel execution conflicts**: Disable parallel execution for debugging

### Error Recovery
```cpp
TestHarness harness;

try {
    harness.loadConfiguration("test_config.json");
} catch (const std::exception& e) {
    qWarning() << "Failed to load config:" << e.what();
    // Use default configuration
    TestSuiteConfig defaultConfig;
    harness.runTestSuite(defaultConfig);
}
```

## Performance Considerations

### Optimization Tips
1. **Use parallel execution** for independent tests
2. **Set appropriate timeouts** to avoid hanging tests
3. **Monitor memory usage** during long test runs
4. **Use test categories** to run only necessary tests

### Memory Management
```cpp
// Proper cleanup after test execution
TestHarness harness;
harness.runTestSuite(config);

// Results are automatically cleaned up when harness is destroyed
// Manual cleanup if needed:
harness.getResults().clear();
```

## Thread Safety

The `TestHarness` class is **not thread-safe**. If you need to run multiple test harnesses concurrently, create separate instances for each thread.

```cpp
// Safe: Separate instances per thread
std::thread t1([&]() {
    TestHarness harness1;
    harness1.runTestCategory(TestCategory::Unit);
});

std::thread t2([&]() {
    TestHarness harness2;
    harness2.runTestCategory(TestCategory::Integration);
});
```

## See Also

- [TestEnvironment](test-environment.md) - Test environment management
- [TestReporting](test-reporting.md) - Test result reporting
- [TestSuiteConfig](test-suite-config.md) - Configuration options
- [TestResults](test-results.md) - Result data structures