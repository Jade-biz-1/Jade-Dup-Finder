# Test Writing Guidelines

## Overview

This document provides comprehensive guidelines for writing high-quality, maintainable tests in the CloneClean testing suite. Following these guidelines ensures consistency, reliability, and maintainability across our test codebase.

## General Principles

### 1. Test Independence
- Each test should be completely independent
- Tests should not depend on execution order
- Clean up all resources after each test
- Use fresh test data for each test

### 2. Clear and Descriptive Names
- Use descriptive test method names
- Include the scenario being tested
- Specify expected behavior
- Follow naming conventions

```cpp
// Good
void testFileScanner_WhenScanningEmptyDirectory_ReturnsEmptyResults()

// Bad  
void testFileScanner()
```

### 3. Single Responsibility
- Each test should verify one specific behavior
- Avoid testing multiple unrelated features
- Split complex tests into smaller, focused tests
- Use helper methods for common setup

### 4. Arrange-Act-Assert Pattern
Structure tests using the AAA pattern:

```cpp
void testExample() {
    // Arrange - Set up test data and conditions
    FileScanner scanner;
    QString testDir = createTestDirectory();
    
    // Act - Execute the behavior being tested
    QStringList results = scanner.scanDirectory(testDir);
    
    // Assert - Verify the expected outcome
    QCOMPARE(results.size(), 0);
}
```

## Naming Conventions

### Test Class Names
- Format: `Test<ComponentName>`
- Example: `TestFileScanner`, `TestDuplicateDetector`

### Test Method Names
- Format: `test<Component>_When<Condition>_<ExpectedBehavior>`
- Use camelCase for method names
- Be specific about conditions and expectations

```cpp
// Unit test examples
void testHashCalculator_WhenCalculatingMD5_ReturnsCorrectHash()
void testFileManager_WhenDeletingNonExistentFile_ThrowsException()

// Integration test examples  
void testScanWorkflow_WhenScanningLargeDirectory_CompletesSuccessfully()
void testUIInteraction_WhenClickingStartButton_InitiatesScan()
```

### Test Data Names
- Use descriptive variable names
- Prefix test data with `test` or `mock`
- Make purpose clear from the name

```cpp
QString testEmptyDirectory = "/tmp/test_empty";
QStringList mockDuplicateFiles = {"file1.txt", "file1_copy.txt"};
```

## Test Categories and Structure

### Unit Tests
Location: `tests/unit/`

```cpp
#include <QtTest>
#include "component_under_test.h"

class TestComponentName : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();    // Run once before all tests
    void init();           // Run before each test
    void cleanup();        // Run after each test
    void cleanupTestCase(); // Run once after all tests
    
    // Test methods
    void testBasicFunctionality();
    void testEdgeCases();
    void testErrorHandling();
};
```

### Integration Tests
Location: `tests/integration/`

```cpp
#include <QtTest>
#include "test_base.h"

class TestComponentIntegration : public TestBase {
    Q_OBJECT

private slots:
    void testComponentInteraction();
    void testDataFlow();
    void testErrorPropagation();
};
```

### UI Tests
Location: `tests/ui_automation.*`

```cpp
#include <QtTest>
#include "ui_automation.h"

class TestUIComponent : public QObject {
    Q_OBJECT

private slots:
    void testUserInteraction();
    void testVisualRegression();
    void testAccessibility();
};
```

## Best Practices by Test Type

### Unit Tests

#### Do:
- Test public interfaces only
- Use mocks for external dependencies
- Test edge cases and error conditions
- Keep tests fast (< 1 second each)

#### Don't:
- Test private methods directly
- Access internal implementation details
- Create dependencies between tests
- Use real file system or network

```cpp
void testFileScanner_WhenGivenInvalidPath_ThrowsException() {
    FileScanner scanner;
    
    QVERIFY_EXCEPTION_THROWN(
        scanner.scanDirectory("/invalid/path"),
        std::invalid_argument
    );
}
```

### Integration Tests

#### Do:
- Test component interactions
- Use realistic test data
- Verify data flow between components
- Test error propagation

#### Don't:
- Test individual component logic (use unit tests)
- Create overly complex test scenarios
- Ignore cleanup of shared resources

```cpp
void testFileScannerIntegration_WhenScanCompletes_NotifiesResultsWindow() {
    // Arrange
    FileScanner scanner;
    ResultsWindow resultsWindow;
    TestSignalSpy spy(&scanner, &FileScanner::scanCompleted);
    
    // Act
    scanner.scanDirectory(createTestDirectory());
    
    // Assert
    QVERIFY(spy.wait(5000));
    QCOMPARE(spy.count(), 1);
}
```

### UI Tests

#### Do:
- Use realistic user interactions
- Wait for UI updates to complete
- Capture screenshots on failure
- Test keyboard navigation

#### Don't:
- Rely on pixel-perfect positioning
- Use hardcoded delays
- Test internal widget state directly

```cpp
void testMainWindow_WhenStartButtonClicked_ShowsProgressDialog() {
    MainWindow window;
    window.show();
    
    // Act
    UIAutomation::clickWidget(&window, "startButton");
    
    // Assert
    QVERIFY(UIAutomation::waitForDialog("Scan Progress", 5000));
}
```

## Test Data Management

### Test Data Principles
- Use minimal, realistic data sets
- Create data programmatically when possible
- Clean up all test data after tests
- Use temporary directories for file operations

### Test Data Helpers
```cpp
class TestDataHelper {
public:
    static QString createTestDirectory(const QString& name = "test_dir");
    static void createTestFiles(const QString& dir, const QStringList& files);
    static void cleanupTestData(const QString& dir);
    static QByteArray generateTestFileContent(int sizeKB);
};
```

### Example Usage
```cpp
void testFileScanner_WhenScanningDirectory_FindsAllFiles() {
    // Arrange
    QString testDir = TestDataHelper::createTestDirectory();
    QStringList testFiles = {"file1.txt", "file2.txt", "subdir/file3.txt"};
    TestDataHelper::createTestFiles(testDir, testFiles);
    
    // Act
    FileScanner scanner;
    QStringList results = scanner.scanDirectory(testDir);
    
    // Assert
    QCOMPARE(results.size(), 3);
    
    // Cleanup
    TestDataHelper::cleanupTestData(testDir);
}
```

## Assertions and Verification

### Choosing the Right Assertion
- `QVERIFY(condition)` - Boolean conditions
- `QCOMPARE(actual, expected)` - Value comparison
- `QVERIFY_EXCEPTION_THROWN(code, exception)` - Exception testing
- `QFAIL(message)` - Explicit failure

### Custom Assertions
```cpp
#define QVERIFY_FILE_EXISTS(path) \
    QVERIFY2(QFile::exists(path), \
             QString("File does not exist: %1").arg(path).toLocal8Bit())

#define QCOMPARE_FILES(file1, file2) \
    QVERIFY2(compareFiles(file1, file2), \
             QString("Files differ: %1 vs %2").arg(file1, file2).toLocal8Bit())
```

## Error Handling and Debugging

### Providing Context
```cpp
void testFileOperation() {
    QString testFile = "/tmp/test.txt";
    
    QVERIFY2(QFile::exists(testFile), 
             QString("Test file missing: %1").arg(testFile).toLocal8Bit());
}
```

### Debug Information
```cpp
void testComplexOperation() {
    // Add debug output for complex tests
    qDebug() << "Starting complex operation test";
    qDebug() << "Test data directory:" << testDataDir;
    
    // Test implementation
    
    qDebug() << "Test completed successfully";
}
```

## Performance Considerations

### Test Performance Guidelines
- Keep unit tests under 1 second each
- Use timeouts for operations that might hang
- Profile slow tests and optimize
- Use parallel execution where appropriate

### Timeout Usage
```cpp
void testLongRunningOperation() {
    QSignalSpy spy(&object, &Object::operationCompleted);
    
    object.startLongOperation();
    
    QVERIFY2(spy.wait(30000), "Operation timed out after 30 seconds");
}
```

## Code Quality

### Code Review Checklist
- [ ] Test names are descriptive and follow conventions
- [ ] Tests are independent and don't rely on execution order
- [ ] Appropriate assertions are used
- [ ] Test data is cleaned up properly
- [ ] Error cases are tested
- [ ] Tests are focused on single responsibility
- [ ] Code is well-commented for complex scenarios

### Static Analysis
Run static analysis tools on test code:
```bash
# Run clang-tidy on test files
clang-tidy tests/**/*.cpp -- -I include -I tests/framework
```

## Documentation Requirements

### Test Documentation
- Document complex test scenarios
- Explain non-obvious test setup
- Provide context for unusual assertions
- Link to related requirements or issues

```cpp
/**
 * Tests the file scanner's behavior when encountering symbolic links.
 * This test verifies that the scanner follows symbolic links by default
 * but can be configured to ignore them.
 * 
 * Related requirement: REQ-FS-001 "Handle symbolic links appropriately"
 */
void testFileScanner_WhenEncounteringSymlinks_FollowsByDefault() {
    // Test implementation
}
```

## Common Pitfalls to Avoid

### 1. Flaky Tests
- Don't use hardcoded delays (`QThread::sleep()`)
- Use proper synchronization mechanisms
- Wait for conditions rather than fixed times

### 2. Test Dependencies
- Don't rely on test execution order
- Don't share mutable state between tests
- Clean up after each test

### 3. Over-Testing
- Don't test framework code (Qt, STL)
- Don't test trivial getters/setters
- Focus on business logic and edge cases

### 4. Under-Testing
- Test error conditions and edge cases
- Test boundary values
- Test integration points

## Examples and Templates

See the [examples directory](examples/) for complete working examples of each test type, and the [templates directory](templates/) for boilerplate code to get started quickly.

## Next Steps

- Review [Test Examples](examples/)
- Check [API Reference](api/)
- Read [Troubleshooting Guide](troubleshooting.md)
- Explore [Performance Optimization](performance-optimization.md)