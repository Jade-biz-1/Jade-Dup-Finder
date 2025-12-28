# Testing Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues encountered when working with the CloneClean testing suite. Issues are organized by category with step-by-step solutions.

## Quick Diagnosis

### Test Execution Issues

#### Tests Won't Run
**Symptoms**: `ctest` fails to find or execute tests

**Common Causes**:
1. Build configuration issues
2. Missing test executables
3. Incorrect CMake configuration

**Solutions**:
```bash
# 1. Verify build configuration
cd build
cmake --build . --target all

# 2. Check if test executables exist
ls -la tests/unit/
ls -la tests/integration/

# 3. Verify CMake test registration
ctest --show-only

# 4. Run with verbose output
ctest --verbose
```

#### Tests Crash on Startup
**Symptoms**: Tests crash immediately or show segmentation faults

**Common Causes**:
1. Missing Qt application instance
2. Uninitialized test environment
3. Missing test data

**Solutions**:
```cpp
// Ensure QApplication exists for UI tests
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);  // Required for UI tests
    TestClass test;
    return QTest::qExec(&test, argc, argv);
}

// Initialize test environment
void TestClass::initTestCase() {
    TestEnvironment::setup();
    // Additional setup
}
```

### Test Failures

#### Flaky Tests
**Symptoms**: Tests pass sometimes, fail other times

**Common Causes**:
1. Race conditions
2. Hardcoded delays
3. Unclean test environment
4. Resource contention

**Solutions**:
```cpp
// Bad: Using hardcoded delays
QThread::sleep(1000);  // Don't do this

// Good: Wait for specific conditions
QSignalSpy spy(&object, &Object::finished);
QVERIFY(spy.wait(5000));

// Good: Use QTest::qWaitFor
QVERIFY(QTest::qWaitFor([&]() { 
    return object.isReady(); 
}, 5000));
```

#### UI Tests Failing
**Symptoms**: UI automation tests fail to find widgets or interact properly

**Common Causes**:
1. Widget not visible or enabled
2. Incorrect widget selectors
3. Timing issues with UI updates
4. Platform-specific behavior

**Solutions**:
```cpp
// Verify widget exists and is visible
QVERIFY(UIAutomation::verifyWidgetExists(parent, "buttonName"));
QVERIFY(UIAutomation::verifyWidgetVisible(parent, "buttonName"));

// Wait for widget to become available
QVERIFY(UIAutomation::waitForWidget(parent, "buttonName", 5000));

// Use more robust selectors
// Bad: Relying on position
UIAutomation::clickWidget(parent, "QPushButton[0]");

// Good: Using object name or text
UIAutomation::clickWidget(parent, "startButton");
UIAutomation::clickWidget(parent, "QPushButton[text='Start Scan']");
```

#### Visual Regression Failures
**Symptoms**: Visual tests fail due to screenshot differences

**Common Causes**:
1. Platform differences (fonts, rendering)
2. Theme changes
3. Window sizing issues
4. Outdated baselines

**Solutions**:
```bash
# Update baselines after intentional UI changes
./update_visual_baselines.sh

# Check difference images
ls -la test_results/visual_diffs/

# Platform-specific baselines
tests/baselines/linux/
tests/baselines/windows/
tests/baselines/macos/
```

### Performance Test Issues

#### Performance Tests Timing Out
**Symptoms**: Performance tests exceed timeout limits

**Common Causes**:
1. Debug builds (use Release for performance tests)
2. System under load
3. Unrealistic performance expectations
4. Memory leaks affecting performance

**Solutions**:
```bash
# Build in Release mode for performance tests
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Run performance tests in isolation
ctest -L performance --parallel 1

# Monitor system resources
top -p $(pgrep test_performance)
```

#### Inconsistent Performance Results
**Symptoms**: Performance benchmarks vary significantly between runs

**Common Causes**:
1. System background processes
2. Thermal throttling
3. Insufficient warm-up
4. Small sample sizes

**Solutions**:
```cpp
// Warm up before measuring
for (int i = 0; i < 10; ++i) {
    performOperation();  // Warm-up runs
}

// Multiple measurements
QList<qint64> measurements;
for (int i = 0; i < 100; ++i) {
    auto start = QDateTime::currentMSecsSinceEpoch();
    performOperation();
    auto end = QDateTime::currentMSecsSinceEpoch();
    measurements.append(end - start);
}

// Use median instead of average
std::sort(measurements.begin(), measurements.end());
qint64 median = measurements[measurements.size() / 2];
```

### Environment Issues

#### Test Data Problems
**Symptoms**: Tests fail due to missing or corrupted test data

**Common Causes**:
1. Test data not created properly
2. Insufficient permissions
3. Disk space issues
4. Cleanup failures

**Solutions**:
```cpp
// Verify test data creation
void TestClass::init() {
    testDataDir = TestDataHelper::createTestDirectory();
    QVERIFY(!testDataDir.isEmpty());
    QVERIFY(QDir(testDataDir).exists());
}

// Robust cleanup
void TestClass::cleanup() {
    if (!testDataDir.isEmpty()) {
        QDir dir(testDataDir);
        if (dir.exists()) {
            dir.removeRecursively();
        }
    }
}
```

#### Permission Issues
**Symptoms**: Tests fail with permission denied errors

**Common Causes**:
1. Running tests as wrong user
2. Restrictive file permissions
3. SELinux or similar security policies

**Solutions**:
```bash
# Check current permissions
ls -la /tmp/test_data/

# Fix permissions
chmod -R 755 /tmp/test_data/

# Run tests with appropriate user
sudo -u testuser ctest
```

### CI/CD Issues

#### Tests Pass Locally but Fail in CI
**Symptoms**: Tests work on developer machines but fail in CI environment

**Common Causes**:
1. Different environment variables
2. Missing dependencies
3. Different Qt versions
4. Headless environment issues

**Solutions**:
```yaml
# GitHub Actions example
- name: Setup test environment
  run: |
    export QT_QPA_PLATFORM=offscreen
    export DISPLAY=:99
    Xvfb :99 -screen 0 1024x768x24 &
    
- name: Run tests
  run: |
    ctest --output-on-failure
```

#### Timeout Issues in CI
**Symptoms**: Tests timeout in CI but not locally

**Common Causes**:
1. Slower CI machines
2. Resource contention
3. Network latency
4. Insufficient timeout values

**Solutions**:
```cpp
// Increase timeouts for CI
#ifdef CI_ENVIRONMENT
const int TIMEOUT_MS = 30000;  // 30 seconds in CI
#else
const int TIMEOUT_MS = 5000;   // 5 seconds locally
#endif
```

## Debugging Techniques

### Verbose Test Output
```bash
# Run with maximum verbosity
ctest --verbose --output-on-failure

# Run specific test with debug output
./tests/unit/test_file_scanner --verbose
```

### Debug Builds
```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Run under debugger
gdb ./tests/unit/test_file_scanner
```

### Test Isolation
```bash
# Run single test to isolate issues
ctest -R "test_file_scanner"

# Run tests sequentially to avoid race conditions
ctest --parallel 1
```

### Memory Debugging
```bash
# Run with Valgrind
valgrind --tool=memcheck --leak-check=full ./tests/unit/test_file_scanner

# Run with AddressSanitizer
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address" ..
make
./tests/unit/test_file_scanner
```

## Common Error Messages

### "QWidget: Must construct a QApplication before a QWidget"
**Solution**: Ensure QApplication is created in test main function
```cpp
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    TestClass test;
    return QTest::qExec(&test, argc, argv);
}
```

### "Cannot create children for a parent that is in a different thread"
**Solution**: Ensure UI operations happen on main thread
```cpp
// Use QMetaObject::invokeMethod for cross-thread calls
QMetaObject::invokeMethod(widget, [&]() {
    widget->show();
}, Qt::BlockingQueuedConnection);
```

### "Test data directory not found"
**Solution**: Verify test data setup
```cpp
void TestClass::initTestCase() {
    QString dataDir = QCoreApplication::applicationDirPath() + "/test_data";
    QVERIFY2(QDir(dataDir).exists(), 
             QString("Test data directory not found: %1").arg(dataDir).toLocal8Bit());
}
```

## Performance Optimization

### Slow Test Execution
1. **Profile test execution**:
   ```bash
   time ctest
   ctest --verbose | grep "Test #"
   ```

2. **Optimize test data**:
   - Use smaller test datasets
   - Cache expensive setup operations
   - Reuse test environments where safe

3. **Parallel execution**:
   ```bash
   ctest --parallel 4
   ```

### Memory Usage Issues
1. **Monitor memory usage**:
   ```bash
   valgrind --tool=massif ./test_executable
   ```

2. **Optimize test cleanup**:
   ```cpp
   void TestClass::cleanup() {
       // Clean up large objects
       largeTestData.clear();
       largeTestData.squeeze();
   }
   ```

## Getting Help

### Internal Resources
1. Check existing test examples in `tests/examples/`
2. Review API documentation in `docs/testing/api/`
3. Search existing issues in the repository

### External Resources
1. [Qt Test Framework Documentation](https://doc.qt.io/qt-6/qtest-overview.html)
2. [CMake CTest Documentation](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
3. [Google Test Best Practices](https://google.github.io/googletest/primer.html)

### Reporting Issues
When reporting test-related issues, include:
1. Test command that failed
2. Complete error output
3. System information (OS, Qt version, compiler)
4. Steps to reproduce
5. Expected vs actual behavior

### Issue Template
```
**Test Command**: `ctest -R test_name`
**Error Output**: [paste complete error]
**System**: Ubuntu 20.04, Qt 6.2, GCC 9.4
**Steps to Reproduce**: 
1. Step 1
2. Step 2
**Expected**: Test should pass
**Actual**: Test fails with error X
```

## Prevention Strategies

### Code Review Checklist
- [ ] Tests are independent and don't rely on execution order
- [ ] Proper cleanup in test teardown methods
- [ ] Appropriate timeouts for async operations
- [ ] Platform-specific considerations addressed
- [ ] Test data management is robust
- [ ] Error messages are descriptive

### Continuous Monitoring
- Set up test result monitoring in CI
- Track test execution times
- Monitor flaky test rates
- Review test coverage regularly

### Regular Maintenance
- Update test baselines when UI changes
- Review and update test timeouts
- Clean up obsolete tests
- Refactor duplicated test code

This troubleshooting guide is a living document. Please contribute solutions for new issues you encounter!