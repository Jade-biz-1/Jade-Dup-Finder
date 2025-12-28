# Load and Stress Testing Framework

## Overview

The Load and Stress Testing Framework provides comprehensive capabilities for validating system performance under various load conditions, stress scenarios, and scalability requirements. It enables automated testing of high-volume scenarios, concurrent operations, resource limits, and system scalability for the CloneClean application.

## Key Features

### 🚀 Load Testing
- **Concurrent Users**: Simulate multiple users performing operations simultaneously
- **High Volume Files**: Test with large numbers of files and directories
- **Large File Sizes**: Validate performance with very large individual files
- **Deep/Wide Directories**: Test complex directory structures
- **Sustained Load**: Long-running operations at controlled rates
- **Burst Load**: Short bursts of high activity with intervals
- **Gradual Ramp**: Gradually increasing load over time

### 💪 Stress Testing
- **Stress to Failure**: Push system until failure point is reached
- **Memory Stress**: Test memory usage limits and leak detection
- **CPU Stress**: Validate CPU usage under intensive operations
- **Concurrency Stress**: Test maximum concurrent operation limits
- **Resource Exhaustion**: Find system resource boundaries

### 📈 Scalability Testing
- **File Count Scaling**: Analyze performance scaling with file quantities
- **File Size Scaling**: Test scaling behavior with different file sizes
- **Thread Scaling**: Validate multi-threading efficiency
- **Comprehensive Analysis**: Combined scaling factor analysis

### 📊 Performance Analysis
- **Statistical Analysis**: Response times, throughput, percentiles
- **Resource Monitoring**: Memory, CPU, and thread usage tracking
- **Baseline Comparison**: Compare against performance baselines
- **Regression Detection**: Automatic performance regression identification

## Core Classes

### LoadStressTesting

The main class providing all load and stress testing capabilities.

```cpp
#include "load_stress_testing.h"

LoadStressTesting loadTester;

// Configure testing framework
loadTester.setMaxConcurrentThreads(8);
loadTester.setResourceMonitoringInterval(1000); // 1 second
```

### LoadTestConfig

Configuration structure for load tests.

```cpp
LoadStressTesting::LoadTestConfig config;
config.name = "My Load Test";
config.type = LoadStressTesting::LoadTestType::ConcurrentUsers;
config.concurrentThreads = 10;
config.totalOperations = 1000;
config.durationMs = 60000; // 1 minute
config.measureMemory = true;
config.measureCpu = true;
```

### StressTestConfig

Configuration structure for stress tests.

```cpp
LoadStressTesting::StressTestConfig config;
config.name = "Memory Stress Test";
config.maxConcurrentOperations = 100;
config.memoryLimitMB = 512;
config.cpuLimitPercent = 90.0;
config.stopOnResourceLimit = true;
```

## Usage Examples

### Basic Load Testing

```cpp
// Simple concurrent user test
bool success = loadTester.runConcurrentUserTest(10, 50, [](int userId) {
    // Simulate user operation
    performUserOperation(userId);
});

// High volume file test
QString testDir = "/tmp/load_test";
success = loadTester.runHighVolumeFileTest(1000, 1024, testDir);

// Sustained load test
success = loadTester.runSustainedLoadTest(30000, 20, []() {
    // Operation to repeat at 20 ops/sec for 30 seconds
    performOperation();
});
```

### Advanced Load Testing

```cpp
// Custom load test with detailed configuration
LoadStressTesting::LoadTestConfig config;
config.name = "Custom File Processing Load";
config.type = LoadStressTesting::LoadTestType::MixedWorkload;
config.concurrentThreads = 8;
config.totalOperations = 500;
config.timeoutMs = 120000; // 2 minutes
config.measureMemory = true;
config.measureCpu = true;
config.collectDetailedMetrics = true;

bool success = loadTester.runLoadTest(config, []() {
    // Custom load test operation
    QString fileName = generateTestFileName();
    processFile(fileName);
    calculateHash(fileName);
    detectDuplicates(fileName);
});

// Retrieve and analyze results
auto result = loadTester.getLoadTestResult("Custom File Processing Load");
qDebug() << "Average response time:" << result.averageResponseTime << "ms";
qDebug() << "Throughput:" << result.operationsPerSecond << "ops/sec";
qDebug() << "95th percentile:" << result.percentile95ResponseTime << "ms";
```

### Stress Testing

```cpp
// Memory stress test
bool success = loadTester.runMemoryStressTest(256, []() {
    // Memory-intensive operation
    static thread_local QList<QByteArray> buffers;
    buffers.append(QByteArray(1024 * 1024, 'M')); // 1MB allocation
    
    // Simulate work
    processMemoryBuffer(buffers.last());
    
    // Occasional cleanup
    if (buffers.size() > 100) {
        buffers.removeFirst();
    }
});

// CPU stress test
success = loadTester.runCpuStressTest(85.0, []() {
    // CPU-intensive calculations
    double result = 0.0;
    for (int i = 0; i < 100000; ++i) {
        result += qSqrt(i) * qSin(i) * qCos(i);
    }
    // Use result to prevent optimization
    static volatile double globalResult = result;
});

// Stress to failure test
LoadStressTesting::StressTestConfig stressConfig;
stressConfig.name = "System Limits Test";
stressConfig.maxConcurrentOperations = 1000;
stressConfig.memoryLimitMB = 1024;
stressConfig.cpuLimitPercent = 95.0;
stressConfig.stopOnResourceLimit = true;

success = loadTester.runStressTest(stressConfig, []() {
    // Combined resource-intensive operation
    allocateMemory();
    performCpuWork();
    createFiles();
});
```

### Scalability Testing

```cpp
// File count scalability test
QList<int> fileCounts = {10, 50, 100, 500, 1000, 5000};
bool success = loadTester.runFileCountScalabilityTest(fileCounts, [](int fileCount) {
    // Create and process specified number of files
    createTestFiles(fileCount);
    processAllFiles();
    detectDuplicates();
});

// Comprehensive scalability analysis
LoadStressTesting::ScalabilityTestConfig scalabilityConfig;
scalabilityConfig.name = "CloneClean Scalability Analysis";
scalabilityConfig.fileCounts = {100, 500, 1000};
scalabilityConfig.fileSizes = {1024, 10240, 102400}; // 1KB, 10KB, 100KB
scalabilityConfig.threadCounts = {1, 2, 4, 8};
scalabilityConfig.iterationsPerConfiguration = 3;
scalabilityConfig.measureMemoryScaling = true;
scalabilityConfig.measureTimeScaling = true;
scalabilityConfig.measureThroughputScaling = true;

success = loadTester.runScalabilityTest(scalabilityConfig, 
    [](int fileCount, qint64 fileSize, int threadCount) {
        // Test with specific configuration
        setThreadPoolSize(threadCount);
        createFiles(fileCount, fileSize);
        processFiles();
    });

// Analyze scaling results
auto scalabilityResult = loadTester.getScalabilityTestResult("CloneClean Scalability Analysis");
qDebug() << "Scaling analysis:" << scalabilityResult.scalingAnalysis;
qDebug() << "Linear time scaling:" << scalabilityResult.linearTimeScaling;
qDebug() << "Linear memory scaling:" << scalabilityResult.linearMemoryScaling;
```

### CloneClean-Specific Load Tests

```cpp
// Duplicate detection load test
QString testDir = "/tmp/duplicate_test";
int fileCount = 1000;
double duplicateRatio = 0.3; // 30% duplicates

bool success = loadTester.runDuplicateDetectionLoadTest(testDir, fileCount, duplicateRatio);

// Hash calculation load test
QStringList filePaths = getTestFiles();
int concurrentHashers = 4;

success = loadTester.runHashCalculationLoadTest(filePaths, concurrentHashers);

// File scanning load test
QString rootDirectory = "/path/to/scan";
int concurrentScanners = 2;

success = loadTester.runFileScanningLoadTest(rootDirectory, concurrentScanners);

// UI load test
QWidget* mainWindow = getMainWindow();
int uiOperations = 100;

success = loadTester.runUILoadTest(mainWindow, uiOperations);
```

### Result Analysis and Reporting

```cpp
// Get individual test results
auto loadResults = loadTester.getLoadTestResults();
auto stressResults = loadTester.getStressTestResults();
auto scalabilityResults = loadTester.getScalabilityTestResults();

// Analyze load test performance
for (const auto& result : loadResults) {
    qDebug() << "Test:" << result.testName;
    qDebug() << "Success rate:" << (result.successfulOperations * 100.0 / result.totalOperations) << "%";
    qDebug() << "Average response time:" << result.averageResponseTime << "ms";
    qDebug() << "Throughput:" << result.operationsPerSecond << "ops/sec";
    qDebug() << "95th percentile:" << result.percentile95ResponseTime << "ms";
    qDebug() << "Peak memory:" << loadTester.formatMemoryUsage(result.peakMemoryUsage);
}

// Analyze stress test limits
for (const auto& result : stressResults) {
    qDebug() << "Stress test:" << result.testName;
    qDebug() << "Max concurrent operations:" << result.maxConcurrentOperationsReached;
    qDebug() << "Peak memory usage:" << result.peakMemoryUsageMB << "MB";
    qDebug() << "Peak CPU usage:" << result.peakCpuUsagePercent << "%";
    
    if (result.hitMemoryLimit) qDebug() << "Hit memory limit";
    if (result.hitCpuLimit) qDebug() << "Hit CPU limit";
    if (result.hitTimeLimit) qDebug() << "Hit time limit";
}

// Generate comprehensive report
QJsonObject report = loadTester.generateComprehensiveReport();

// Export results
bool success = loadTester.exportResults("load_stress_results.json", "json");
```

## Configuration Options

### LoadTestConfig Structure

```cpp
struct LoadTestConfig {
    QString name;                           // Test name
    LoadTestType type;                      // Type of load test
    int concurrentThreads = 1;             // Number of concurrent threads
    int totalOperations = 100;             // Total operations to perform
    qint64 durationMs = 60000;             // Test duration in milliseconds
    qint64 rampUpTimeMs = 5000;            // Ramp-up time
    qint64 rampDownTimeMs = 5000;          // Ramp-down time
    int operationsPerSecond = 10;          // Target operations per second
    qint64 maxMemoryMB = 1024;             // Maximum memory usage
    double maxCpuPercent = 80.0;           // Maximum CPU usage
    qint64 timeoutMs = 300000;             // Overall test timeout
    bool failOnError = false;              // Whether to fail on first error
    bool collectDetailedMetrics = true;    // Collect detailed metrics
    QString description;                    // Test description
    QStringList tags;                      // Tags for categorization
};
```

### StressTestConfig Structure

```cpp
struct StressTestConfig {
    QString name;                           // Test name
    int maxConcurrentOperations = 100;     // Maximum concurrent operations
    qint64 maxFileSize = 1073741824;       // Maximum file size (1GB)
    int maxFileCount = 10000;              // Maximum number of files
    int maxDirectoryDepth = 20;            // Maximum directory depth
    int maxDirectoryWidth = 1000;          // Maximum files per directory
    qint64 maxTestDurationMs = 600000;     // Maximum test duration (10 min)
    qint64 memoryLimitMB = 2048;           // Memory limit in MB
    double cpuLimitPercent = 95.0;         // CPU usage limit
    bool stopOnResourceLimit = true;       // Stop when limits hit
    bool stopOnFirstFailure = false;      // Stop on first failure
    int failureThreshold = 10;            // Failure count threshold
    QString description;                    // Test description
    QStringList tags;                      // Tags for categorization
};
```

### ScalabilityTestConfig Structure

```cpp
struct ScalabilityTestConfig {
    QString name;                           // Test name
    QList<int> fileCounts;                 // File counts to test
    QList<qint64> fileSizes;               // File sizes to test
    QList<int> threadCounts;               // Thread counts to test
    int iterationsPerConfiguration = 3;    // Iterations per configuration
    qint64 maxDurationPerTest = 300000;    // Max duration per test
    bool measureMemoryScaling = true;      // Measure memory scaling
    bool measureTimeScaling = true;        // Measure time scaling
    bool measureThroughputScaling = true;  // Measure throughput scaling
    QString description;                    // Test description
    QStringList tags;                      // Tags for categorization
};
```

## Integration with Test Framework

### Qt Test Integration

```cpp
class MyLoadStressTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase() {
        m_loadTester = new LoadStressTesting(this);
        m_loadTester->setMaxConcurrentThreads(QThread::idealThreadCount());
    }
    
    void testConcurrentFileProcessing() {
        bool success = m_loadTester->runConcurrentUserTest(10, 20, [](int userId) {
            processFilesForUser(userId);
        });
        QVERIFY2(success, "Concurrent file processing test failed");
        
        // Verify performance requirements
        auto result = m_loadTester->getLoadTestResult("concurrent_users_10_users_20_ops");
        QVERIFY(result.averageResponseTime < 2000); // Must be under 2 seconds
        QVERIFY(result.operationsPerSecond > 5.0);  // Must achieve 5 ops/sec
    }
    
    void testMemoryLimits() {
        bool success = m_loadTester->runMemoryStressTest(512, []() {
            allocateAndProcessMemory();
        });
        
        auto result = m_loadTester->getStressTestResult("memory_stress_512MB");
        QVERIFY(result.peakMemoryUsageMB <= 512); // Should not exceed limit
    }

private:
    LoadStressTesting* m_loadTester;
};
```

### Convenience Macros

```cpp
// Load test with automatic failure handling
LOAD_TEST("file_processing", config, []() {
    processFiles();
});

// Stress test with automatic failure handling
STRESS_TEST("memory_stress", stressConfig, []() {
    allocateMemory();
});

// Scalability test with automatic failure handling
SCALABILITY_TEST("file_count_scaling", scalabilityConfig, [](int files, qint64 size, int threads) {
    processFiles(files, size, threads);
});

// Verify load performance requirements
VERIFY_LOAD_PERFORMANCE("my_test", 1000.0, 10.0); // Max 1000ms, min 10 ops/sec
```

## Performance Requirements Validation

### Automated Performance Validation

```cpp
// Define performance requirements
struct PerformanceRequirement {
    QString testPattern;
    double maxResponseTimeMs;
    double minThroughputOpsPerSec;
    QString description;
};

QList<PerformanceRequirement> requirements = {
    {"concurrent_users_*", 1000.0, 5.0, "Concurrent users"},
    {"hash_calculation_*", 2000.0, 20.0, "Hash calculations"},
    {"file_scanning_*", 3000.0, 15.0, "File scanning"}
};

// Validate all requirements
auto loadResults = loadTester.getLoadTestResults();
for (const auto& req : requirements) {
    for (const auto& result : loadResults) {
        if (QRegExp(req.testPattern, Qt::CaseInsensitive, QRegExp::Wildcard).exactMatch(result.testName)) {
            QVERIFY2(result.averageResponseTime <= req.maxResponseTimeMs,
                    QString("Response time requirement failed for %1").arg(result.testName).toUtf8());
            QVERIFY2(result.operationsPerSecond >= req.minThroughputOpsPerSec,
                    QString("Throughput requirement failed for %1").arg(result.testName).toUtf8());
        }
    }
}
```

## Best Practices

### 1. Test Design
- Start with simple load tests before complex stress scenarios
- Use realistic data sizes and operation patterns
- Include both positive and negative test cases
- Test edge cases and boundary conditions

### 2. Resource Management
- Monitor system resources during testing
- Set appropriate limits to prevent system crashes
- Clean up test data between iterations
- Use temporary directories for test files

### 3. Performance Baselines
- Establish performance baselines on target hardware
- Update baselines when making intentional changes
- Use consistent test environments
- Document baseline creation conditions

### 4. Scalability Analysis
- Test with multiple data points for accurate scaling analysis
- Include both small and large scale scenarios
- Analyze scaling factors for different metrics
- Consider non-linear scaling patterns

### 5. Stress Testing Safety
- Use resource limits to prevent system damage
- Monitor for memory leaks and resource exhaustion
- Test in isolated environments when possible
- Have recovery procedures for failed stress tests

## Troubleshooting

### Common Issues

1. **Test Timeouts**
   - Increase timeout values for slow operations
   - Reduce test data size for initial validation
   - Check for deadlocks or infinite loops

2. **Resource Exhaustion**
   - Set appropriate resource limits
   - Monitor system resources during tests
   - Implement proper cleanup procedures

3. **Inconsistent Results**
   - Run multiple iterations for statistical significance
   - Control system load during testing
   - Use dedicated test environments

4. **Memory Issues**
   - Monitor for memory leaks in test code
   - Implement proper resource cleanup
   - Use appropriate memory limits

### Performance Tips

1. **Optimize Test Execution**
   - Use appropriate thread counts for the system
   - Balance test thoroughness with execution time
   - Implement efficient test data generation

2. **Resource Monitoring**
   - Use appropriate monitoring intervals
   - Focus on relevant metrics for each test type
   - Implement efficient data collection

3. **Result Analysis**
   - Use statistical analysis for meaningful results
   - Focus on trends rather than individual measurements
   - Implement automated analysis where possible

## Example Test Suite

See `example_load_stress_testing.cpp` for a comprehensive example that demonstrates:

- Concurrent user load testing
- High volume file processing
- Memory and CPU stress testing
- Scalability analysis
- CloneClean-specific load scenarios
- Performance requirement validation
- Result analysis and reporting

This example serves as both documentation and a working test suite that can be adapted for specific load and stress testing needs.

## Integration with CloneClean

The load and stress testing framework is specifically designed to validate CloneClean's performance under various conditions:

- **File Processing Load**: Test with thousands of files of varying sizes
- **Duplicate Detection Stress**: Validate performance with high duplicate ratios
- **Hash Calculation Load**: Test concurrent hash computation efficiency
- **Memory Usage Validation**: Ensure memory usage stays within acceptable limits
- **UI Responsiveness**: Validate user interface performance under load
- **Scalability Analysis**: Understand how performance scales with data size and complexity

This framework ensures that CloneClean maintains optimal performance and stability under all expected usage scenarios and can handle extreme conditions gracefully.