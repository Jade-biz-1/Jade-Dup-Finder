# Performance Benchmarking Framework

## Overview

The Performance Benchmarking Framework provides comprehensive performance measurement, analysis, and reporting capabilities for the DupFinder application. It enables automated performance testing, baseline management, regression detection, and detailed performance reporting.

## Key Features

### 🚀 Performance Measurement
- **Execution Time**: Precise timing of operations and functions
- **Memory Usage**: Real-time memory consumption monitoring
- **CPU Usage**: CPU utilization tracking during operations
- **Disk I/O**: File system operation performance measurement
- **UI Responsiveness**: User interface performance and frame rate analysis
- **Throughput**: Operations per second and data processing rates

### 📊 Statistical Analysis
- **Descriptive Statistics**: Mean, median, standard deviation, min/max
- **Percentile Analysis**: 95th and 99th percentile calculations
- **Sample Distribution**: Raw value collection and analysis
- **Trend Analysis**: Performance trends over multiple runs

### 🎯 Baseline Management
- **Baseline Creation**: Establish performance expectations
- **Baseline Comparison**: Compare current performance against baselines
- **Tolerance Configuration**: Set acceptable performance deviation ranges
- **Regression Detection**: Automatic identification of performance regressions
- **Improvement Tracking**: Recognition of performance improvements

### 📈 Reporting and Export
- **JSON Reports**: Comprehensive performance reports in JSON format
- **Comparison Reports**: Detailed baseline comparison analysis
- **Export/Import**: Baseline and result data portability
- **Real-time Monitoring**: Live performance metric updates

## Core Classes

### PerformanceBenchmark

The main benchmarking class that provides all performance measurement capabilities.

```cpp
#include "performance_benchmark.h"

PerformanceBenchmark benchmark;

// Configure benchmark settings
PerformanceBenchmark::BenchmarkConfig config;
config.name = "File Operations Test";
config.iterations = 5;
config.warmupIterations = 2;
config.measureMemory = true;
config.measureCpu = true;
benchmark.setBenchmarkConfig(config);
```

### BenchmarkRunner

A convenience class for managing and executing multiple benchmarks as a suite.

```cpp
BenchmarkRunner runner;

// Add benchmarks to the suite
runner.addBenchmark("file_creation", []() {
    // File creation benchmark code
});

runner.addBenchmark("hash_calculation", []() {
    // Hash calculation benchmark code
});

// Run all benchmarks
runner.runAllBenchmarks();
```

## Usage Examples

### Basic Performance Measurement

```cpp
// Simple function benchmarking
benchmark.runBenchmark("my_operation", []() {
    // Code to benchmark
    performSomeOperation();
});

// Manual timing
benchmark.startMeasurement("custom_operation");
performCustomOperation();
benchmark.stopMeasurement("custom_operation");

// Record custom metrics
benchmark.recordMetric("files_processed", 1000, "count");
benchmark.recordThroughput("processing_rate", 150.5); // 150.5 ops/sec
```

### File Operation Benchmarking

```cpp
// Benchmark file operations
QString testDir = "/tmp/benchmark_test";
benchmark.benchmarkFileOperations(testDir, 100, 1024); // 100 files, 1KB each

// Benchmark directory scanning
benchmark.benchmarkDirectoryScanning("/path/to/scan", true); // recursive

// Benchmark hash calculation
QStringList files = getTestFiles();
benchmark.benchmarkHashCalculation(files, "SHA256");

// Benchmark duplicate detection
benchmark.benchmarkDuplicateDetection(testDir);
```

### UI Performance Benchmarking

```cpp
QWidget* mainWindow = getMainWindow();

// Benchmark UI responsiveness
benchmark.benchmarkUIResponsiveness(mainWindow, 100);

// Benchmark widget rendering
benchmark.benchmarkWidgetRendering(mainWindow, 60);

// Benchmark theme switching
QStringList themes = {"Light", "Dark", "Blue"};
benchmark.benchmarkThemeSwitching(mainWindow, themes);
```

### Resource Monitoring

```cpp
// Start continuous resource monitoring
benchmark.startResourceMonitoring();

// Perform operations while monitoring
performLongRunningOperation();

// Stop monitoring
benchmark.stopResourceMonitoring();

// Get monitoring results
auto resourceResults = benchmark.getResourceMonitoringResults();
```

### Baseline Management

```cpp
// Create baseline from current performance data
benchmark.createBaseline("file_ops_baseline", "file_operations", "execution_time");

// Compare current performance with baseline
auto comparison = benchmark.compareWithBaseline(
    "file_ops_baseline", 
    "file_operations", 
    "execution_time"
);

if (comparison.isRegression) {
    qWarning() << "Performance regression detected:" << comparison.deviationPercent << "%";
}

// Detect all regressions
bool hasRegressions = benchmark.detectPerformanceRegressions(10.0); // 10% threshold
```

### Statistical Analysis

```cpp
// Calculate statistics for specific benchmark
auto stats = benchmark.calculateStatistics("file_operations", "execution_time");
qDebug() << "Mean execution time:" << stats.mean << "ms";
qDebug() << "95th percentile:" << stats.percentile95 << "ms";
qDebug() << "Standard deviation:" << stats.standardDeviation << "ms";

// Get all statistics
auto allStats = benchmark.calculateAllStatistics();
for (const auto& stat : allStats) {
    qDebug() << stat.benchmarkName << stat.metricName 
             << "mean:" << stat.mean << stat.unit;
}
```

### Reporting and Export

```cpp
// Generate comprehensive report
QJsonObject report = benchmark.generateReport();

// Export results to file
benchmark.exportResults("performance_report.json", "json");

// Export baselines
benchmark.exportBaselines("baselines.json");

// Import baselines
benchmark.importBaselines("baselines.json");

// Generate comparison report
auto comparisons = benchmark.compareAllWithBaselines();
QJsonObject comparisonReport = benchmark.generateComparisonReport(comparisons);
```

## Convenience Macros

The framework provides convenient macros for common benchmarking tasks:

```cpp
// Simple measurement macros
BENCHMARK_START("operation_name");
performOperation();
BENCHMARK_STOP("operation_name");

// Record custom metrics
BENCHMARK_RECORD("files_processed", fileCount, "count");

// Function benchmarking with automatic failure handling
BENCHMARK_FUNCTION("my_benchmark", []() {
    performBenchmarkedOperation();
});

// Baseline comparison with automatic regression detection
BENCHMARK_COMPARE_BASELINE("my_baseline", "my_benchmark", "execution_time");
```

## Configuration Options

### BenchmarkConfig Structure

```cpp
struct BenchmarkConfig {
    QString name;                           // Benchmark name
    int iterations = 1;                     // Number of iterations
    int warmupIterations = 0;              // Warmup iterations (not measured)
    qint64 timeoutMs = 60000;              // Timeout per iteration
    bool measureMemory = true;              // Measure memory usage
    bool measureCpu = true;                 // Measure CPU usage
    bool measureDiskIO = false;             // Measure disk I/O
    bool measureNetworkIO = false;          // Measure network I/O
    bool measureUIResponsiveness = false;   // Measure UI responsiveness
    int samplingIntervalMs = 100;          // Sampling interval for continuous metrics
    QMap<QString, QVariant> customParams;  // Custom parameters
    QString description;                    // Benchmark description
    QStringList tags;                      // Tags for categorization
};
```

### Performance Baseline Configuration

```cpp
struct PerformanceBaseline {
    QString name;                          // Baseline name
    QString benchmarkName;                 // Associated benchmark
    QString metricName;                    // Metric name
    double expectedValue = 0.0;           // Expected performance value
    double tolerancePercent = 10.0;       // Acceptable deviation (%)
    double warningThreshold = 5.0;        // Warning threshold (%)
    QDateTime created;                     // Creation timestamp
    QString platform;                      // Platform information
    QString version;                       // Software version
    QMap<QString, QVariant> environment;   // Environment details
};
```

## Integration with Test Framework

### Qt Test Integration

```cpp
class MyPerformanceTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase() {
        m_benchmark = new PerformanceBenchmark(this);
        // Configure benchmark
    }
    
    void testFileOperationPerformance() {
        bool success = m_benchmark->benchmarkFileOperations(testDir, 100, 1024);
        QVERIFY2(success, "File operation benchmark failed");
        
        // Verify performance meets requirements
        auto stats = m_benchmark->calculateStatistics("file_operations", "execution_time");
        QVERIFY2(stats.mean < 5000, "File operations too slow"); // Must be under 5 seconds
    }
    
    void testPerformanceRegression() {
        // Run benchmark
        BENCHMARK_FUNCTION("regression_test", []() {
            performCriticalOperation();
        });
        
        // Check against baseline
        BENCHMARK_COMPARE_BASELINE("critical_baseline", "regression_test", "execution_time");
    }

private:
    PerformanceBenchmark* m_benchmark;
};
```

### CI/CD Integration

```cpp
// In your CI/CD pipeline test
class CIPipelinePerformanceTest : public QObject {
    Q_OBJECT

private slots:
    void validatePerformanceRequirements() {
        BenchmarkRunner runner;
        
        // Add all critical benchmarks
        runner.addBenchmark("file_scanning", []() { /* ... */ });
        runner.addBenchmark("hash_calculation", []() { /* ... */ });
        runner.addBenchmark("duplicate_detection", []() { /* ... */ });
        
        // Run benchmark suite
        bool success = runner.runAllBenchmarks();
        QVERIFY2(success, "Performance benchmark suite failed");
        
        // Check for regressions
        auto benchmark = runner.getAllResults();
        bool hasRegressions = benchmark.detectPerformanceRegressions(15.0); // 15% threshold for CI
        QVERIFY2(!hasRegressions, "Performance regressions detected in CI pipeline");
        
        // Export results for analysis
        runner.generateSuiteReport("ci_performance_report.json");
    }
};
```

## Best Practices

### 1. Benchmark Configuration
- Use appropriate iteration counts (3-10 for most cases)
- Include warmup iterations for JIT-compiled or cached operations
- Set reasonable timeouts to prevent hanging tests
- Enable only necessary measurements to reduce overhead

### 2. Test Environment
- Run benchmarks on consistent hardware
- Minimize background processes during benchmarking
- Use dedicated test data that doesn't change between runs
- Consider system load and resource availability

### 3. Baseline Management
- Create baselines on representative hardware
- Update baselines when making intentional performance changes
- Use appropriate tolerance levels (5-15% typically)
- Document baseline creation conditions

### 4. Statistical Interpretation
- Use multiple iterations for statistical significance
- Focus on median and percentiles for skewed distributions
- Consider standard deviation for consistency assessment
- Analyze trends over time, not just single measurements

### 5. Regression Detection
- Set conservative thresholds for automated regression detection
- Investigate all detected regressions promptly
- Consider performance improvements as opportunities to update baselines
- Use different thresholds for different types of operations

## Troubleshooting

### Common Issues

1. **Inconsistent Results**
   - Increase iteration count
   - Add warmup iterations
   - Check for background processes
   - Verify test data consistency

2. **Memory Measurement Issues**
   - Ensure proper cleanup between iterations
   - Consider garbage collection in managed environments
   - Check for memory leaks in test code

3. **Timeout Issues**
   - Increase timeout values for slow operations
   - Optimize test data size for reasonable execution times
   - Consider breaking large benchmarks into smaller ones

4. **Platform Differences**
   - Create platform-specific baselines
   - Use relative performance comparisons
   - Account for hardware differences in tolerances

### Performance Tips

1. **Minimize Measurement Overhead**
   - Use sampling for continuous monitoring
   - Disable unnecessary measurements
   - Batch metric recordings when possible

2. **Optimize Test Data**
   - Use representative but manageable data sizes
   - Pre-generate test data when possible
   - Clean up test data between runs

3. **Resource Management**
   - Monitor system resources during benchmarking
   - Ensure adequate disk space for test files
   - Consider memory constraints for large datasets

## Example Test Suite

See `example_performance_benchmark.cpp` for a comprehensive example that demonstrates:

- File operation benchmarking
- UI performance testing
- Memory usage monitoring
- Baseline creation and comparison
- Regression detection
- Report generation
- Statistical analysis

This example serves as both documentation and a working test suite that can be adapted for specific performance testing needs.

## Integration with DupFinder

The performance benchmarking framework is specifically designed to test DupFinder's core operations:

- **File Scanning**: Directory traversal and file discovery performance
- **Hash Calculation**: Cryptographic hash computation efficiency
- **Duplicate Detection**: Algorithm performance for finding duplicates
- **UI Responsiveness**: User interface performance during operations
- **Memory Management**: Memory usage patterns and leak detection
- **Scalability**: Performance with large datasets and file counts

This framework ensures that DupFinder maintains optimal performance across all supported platforms and use cases.