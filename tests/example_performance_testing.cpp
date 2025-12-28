#include "test_base.h"
#include "performance_benchmarks.h"
#include "test_data_generator.h"
#include <QTest>
#include <QCoreApplication>
#include <QThread>
#include <QDebug>
#include <QRandomGenerator>
#include <QCryptographicHash>
#include <QFile>
#include <QDir>
#include <QElapsedTimer>
#include <QtConcurrent>
#include <QFuture>
#include <algorithm>
#include <vector>

/**
 * @brief Example test class demonstrating the performance benchmarking system
 */
DECLARE_TEST_CLASS(PerformanceTestingExample, Performance, High, "performance", "benchmarks", "example")

private:
    PerformanceBenchmarks* m_performanceBenchmarks;
    TestDataGenerator* m_testDataGenerator;
    QString m_testDataDirectory;

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        logTestInfo("Setting up performance testing example");
        
        // Create performance benchmarks instance
        m_performanceBenchmarks = new PerformanceBenchmarks(this);
        
        // Configure performance testing
        PerformanceBenchmarks::BenchmarkConfig config;
        config.iterations = 5;
        config.warmupIterations = 2;
        config.measureMemory = true;
        config.measureCPU = false; // Disable CPU measurement for CI compatibility
        config.timeoutSeconds = 30.0;
        m_performanceBenchmarks->setBenchmarkConfig(config);
        
        // Create test data generator
        m_testDataGenerator = new TestDataGenerator();
        m_testDataDirectory = createTestDirectory("performance_test_data");
        m_testDataGenerator->setTemporaryDirectory(m_testDataDirectory);
    }

    void cleanupTestCase() {
        logTestInfo("Cleaning up performance testing example");
        TestBase::cleanupTestCase();
    }

    // Test basic performance measurement
    TEST_METHOD(test_performanceMeasurement_basicTiming_measuresAccurately) {
        logTestStep("Testing basic performance measurement");
        
        // Test simple computation benchmark
        auto result = m_performanceBenchmarks->measureExecutionTime("simple_computation", []() {
            // Simulate some computation work
            volatile int sum = 0;
            for (int i = 0; i < 100000; ++i) {
                sum += i * i;
            }
        });
        
        TEST_VERIFY_WITH_MSG(result.value > 0, "Execution time should be positive");
        TEST_VERIFY_WITH_MSG(result.value < 1000, "Simple computation should complete quickly");
        TEST_COMPARE_WITH_MSG(result.unit, QString("ms"), "Unit should be milliseconds");
        
        logTestInfo(QString("Simple computation took: %1 %2").arg(result.value).arg(result.unit));
        
        // Test string processing benchmark
        auto stringResult = m_performanceBenchmarks->measureExecutionTime("string_processing", []() {
            QString text = "Performance testing with Qt framework and C++ implementation";
            for (int i = 0; i < 1000; ++i) {
                text = text.toUpper().toLower().trimmed();
                text.replace("performance", "PERFORMANCE");
            }
        });
        
        TEST_VERIFY_WITH_MSG(stringResult.value > 0, "String processing time should be positive");
        logTestInfo(QString("String processing took: %1 %2").arg(stringResult.value).arg(stringResult.unit));
        
        logTestStep("Basic performance measurement test completed successfully");
    }

    TEST_METHOD(test_performanceMeasurement_memoryUsage_tracksCorrectly) {
        logTestStep("Testing memory usage measurement");
        
        // Test memory allocation benchmark
        auto memoryStats = m_performanceBenchmarks->measureMemoryUsage("memory_allocation", []() {
            // Allocate and use memory
            std::vector<QByteArray> data;
            for (int i = 0; i < 100; ++i) {
                QByteArray chunk(10000, 'A' + (i % 26)); // 10KB chunks
                data.push_back(chunk);
            }
            
            // Process the data to ensure it's not optimized away
            for (auto& chunk : data) {
                chunk[0] = 'X';
            }
        });
        
        TEST_VERIFY_WITH_MSG(memoryStats.peakUsage > memoryStats.initialUsage, 
                           "Peak memory should be higher than initial");
        TEST_VERIFY_WITH_MSG(memoryStats.allocated >= 0, 
                           "Allocated memory should be non-negative");
        
        logTestInfo(QString("Memory stats - Initial: %1, Peak: %2, Allocated: %3")
                   .arg(PerformanceBenchmarks::formatBytes(memoryStats.initialUsage))
                   .arg(PerformanceBenchmarks::formatBytes(memoryStats.peakUsage))
                   .arg(PerformanceBenchmarks::formatBytes(memoryStats.allocated)));
        
        logTestStep("Memory usage measurement test completed successfully");
    }

    TEST_METHOD(test_performanceMeasurement_throughputTesting_calculatesCorrectly) {
        logTestStep("Testing throughput measurement");
        
        // Test operations throughput
        auto throughputResult = m_performanceBenchmarks->measureThroughput("hash_operations", []() -> int {
            int operations = 0;
            QCryptographicHash hash(QCryptographicHash::Md5);
            
            for (int i = 0; i < 1000; ++i) {
                QString data = QString("test_data_%1").arg(i);
                hash.addData(data.toUtf8());
                hash.result(); // Force calculation
                hash.reset();
                operations++;
            }
            
            return operations;
        });
        
        TEST_VERIFY_WITH_MSG(throughputResult.value > 0, "Throughput should be positive");
        TEST_COMPARE_WITH_MSG(throughputResult.unit, QString("ops/sec"), "Unit should be ops/sec");
        TEST_VERIFY_WITH_MSG(throughputResult.metadata.contains("operations"), 
                           "Should contain operations metadata");
        
        logTestInfo(QString("Hash operations throughput: %1 %2")
                   .arg(throughputResult.value, 0, 'f', 2).arg(throughputResult.unit));
        
        // Test data throughput
        auto dataResult = m_performanceBenchmarks->measureDataThroughput("file_processing", []() -> qint64 {
            qint64 bytesProcessed = 0;
            
            // Simulate file processing
            for (int i = 0; i < 100; ++i) {
                QByteArray data(1024, 'A' + (i % 26)); // 1KB per iteration
                
                // Simulate processing (checksum calculation)
                QCryptographicHash hash(QCryptographicHash::Sha256);
                hash.addData(data);
                hash.result();
                
                bytesProcessed += data.size();
            }
            
            return bytesProcessed;
        });
        
        TEST_VERIFY_WITH_MSG(dataResult.value > 0, "Data throughput should be positive");
        logTestInfo(QString("File processing throughput: %1 bytes/sec")
                   .arg(dataResult.value, 0, 'f', 2));
        
        logTestStep("Throughput measurement test completed successfully");
    }

    TEST_METHOD(test_performanceBaselines_createAndCompare_worksCorrectly) {
        logTestStep("Testing performance baseline management");
        
        // Create a baseline benchmark
        auto baselineResult = m_performanceBenchmarks->measureExecutionTime("baseline_test", []() {
            // Consistent workload for baseline
            QStringList items;
            for (int i = 0; i < 1000; ++i) {
                items.append(QString("item_%1").arg(i));
            }
            items.sort();
        });
        
        // Create baseline
        QString baselineName = "baseline_test";
        TEST_VERIFY_WITH_MSG(
            m_performanceBenchmarks->createBaseline(baselineName, baselineResult),
            "Should create baseline successfully"
        );
        
        TEST_VERIFY_WITH_MSG(
            m_performanceBenchmarks->baselineExists(baselineName),
            "Baseline should exist after creation"
        );
        
        // Test comparison with baseline (should match closely)
        auto comparisonResult = m_performanceBenchmarks->measureExecutionTime("baseline_test", []() {
            // Same workload as baseline
            QStringList items;
            for (int i = 0; i < 1000; ++i) {
                items.append(QString("item_%1").arg(i));
            }
            items.sort();
        });
        
        auto comparison = m_performanceBenchmarks->compareWithBaseline(baselineName, comparisonResult);
        TEST_VERIFY_WITH_MSG(comparison.baseline > 0, "Baseline value should be positive");
        TEST_VERIFY_WITH_MSG(comparison.metadata.contains("regression"), 
                           "Should contain regression information");
        
        double regression = comparison.metadata["regression"].toDouble();
        logTestInfo(QString("Performance comparison - Current: %1ms, Baseline: %2ms, Regression: %3%")
                   .arg(comparison.value).arg(comparison.baseline).arg(regression, 0, 'f', 2));
        
        // Test with intentionally slower operation (should detect regression)
        auto slowerResult = m_performanceBenchmarks->measureExecutionTime("baseline_test", []() {
            // Same workload but with added delay
            QStringList items;
            for (int i = 0; i < 1000; ++i) {
                items.append(QString("item_%1").arg(i));
            }
            items.sort();
            QThread::msleep(10); // Add 10ms delay
        });
        
        auto regressionComparison = m_performanceBenchmarks->compareWithBaseline(baselineName, slowerResult);
        double regressionPercent = regressionComparison.metadata["regression"].toDouble();
        
        TEST_VERIFY_WITH_MSG(regressionPercent > 0, "Should detect performance regression");
        logTestInfo(QString("Intentional regression detected: %1%").arg(regressionPercent, 0, 'f', 2));
        
        logTestStep("Performance baseline management test completed successfully");
    }

    TEST_METHOD(test_performanceBenchmarks_fileOperations_measuresRealWorldScenarios) {
        logTestStep("Testing real-world file operation performance");
        
        // Generate test files for performance testing
        TestDataGenerator::DirectorySpec spec;
        spec.name = "perf_test_files";
        spec.filesPerDirectory = 50;
        spec.subdirectories = 2;
        spec.minFileSize = 1024;   // 1KB
        spec.maxFileSize = 10240;  // 10KB
        
        QString testFilesDir = m_testDataGenerator->generateTestDirectory(spec, m_testDataDirectory);
        TEST_VERIFY_WITH_MSG(!testFilesDir.isEmpty(), "Should generate test files");
        
        // Benchmark file scanning performance
        auto scanResult = m_performanceBenchmarks->measureExecutionTime("file_scanning", [testFilesDir]() {
            QDir dir(testFilesDir);
            QStringList files = dir.entryList(QDir::Files, QDir::Name);
            
            // Simulate file processing
            for (const QString& fileName : files) {
                QString filePath = dir.absoluteFilePath(fileName);
                QFileInfo info(filePath);
                
                // Get file information (simulating what CloneClean might do)
                qint64 size = info.size();
                QDateTime modified = info.lastModified();
                Q_UNUSED(size)
                Q_UNUSED(modified)
            }
        });
        
        TEST_VERIFY_WITH_MSG(scanResult.value > 0, "File scanning should take measurable time");
        logTestInfo(QString("File scanning performance: %1 %2").arg(scanResult.value).arg(scanResult.unit));
        
        // Benchmark hash calculation performance
        auto hashResult = m_performanceBenchmarks->measureThroughput("hash_calculation", [testFilesDir]() -> int {
            QDir dir(testFilesDir);
            QStringList files = dir.entryList(QDir::Files, QDir::Name);
            int filesProcessed = 0;
            
            for (const QString& fileName : files) {
                QString filePath = dir.absoluteFilePath(fileName);
                QFile file(filePath);
                
                if (file.open(QIODevice::ReadOnly)) {
                    QCryptographicHash hash(QCryptographicHash::Md5);
                    hash.addData(&file);
                    hash.result(); // Force calculation
                    filesProcessed++;
                }
            }
            
            return filesProcessed;
        });
        
        TEST_VERIFY_WITH_MSG(hashResult.value > 0, "Hash calculation throughput should be positive");
        logTestInfo(QString("Hash calculation throughput: %1 files/sec").arg(hashResult.value, 0, 'f', 2));
        
        logTestStep("Real-world file operation performance test completed successfully");
    }

    TEST_METHOD(test_performanceBenchmarks_statisticalAnalysis_calculatesCorrectly) {
        logTestStep("Testing statistical analysis of performance data");
        
        // Generate multiple performance measurements
        QList<PerformanceBenchmarks::PerformanceResult> results;
        
        for (int i = 0; i < 10; ++i) {
            auto result = m_performanceBenchmarks->measureExecutionTime("statistical_test", [i]() {
                // Variable workload to create different execution times
                volatile int sum = 0;
                int iterations = 10000 + (i * 1000); // Varying workload
                for (int j = 0; j < iterations; ++j) {
                    sum += j * j;
                }
            });
            results.append(result);
        }
        
        TEST_VERIFY_WITH_MSG(results.size() == 10, "Should have 10 measurements");
        
        // Extract values for analysis
        QList<double> values;
        for (const auto& result : results) {
            values.append(result.value);
        }
        
        // Test statistical calculations
        double mean = m_performanceBenchmarks->calculateMean(values);
        double median = m_performanceBenchmarks->calculateMedian(values);
        double stddev = m_performanceBenchmarks->calculateStandardDeviation(values);
        double p95 = m_performanceBenchmarks->calculatePercentile(values, 95.0);
        
        TEST_VERIFY_WITH_MSG(mean > 0, "Mean should be positive");
        TEST_VERIFY_WITH_MSG(median > 0, "Median should be positive");
        TEST_VERIFY_WITH_MSG(stddev >= 0, "Standard deviation should be non-negative");
        TEST_VERIFY_WITH_MSG(p95 > 0, "95th percentile should be positive");
        
        logTestInfo(QString("Statistical analysis:"));
        logTestInfo(QString("  Mean: %1ms").arg(mean, 0, 'f', 2));
        logTestInfo(QString("  Median: %1ms").arg(median, 0, 'f', 2));
        logTestInfo(QString("  Std Dev: %1ms").arg(stddev, 0, 'f', 2));
        logTestInfo(QString("  95th Percentile: %1ms").arg(p95, 0, 'f', 2));
        
        // Verify statistical relationships
        TEST_VERIFY_WITH_MSG(p95 >= median, "95th percentile should be >= median");
        
        logTestStep("Statistical analysis test completed successfully");
    }

    TEST_METHOD(test_performanceBenchmarks_concurrentOperations_measuresParallelPerformance) {
        logTestStep("Testing concurrent operation performance measurement");
        
        // Skip this test in CI environments to avoid resource contention
        skipIfCI("Concurrent performance test may be unreliable in CI");
        
        // Test sequential vs parallel performance
        auto sequentialResult = m_performanceBenchmarks->measureExecutionTime("sequential_processing", []() {
            for (int i = 0; i < 100; ++i) {
                // Simulate CPU-intensive work
                QCryptographicHash hash(QCryptographicHash::Sha256);
                QByteArray data(1000, 'A' + (i % 26));
                hash.addData(data);
                hash.result();
            }
        });
        
        auto parallelResult = m_performanceBenchmarks->measureExecutionTime("parallel_processing", []() {
            // Use QtConcurrent for parallel processing
            QList<int> indices;
            for (int i = 0; i < 100; ++i) {
                indices.append(i);
            }
            
            QtConcurrent::blockingMap(indices, [](int i) {
                // Same CPU-intensive work as sequential version
                QCryptographicHash hash(QCryptographicHash::Sha256);
                QByteArray data(1000, 'A' + (i % 26));
                hash.addData(data);
                hash.result();
            });
        });
        
        TEST_VERIFY_WITH_MSG(sequentialResult.value > 0, "Sequential processing should take time");
        TEST_VERIFY_WITH_MSG(parallelResult.value > 0, "Parallel processing should take time");
        
        double speedup = sequentialResult.value / parallelResult.value;
        logTestInfo(QString("Performance comparison:"));
        logTestInfo(QString("  Sequential: %1ms").arg(sequentialResult.value, 0, 'f', 2));
        logTestInfo(QString("  Parallel: %1ms").arg(parallelResult.value, 0, 'f', 2));
        logTestInfo(QString("  Speedup: %1x").arg(speedup, 0, 'f', 2));
        
        // Parallel should generally be faster (though not guaranteed in all environments)
        if (speedup > 1.0) {
            logTestInfo("Parallel processing showed performance improvement");
        } else {
            logTestWarning("Parallel processing did not show improvement (may be environment-dependent)");
        }
        
        logTestStep("Concurrent operation performance test completed successfully");
    }

    TEST_METHOD(test_performanceBenchmarks_macroUsage_simplifiesTesting) {
        logTestStep("Testing performance benchmark macros");
        
        // Test BENCHMARK_MEASURE macro
        BENCHMARK_MEASURE("macro_test_simple", {
            QStringList list;
            for (int i = 0; i < 1000; ++i) {
                list.append(QString::number(i));
            }
            list.sort();
        });
        
        // Test BENCHMARK_MEMORY macro
        BENCHMARK_MEMORY("macro_test_memory", {
            QList<QByteArray> data;
            for (int i = 0; i < 50; ++i) {
                data.append(QByteArray(1000, 'X'));
            }
        });
        
        // Test BENCHMARK_THROUGHPUT macro
        BENCHMARK_THROUGHPUT("macro_test_throughput", {
            // Return number of operations performed
            int ops = 0;
            for (int i = 0; i < 100; ++i) {
                QString str = QString("operation_%1").arg(i);
                str.toUpper();
                ops++;
            }
            return ops;
        }, 50); // Expect at least 50 ops/sec
        
        logTestStep("Performance benchmark macros test completed successfully");
    }

END_TEST_CLASS()

/**
 * @brief Main function for running the performance testing example
 */
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    qDebug() << "========================================";
    qDebug() << "Performance Benchmarking System Example";
    qDebug() << "========================================";
    
    // Load test configuration
    TestConfig::instance().loadConfiguration();
    
    // Create and run the test
    PerformanceTestingExample test;
    
    if (test.shouldRunTest()) {
        int result = QTest::qExec(&test, argc, argv);
        
        if (result == 0) {
            qDebug() << "✅ Performance testing example PASSED";
        } else {
            qDebug() << "❌ Performance testing example FAILED";
        }
        
        return result;
    } else {
        qDebug() << "⏭️  Performance testing example SKIPPED (disabled by configuration)";
        return 0;
    }
}

#include "example_performance_testing.moc"