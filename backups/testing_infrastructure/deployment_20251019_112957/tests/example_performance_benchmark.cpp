#include <QtTest>
#include <QApplication>
#include <QWidget>
#include <QDir>
#include <QTemporaryDir>
#include "performance_benchmark.h"
#include "test_data_generator.h"

/**
 * @brief Example performance benchmark tests for DupFinder
 * 
 * This class demonstrates how to use the PerformanceBenchmark framework
 * to measure and validate performance of various DupFinder operations.
 */
class ExamplePerformanceBenchmark : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // File operation benchmarks
    void benchmarkFileCreation();
    void benchmarkFileReading();
    void benchmarkDirectoryScanning();
    void benchmarkHashCalculation();
    void benchmarkDuplicateDetection();
    
    // UI performance benchmarks
    void benchmarkUIResponsiveness();
    void benchmarkWidgetRendering();
    void benchmarkThemeSwitching();
    
    // Memory and resource benchmarks
    void benchmarkMemoryUsage();
    void benchmarkResourceMonitoring();
    
    // Baseline and comparison tests
    void testBaselineCreation();
    void testPerformanceComparison();
    void testRegressionDetection();
    
    // Reporting and export tests
    void testReportGeneration();
    void testBaselineExportImport();

private:
    PerformanceBenchmark* m_benchmark;
    BenchmarkRunner* m_runner;
    TestDataGenerator* m_dataGenerator;
    QTemporaryDir* m_testDir;
    QString m_testPath;
};

void ExamplePerformanceBenchmark::initTestCase() {
    // Initialize performance benchmark framework
    m_benchmark = new PerformanceBenchmark(this);
    m_runner = new BenchmarkRunner(this);
    m_dataGenerator = new TestDataGenerator(this);
    
    // Create temporary directory for test files
    m_testDir = new QTemporaryDir();
    QVERIFY(m_testDir->isValid());
    m_testPath = m_testDir->path();
    
    // Configure benchmark settings
    PerformanceBenchmark::BenchmarkConfig config;
    config.name = "DupFinder Performance Tests";
    config.iterations = 3;
    config.warmupIterations = 1;
    config.timeoutMs = 30000; // 30 seconds timeout
    config.measureMemory = true;
    config.measureCpu = true;
    config.samplingIntervalMs = 100;
    config.description = "Performance benchmarks for DupFinder file operations";
    config.tags << "file_operations" << "performance" << "automated";
    
    m_benchmark->setBenchmarkConfig(config);
    
    qDebug() << "Performance benchmark test suite initialized";
    qDebug() << "Test directory:" << m_testPath;
    qDebug() << "System info:" << m_benchmark->getSystemInfo();
}

void ExamplePerformanceBenchmark::cleanupTestCase() {
    // Export final results
    QString reportPath = QDir(m_testPath).absoluteFilePath("performance_report.json");
    m_benchmark->exportResults(reportPath);
    qDebug() << "Performance report exported to:" << reportPath;
    
    // Cleanup
    delete m_testDir;
    qDebug() << "Performance benchmark test suite completed";
}

void ExamplePerformanceBenchmark::benchmarkFileCreation() {
    qDebug() << "Running file creation benchmark...";
    
    // Test different file sizes and counts
    QList<QPair<int, qint64>> testCases = {
        {100, 1024},        // 100 files of 1KB each
        {50, 10240},        // 50 files of 10KB each
        {10, 102400},       // 10 files of 100KB each
        {5, 1048576}        // 5 files of 1MB each
    };
    
    for (const auto& testCase : testCases) {
        int fileCount = testCase.first;
        qint64 fileSize = testCase.second;
        
        QString testSubDir = QString("file_creation_%1_%2").arg(fileCount).arg(fileSize);
        QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
        
        bool success = m_benchmark->benchmarkFileOperations(testPath, fileCount, fileSize);
        QVERIFY2(success, QString("File creation benchmark failed for %1 files of %2 bytes")
                 .arg(fileCount).arg(fileSize).toUtf8().constData());
        
        // Verify results were recorded
        QList<PerformanceBenchmark::PerformanceResult> results = m_benchmark->getResults();
        QVERIFY(!results.isEmpty());
        
        qDebug() << QString("File creation: %1 files (%2 bytes each) completed")
                    .arg(fileCount).arg(m_benchmark->formatBytes(fileSize));
    }
}

void ExamplePerformanceBenchmark::benchmarkFileReading() {
    qDebug() << "Running file reading benchmark...";
    
    // Create test files first
    QString testSubDir = "file_reading_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    QDir().mkpath(testPath);
    
    // Generate test files
    QStringList testFiles = m_dataGenerator->createTestFiles(testPath, 20, 50000); // 20 files of ~50KB each
    QVERIFY(!testFiles.isEmpty());
    
    // Benchmark file reading
    BENCHMARK_FUNCTION("file_reading_performance", [this, testFiles]() {
        BENCHMARK_START("file_reading");
        
        qint64 totalBytes = 0;
        for (const QString& filePath : testFiles) {
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QByteArray data = file.readAll();
                totalBytes += data.size();
            }
        }
        
        BENCHMARK_STOP("file_reading");
        BENCHMARK_RECORD("total_bytes_read", totalBytes, "bytes");
    });
    
    // Verify performance metrics
    auto stats = m_benchmark->calculateStatistics("file_reading_performance", "file_reading");
    QVERIFY(stats.sampleCount > 0);
    QVERIFY(stats.mean > 0);
    
    qDebug() << QString("File reading: %1 files, average time: %2")
                .arg(testFiles.size()).arg(m_benchmark->formatDuration(stats.mean));
}

void ExamplePerformanceBenchmark::benchmarkDirectoryScanning() {
    qDebug() << "Running directory scanning benchmark...";
    
    // Create a complex directory structure
    QString testSubDir = "directory_scan_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Generate nested directory structure with files
    m_dataGenerator->createNestedDirectoryStructure(testPath, 3, 5, 10); // 3 levels, 5 dirs per level, 10 files per dir
    
    // Benchmark recursive directory scanning
    bool success = m_benchmark->benchmarkDirectoryScanning(testPath, true);
    QVERIFY2(success, "Recursive directory scanning benchmark failed");
    
    // Benchmark flat directory scanning
    success = m_benchmark->benchmarkDirectoryScanning(testPath, false);
    QVERIFY2(success, "Flat directory scanning benchmark failed");
    
    // Compare recursive vs flat scanning performance
    auto recursiveStats = m_benchmark->calculateStatistics("directory_scan_recursive", "directory_scan");
    auto flatStats = m_benchmark->calculateStatistics("directory_scan_flat", "directory_scan");
    
    QVERIFY(recursiveStats.sampleCount > 0);
    QVERIFY(flatStats.sampleCount > 0);
    
    qDebug() << QString("Directory scanning - Recursive: %1, Flat: %2")
                .arg(m_benchmark->formatDuration(recursiveStats.mean))
                .arg(m_benchmark->formatDuration(flatStats.mean));
}

void ExamplePerformanceBenchmark::benchmarkHashCalculation() {
    qDebug() << "Running hash calculation benchmark...";
    
    // Create test files for hashing
    QString testSubDir = "hash_calculation_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    QStringList testFiles = m_dataGenerator->createTestFiles(testPath, 15, 100000); // 15 files of ~100KB each
    QVERIFY(!testFiles.isEmpty());
    
    // Benchmark different hash algorithms
    QStringList algorithms = {"MD5", "SHA1", "SHA256"};
    
    for (const QString& algorithm : algorithms) {
        bool success = m_benchmark->benchmarkHashCalculation(testFiles, algorithm);
        QVERIFY2(success, QString("Hash calculation benchmark failed for %1").arg(algorithm).toUtf8().constData());
        
        auto stats = m_benchmark->calculateStatistics(QString("hash_calculation_%1").arg(algorithm.toLower()), "hash_calculation");
        QVERIFY(stats.sampleCount > 0);
        
        qDebug() << QString("Hash calculation (%1): %2, throughput: %3 MB/s")
                    .arg(algorithm)
                    .arg(m_benchmark->formatDuration(stats.mean))
                    .arg(stats.mean > 0 ? QString::number(100.0 / (stats.mean / 1000.0), 'f', 2) : "N/A");
    }
}

void ExamplePerformanceBenchmark::benchmarkDuplicateDetection() {
    qDebug() << "Running duplicate detection benchmark...";
    
    // Create test directory with duplicates
    QString testSubDir = "duplicate_detection_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Generate files with some duplicates
    m_dataGenerator->createDuplicateTestSet(testPath, 50, 20000, 0.3); // 50 files, ~20KB each, 30% duplicates
    
    // Benchmark duplicate detection
    bool success = m_benchmark->benchmarkDuplicateDetection(testPath);
    QVERIFY2(success, "Duplicate detection benchmark failed");
    
    // Verify results
    auto stats = m_benchmark->calculateStatistics("duplicate_detection", "duplicate_detection");
    QVERIFY(stats.sampleCount > 0);
    
    auto results = m_benchmark->getResults("duplicate_detection");
    int duplicateGroups = 0;
    int totalDuplicates = 0;
    
    for (const auto& result : results) {
        if (result.metricName == "duplicate_groups_found") {
            duplicateGroups = static_cast<int>(result.value);
        } else if (result.metricName == "total_duplicates_found") {
            totalDuplicates = static_cast<int>(result.value);
        }
    }
    
    qDebug() << QString("Duplicate detection: %1, found %2 groups with %3 total duplicates")
                .arg(m_benchmark->formatDuration(stats.mean))
                .arg(duplicateGroups)
                .arg(totalDuplicates);
}

void ExamplePerformanceBenchmark::benchmarkUIResponsiveness() {
    qDebug() << "Running UI responsiveness benchmark...";
    
    // Create a test widget
    QWidget testWidget;
    testWidget.resize(800, 600);
    testWidget.show();
    
    // Wait for widget to be fully displayed
    QTest::qWaitForWindowExposed(&testWidget);
    
    // Benchmark UI responsiveness
    bool success = m_benchmark->benchmarkUIResponsiveness(&testWidget, 100);
    QVERIFY2(success, "UI responsiveness benchmark failed");
    
    // Verify results
    auto stats = m_benchmark->calculateStatistics("ui_responsiveness_100_ops", "ui_updates");
    QVERIFY(stats.sampleCount > 0);
    
    qDebug() << QString("UI responsiveness: %1, update rate: %2 ops/sec")
                .arg(m_benchmark->formatDuration(stats.mean))
                .arg(stats.mean > 0 ? QString::number(100000.0 / stats.mean, 'f', 2) : "N/A");
}

void ExamplePerformanceBenchmark::benchmarkWidgetRendering() {
    qDebug() << "Running widget rendering benchmark...";
    
    // Create a test widget with some content
    QWidget testWidget;
    testWidget.resize(1024, 768);
    testWidget.setStyleSheet("background-color: lightblue; border: 2px solid darkblue;");
    testWidget.show();
    
    QTest::qWaitForWindowExposed(&testWidget);
    
    // Benchmark widget rendering
    bool success = m_benchmark->benchmarkWidgetRendering(&testWidget, 60);
    QVERIFY2(success, "Widget rendering benchmark failed");
    
    // Verify results
    auto stats = m_benchmark->calculateStatistics("widget_rendering_60_frames", "widget_rendering");
    QVERIFY(stats.sampleCount > 0);
    
    qDebug() << QString("Widget rendering: %1 for 60 frames")
                .arg(m_benchmark->formatDuration(stats.mean));
}

void ExamplePerformanceBenchmark::benchmarkThemeSwitching() {
    qDebug() << "Running theme switching benchmark...";
    
    // Create a test widget
    QWidget testWidget;
    testWidget.resize(800, 600);
    testWidget.show();
    
    QTest::qWaitForWindowExposed(&testWidget);
    
    // Define test themes
    QStringList themes = {
        "Light Theme",
        "Dark Theme", 
        "Blue Theme",
        "Green Theme",
        "High Contrast Theme"
    };
    
    // Benchmark theme switching
    bool success = m_benchmark->benchmarkThemeSwitching(&testWidget, themes);
    QVERIFY2(success, "Theme switching benchmark failed");
    
    // Verify results
    auto stats = m_benchmark->calculateStatistics(QString("theme_switching_%1_themes").arg(themes.size()), "theme_switching");
    QVERIFY(stats.sampleCount > 0);
    
    qDebug() << QString("Theme switching: %1 for %2 themes")
                .arg(m_benchmark->formatDuration(stats.mean))
                .arg(themes.size());
}

void ExamplePerformanceBenchmark::benchmarkMemoryUsage() {
    qDebug() << "Running memory usage benchmark...";
    
    // Benchmark memory usage during file operations
    BENCHMARK_FUNCTION("memory_usage_test", [this]() {
        // Record initial memory usage
        qint64 initialMemory = m_benchmark->getCurrentResourceUsage()["memory_usage"].toLongLong();
        BENCHMARK_RECORD("initial_memory", initialMemory, "bytes");
        
        // Perform memory-intensive operations
        QList<QByteArray> dataBuffers;
        for (int i = 0; i < 100; ++i) {
            dataBuffers.append(QByteArray(10240, 'X')); // 10KB buffers
        }
        
        // Record peak memory usage
        qint64 peakMemory = m_benchmark->getCurrentResourceUsage()["memory_usage"].toLongLong();
        BENCHMARK_RECORD("peak_memory", peakMemory, "bytes");
        
        // Calculate memory increase
        qint64 memoryIncrease = peakMemory - initialMemory;
        BENCHMARK_RECORD("memory_increase", memoryIncrease, "bytes");
        
        // Clear buffers
        dataBuffers.clear();
        
        // Record final memory usage
        qint64 finalMemory = m_benchmark->getCurrentResourceUsage()["memory_usage"].toLongLong();
        BENCHMARK_RECORD("final_memory", finalMemory, "bytes");
    });
    
    // Verify memory measurements
    auto results = m_benchmark->getResults("memory_usage_test");
    QVERIFY(!results.isEmpty());
    
    qDebug() << "Memory usage benchmark completed with" << results.size() << "measurements";
}

void ExamplePerformanceBenchmark::benchmarkResourceMonitoring() {
    qDebug() << "Running resource monitoring benchmark...";
    
    // Start resource monitoring
    m_benchmark->startResourceMonitoring();
    
    // Perform some operations while monitoring
    QElapsedTimer timer;
    timer.start();
    
    while (timer.elapsed() < 2000) { // Monitor for 2 seconds
        // Simulate some work
        QByteArray data(1024, 'M');
        QCryptographicHash hash(QCryptographicHash::Md5);
        hash.addData(data);
        hash.result();
        
        QApplication::processEvents();
        QThread::msleep(50);
    }
    
    // Stop resource monitoring
    m_benchmark->stopResourceMonitoring();
    
    // Verify monitoring results
    auto monitoringResults = m_benchmark->getResourceMonitoringResults();
    QVERIFY(!monitoringResults.isEmpty());
    
    qDebug() << QString("Resource monitoring: collected %1 samples over %2")
                .arg(monitoringResults.size())
                .arg(m_benchmark->formatDuration(timer.elapsed()));
}

void ExamplePerformanceBenchmark::testBaselineCreation() {
    qDebug() << "Testing baseline creation...";
    
    // Run a simple benchmark to generate data
    BENCHMARK_FUNCTION("baseline_test", [this]() {
        QThread::msleep(100); // Simulate 100ms operation
        BENCHMARK_RECORD("test_metric", 100.0, "ms");
    });
    
    // Create baseline from the benchmark data
    bool success = m_benchmark->createBaseline("test_baseline", "baseline_test", "test_metric");
    QVERIFY2(success, "Failed to create performance baseline");
    
    // Verify baseline was created
    auto baseline = m_benchmark->getBaseline("test_baseline");
    QVERIFY(!baseline.name.isEmpty());
    QCOMPARE(baseline.name, QString("test_baseline"));
    QCOMPARE(baseline.benchmarkName, QString("baseline_test"));
    QCOMPARE(baseline.metricName, QString("test_metric"));
    QVERIFY(baseline.expectedValue > 0);
    
    qDebug() << QString("Baseline created: %1 = %2 %3")
                .arg(baseline.name)
                .arg(baseline.expectedValue)
                .arg("ms");
}

void ExamplePerformanceBenchmark::testPerformanceComparison() {
    qDebug() << "Testing performance comparison...";
    
    // Ensure we have a baseline
    testBaselineCreation();
    
    // Run the same benchmark again
    BENCHMARK_FUNCTION("baseline_test", [this]() {
        QThread::msleep(105); // Slightly slower (5% regression)
        BENCHMARK_RECORD("test_metric", 105.0, "ms");
    });
    
    // Compare with baseline
    auto comparison = m_benchmark->compareWithBaseline("test_baseline", "baseline_test", "test_metric");
    QVERIFY(!comparison.status.isEmpty());
    QVERIFY(comparison.currentValue > 0);
    QVERIFY(comparison.baselineValue > 0);
    
    qDebug() << QString("Performance comparison: %1 (deviation: %2%)")
                .arg(comparison.status)
                .arg(comparison.deviationPercent, 0, 'f', 2);
}

void ExamplePerformanceBenchmark::testRegressionDetection() {
    qDebug() << "Testing regression detection...";
    
    // Create a baseline with tight tolerance
    BENCHMARK_FUNCTION("regression_test", [this]() {
        QThread::msleep(50);
        BENCHMARK_RECORD("regression_metric", 50.0, "ms");
    });
    
    m_benchmark->createBaseline("regression_baseline", "regression_test", "regression_metric");
    
    // Update baseline tolerance for testing
    auto baseline = m_benchmark->getBaseline("regression_baseline");
    baseline.tolerancePercent = 5.0; // 5% tolerance
    
    // Run benchmark with significant regression (20% slower)
    BENCHMARK_FUNCTION("regression_test", [this]() {
        QThread::msleep(60); // 20% slower
        BENCHMARK_RECORD("regression_metric", 60.0, "ms");
    });
    
    // Check for regressions
    bool hasRegression = m_benchmark->detectPerformanceRegressions(10.0); // 10% threshold
    QVERIFY2(hasRegression, "Failed to detect performance regression");
    
    qDebug() << "Regression detection test passed";
}

void ExamplePerformanceBenchmark::testReportGeneration() {
    qDebug() << "Testing report generation...";
    
    // Generate comprehensive report
    QJsonObject report = m_benchmark->generateReport();
    QVERIFY(!report.isEmpty());
    QVERIFY(report.contains("timestamp"));
    QVERIFY(report.contains("platform"));
    QVERIFY(report.contains("results"));
    QVERIFY(report.contains("statistics"));
    
    // Export report to file
    QString reportPath = QDir(m_testPath).absoluteFilePath("test_report.json");
    bool success = m_benchmark->exportResults(reportPath, "json");
    QVERIFY2(success, "Failed to export performance report");
    QVERIFY(QFile::exists(reportPath));
    
    // Verify report file content
    QFile reportFile(reportPath);
    QVERIFY(reportFile.open(QIODevice::ReadOnly));
    QJsonDocument doc = QJsonDocument::fromJson(reportFile.readAll());
    QVERIFY(!doc.isNull());
    
    qDebug() << QString("Report generated: %1 (%2)")
                .arg(reportPath)
                .arg(m_benchmark->formatBytes(reportFile.size()));
}

void ExamplePerformanceBenchmark::testBaselineExportImport() {
    qDebug() << "Testing baseline export/import...";
    
    // Ensure we have baselines
    testBaselineCreation();
    
    // Export baselines
    QString baselinePath = QDir(m_testPath).absoluteFilePath("test_baselines.json");
    bool success = m_benchmark->exportBaselines(baselinePath);
    QVERIFY2(success, "Failed to export baselines");
    QVERIFY(QFile::exists(baselinePath));
    
    // Clear current baselines
    m_benchmark->clearBaselines();
    QVERIFY(m_benchmark->getBaselines().isEmpty());
    
    // Import baselines
    success = m_benchmark->importBaselines(baselinePath);
    QVERIFY2(success, "Failed to import baselines");
    
    // Verify baselines were imported
    auto baselines = m_benchmark->getBaselines();
    QVERIFY(!baselines.isEmpty());
    
    qDebug() << QString("Baseline export/import: %1 baselines processed")
                .arg(baselines.size());
}

QTEST_MAIN(ExamplePerformanceBenchmark)
#include "example_performance_benchmark.moc"