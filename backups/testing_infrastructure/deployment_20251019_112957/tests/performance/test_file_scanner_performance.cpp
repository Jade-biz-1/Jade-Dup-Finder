#include "performance_test_framework.h"
#include "file_scanner.h"
#include <QtTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QDebug>

using namespace PerformanceTest;

/**
 * @brief Performance benchmarks for FileScanner component
 * 
 * Tests:
 * - Scan rate with various file counts
 * - Memory usage with 100,000+ files
 * - Progress update latency
 * - Pattern matching overhead
 * - Streaming mode vs normal mode
 * - Metadata caching performance
 */
class FileScannerPerformanceTest : public PerformanceTestCase {
    Q_OBJECT

public:
    FileScannerPerformanceTest() : PerformanceTestCase() {}

private slots:
    void initTestCase();
    void cleanupTestCase();
    void runPerformanceTests() override;

private:
    // Individual benchmark tests
    void benchmarkScanRate_SmallFiles();
    void benchmarkScanRate_LargeFiles();
    void benchmarkScanRate_MixedSizes();
    void benchmarkMemoryUsage_100kFiles();
    void benchmarkProgressUpdateLatency();
    void benchmarkPatternMatchingOverhead();
    void benchmarkStreamingMode();
    void benchmarkMetadataCache();
    void benchmarkProgressBatching();
    
    // Helper methods
    bool createTestFiles(int count, qint64 avgSize);
    void cleanupTestFiles();
    qint64 measureMemoryUsage();
    
    QTemporaryDir* m_testDir;
    QString m_testPath;
    int m_testFileCount;
};

void FileScannerPerformanceTest::initTestCase()
{
    qDebug() << "=== FileScanner Performance Tests ===";
    SystemProfiler::logSystemInfo();
    
    m_testDir = new QTemporaryDir();
    QVERIFY(m_testDir->isValid());
    m_testPath = m_testDir->path();
    m_testFileCount = 0;
    
    qDebug() << "Test directory:" << m_testPath;
}

void FileScannerPerformanceTest::cleanupTestCase()
{
    cleanupTestFiles();
    delete m_testDir;
    
    qDebug() << "\n=== Performance Test Summary ===";
    benchmarkRunner()->logResults();
}

void FileScannerPerformanceTest::runPerformanceTests()
{
    // Configure benchmark runner
    BenchmarkRunner::BenchmarkConfig config;
    config.warmupRuns = 2;
    config.measurementRuns = 5;
    config.enableResourceMonitoring = true;
    config.saveResults = true;
    config.resultsPath = "file_scanner_performance_results.json";
    benchmarkRunner()->setBenchmarkConfig(config);
    
    // Run all benchmarks
    qDebug() << "\n--- Running FileScanner Performance Benchmarks ---\n";
    
    benchmarkScanRate_SmallFiles();
    benchmarkScanRate_LargeFiles();
    benchmarkScanRate_MixedSizes();
    benchmarkMemoryUsage_100kFiles();
    benchmarkProgressUpdateLatency();
    benchmarkPatternMatchingOverhead();
    benchmarkStreamingMode();
    benchmarkMetadataCache();
    benchmarkProgressBatching();
}

void FileScannerPerformanceTest::benchmarkScanRate_SmallFiles()
{
    qDebug() << "\n[Benchmark] Scan Rate - Small Files (1KB each)";
    
    // Create 10,000 small files
    const int fileCount = 10000;
    const qint64 fileSize = 1024; // 1KB
    
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, fileSize));
    m_testPath = dataGenerator()->getTestDataPath();
    
    benchmarkRunner()->registerBenchmark(
        "ScanRate_SmallFiles_10k",
        "scan_rate",
        [this]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            
            scanner.startScan(options);
            
            // Wait for completion (max 60 seconds)
            bool completed = completedSpy.wait(60000);
            if (!completed) return false;
            
            int filesFound = scanner.getTotalFilesFound();
            qDebug() << "  Files found:" << filesFound;
            
            return filesFound >= fileCount * 0.95; // Allow 5% tolerance
        }
    );
    
    benchmarkRunner()->runBenchmark("ScanRate_SmallFiles_10k");
    
    // Calculate and verify scan rate
    auto result = benchmarkRunner()->getResult("ScanRate_SmallFiles_10k");
    double filesPerSecond = (fileCount / result.meanTime) * 1000.0;
    qDebug() << "  Scan rate:" << filesPerSecond << "files/second";
    
    // Target: >= 1,000 files/minute = 16.67 files/second minimum
    QVERIFY2(filesPerSecond >= 16.67, 
             QString("Scan rate too slow: %1 files/sec (target: >= 16.67)").arg(filesPerSecond).toLatin1());
}

void FileScannerPerformanceTest::benchmarkScanRate_LargeFiles()
{
    qDebug() << "\n[Benchmark] Scan Rate - Large Files (10MB each)";
    
    const int fileCount = 100;
    const qint64 fileSize = 10 * 1024 * 1024; // 10MB
    
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateLargeFilesScenario(fileCount, fileSize));
    m_testPath = dataGenerator()->getTestDataPath();
    
    benchmarkRunner()->registerBenchmark(
        "ScanRate_LargeFiles_100",
        "scan_rate",
        [this]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            if (!completed) return false;
            
            return scanner.getTotalFilesFound() >= fileCount * 0.95;
        }
    );
    
    benchmarkRunner()->runBenchmark("ScanRate_LargeFiles_100");
}

void FileScannerPerformanceTest::benchmarkScanRate_MixedSizes()
{
    qDebug() << "\n[Benchmark] Scan Rate - Mixed File Sizes";
    
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateMixedSizesScenario(5000));
    m_testPath = dataGenerator()->getTestDataPath();
    
    benchmarkRunner()->registerBenchmark(
        "ScanRate_MixedSizes_5k",
        "scan_rate",
        [this]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            if (!completed) return false;
            
            return scanner.getTotalFilesFound() >= 4500; // 90% tolerance
        }
    );
    
    benchmarkRunner()->runBenchmark("ScanRate_MixedSizes_5k");
}

void FileScannerPerformanceTest::benchmarkMemoryUsage_100kFiles()
{
    qDebug() << "\n[Benchmark] Memory Usage - 100,000 Files";
    
    // Note: This test may take a while and requires significant disk space
    // Adjust file count based on available resources
    const int fileCount = 100000;
    const qint64 fileSize = 1024; // 1KB each
    
    cleanupTestFiles();
    
    qDebug() << "  Generating" << fileCount << "test files (this may take a while)...";
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, fileSize));
    m_testPath = dataGenerator()->getTestDataPath();
    
    qint64 initialMemory = measureMemoryUsage();
    qint64 peakMemory = initialMemory;
    
    benchmarkRunner()->registerBenchmark(
        "MemoryUsage_100kFiles",
        "memory",
        [this, &peakMemory]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.estimatedFileCount = 100000; // Use capacity reservation
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            QSignalSpy progressSpy(&scanner, &FileScanner::scanProgress);
            
            scanner.startScan(options);
            
            // Monitor memory during scan
            while (scanner.isScanning()) {
                QCoreApplication::processEvents();
                qint64 currentMemory = measureMemoryUsage();
                if (currentMemory > peakMemory) {
                    peakMemory = currentMemory;
                }
                QThread::msleep(100);
            }
            
            bool completed = completedSpy.wait(300000); // 5 minutes max
            if (!completed) return false;
            
            return scanner.getTotalFilesFound() >= 95000; // 95% tolerance
        }
    );
    
    benchmarkRunner()->runBenchmark("MemoryUsage_100kFiles");
    
    qint64 memoryUsed = (peakMemory - initialMemory) / (1024 * 1024); // Convert to MB
    qDebug() << "  Peak memory usage:" << memoryUsed << "MB";
    
    // Target: < 100MB for 100k files
    QVERIFY2(memoryUsed < 100, 
             QString("Memory usage too high: %1 MB (target: < 100 MB)").arg(memoryUsed).toLatin1());
}

void FileScannerPerformanceTest::benchmarkProgressUpdateLatency()
{
    qDebug() << "\n[Benchmark] Progress Update Latency";
    
    const int fileCount = 5000;
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, 1024));
    m_testPath = dataGenerator()->getTestDataPath();
    
    benchmarkRunner()->registerBenchmark(
        "ProgressUpdateLatency",
        "latency",
        [this]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.progressBatchSize = 100; // Update every 100 files
            
            QElapsedTimer timer;
            QVector<qint64> updateLatencies;
            qint64 lastUpdate = 0;
            
            connect(&scanner, &FileScanner::scanProgress, 
                    [&timer, &updateLatencies, &lastUpdate](int, int, const QString&) {
                if (lastUpdate > 0) {
                    qint64 latency = timer.elapsed() - lastUpdate;
                    updateLatencies.append(latency);
                }
                lastUpdate = timer.elapsed();
            });
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            
            timer.start();
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            if (!completed) return false;
            
            // Calculate average latency
            if (!updateLatencies.isEmpty()) {
                qint64 totalLatency = 0;
                for (qint64 latency : updateLatencies) {
                    totalLatency += latency;
                }
                double avgLatency = static_cast<double>(totalLatency) / static_cast<double>(updateLatencies.size());
                qDebug() << "  Average progress update latency:" << avgLatency << "ms";
                
                // Target: < 100ms average latency
                if (avgLatency >= 100.0) {
                    qWarning() << "Progress latency too high:" << avgLatency << "ms";
                    return false;
                }
            }
            
            return true;
        }
    );
    
    benchmarkRunner()->runBenchmark("ProgressUpdateLatency");
}

void FileScannerPerformanceTest::benchmarkPatternMatchingOverhead()
{
    qDebug() << "\n[Benchmark] Pattern Matching Overhead";
    
    const int fileCount = 5000;
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, 1024));
    m_testPath = dataGenerator()->getTestDataPath();
    
    // Benchmark without pattern matching
    double timeWithoutPatterns = 0.0;
    benchmarkRunner()->registerBenchmark(
        "Scan_NoPatterns",
        "pattern_overhead",
        [this, &timeWithoutPatterns]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            
            QElapsedTimer timer;
            timer.start();
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            
            timeWithoutPatterns = timer.elapsed();
            
            return completed;
        }
    );
    benchmarkRunner()->runBenchmark("Scan_NoPatterns");
    
    // Benchmark with pattern matching
    double timeWithPatterns = 0.0;
    benchmarkRunner()->registerBenchmark(
        "Scan_WithPatterns",
        "pattern_overhead",
        [this, &timeWithPatterns]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.includePatterns << "*.txt" << "*.dat" << "*.bin";
            options.excludePatterns << "*.tmp" << "*.log";
            
            QElapsedTimer timer;
            timer.start();
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            
            timeWithPatterns = timer.elapsed();
            
            return completed;
        }
    );
    benchmarkRunner()->runBenchmark("Scan_WithPatterns");
    
    // Calculate overhead
    double overhead = ((timeWithPatterns - timeWithoutPatterns) / timeWithoutPatterns) * 100.0;
    qDebug() << "  Pattern matching overhead:" << overhead << "%";
    
    // Target: < 5% overhead
    QVERIFY2(overhead < 5.0,
             QString("Pattern matching overhead too high: %1%").arg(overhead).toLatin1());
}

void FileScannerPerformanceTest::benchmarkStreamingMode()
{
    qDebug() << "\n[Benchmark] Streaming Mode vs Normal Mode";
    
    const int fileCount = 10000;
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, 1024));
    m_testPath = dataGenerator()->getTestDataPath();
    
    // Normal mode
    qint64 normalMemory = 0;
    benchmarkRunner()->registerBenchmark(
        "Scan_NormalMode",
        "streaming",
        [this, &normalMemory]() -> bool {
            qint64 initialMem = measureMemoryUsage();
            
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.streamingMode = false;
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            
            normalMemory = measureMemoryUsage() - initialMem;
            qDebug() << "  Normal mode memory:" << (normalMemory / 1024 / 1024) << "MB";
            
            return completed;
        }
    );
    benchmarkRunner()->runBenchmark("Scan_NormalMode");
    
    // Streaming mode
    qint64 streamingMemory = 0;
    benchmarkRunner()->registerBenchmark(
        "Scan_StreamingMode",
        "streaming",
        [this, &streamingMemory]() -> bool {
            qint64 initialMem = measureMemoryUsage();
            
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.streamingMode = true;
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            
            streamingMemory = measureMemoryUsage() - initialMem;
            qDebug() << "  Streaming mode memory:" << (streamingMemory / 1024 / 1024) << "MB";
            
            return completed;
        }
    );
    benchmarkRunner()->runBenchmark("Scan_StreamingMode");
    
    // Verify streaming mode uses less memory
    qDebug() << "  Memory savings:" << ((normalMemory - streamingMemory) / 1024 / 1024) << "MB";
    QVERIFY2(streamingMemory < normalMemory,
             "Streaming mode should use less memory than normal mode");
}

void FileScannerPerformanceTest::benchmarkMetadataCache()
{
    qDebug() << "\n[Benchmark] Metadata Caching Performance";
    
    const int fileCount = 5000;
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, 1024));
    m_testPath = dataGenerator()->getTestDataPath();
    
    // First scan without cache
    double firstScanTime = 0.0;
    benchmarkRunner()->registerBenchmark(
        "Scan_FirstRun_NoCache",
        "caching",
        [this, &firstScanTime]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.enableMetadataCache = false;
            
            QElapsedTimer timer;
            timer.start();
            
            QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            bool completed = completedSpy.wait(60000);
            
            firstScanTime = timer.elapsed();
            
            return completed;
        }
    );
    benchmarkRunner()->runBenchmark("Scan_FirstRun_NoCache");
    
    // Second scan with cache enabled
    double cachedScanTime = 0.0;
    benchmarkRunner()->registerBenchmark(
        "Scan_SecondRun_WithCache",
        "caching",
        [this, &cachedScanTime]() -> bool {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath;
            options.minimumFileSize = 0;
            options.enableMetadataCache = true;
            
            // First scan to populate cache
            QSignalSpy completedSpy1(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            completedSpy1.wait(60000);
            
            // Second scan using cache
            QElapsedTimer timer;
            timer.start();
            
            QSignalSpy completedSpy2(&scanner, &FileScanner::scanCompleted);
            scanner.startScan(options);
            bool completed = completedSpy2.wait(60000);
            
            cachedScanTime = timer.elapsed();
            
            return completed;
        }
    );
    benchmarkRunner()->runBenchmark("Scan_SecondRun_WithCache");
    
    double improvement = ((firstScanTime - cachedScanTime) / firstScanTime) * 100.0;
    qDebug() << "  Cache performance improvement:" << improvement << "%";
}

void FileScannerPerformanceTest::benchmarkProgressBatching()
{
    qDebug() << "\n[Benchmark] Progress Batching Impact";
    
    const int fileCount = 10000;
    cleanupTestFiles();
    QVERIFY(dataGenerator()->generateSmallFilesScenario(fileCount, 1024));
    m_testPath = dataGenerator()->getTestDataPath();
    
    // Test different batch sizes
    QVector<int> batchSizes = {1, 10, 100, 1000};
    
    for (int batchSize : batchSizes) {
        QString testName = QString("ProgressBatch_%1").arg(batchSize);
        
        benchmarkRunner()->registerBenchmark(
            testName,
            "progress_batching",
            [this, batchSize]() -> bool {
                FileScanner scanner;
                FileScanner::ScanOptions options;
                options.targetPaths << m_testPath;
                options.minimumFileSize = 0;
                options.progressBatchSize = batchSize;
                
                int progressCount = 0;
                connect(&scanner, &FileScanner::scanProgress,
                        [&progressCount](int, int, const QString&) {
                    progressCount++;
                });
                
                QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
                scanner.startScan(options);
                bool completed = completedSpy.wait(60000);
                
                qDebug() << "    Batch size" << batchSize << "- Progress signals:" << progressCount;
                
                return completed;
            }
        );
        
        benchmarkRunner()->runBenchmark(testName);
    }
}

bool FileScannerPerformanceTest::createTestFiles(int count, qint64 avgSize)
{
    return dataGenerator()->generateSmallFilesScenario(count, avgSize);
}

void FileScannerPerformanceTest::cleanupTestFiles()
{
    if (dataGenerator()) {
        dataGenerator()->cleanupTestData();
    }
}

qint64 FileScannerPerformanceTest::measureMemoryUsage()
{
    // Platform-specific memory measurement
    // This is a simplified version - actual implementation would use platform APIs
    
#ifdef Q_OS_LINUX
    QFile file("/proc/self/status");
    if (file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        while (stream.readLineInto(&line)) {
            if (line.startsWith("VmRSS:")) {
                QStringList parts = line.split(QRegularExpression("\\s+"));
                if (parts.size() >= 2) {
                    return parts[1].toLongLong() * 1024; // Convert KB to bytes
                }
            }
        }
    }
#endif
    
    // Fallback: return 0 if platform-specific measurement not available
    return 0;
}

QTEST_MAIN(FileScannerPerformanceTest)
#include "test_file_scanner_performance.moc"
