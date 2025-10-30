#pragma once

#include <QObject>
#include <QString>
#include <QList>
#include <QElapsedTimer>
#include <QMap>

/**
 * @brief Performance benchmarking framework for DupFinder
 * 
 * Provides comprehensive performance testing for:
 * - Hash calculation
 * - File scanning
 * - Duplicate detection
 * - Memory usage
 * - Threading efficiency
 */
class PerformanceBenchmark : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Benchmark result for a single test
     */
    struct BenchmarkResult {
        QString testName;
        QString category;
        qint64 durationMs = 0;
        double throughputMBps = 0.0;
        int filesProcessed = 0;
        qint64 bytesProcessed = 0;
        double filesPerSecond = 0.0;
        qint64 peakMemoryMB = 0;
        QMap<QString, QVariant> metadata;
        bool success = true;
        QString errorMessage;
    };

    /**
     * @brief Benchmark configuration
     */
    struct BenchmarkConfig {
        QString outputDirectory = "benchmark_results";
        bool generateReport = true;
        bool generateCsv = true;
        bool generateJson = true;
        int warmupIterations = 1;
        int benchmarkIterations = 3;
        bool enableMemoryProfiling = true;
        bool enableThreadProfiling = true;
    };

    explicit PerformanceBenchmark(QObject* parent = nullptr);
    ~PerformanceBenchmark() = default;

    /**
     * @brief Run all benchmarks
     */
    void runAllBenchmarks();

    /**
     * @brief Run hash calculation benchmarks
     */
    void runHashBenchmarks();

    /**
     * @brief Run file scanning benchmarks
     */
    void runScanBenchmarks();

    /**
     * @brief Run duplicate detection benchmarks
     */
    void runDetectionBenchmarks();

    /**
     * @brief Set benchmark configuration
     */
    void setConfiguration(const BenchmarkConfig& config);

    /**
     * @brief Get benchmark results
     */
    QList<BenchmarkResult> getResults() const;

    /**
     * @brief Generate reports
     */
    void generateReports();

signals:
    void benchmarkStarted(const QString& testName);
    void benchmarkCompleted(const BenchmarkResult& result);
    void allBenchmarksCompleted();
    void progressUpdated(int current, int total);

private:
    // Hash calculation benchmarks
    BenchmarkResult benchmarkSmallFiles();           // < 1MB
    BenchmarkResult benchmarkMediumFiles();          // 1-100MB
    BenchmarkResult benchmarkLargeFiles();           // 100MB-1GB
    BenchmarkResult benchmarkMassiveFiles();         // > 1GB
    BenchmarkResult benchmarkManySmallFiles();       // 10,000+ files
    
    // Threading benchmarks
    BenchmarkResult benchmarkSequentialProcessing();
    BenchmarkResult benchmarkParallelProcessing();
    BenchmarkResult benchmarkThreadScaling();        // 1, 2, 4, 8 threads
    
    // Memory benchmarks
    BenchmarkResult benchmarkMemoryUsage();
    BenchmarkResult benchmarkCacheEfficiency();
    
    // Real-world scenarios
    BenchmarkResult benchmarkPhotoLibrary();         // Typical photo collection
    BenchmarkResult benchmarkDownloadsFolder();      // Mixed file types
    BenchmarkResult benchmarkCodeRepository();       // Many small text files
    
    // Helper functions
    void createTestFiles(const QString& directory, int count, qint64 sizeBytes);
    void cleanupTestFiles(const QString& directory);
    qint64 measurePeakMemory();
    double calculateThroughput(qint64 bytes, qint64 durationMs);
    void generateHtmlReport(const QString& outputPath);
    void generateCsvReport(const QString& outputPath);
    void generateJsonReport(const QString& outputPath);
    
    BenchmarkConfig m_config;
    QList<BenchmarkResult> m_results;
    QString m_testDataDirectory;
};
