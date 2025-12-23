#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QElapsedTimer>
#include <QDateTime>
#include <QMap>
#include <QVariant>
#include <QJsonObject>
#include <QJsonDocument>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <functional>

class QWidget;
class TestDataGenerator;

/**
 * @brief Comprehensive performance benchmarking framework for CloneClean testing
 * 
 * Provides advanced performance measurement, analysis, and reporting capabilities
 * including CPU usage, memory consumption, I/O operations, and UI responsiveness.
 */
class PerformanceBenchmark : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Performance metric types
     */
    enum class MetricType {
        ExecutionTime,      ///< Execution time in milliseconds
        MemoryUsage,        ///< Memory usage in bytes
        CpuUsage,          ///< CPU usage percentage
        DiskIO,            ///< Disk I/O operations and throughput
        NetworkIO,         ///< Network I/O operations and throughput
        UIResponsiveness,  ///< UI response time and frame rate
        ThroughputRate,    ///< Operations per second
        Latency,           ///< Response latency in milliseconds
        Custom             ///< Custom user-defined metrics
    };

    /**
     * @brief Benchmark configuration
     */
    struct BenchmarkConfig {
        QString name;                           ///< Benchmark name
        int iterations = 1;                     ///< Number of iterations to run
        int warmupIterations = 0;              ///< Warmup iterations (not measured)
        qint64 timeoutMs = 60000;              ///< Timeout per iteration in milliseconds
        bool measureMemory = true;              ///< Measure memory usage
        bool measureCpu = true;                 ///< Measure CPU usage
        bool measureDiskIO = false;             ///< Measure disk I/O
        bool measureNetworkIO = false;          ///< Measure network I/O
        bool measureUIResponsiveness = false;   ///< Measure UI responsiveness
        int samplingIntervalMs = 100;          ///< Sampling interval for continuous metrics
        QMap<QString, QVariant> customParams;  ///< Custom benchmark parameters
        QString description;                    ///< Benchmark description
        QStringList tags;                      ///< Tags for categorization
    };

    /**
     * @brief Performance measurement result
     */
    struct PerformanceResult {
        QString benchmarkName;                  ///< Name of the benchmark
        MetricType metricType;                  ///< Type of metric measured
        QString metricName;                     ///< Name of the specific metric
        double value = 0.0;                    ///< Measured value
        QString unit;                          ///< Unit of measurement
        QDateTime timestamp;                    ///< When measurement was taken
        int iteration = 0;                      ///< Iteration number
        QMap<QString, QVariant> metadata;      ///< Additional metadata
        QString description;                    ///< Result description
    };

    /**
     * @brief Statistical analysis of benchmark results
     */
    struct BenchmarkStatistics {
        QString benchmarkName;                  ///< Benchmark name
        QString metricName;                     ///< Metric name
        int sampleCount = 0;                   ///< Number of samples
        double mean = 0.0;                     ///< Mean value
        double median = 0.0;                   ///< Median value
        double standardDeviation = 0.0;        ///< Standard deviation
        double minimum = 0.0;                  ///< Minimum value
        double maximum = 0.0;                  ///< Maximum value
        double percentile95 = 0.0;             ///< 95th percentile
        double percentile99 = 0.0;             ///< 99th percentile
        QString unit;                          ///< Unit of measurement
        QList<double> rawValues;               ///< Raw measurement values
    };

    /**
     * @brief Performance baseline for comparison
     */
    struct PerformanceBaseline {
        QString name;                          ///< Baseline name
        QString benchmarkName;                 ///< Associated benchmark
        QString metricName;                    ///< Metric name
        double expectedValue = 0.0;           ///< Expected performance value
        double tolerancePercent = 10.0;       ///< Acceptable deviation percentage
        double warningThreshold = 5.0;        ///< Warning threshold percentage
        QDateTime created;                     ///< When baseline was created
        QString platform;                      ///< Platform information
        QString version;                       ///< Software version
        QMap<QString, QVariant> environment;   ///< Environment details
    };

    /**
     * @brief Performance comparison result
     */
    struct ComparisonResult {
        QString benchmarkName;                 ///< Benchmark name
        QString metricName;                    ///< Metric name
        double currentValue = 0.0;            ///< Current measured value
        double baselineValue = 0.0;           ///< Baseline value
        double deviationPercent = 0.0;        ///< Deviation from baseline
        bool withinTolerance = false;         ///< Whether within acceptable range
        bool isRegression = false;            ///< Whether this is a performance regression
        bool isImprovement = false;           ///< Whether this is a performance improvement
        QString status;                       ///< Status description
        QString recommendation;               ///< Performance recommendation
    };

    explicit PerformanceBenchmark(QObject* parent = nullptr);
    ~PerformanceBenchmark();

    // Benchmark configuration and execution
    void setBenchmarkConfig(const BenchmarkConfig& config);
    BenchmarkConfig getBenchmarkConfig() const;
    bool runBenchmark(const QString& name, std::function<void()> benchmarkFunction);
    bool runBenchmark(const QString& name, std::function<void(int)> benchmarkFunction); // With iteration parameter
    QList<PerformanceResult> getResults(const QString& benchmarkName = "") const;

    // Performance measurement methods
    void startMeasurement(const QString& measurementName);
    void stopMeasurement(const QString& measurementName);
    void recordMetric(const QString& metricName, double value, const QString& unit, MetricType type = MetricType::Custom);
    void recordExecutionTime(const QString& operationName, qint64 timeMs);
    void recordMemoryUsage(const QString& operationName, qint64 memoryBytes);
    void recordThroughput(const QString& operationName, double operationsPerSecond);

    // System resource monitoring
    void startResourceMonitoring();
    void stopResourceMonitoring();
    QMap<QString, QVariant> getCurrentResourceUsage() const;
    QList<PerformanceResult> getResourceMonitoringResults() const;

    // File operation benchmarking
    bool benchmarkFileOperations(const QString& testDirectory, int fileCount, qint64 fileSize);
    bool benchmarkDirectoryScanning(const QString& directory, bool recursive = true);
    bool benchmarkHashCalculation(const QStringList& filePaths, const QString& algorithm = "MD5");
    bool benchmarkDuplicateDetection(const QString& testDirectory);

    // UI performance benchmarking
    bool benchmarkUIResponsiveness(QWidget* widget, int operationCount = 100);
    bool benchmarkWidgetRendering(QWidget* widget, int frameCount = 60);
    bool benchmarkThemeSwitching(QWidget* widget, const QStringList& themes);

    // Statistical analysis
    BenchmarkStatistics calculateStatistics(const QString& benchmarkName, const QString& metricName) const;
    QList<BenchmarkStatistics> calculateAllStatistics() const;
    QMap<QString, BenchmarkStatistics> groupStatisticsByBenchmark() const;

    // Baseline management
    bool createBaseline(const QString& name, const QString& benchmarkName, const QString& metricName);
    bool updateBaseline(const QString& name, double newValue);
    bool deleteBaseline(const QString& name);
    QList<PerformanceBaseline> getBaselines() const;
    PerformanceBaseline getBaseline(const QString& name) const;

    // Performance comparison
    ComparisonResult compareWithBaseline(const QString& baselineName, const QString& benchmarkName, const QString& metricName) const;
    QList<ComparisonResult> compareAllWithBaselines() const;
    bool detectPerformanceRegressions(double regressionThreshold = 10.0) const;

    // Reporting and export
    QJsonObject generateReport() const;
    QJsonObject generateComparisonReport(const QList<ComparisonResult>& comparisons) const;
    bool exportResults(const QString& filePath, const QString& format = "json") const;
    bool exportBaselines(const QString& filePath) const;
    bool importBaselines(const QString& filePath);

    // Utility methods
    void clearResults();
    void clearBaselines();
    QString formatDuration(qint64 milliseconds) const;
    QString formatBytes(qint64 bytes) const;
    QString formatRate(double rate, const QString& unit) const;

    // Platform and environment information
    QMap<QString, QVariant> getSystemInfo() const;
    QMap<QString, QVariant> getEnvironmentInfo() const;
    QString getPlatformIdentifier() const;

signals:
    void benchmarkStarted(const QString& name);
    void benchmarkCompleted(const QString& name, const QList<PerformanceResult>& results);
    void measurementRecorded(const PerformanceResult& result);
    void baselineCreated(const QString& name);
    void baselineUpdated(const QString& name);
    void performanceRegression(const ComparisonResult& regression);
    void performanceImprovement(const ComparisonResult& improvement);
    void resourceMonitoringUpdate(const QMap<QString, QVariant>& resources);

private slots:
    void onResourceMonitoringTimer();

private:
    BenchmarkConfig m_config;
    QList<PerformanceResult> m_results;
    QList<PerformanceBaseline> m_baselines;
    QMap<QString, QElapsedTimer> m_activeTimers;
    QMap<QString, QVariant> m_activeMeasurements;
    
    // Resource monitoring
    QTimer* m_resourceTimer;
    bool m_resourceMonitoringActive;
    QList<PerformanceResult> m_resourceResults;
    
    // Thread safety
    mutable QMutex m_resultsMutex;
    mutable QMutex m_baselinesMutex;
    
    // Internal helper methods
    void recordResult(const PerformanceResult& result);
    PerformanceResult createResult(const QString& benchmarkName, MetricType type, 
                                 const QString& metricName, double value, const QString& unit) const;
    
    // System monitoring helpers
    qint64 getCurrentMemoryUsage() const;
    double getCurrentCpuUsage() const;
    QMap<QString, qint64> getDiskIOStats() const;
    QMap<QString, qint64> getNetworkIOStats() const;
    
    // File operation helpers
    qint64 measureFileCreation(const QString& directory, int fileCount, qint64 fileSize);
    qint64 measureFileReading(const QStringList& filePaths);
    qint64 measureFileWriting(const QStringList& filePaths, const QByteArray& data);
    qint64 measureDirectoryTraversal(const QString& directory, bool recursive);
    
    // UI performance helpers
    qint64 measureWidgetUpdate(QWidget* widget, int updateCount);
    qint64 measureWidgetResize(QWidget* widget, int resizeCount);
    double measureFrameRate(QWidget* widget, int durationMs);
    
    // Statistical calculation helpers
    double calculateMean(const QList<double>& values) const;
    double calculateMedian(QList<double> values) const;
    double calculateStandardDeviation(const QList<double>& values, double mean) const;
    double calculatePercentile(QList<double> values, double percentile) const;
    
    // Baseline helpers
    void saveBaselines() const;
    void loadBaselines();
    QString getBaselinesFilePath() const;
    
    // Platform detection helpers
    QString detectPlatform() const;
    QString detectCpuInfo() const;
    qint64 detectTotalMemory() const;
};

/**
 * @brief Performance benchmark runner for automated testing
 */
class BenchmarkRunner : public QObject {
    Q_OBJECT

public:
    explicit BenchmarkRunner(QObject* parent = nullptr);

    // Benchmark suite management
    void addBenchmark(const QString& name, std::function<void()> benchmarkFunction, 
                     const PerformanceBenchmark::BenchmarkConfig& config = {});
    void removeBenchmark(const QString& name);
    QStringList getBenchmarkNames() const;

    // Execution control
    bool runAllBenchmarks();
    bool runBenchmark(const QString& name);
    bool runBenchmarkSuite(const QStringList& names);

    // Results and reporting
    QList<PerformanceBenchmark::PerformanceResult> getAllResults() const;
    QMap<QString, PerformanceBenchmark::BenchmarkStatistics> getAllStatistics() const;
    bool generateSuiteReport(const QString& outputPath) const;

signals:
    void suiteStarted(int benchmarkCount);
    void suiteCompleted(int totalBenchmarks, int successfulBenchmarks);
    void benchmarkProgress(const QString& name, int current, int total);

private:
    struct BenchmarkInfo {
        QString name;
        std::function<void()> function;
        PerformanceBenchmark::BenchmarkConfig config;
    };
    
    QList<BenchmarkInfo> m_benchmarks;
    PerformanceBenchmark* m_benchmark;
};

/**
 * @brief Convenience macros for performance benchmarking
 */
#define BENCHMARK_START(name) \
    performanceBenchmark.startMeasurement(name)

#define BENCHMARK_STOP(name) \
    performanceBenchmark.stopMeasurement(name)

#define BENCHMARK_RECORD(name, value, unit) \
    performanceBenchmark.recordMetric(name, value, unit)

#define BENCHMARK_FUNCTION(benchmarkName, function) \
    do { \
        if (!performanceBenchmark.runBenchmark(benchmarkName, function)) { \
            QFAIL(QString("Benchmark failed: %1").arg(benchmarkName).toUtf8().constData()); \
        } \
    } while(0)

#define BENCHMARK_COMPARE_BASELINE(baselineName, benchmarkName, metricName) \
    do { \
        auto result = performanceBenchmark.compareWithBaseline(baselineName, benchmarkName, metricName); \
        if (result.isRegression) { \
            QFAIL(QString("Performance regression detected: %1 (deviation: %2%)") \
                  .arg(result.benchmarkName).arg(result.deviationPercent).toUtf8().constData()); \
        } \
    } while(0)