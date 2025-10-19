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
#include <QWaitCondition>
#include <functional>

/**
 * @brief Comprehensive performance benchmarking and measurement system
 * 
 * Provides advanced performance testing capabilities including execution time measurement,
 * memory usage monitoring, CPU utilization tracking, and performance baseline management.
 */
class PerformanceBenchmarks : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Performance metric types
     */
    enum class MetricType {
        ExecutionTime,      ///< Execution time in milliseconds
        MemoryUsage,        ///< Memory usage in bytes
        CPUUsage,          ///< CPU usage percentage
        ThroughputOps,     ///< Operations per second
        ThroughputData,    ///< Data throughput (bytes/second)
        Latency,           ///< Response latency in milliseconds
        Custom             ///< Custom metric type
    };

    /**
     * @brief Performance measurement result
     */
    struct PerformanceResult {
        QString benchmarkName;              ///< Name of the benchmark
        MetricType metricType;              ///< Type of metric measured
        double value = 0.0;                 ///< Measured value
        QString unit;                       ///< Unit of measurement
        QDateTime timestamp;                ///< When measurement was taken
        double baseline = 0.0;              ///< Baseline value for comparison
        double threshold = 0.0;             ///< Performance threshold
        bool withinThreshold = true;        ///< Whether result is within acceptable range
        QMap<QString, QVariant> metadata;  ///< Additional measurement metadata
        QString description;                ///< Human-readable description
    };

    /**
     * @brief Performance baseline information
     */
    struct PerformanceBaseline {
        QString name;                       ///< Baseline name
        MetricType metricType;              ///< Type of metric
        double value = 0.0;                 ///< Baseline value
        double tolerance = 0.1;             ///< Acceptable deviation (10% default)
        QDateTime created;                  ///< When baseline was created
        QDateTime lastUpdated;              ///< Last update timestamp
        QString platform;                   ///< Platform information
        QString configuration;              ///< Test configuration
        QMap<QString, QVariant> environment; ///< Environment details
        QString description;                ///< Baseline description
    };

    /**
     * @brief Benchmark configuration
     */
    struct BenchmarkConfig {
        int iterations = 1;                 ///< Number of iterations to run
        int warmupIterations = 0;           ///< Warmup iterations (not measured)
        bool measureMemory = true;          ///< Whether to measure memory usage
        bool measureCPU = false;            ///< Whether to measure CPU usage
        bool measureThroughput = false;     ///< Whether to measure throughput
        int samplingIntervalMs = 100;       ///< Sampling interval for continuous metrics
        double timeoutSeconds = 60.0;       ///< Maximum benchmark execution time
        bool failOnTimeout = true;          ///< Whether to fail on timeout
        bool collectGCStats = false;        ///< Whether to collect garbage collection stats
        QMap<QString, QVariant> customSettings; ///< Custom benchmark settings
    };

    /**
     * @brief Memory usage statistics
     */
    struct MemoryStats {
        qint64 peakUsage = 0;              ///< Peak memory usage in bytes
        qint64 averageUsage = 0;           ///< Average memory usage in bytes
        qint64 initialUsage = 0;           ///< Memory usage at start
        qint64 finalUsage = 0;             ///< Memory usage at end
        qint64 allocated = 0;              ///< Total memory allocated
        qint64 deallocated = 0;            ///< Total memory deallocated
        int allocationCount = 0;           ///< Number of allocations
        int deallocationCount = 0;         ///< Number of deallocations
    };

    /**
     * @brief CPU usage statistics
     */
    struct CPUStats {
        double peakUsage = 0.0;            ///< Peak CPU usage percentage
        double averageUsage = 0.0;         ///< Average CPU usage percentage
        double userTime = 0.0;             ///< User CPU time in seconds
        double systemTime = 0.0;           ///< System CPU time in seconds
        int contextSwitches = 0;           ///< Number of context switches
        int pageFaults = 0;                ///< Number of page faults
    };

    explicit PerformanceBenchmarks(QObject* parent = nullptr);
    ~PerformanceBenchmarks();

    // Configuration
    void setBenchmarkConfig(const BenchmarkConfig& config);
    BenchmarkConfig getBenchmarkConfig() const;
    void setOutputDirectory(const QString& directory);
    QString getOutputDirectory() const;

    // Benchmark execution
    PerformanceResult runBenchmark(const QString& name, std::function<void()> benchmarkFunction);
    PerformanceResult runBenchmark(const QString& name, std::function<void()> benchmarkFunction, 
                                 const BenchmarkConfig& config);
    QList<PerformanceResult> runBenchmarkSuite(const QMap<QString, std::function<void()>>& benchmarks);
    
    // Timing measurements
    void startTimer(const QString& name);
    qint64 stopTimer(const QString& name);
    qint64 getElapsedTime(const QString& name) const;
    PerformanceResult measureExecutionTime(const QString& name, std::function<void()> function);
    
    // Memory measurements
    MemoryStats measureMemoryUsage(const QString& name, std::function<void()> function);
    qint64 getCurrentMemoryUsage() const;
    qint64 getPeakMemoryUsage() const;
    void resetMemoryTracking();
    
    // CPU measurements
    CPUStats measureCPUUsage(const QString& name, std::function<void()> function);
    double getCurrentCPUUsage() const;
    void startCPUMonitoring();
    void stopCPUMonitoring();
    
    // Throughput measurements
    PerformanceResult measureThroughput(const QString& name, std::function<int()> function, 
                                      const QString& unit = "ops/sec");
    PerformanceResult measureDataThroughput(const QString& name, std::function<qint64()> function);
    
    // Baseline management
    bool createBaseline(const QString& name, const PerformanceResult& result);
    bool updateBaseline(const QString& name, const PerformanceResult& result);
    bool deleteBaseline(const QString& name);
    PerformanceBaseline getBaseline(const QString& name) const;
    QStringList getAvailableBaselines() const;
    bool baselineExists(const QString& name) const;
    
    // Performance comparison
    PerformanceResult compareWithBaseline(const QString& baselineName, const PerformanceResult& result) const;
    QList<PerformanceResult> compareAllWithBaselines(const QList<PerformanceResult>& results) const;
    double calculatePerformanceRegression(const PerformanceResult& current, 
                                        const PerformanceBaseline& baseline) const;
    
    // Statistical analysis
    PerformanceResult calculateStatistics(const QList<PerformanceResult>& results) const;
    double calculateMean(const QList<double>& values) const;
    double calculateMedian(const QList<double>& values) const;
    double calculateStandardDeviation(const QList<double>& values) const;
    double calculatePercentile(const QList<double>& values, double percentile) const;
    
    // Reporting and export
    bool generatePerformanceReport(const QString& outputPath, 
                                 const QList<PerformanceResult>& results) const;
    bool generateTrendReport(const QString& outputPath, const QString& benchmarkName) const;
    QJsonObject exportResults(const QList<PerformanceResult>& results) const;
    bool importResults(const QString& filePath);
    
    // Continuous monitoring
    void startContinuousMonitoring(const QString& name, int intervalMs = 1000);
    void stopContinuousMonitoring(const QString& name);
    QList<PerformanceResult> getContinuousResults(const QString& name) const;
    
    // Stress testing utilities
    PerformanceResult runStressTest(const QString& name, std::function<void()> stressFunction, 
                                  int durationSeconds, int concurrentThreads = 1);
    PerformanceResult runLoadTest(const QString& name, std::function<void()> loadFunction,
                                int requestsPerSecond, int durationSeconds);
    
    // Platform and environment info
    QMap<QString, QVariant> getSystemInfo() const;
    QMap<QString, QVariant> getEnvironmentInfo() const;
    QString getPlatformIdentifier() const;
    
    // Utility methods
    static QString formatDuration(qint64 milliseconds);
    static QString formatBytes(qint64 bytes);
    static QString formatRate(double rate, const QString& unit);
    static double convertToSeconds(qint64 milliseconds);
    static qint64 convertToMilliseconds(double seconds);

signals:
    void benchmarkStarted(const QString& name);
    void benchmarkCompleted(const QString& name, const PerformanceResult& result);
    void benchmarkFailed(const QString& name, const QString& error);
    void baselineCreated(const QString& name);
    void baselineUpdated(const QString& name);
    void performanceRegression(const QString& name, double regressionPercent);
    void continuousResultAvailable(const QString& name, const PerformanceResult& result);

private:
    BenchmarkConfig m_config;
    QString m_outputDirectory;
    
    // Timing
    QMap<QString, QElapsedTimer> m_timers;
    QMap<QString, qint64> m_elapsedTimes;
    
    // Memory tracking
    qint64 m_initialMemory;
    qint64 m_peakMemory;
    QMutex m_memoryMutex;
    
    // CPU monitoring
    QThread* m_cpuMonitorThread;
    bool m_cpuMonitoringActive;
    QMutex m_cpuMutex;
    QList<double> m_cpuSamples;
    
    // Continuous monitoring
    QMap<QString, QThread*> m_monitoringThreads;
    QMap<QString, QList<PerformanceResult>> m_continuousResults;
    QMutex m_continuousResultsMutex;
    
    // Baselines
    QMap<QString, PerformanceBaseline> m_baselines;
    mutable QMutex m_baselinesMutex;
    
    // Internal helper methods
    void initializePerformanceCounters();
    void cleanupPerformanceCounters();
    PerformanceResult createResult(const QString& name, MetricType type, double value, 
                                 const QString& unit) const;
    void saveBaseline(const QString& name, const PerformanceBaseline& baseline);
    PerformanceBaseline loadBaseline(const QString& name) const;
    QString getBaselinePath(const QString& name) const;
    
    // Platform-specific implementations
    qint64 getPlatformMemoryUsage() const;
    double getPlatformCPUUsage() const;
    QMap<QString, QVariant> getPlatformSystemInfo() const;
    
    // Statistical helpers
    QList<double> extractValues(const QList<PerformanceResult>& results) const;
    void sortValues(QList<double>& values) const;
    
    // Monitoring thread functions
    void cpuMonitoringLoop();
    void continuousMonitoringLoop(const QString& name, int intervalMs);
};

/**
 * @brief Performance test runner for automated benchmark execution
 */
class PerformanceTestRunner : public QObject {
    Q_OBJECT

public:
    explicit PerformanceTestRunner(PerformanceBenchmarks* benchmarks, QObject* parent = nullptr);

    // Test suite management
    void registerBenchmark(const QString& name, std::function<void()> benchmark);
    void registerBenchmark(const QString& name, std::function<void()> benchmark, 
                          const PerformanceBenchmarks::BenchmarkConfig& config);
    void unregisterBenchmark(const QString& name);
    QStringList getRegisteredBenchmarks() const;

    // Execution
    bool runAllBenchmarks();
    bool runBenchmark(const QString& name);
    bool runBenchmarkCategory(const QString& category);
    QList<PerformanceBenchmarks::PerformanceResult> getResults() const;

    // Configuration
    void setCategoryConfig(const QString& category, const PerformanceBenchmarks::BenchmarkConfig& config);
    void setGlobalConfig(const PerformanceBenchmarks::BenchmarkConfig& config);

signals:
    void testSuiteStarted(int totalBenchmarks);
    void testSuiteCompleted(const QList<PerformanceBenchmarks::PerformanceResult>& results);
    void benchmarkProgress(int completed, int total);

private:
    struct RegisteredBenchmark {
        QString name;
        QString category;
        std::function<void()> function;
        PerformanceBenchmarks::BenchmarkConfig config;
    };

    PerformanceBenchmarks* m_benchmarks;
    QMap<QString, RegisteredBenchmark> m_registeredBenchmarks;
    QMap<QString, PerformanceBenchmarks::BenchmarkConfig> m_categoryConfigs;
    PerformanceBenchmarks::BenchmarkConfig m_globalConfig;
    QList<PerformanceBenchmarks::PerformanceResult> m_results;
};

/**
 * @brief Convenience macros for performance testing
 */
#define BENCHMARK_MEASURE(name, code) \
    do { \
        auto result = performanceBenchmarks.measureExecutionTime(name, [&]() { code; }); \
        if (!result.withinThreshold) { \
            QFAIL(QString("Performance benchmark failed: %1 (%2 %3)") \
                  .arg(name).arg(result.value).arg(result.unit).toUtf8().constData()); \
        } \
    } while(0)

#define BENCHMARK_COMPARE_BASELINE(name, code) \
    do { \
        auto result = performanceBenchmarks.measureExecutionTime(name, [&]() { code; }); \
        auto comparison = performanceBenchmarks.compareWithBaseline(name, result); \
        if (!comparison.withinThreshold) { \
            QFAIL(QString("Performance regression detected: %1 (current: %2, baseline: %3)") \
                  .arg(name).arg(result.value).arg(comparison.baseline).toUtf8().constData()); \
        } \
    } while(0)

#define BENCHMARK_MEMORY(name, code) \
    do { \
        auto memStats = performanceBenchmarks.measureMemoryUsage(name, [&]() { code; }); \
        qDebug() << "Memory usage for" << name << ":" << memStats.peakUsage << "bytes"; \
    } while(0)

#define BENCHMARK_THROUGHPUT(name, code, expectedMinOps) \
    do { \
        auto result = performanceBenchmarks.measureThroughput(name, [&]() -> int { code; }); \
        if (result.value < expectedMinOps) { \
            QFAIL(QString("Throughput below threshold: %1 (%2 ops/sec, expected >= %3)") \
                  .arg(name).arg(result.value).arg(expectedMinOps).toUtf8().constData()); \
        } \
    } while(0)