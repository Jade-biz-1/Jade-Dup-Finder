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
#include <QThreadPool>
#include <QRunnable>
#include <QAtomicInt>
#include <QWaitCondition>
#include <functional>

class PerformanceBenchmark;
class TestDataGenerator;

/**
 * @brief Comprehensive load and stress testing framework for CloneClean
 * 
 * Provides advanced load testing, stress testing, and scalability validation
 * capabilities including concurrent operations, high-volume scenarios, and
 * system limit validation.
 */
class LoadStressTesting : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Load test types
     */
    enum class LoadTestType {
        ConcurrentUsers,        ///< Simulate multiple concurrent users
        HighVolumeFiles,       ///< Test with large numbers of files
        LargeFileSize,         ///< Test with very large individual files
        DeepDirectories,       ///< Test with deeply nested directory structures
        WideDirectories,       ///< Test with directories containing many files
        MixedWorkload,         ///< Combined load patterns
        SustainedLoad,         ///< Long-running sustained operations
        BurstLoad,             ///< Short bursts of high activity
        GradualRamp,           ///< Gradually increasing load
        StressToFailure        ///< Push system until failure point
    };

    /**
     * @brief Load test configuration
     */
    struct LoadTestConfig {
        QString name;                           ///< Test name
        LoadTestType type;                      ///< Type of load test
        int concurrentThreads = 1;             ///< Number of concurrent threads
        int totalOperations = 100;             ///< Total operations to perform
        qint64 durationMs = 60000;             ///< Test duration in milliseconds
        qint64 rampUpTimeMs = 5000;            ///< Ramp-up time in milliseconds
        qint64 rampDownTimeMs = 5000;          ///< Ramp-down time in milliseconds
        int operationsPerSecond = 10;          ///< Target operations per second
        qint64 maxMemoryMB = 1024;             ///< Maximum memory usage in MB
        double maxCpuPercent = 80.0;           ///< Maximum CPU usage percentage
        qint64 timeoutMs = 300000;             ///< Overall test timeout (5 minutes)
        bool failOnError = false;              ///< Whether to fail test on first error
        bool collectDetailedMetrics = true;    ///< Collect detailed performance metrics
        QMap<QString, QVariant> customParams;  ///< Custom test parameters
        QString description;                    ///< Test description
        QStringList tags;                      ///< Tags for categorization
    };

    /**
     * @brief Load test result
     */
    struct LoadTestResult {
        QString testName;                       ///< Name of the test
        LoadTestType testType;                  ///< Type of load test
        QDateTime startTime;                    ///< Test start time
        QDateTime endTime;                      ///< Test end time
        qint64 actualDurationMs = 0;           ///< Actual test duration
        int totalOperations = 0;               ///< Total operations performed
        int successfulOperations = 0;          ///< Successful operations
        int failedOperations = 0;              ///< Failed operations
        double operationsPerSecond = 0.0;      ///< Actual operations per second
        double averageResponseTime = 0.0;      ///< Average response time in ms
        double minResponseTime = 0.0;          ///< Minimum response time in ms
        double maxResponseTime = 0.0;          ///< Maximum response time in ms
        double percentile95ResponseTime = 0.0; ///< 95th percentile response time
        double percentile99ResponseTime = 0.0; ///< 99th percentile response time
        qint64 peakMemoryUsage = 0;            ///< Peak memory usage in bytes
        double averageCpuUsage = 0.0;          ///< Average CPU usage percentage
        double peakCpuUsage = 0.0;             ///< Peak CPU usage percentage
        int concurrentThreadsUsed = 0;        ///< Number of concurrent threads used
        bool completedSuccessfully = false;    ///< Whether test completed successfully
        QString errorMessage;                   ///< Error message if failed
        QMap<QString, QVariant> customMetrics; ///< Custom metrics collected
        QList<double> responseTimes;           ///< All response times for analysis
    };

    /**
     * @brief Stress test configuration
     */
    struct StressTestConfig {
        QString name;                           ///< Test name
        int maxConcurrentOperations = 100;     ///< Maximum concurrent operations
        qint64 maxFileSize = 1073741824;       ///< Maximum file size (1GB)
        int maxFileCount = 10000;              ///< Maximum number of files
        int maxDirectoryDepth = 20;            ///< Maximum directory nesting depth
        int maxDirectoryWidth = 1000;          ///< Maximum files per directory
        qint64 maxTestDurationMs = 600000;     ///< Maximum test duration (10 minutes)
        qint64 memoryLimitMB = 2048;           ///< Memory limit in MB
        double cpuLimitPercent = 95.0;         ///< CPU usage limit percentage
        bool stopOnResourceLimit = true;       ///< Stop when resource limits hit
        bool stopOnFirstFailure = false;      ///< Stop on first operation failure
        int failureThreshold = 10;            ///< Number of failures before stopping
        QString description;                    ///< Test description
        QStringList tags;                      ///< Tags for categorization
    };

    /**
     * @brief Stress test result
     */
    struct StressTestResult {
        QString testName;                       ///< Name of the test
        QDateTime startTime;                    ///< Test start time
        QDateTime endTime;                      ///< Test end time
        qint64 actualDurationMs = 0;           ///< Actual test duration
        int maxConcurrentOperationsReached = 0; ///< Maximum concurrent operations reached
        qint64 maxFileSizeProcessed = 0;       ///< Largest file size processed
        int maxFileCountProcessed = 0;         ///< Maximum files processed simultaneously
        int maxDirectoryDepthReached = 0;      ///< Maximum directory depth reached
        qint64 peakMemoryUsageMB = 0;          ///< Peak memory usage in MB
        double peakCpuUsagePercent = 0.0;      ///< Peak CPU usage percentage
        bool hitMemoryLimit = false;           ///< Whether memory limit was hit
        bool hitCpuLimit = false;              ///< Whether CPU limit was hit
        bool hitTimeLimit = false;             ///< Whether time limit was hit
        int totalFailures = 0;                 ///< Total number of failures
        QString failureReason;                 ///< Reason for test termination
        bool completedSuccessfully = false;    ///< Whether test completed successfully
        QMap<QString, QVariant> resourceMetrics; ///< Resource usage metrics
        QStringList errorMessages;            ///< List of error messages encountered
    };

    /**
     * @brief Scalability test configuration
     */
    struct ScalabilityTestConfig {
        QString name;                           ///< Test name
        QList<int> fileCounts;                 ///< File counts to test
        QList<qint64> fileSizes;               ///< File sizes to test
        QList<int> threadCounts;               ///< Thread counts to test
        int iterationsPerConfiguration = 3;    ///< Iterations per configuration
        qint64 maxDurationPerTest = 300000;    ///< Max duration per test (5 minutes)
        bool measureMemoryScaling = true;      ///< Measure memory scaling
        bool measureTimeScaling = true;        ///< Measure time scaling
        bool measureThroughputScaling = true;  ///< Measure throughput scaling
        QString description;                    ///< Test description
        QStringList tags;                      ///< Tags for categorization
    };

    /**
     * @brief Scalability test result
     */
    struct ScalabilityTestResult {
        QString testName;                       ///< Name of the test
        QDateTime startTime;                    ///< Test start time
        QDateTime endTime;                      ///< Test end time
        QMap<QString, LoadTestResult> results; ///< Results for each configuration
        QMap<QString, double> scalingFactors;  ///< Scaling factors for different metrics
        bool linearTimeScaling = false;        ///< Whether time scales linearly
        bool linearMemoryScaling = false;      ///< Whether memory scales linearly
        bool linearThroughputScaling = false;  ///< Whether throughput scales linearly
        QString scalingAnalysis;               ///< Analysis of scaling behavior
        bool completedSuccessfully = false;    ///< Whether test completed successfully
    };

    explicit LoadStressTesting(QObject* parent = nullptr);
    ~LoadStressTesting();

    // Load testing methods
    bool runLoadTest(const LoadTestConfig& config, std::function<void()> operation);
    bool runConcurrentUserTest(int userCount, int operationsPerUser, std::function<void(int)> userOperation);
    bool runHighVolumeFileTest(int fileCount, qint64 fileSize, const QString& testDirectory);
    bool runLargeFileSizeTest(qint64 maxFileSize, const QString& testDirectory);
    bool runDeepDirectoryTest(int maxDepth, int filesPerLevel, const QString& testDirectory);
    bool runWideDirectoryTest(int filesPerDirectory, int directoryCount, const QString& testDirectory);
    bool runSustainedLoadTest(qint64 durationMs, int operationsPerSecond, std::function<void()> operation);
    bool runBurstLoadTest(int burstCount, int operationsPerBurst, qint64 burstIntervalMs, std::function<void()> operation);
    bool runGradualRampTest(int startThreads, int endThreads, qint64 rampDurationMs, std::function<void()> operation);

    // Stress testing methods
    bool runStressTest(const StressTestConfig& config, std::function<void()> operation);
    bool runStressToFailureTest(std::function<void()> operation, const StressTestConfig& config = {});
    bool runMemoryStressTest(qint64 maxMemoryMB, std::function<void()> operation);
    bool runCpuStressTest(double maxCpuPercent, std::function<void()> operation);
    bool runConcurrencyStressTest(int maxConcurrentOperations, std::function<void()> operation);
    bool runResourceExhaustionTest(const StressTestConfig& config, std::function<void()> operation);

    // Scalability testing methods
    bool runScalabilityTest(const ScalabilityTestConfig& config, std::function<void(int, qint64, int)> operation);
    bool runFileCountScalabilityTest(const QList<int>& fileCounts, std::function<void(int)> operation);
    bool runFileSizeScalabilityTest(const QList<qint64>& fileSizes, std::function<void(qint64)> operation);
    bool runThreadScalabilityTest(const QList<int>& threadCounts, std::function<void(int)> operation);

    // CloneClean-specific load tests
    bool runDuplicateDetectionLoadTest(const QString& testDirectory, int fileCount, double duplicateRatio);
    bool runHashCalculationLoadTest(const QStringList& filePaths, int concurrentHashers);
    bool runFileScanningLoadTest(const QString& rootDirectory, int concurrentScanners);
    bool runUILoadTest(QWidget* mainWindow, int concurrentUIOperations);

    // Result management
    QList<LoadTestResult> getLoadTestResults() const;
    QList<StressTestResult> getStressTestResults() const;
    QList<ScalabilityTestResult> getScalabilityTestResults() const;
    LoadTestResult getLoadTestResult(const QString& testName) const;
    StressTestResult getStressTestResult(const QString& testName) const;
    ScalabilityTestResult getScalabilityTestResult(const QString& testName) const;

    // Analysis and reporting
    QJsonObject generateLoadTestReport() const;
    QJsonObject generateStressTestReport() const;
    QJsonObject generateScalabilityTestReport() const;
    QJsonObject generateComprehensiveReport() const;
    bool exportResults(const QString& filePath, const QString& format = "json") const;

    // Configuration and control
    void setPerformanceBenchmark(PerformanceBenchmark* benchmark);
    void setTestDataGenerator(TestDataGenerator* generator);
    void setMaxConcurrentThreads(int maxThreads);
    void setResourceMonitoringInterval(int intervalMs);
    void clearResults();

    // Utility methods
    QString formatLoadTestType(LoadTestType type) const;
    QString formatDuration(qint64 milliseconds) const;
    QString formatThroughput(double operationsPerSecond) const;
    QString formatMemoryUsage(qint64 bytes) const;

signals:
    void loadTestStarted(const QString& testName);
    void loadTestCompleted(const QString& testName, const LoadTestResult& result);
    void stressTestStarted(const QString& testName);
    void stressTestCompleted(const QString& testName, const StressTestResult& result);
    void scalabilityTestStarted(const QString& testName);
    void scalabilityTestCompleted(const QString& testName, const ScalabilityTestResult& result);
    void operationCompleted(const QString& testName, int operationIndex, qint64 responseTime);
    void resourceLimitReached(const QString& testName, const QString& resourceType, double currentValue, double limit);
    void testProgress(const QString& testName, int completedOperations, int totalOperations);
    void errorOccurred(const QString& testName, const QString& errorMessage);

private slots:
    void onResourceMonitoringTimer();

private:
    // Configuration
    int m_maxConcurrentThreads;
    int m_resourceMonitoringInterval;
    PerformanceBenchmark* m_performanceBenchmark;
    TestDataGenerator* m_testDataGenerator;

    // Results storage
    QList<LoadTestResult> m_loadTestResults;
    QList<StressTestResult> m_stressTestResults;
    QList<ScalabilityTestResult> m_scalabilityTestResults;

    // Resource monitoring
    QTimer* m_resourceTimer;
    bool m_resourceMonitoringActive;
    QMap<QString, QVariant> m_currentResourceUsage;

    // Thread management
    QThreadPool* m_threadPool;
    QAtomicInt m_activeOperations;
    QMutex m_resultsMutex;
    QWaitCondition m_operationsComplete;

    // Internal helper methods
    void recordLoadTestResult(const LoadTestResult& result);
    void recordStressTestResult(const StressTestResult& result);
    void recordScalabilityTestResult(const ScalabilityTestResult& result);

    // Resource monitoring helpers
    qint64 getCurrentMemoryUsage() const;
    double getCurrentCpuUsage() const;
    int getCurrentActiveThreads() const;
    QMap<QString, QVariant> collectResourceMetrics() const;

    // Load test execution helpers
    LoadTestResult executeLoadTest(const LoadTestConfig& config, std::function<void(int)> operation);
    void executeOperationWithTiming(std::function<void()> operation, QList<double>& responseTimes, QAtomicInt& successCount, QAtomicInt& failureCount);
    void rampUpOperations(const LoadTestConfig& config, std::function<void()> operation, QAtomicInt& activeOperations);
    void rampDownOperations(qint64 rampDownTimeMs, QAtomicInt& activeOperations);

    // Stress test execution helpers
    StressTestResult executeStressTest(const StressTestConfig& config, std::function<void()> operation);
    bool checkResourceLimits(const StressTestConfig& config, StressTestResult& result);
    void monitorResourceUsage(StressTestResult& result);

    // Scalability test execution helpers
    ScalabilityTestResult executeScalabilityTest(const ScalabilityTestConfig& config, std::function<void(int, qint64, int)> operation);
    double calculateScalingFactor(const QList<double>& values, const QList<double>& inputs) const;
    QString analyzeScalingBehavior(const QMap<QString, double>& scalingFactors) const;

    // Statistical analysis helpers
    double calculateMean(const QList<double>& values) const;
    double calculatePercentile(QList<double> values, double percentile) const;
    double calculateStandardDeviation(const QList<double>& values, double mean) const;

    // File system test helpers
    bool createTestFileStructure(const QString& baseDirectory, int fileCount, qint64 fileSize);
    bool createDeepDirectoryStructure(const QString& baseDirectory, int depth, int filesPerLevel);
    bool createWideDirectoryStructure(const QString& baseDirectory, int filesPerDirectory, int directoryCount);
    void cleanupTestDirectory(const QString& directory);

    // Platform-specific helpers
    QString detectPlatform() const;
    qint64 getAvailableMemory() const;
    int getAvailableCpuCores() const;
};

/**
 * @brief Worker class for concurrent load test operations
 */
class LoadTestWorker : public QObject, public QRunnable {
    Q_OBJECT

public:
    LoadTestWorker(std::function<void()> operation, QAtomicInt* successCount, QAtomicInt* failureCount, QList<double>* responseTimes, QMutex* mutex);
    void run() override;

signals:
    void operationCompleted(qint64 responseTime, bool success);

private:
    std::function<void()> m_operation;
    QAtomicInt* m_successCount;
    QAtomicInt* m_failureCount;
    QList<double>* m_responseTimes;
    QMutex* m_mutex;
};

/**
 * @brief Convenience macros for load and stress testing
 */
#define LOAD_TEST(testName, config, operation) \
    do { \
        if (!loadStressTesting.runLoadTest(config, operation)) { \
            QFAIL(QString("Load test failed: %1").arg(testName).toUtf8().constData()); \
        } \
    } while(0)

#define STRESS_TEST(testName, config, operation) \
    do { \
        if (!loadStressTesting.runStressTest(config, operation)) { \
            QFAIL(QString("Stress test failed: %1").arg(testName).toUtf8().constData()); \
        } \
    } while(0)

#define SCALABILITY_TEST(testName, config, operation) \
    do { \
        if (!loadStressTesting.runScalabilityTest(config, operation)) { \
            QFAIL(QString("Scalability test failed: %1").arg(testName).toUtf8().constData()); \
        } \
    } while(0)

#define VERIFY_LOAD_PERFORMANCE(testName, maxResponseTime, minThroughput) \
    do { \
        auto result = loadStressTesting.getLoadTestResult(testName); \
        QVERIFY2(result.averageResponseTime <= maxResponseTime, \
                QString("Average response time %1ms exceeds limit %2ms") \
                .arg(result.averageResponseTime).arg(maxResponseTime).toUtf8().constData()); \
        QVERIFY2(result.operationsPerSecond >= minThroughput, \
                QString("Throughput %1 ops/sec below minimum %2 ops/sec") \
                .arg(result.operationsPerSecond).arg(minThroughput).toUtf8().constData()); \
    } while(0)