#ifndef PERFORMANCE_TEST_FRAMEWORK_H
#define PERFORMANCE_TEST_FRAMEWORK_H

#include <QtTest>
#include <QObject>
#include <QElapsedTimer>
#include <QDateTime>
#include <QHash>
#include <QStringList>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDir>
#include <QTemporaryDir>
#include <QDebug>
#include <QCoreApplication>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QVector>
#include <QRandomGenerator>
#include <QFileInfo>
#include <QStandardPaths>

/**
 * @brief Performance testing framework for DupFinder components
 * 
 * This framework provides comprehensive performance testing capabilities including:
 * - High-precision timing measurements
 * - Test data generation utilities
 * - Performance metrics collection and analysis
 * - Statistical analysis of results
 * - Regression testing support
 * - Hardware and environment profiling
 */

namespace PerformanceTest {

    /**
     * @brief Performance measurement result container
     */
    struct PerformanceResult {
        QString testName;
        QString category;
        QDateTime timestamp;
        
        // Timing measurements (in milliseconds)
        double executionTime = 0.0;
        double cpuTime = 0.0;
        double wallTime = 0.0;
        
        // Throughput measurements
        double throughputMBps = 0.0;
        double filesPerSecond = 0.0;
        double operationsPerSecond = 0.0;
        
        // Resource utilization
        qint64 peakMemoryUsage = 0;
        qint64 averageMemoryUsage = 0;
        double cpuUtilization = 0.0;
        double ioUtilization = 0.0;
        
        // Statistical data for multiple runs
        QVector<double> executionTimes;
        double meanTime = 0.0;
        double medianTime = 0.0;
        double stdDeviation = 0.0;
        double minTime = 0.0;
        double maxTime = 0.0;
        double confidenceInterval95 = 0.0;
        
        // Test-specific metrics
        QHash<QString, double> customMetrics;
        
        // System information
        QString systemInfo;
        QString qtVersion;
        QString buildConfig;
        int threadCount = 0;
        
        // Test validation
        bool validationPassed = true;
        QString errorMessage;
        
        QJsonObject toJson() const;
        void fromJson(const QJsonObject& json);
    };

    /**
     * @brief System profiler for hardware and environment information
     */
    class SystemProfiler {
    public:
        struct SystemInfo {
            QString cpuModel;
            int cpuCores;
            int logicalProcessors;
            QString cpuArchitecture;
            double cpuFrequencyGHz;
            
            qint64 totalMemoryMB;
            qint64 availableMemoryMB;
            
            QString osName;
            QString osVersion;
            QString qtVersion;
            QString buildType;
            
            QStringList storageDevices;
            QString primaryStorageType; // SSD, HDD, etc.
            
            QJsonObject toJson() const;
        };
        
        static SystemInfo getSystemInfo();
        static QString getSystemSummary();
        static void logSystemInfo();
    };

    /**
     * @brief Test data generator for performance testing
     */
    class TestDataGenerator {
    public:
        struct DataGenerationOptions {
            qint64 totalSize = 100 * 1024 * 1024; // 100MB default
            QVector<qint64> fileSizes; // Specific file sizes to generate
            int fileCount = 100;
            QString basePath;
            bool createDuplicates = false;
            double duplicateRatio = 0.3; // 30% duplicates
            QStringList fileExtensions = {"txt", "dat", "bin"};
            bool randomContent = true;
            QString contentPattern; // Optional fixed pattern
            bool compressible = false; // Generate compressible content
            int directoryDepth = 3;
            int subdirectoryCount = 5;
        };
        
        explicit TestDataGenerator(QObject* parent = nullptr);
        ~TestDataGenerator();
        
        bool generateTestFiles(const DataGenerationOptions& options);
        bool generateTestFile(const QString& filePath, qint64 size, const QString& pattern = QString());
        QStringList getGeneratedFiles() const { return m_generatedFiles; }
        QString getTestDataPath() const { return m_testDataPath; }
        
        // Cleanup
        void cleanupTestData();
        
        // Predefined test scenarios
        bool generateSmallFilesScenario(int fileCount = 1000, qint64 avgSize = 1024);
        bool generateLargeFilesScenario(int fileCount = 10, qint64 avgSize = 100 * 1024 * 1024);
        bool generateMixedSizesScenario(int totalFiles = 500);
        bool generateDuplicateScenario(int uniqueFiles = 100, int duplicatesPerFile = 3);
        bool generateDirectoryStructureScenario(int depth = 5, int filesPerDir = 20);
        
    private:
        QString m_testDataPath;
        QStringList m_generatedFiles;
        QScopedPointer<QTemporaryDir> m_tempDir;
        
        QByteArray generateRandomContent(qint64 size, bool compressible = false);
        QString createDirectoryStructure(const QString& basePath, int depth, int subdirCount);
    };

    /**
     * @brief Performance timer with high precision and statistical analysis
     */
    class PerformanceTimer {
    public:
        PerformanceTimer();
        ~PerformanceTimer();
        
        // Basic timing
        void start();
        void stop();
        void reset();
        
        // Multi-run timing
        void startRun();
        void endRun();
        void beginSeries(const QString& name, int expectedRuns = 1);
        void endSeries();
        
        // Results
        double elapsedTime() const; // Current timing in ms
        double averageTime() const;
        double medianTime() const;
        double standardDeviation() const;
        double confidenceInterval95() const;
        QVector<double> getAllTimes() const { return m_runTimes; }
        
        // Resource monitoring
        void enableResourceMonitoring(bool enabled = true);
        qint64 getCurrentMemoryUsage() const;
        qint64 getPeakMemoryUsage() const;
        double getCPUUsage() const;
        
        // Statistics
        PerformanceResult getResult() const;
        void setTestName(const QString& name) { m_testName = name; }
        void setCategory(const QString& category) { m_category = category; }
        
    private:
        QElapsedTimer m_timer;
        QDateTime m_startTime;
        QDateTime m_endTime;
        
        QString m_testName;
        QString m_category;
        QString m_seriesName;
        
        QVector<double> m_runTimes;
        qint64 m_startMemory;
        qint64 m_peakMemory;
        bool m_resourceMonitoring;
        
        double calculateStandardDeviation() const;
        double calculateMedian() const;
        double calculateConfidenceInterval() const;
        void updateMemoryUsage();
    };

    /**
     * @brief Performance benchmark runner and coordinator
     */
    class BenchmarkRunner : public QObject {
        Q_OBJECT
        
    public:
        struct BenchmarkConfig {
            int warmupRuns = 3;
            int measurementRuns = 10;
            int maxRunTimeSeconds = 300; // 5 minutes max per test
            bool enableResourceMonitoring = true;
            bool enableStatistics = true;
            bool saveResults = true;
            QString resultsPath;
            double acceptableVariation = 0.15; // 15% variation is acceptable
            bool enableRegressionDetection = false;
            QString baselineResultsPath;
        };
        
        explicit BenchmarkRunner(QObject* parent = nullptr);
        ~BenchmarkRunner();
        
        // Configuration
        void setBenchmarkConfig(const BenchmarkConfig& config);
        BenchmarkConfig getBenchmarkConfig() const { return m_config; }
        
        // Test registration
        void registerBenchmark(const QString& name, const QString& category, 
                              std::function<bool()> testFunction);
        void registerSetup(std::function<bool()> setupFunction);
        void registerTeardown(std::function<void()> teardownFunction);
        
        // Execution
        bool runAllBenchmarks();
        bool runBenchmark(const QString& name);
        bool runCategory(const QString& category);
        
        // Results
        QVector<PerformanceResult> getAllResults() const { return m_results; }
        PerformanceResult getResult(const QString& testName) const;
        bool saveResultsToFile(const QString& filePath = QString()) const;
        bool loadBaselineResults(const QString& filePath);
        
        // Analysis
        bool detectRegressions() const;
        QString generateReport() const;
        QString generateSummaryReport() const;
        void logResults() const;
        
        // Utilities
        TestDataGenerator* getDataGenerator() { return &m_dataGenerator; }
        SystemProfiler::SystemInfo getSystemInfo() const { return m_systemInfo; }
        
    signals:
        void benchmarkStarted(const QString& name);
        void benchmarkCompleted(const QString& name, const PerformanceResult& result);
        void allBenchmarksCompleted();
        void progressUpdate(int current, int total);
        
    private slots:
        void onBenchmarkTimeout();
        
    private:
        struct BenchmarkInfo {
            QString name;
            QString category;
            std::function<bool()> testFunction;
        };
        
        BenchmarkConfig m_config;
        QVector<BenchmarkInfo> m_benchmarks;
        QVector<PerformanceResult> m_results;
        QVector<PerformanceResult> m_baselineResults;
        
        std::function<bool()> m_setupFunction;
        std::function<void()> m_teardownFunction;
        
        TestDataGenerator m_dataGenerator;
        SystemProfiler::SystemInfo m_systemInfo;
        QTimer* m_timeoutTimer;
        
        bool runSingleBenchmark(const BenchmarkInfo& benchmark);
        void performWarmupRuns(const BenchmarkInfo& benchmark);
        PerformanceResult measureBenchmark(const BenchmarkInfo& benchmark);
        bool validateResult(const PerformanceResult& result) const;
        QString formatDuration(double milliseconds) const;
        QString formatThroughput(double mbps) const;
        QString formatMemory(qint64 bytes) const;
    };

    /**
     * @brief Regression testing utilities
     */
    class RegressionTester {
    public:
        struct RegressionResult {
            QString testName;
            double currentValue;
            double baselineValue;
            double changePercent;
            bool isRegression;
            bool isImprovement;
            QString status;
        };
        
        static QVector<RegressionResult> compareResults(
            const QVector<PerformanceResult>& current,
            const QVector<PerformanceResult>& baseline,
            double regressionThreshold = 0.20); // 20% slowdown is regression
        
        static QString generateRegressionReport(const QVector<RegressionResult>& results);
        static bool hasRegressions(const QVector<RegressionResult>& results);
        static bool hasImprovements(const QVector<RegressionResult>& results);
    };

    /**
     * @brief Base class for performance test cases
     */
    class PerformanceTestCase : public QObject {
        Q_OBJECT
        
    public:
        explicit PerformanceTestCase(QObject* parent = nullptr);
        virtual ~PerformanceTestCase();
        
    protected:
        BenchmarkRunner* benchmarkRunner() { return &m_runner; }
        TestDataGenerator* dataGenerator() { return m_runner.getDataGenerator(); }
        
        // Convenience methods for common test patterns
        void measureFunction(const QString& testName, std::function<void()> func);
        void measureThroughput(const QString& testName, qint64 dataSize, std::function<void()> func);
        
        // Test data helpers
        bool setupTestEnvironment();
        void cleanupTestEnvironment();
        
        // Assertion helpers with performance context
        void QVERIFY_PERFORMANCE(bool condition, const QString& message = QString());
        void QCOMPARE_PERFORMANCE(double actual, double expected, double tolerance, 
                                 const QString& metric = QString());
        
    protected slots:
        virtual void initTestCase();
        virtual void cleanupTestCase();
        virtual void runPerformanceTests() = 0;
        
    private:
        BenchmarkRunner m_runner;
        SystemProfiler::SystemInfo m_systemInfo;
    };

} // namespace PerformanceTest

// Convenience macros for performance testing
#define PERFORMANCE_TEST_MAIN(TestClass) \
    int main(int argc, char *argv[]) \
    { \
        QCoreApplication app(argc, argv); \
        TestClass test; \
        return QTest::qExec(&test, argc, argv); \
    }

#define BENCHMARK_FUNCTION(runner, name, func) \
    runner->registerBenchmark(name, "function", [&]() -> bool { \
        func(); \
        return true; \
    })

#define BENCHMARK_THROUGHPUT(runner, name, dataSize, func) \
    runner->registerBenchmark(name, "throughput", [&]() -> bool { \
        auto timer = PerformanceTest::PerformanceTimer(); \
        timer.start(); \
        func(); \
        timer.stop(); \
        double mbps = (dataSize / 1024.0 / 1024.0) / (timer.elapsedTime() / 1000.0); \
        qDebug() << name << "throughput:" << mbps << "MB/s"; \
        return true; \
    })

Q_DECLARE_METATYPE(PerformanceTest::PerformanceResult)
Q_DECLARE_METATYPE(PerformanceTest::SystemProfiler::SystemInfo)

#endif // PERFORMANCE_TEST_FRAMEWORK_H