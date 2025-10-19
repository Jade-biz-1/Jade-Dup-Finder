#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <memory>
#include <functional>

// Forward declarations
class TestEnvironment;
class TestReporting;
class TestSuite;

/**
 * @brief Test execution categories for organizing and filtering tests
 */
enum class TestCategory {
    Unit,           // Fast, isolated component tests
    Integration,    // Component interaction tests
    UI,            // User interface tests
    EndToEnd,      // Complete workflow tests
    Performance,   // Performance and benchmarking tests
    Accessibility, // Accessibility compliance tests
    CrossPlatform, // Platform-specific behavior tests
    Security,      // Security and safety tests
    Visual,        // Visual regression tests
    All            // Run all categories
};

/**
 * @brief Configuration for test suite execution
 */
struct TestSuiteConfig {
    QStringList enabledCategories;
    QStringList disabledTests;
    int timeoutSeconds = 300;
    bool parallelExecution = true;
    int maxParallelThreads = 4;
    QString outputDirectory = "test_results";
    bool generateHtmlReport = true;
    bool generateJunitXml = true;
    bool captureScreenshots = true;
    double visualThreshold = 0.95;
    bool stopOnFirstFailure = false;
    bool verboseOutput = false;
    QString logLevel = "INFO";
};

/**
 * @brief Individual test failure information
 */
struct TestFailure {
    QString testName;
    QString category;
    QString errorMessage;
    QString stackTrace;
    QString screenshotPath;
    qint64 timestamp;
    QMap<QString, QVariant> additionalData;
};

/**
 * @brief Test warning information
 */
struct TestWarning {
    QString testName;
    QString message;
    QString category;
    qint64 timestamp;
};

/**
 * @brief Comprehensive test execution results
 */
struct TestResults {
    int totalTests = 0;
    int passedTests = 0;
    int failedTests = 0;
    int skippedTests = 0;
    qint64 executionTimeMs = 0;
    double codeCoverage = 0.0;
    
    QList<TestFailure> failures;
    QList<TestWarning> warnings;
    QMap<QString, QVariant> metrics;
    QMap<QString, qint64> categoryExecutionTimes;
    
    // Helper methods
    bool hasFailures() const { return failedTests > 0; }
    double successRate() const { 
        return totalTests > 0 ? (double)passedTests / totalTests * 100.0 : 0.0; 
    }
    QString summary() const {
        return QString("Tests: %1, Passed: %2, Failed: %3, Skipped: %4, Success Rate: %5%")
               .arg(totalTests).arg(passedTests).arg(failedTests).arg(skippedTests)
               .arg(successRate(), 0, 'f', 1);
    }
};

/**
 * @brief Abstract base class for test suites
 */
class TestSuite : public QObject {
    Q_OBJECT

public:
    explicit TestSuite(const QString& name, TestCategory category, QObject* parent = nullptr);
    virtual ~TestSuite() = default;

    // Test suite information
    QString name() const { return m_name; }
    TestCategory category() const { return m_category; }
    QStringList testNames() const { return m_testNames; }
    
    // Test execution
    virtual bool runAllTests() = 0;
    virtual bool runTest(const QString& testName) = 0;
    virtual void setUp() {}
    virtual void tearDown() {}
    
    // Results
    TestResults getResults() const { return m_results; }
    void clearResults();

signals:
    void testStarted(const QString& testName);
    void testCompleted(const QString& testName, bool passed);
    void testFailed(const QString& testName, const QString& error);
    void suiteCompleted(bool allPassed);

protected:
    void addTest(const QString& testName);
    void recordTestResult(const QString& testName, bool passed, const QString& error = QString());
    void recordTestWarning(const QString& testName, const QString& warning);
    void addMetric(const QString& name, const QVariant& value);

private:
    QString m_name;
    TestCategory m_category;
    QStringList m_testNames;
    TestResults m_results;
    QElapsedTimer m_timer;
};

/**
 * @brief Central test execution and coordination system
 */
class TestHarness : public QObject {
    Q_OBJECT

public:
    explicit TestHarness(QObject* parent = nullptr);
    ~TestHarness();

    // Configuration
    void loadConfiguration(const QString& configFile);
    void setConfiguration(const TestSuiteConfig& config);
    TestSuiteConfig getConfiguration() const { return m_config; }

    // Test suite management
    void registerTestSuite(std::shared_ptr<TestSuite> suite);
    void unregisterTestSuite(const QString& suiteName);
    QStringList getRegisteredSuites() const;
    
    // Environment and reporting
    void setTestEnvironment(std::shared_ptr<TestEnvironment> env);
    void setReportGenerator(std::shared_ptr<TestReporting> reporter);
    
    // Test execution
    bool runTestSuite(const QString& suiteName);
    bool runTestCategory(TestCategory category);
    bool runSpecificTest(const QString& suiteName, const QString& testName);
    bool runAllTests();
    
    // Parallel execution
    void setParallelExecution(bool enabled, int maxThreads = 0);
    bool isParallelExecutionEnabled() const { return m_config.parallelExecution; }
    
    // Timeout and control
    void setTimeout(int seconds);
    void stopExecution();
    bool isExecutionStopped() const { return m_stopRequested; }
    
    // Results and reporting
    TestResults getResults() const { return m_overallResults; }
    TestResults getResults(const QString& suiteName) const;
    bool hasFailures() const { return m_overallResults.hasFailures(); }
    void generateReport(const QString& outputPath = QString());
    
    // Utilities
    static QString categoryToString(TestCategory category);
    static TestCategory stringToCategory(const QString& categoryStr);

signals:
    void executionStarted();
    void executionCompleted(const TestResults& results);
    void suiteStarted(const QString& suiteName);
    void suiteCompleted(const QString& suiteName, const TestResults& results);
    void testStarted(const QString& suiteName, const QString& testName);
    void testCompleted(const QString& suiteName, const QString& testName, bool passed);
    void progressUpdated(int current, int total);

private slots:
    void onSuiteCompleted(bool allPassed);
    void onTestStarted(const QString& testName);
    void onTestCompleted(const QString& testName, bool passed);
    void onTestFailed(const QString& testName, const QString& error);

private:
    // Execution methods
    bool executeTestSuite(std::shared_ptr<TestSuite> suite);
    bool executeTestSuiteParallel(const QList<std::shared_ptr<TestSuite>>& suites);
    void aggregateResults();
    void setupTestEnvironment();
    void cleanupTestEnvironment();
    
    // Configuration and state
    TestSuiteConfig m_config;
    QMap<QString, std::shared_ptr<TestSuite>> m_testSuites;
    std::shared_ptr<TestEnvironment> m_testEnvironment;
    std::shared_ptr<TestReporting> m_reportGenerator;
    
    // Execution state
    TestResults m_overallResults;
    QMap<QString, TestResults> m_suiteResults;
    QElapsedTimer m_executionTimer;
    bool m_stopRequested = false;
    
    // Threading
    QMutex m_mutex;
    QWaitCondition m_waitCondition;
    QList<QThread*> m_workerThreads;
};

/**
 * @brief Utility macros for test implementation
 */
#define TEST_SUITE(className, suiteName, category) \
    class className : public TestSuite { \
        Q_OBJECT \
    public: \
        className(QObject* parent = nullptr) : TestSuite(suiteName, category, parent) { \
            setupTests(); \
        } \
    private: \
        void setupTests(); \
        bool runAllTests() override; \
        bool runTest(const QString& testName) override; \
    };

#define TEST_METHOD(methodName) \
    void methodName(); \
    bool test_##methodName() { \
        try { \
            methodName(); \
            return true; \
        } catch (const std::exception& e) { \
            recordTestResult(#methodName, false, QString::fromStdString(e.what())); \
            return false; \
        } catch (...) { \
            recordTestResult(#methodName, false, "Unknown exception"); \
            return false; \
        } \
    }

#define REGISTER_TEST(methodName) \
    addTest(#methodName);

#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(QString("Assertion failed: %1").arg(message).toStdString()); \
    }

#define TEST_VERIFY(condition) \
    TEST_ASSERT(condition, #condition)

#define TEST_COMPARE(actual, expected) \
    if ((actual) != (expected)) { \
        throw std::runtime_error(QString("Comparison failed: %1 != %2") \
                                .arg(QVariant(actual).toString()) \
                                .arg(QVariant(expected).toString()).toStdString()); \
    }