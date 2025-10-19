#pragma once

#include "test_config.h"
#include <QObject>
#include <QTest>
#include <QString>
#include <QStringList>
#include <QElapsedTimer>
#include <QDebug>
#include <QDir>
#include <QStandardPaths>

/**
 * @brief Base class for all DupFinder tests
 * 
 * Provides standardized test infrastructure, naming conventions, and common utilities.
 * All test classes should inherit from this base class to ensure consistency.
 */
class TestBase : public QObject {
    Q_OBJECT
    
    friend class EnhancedTestRunner;

public:
    explicit TestBase(QObject* parent = nullptr);
    virtual ~TestBase() = default;

    // Test identification
    virtual QString testSuiteName() const = 0;
    virtual TestConfig::Category testCategory() const = 0;
    virtual TestConfig::Priority testPriority() const { return TestConfig::Priority::Medium; }
    virtual QStringList testTags() const { return {}; }
    
    // Test lifecycle hooks
    virtual void globalSetUp() {}
    virtual void globalTearDown() {}

protected slots:
    // Standard Qt Test Framework hooks
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

protected:
    // Standardized assertion macros with enhanced reporting
    #define TEST_VERIFY_WITH_MSG(condition, message) \
        do { \
            if (!(condition)) { \
                logTestFailure(__FUNCTION__, __LINE__, #condition, message); \
                QVERIFY2(condition, message.toUtf8().constData()); \
            } else { \
                logTestSuccess(__FUNCTION__, #condition); \
            } \
        } while(0)

    #define TEST_COMPARE_WITH_MSG(actual, expected, message) \
        do { \
            if (!QTest::qCompare(actual, expected, #actual, #expected, __FILE__, __LINE__)) { \
                logTestFailure(__FUNCTION__, __LINE__, \
                    QString("%1 != %2").arg(QTest::toString(actual)).arg(QTest::toString(expected)), \
                    message); \
                QFAIL(message.toUtf8().constData()); \
            } else { \
                logTestSuccess(__FUNCTION__, QString("%1 == %2").arg(#actual).arg(#expected)); \
            } \
        } while(0)

    // Enhanced test utilities
    void logTestStep(const QString& step);
    void logTestInfo(const QString& info);
    void logTestWarning(const QString& warning);
    void logTestSuccess(const QString& function, const QString& condition);
    void logTestFailure(const QString& function, int line, const QString& condition, const QString& message);
    
    // Performance measurement
    void startPerformanceMeasurement(const QString& operationName);
    qint64 stopPerformanceMeasurement(const QString& operationName);
    void recordPerformanceMetric(const QString& metricName, qint64 value, const QString& unit = "ms");
    
    // Test data management
    QString createTestDirectory(const QString& suffix = "");
    QString createTestFile(const QString& fileName, const QString& content = "");
    void cleanupTestData();
    
    // Test environment utilities
    bool isRunningInCI() const;
    QString getPlatformName() const;
    void skipIfPlatformNot(const QStringList& supportedPlatforms);
    void skipIfCI(const QString& reason = "Test not suitable for CI environment");
    void skipIfNotCI(const QString& reason = "Test only runs in CI environment");
    
    // Configuration access
    TestConfig::TestSuiteConfig getTestConfig() const;
    bool shouldRunTest() const;

private:
    QString m_testDataDirectory;
    QStringList m_createdFiles;
    QStringList m_createdDirectories;
    QMap<QString, QElapsedTimer> m_performanceTimers;
    QMap<QString, qint64> m_performanceMetrics;
    
    void setupTestEnvironment();
    void cleanupTestEnvironment();
    void registerTestSuite();
};

/**
 * @brief Macro for declaring a test class with standardized naming
 */
#define DECLARE_TEST_CLASS(className, category, priority, ...) \
    class className : public TestBase { \
        Q_OBJECT \
    public: \
        explicit className(QObject* parent = nullptr) : TestBase(parent) {} \
        QString testSuiteName() const override { return #className; } \
        TestConfig::Category testCategory() const override { return TestConfig::Category::category; } \
        TestConfig::Priority testPriority() const override { return TestConfig::Priority::priority; } \
        QStringList testTags() const override { return QStringList{__VA_ARGS__}; } \
    private slots:

/**
 * @brief Macro for ending a test class declaration
 */
#define END_TEST_CLASS() \
    };

/**
 * @brief Macro for running a test class
 */
#define RUN_TEST_CLASS(className) \
    do { \
        className test; \
        if (test.shouldRunTest()) { \
            QTest::qExec(&test); \
        } else { \
            qDebug() << "Skipping test class:" << #className << "(disabled by configuration)"; \
        } \
    } while(0)

/**
 * @brief Macro for test method naming convention
 * Format: test_<component>_<scenario>_<expectedResult>
 */
#define TEST_METHOD(methodName) void methodName()

/**
 * @brief Macro for benchmark test methods
 */
#define BENCHMARK_METHOD(methodName) void methodName()

/**
 * @brief Macro for data-driven test methods
 */
#define DATA_DRIVEN_TEST_METHOD(methodName) \
    void methodName(); \
    void methodName##_data()

/**
 * @brief Test naming convention helpers
 */
namespace TestNaming {
    /**
     * @brief Generate standardized test method name
     * @param component Component being tested (e.g., "FileScanner", "HashCalculator")
     * @param scenario Test scenario (e.g., "EmptyDirectory", "LargeFile")
     * @param expectedResult Expected outcome (e.g., "ReturnsEmpty", "ThrowsException")
     * @return Formatted test method name
     */
    QString generateTestMethodName(const QString& component, const QString& scenario, const QString& expectedResult);
    
    /**
     * @brief Generate standardized test data name
     * @param testCase Test case description
     * @param variant Variant identifier (optional)
     * @return Formatted test data name
     */
    QString generateTestDataName(const QString& testCase, const QString& variant = "");
    
    /**
     * @brief Validate test method name follows conventions
     * @param methodName Method name to validate
     * @return True if name follows conventions
     */
    bool validateTestMethodName(const QString& methodName);
}

/**
 * @brief Test execution statistics
 */
struct TestExecutionStats {
    QString testSuiteName;
    TestConfig::Category category;
    int totalTests = 0;
    int passedTests = 0;
    int failedTests = 0;
    int skippedTests = 0;
    qint64 executionTimeMs = 0;
    QMap<QString, qint64> performanceMetrics;
    QStringList failureMessages;
};

/**
 * @brief Test execution coordinator
 */
class TestExecutionCoordinator : public QObject {
    Q_OBJECT

public:
    static TestExecutionCoordinator& instance();
    
    void registerTestExecution(const TestExecutionStats& stats);
    QList<TestExecutionStats> getExecutionStats() const;
    void generateExecutionReport(const QString& outputPath = "");
    void clearStats();

signals:
    void testSuiteStarted(const QString& suiteName);
    void testSuiteFinished(const QString& suiteName, const TestExecutionStats& stats);

private:
    TestExecutionCoordinator() = default;
    QList<TestExecutionStats> m_executionStats;
};