#pragma once

#include "test_config.h"
#include "test_base.h"
#include <QObject>
#include <QStringList>
#include <QMap>
#include <QElapsedTimer>
#include <QProcess>
#include <QThreadPool>
#include <QRunnable>
#include <QMutex>
#include <QWaitCondition>

/**
 * @brief Enhanced test runner with configuration support and parallel execution
 * 
 * Provides advanced test execution capabilities including:
 * - Configuration-based test filtering and selection
 * - Parallel test execution with resource management
 * - Comprehensive reporting and statistics
 * - Integration with CI/CD systems
 */
class EnhancedTestRunner : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Test execution result
     */
    struct TestResult {
        QString testName;
        TestConfig::Category category;
        bool passed = false;
        qint64 executionTimeMs = 0;
        QString errorMessage;
        QMap<QString, qint64> performanceMetrics;
        int exitCode = 0;
    };

    /**
     * @brief Test execution summary
     */
    struct ExecutionSummary {
        int totalTests = 0;
        int passedTests = 0;
        int failedTests = 0;
        int skippedTests = 0;
        qint64 totalExecutionTimeMs = 0;
        QList<TestResult> results;
        QStringList failedTestNames;
    };

    explicit EnhancedTestRunner(QObject* parent = nullptr);
    ~EnhancedTestRunner();

    // Configuration
    void loadConfiguration(const QString& configFile = "");
    void setGlobalConfig(const TestConfig::GlobalConfig& config);
    TestConfig::GlobalConfig getGlobalConfig() const;

    // Test discovery and registration
    void discoverTests(const QString& testDirectory = "");
    void registerTestExecutable(const QString& testName, const QString& executablePath);
    void registerTestClass(const QString& testName, TestBase* testInstance);
    QStringList getAvailableTests() const;

    // Test filtering and selection
    void setEnabledCategories(const QStringList& categories);
    void setEnabledTags(const QStringList& tags);
    void setDisabledTests(const QStringList& tests);
    QStringList getFilteredTests() const;

    // Test execution
    bool runAllTests();
    bool runTestsByCategory(TestConfig::Category category);
    bool runTestsByTag(const QString& tag);
    bool runSpecificTests(const QStringList& testNames);
    bool runSingleTest(const QString& testName);

    // Results and reporting
    ExecutionSummary getExecutionSummary() const;
    void generateReport(const QString& outputPath = "");
    void generateJUnitReport(const QString& outputPath = "");
    void generateHtmlReport(const QString& outputPath = "");

    // Parallel execution control
    void setMaxParallelTests(int maxTests);
    void setExecutionMode(TestConfig::ExecutionMode mode);

signals:
    void testStarted(const QString& testName);
    void testFinished(const QString& testName, const TestResult& result);
    void executionStarted(int totalTests);
    void executionFinished(const ExecutionSummary& summary);
    void progressUpdated(int completedTests, int totalTests);

private slots:
    void onTestProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

public:
    struct TestExecutable {
        QString name;
        QString executablePath;
        TestConfig::TestSuiteConfig config;
        TestBase* testInstance = nullptr; // For in-process tests
    };

private:

    TestConfig::GlobalConfig m_globalConfig;
    QMap<QString, TestExecutable> m_availableTests;
    QStringList m_enabledCategories;
    QStringList m_enabledTags;
    QStringList m_disabledTests;
    
    ExecutionSummary m_executionSummary;
    QElapsedTimer m_executionTimer;
    
    // Parallel execution
    QThreadPool* m_threadPool;
    QMutex m_resultsMutex;
    QWaitCondition m_executionComplete;
    int m_runningTests = 0;
    int m_completedTests = 0;
    int m_totalTests = 0;

    // Helper methods
    bool shouldRunTest(const QString& testName) const;
    TestResult executeTest(const QString& testName);
    TestResult executeTestProcess(const TestExecutable& test);
    TestResult executeTestClass(const TestExecutable& test);
    void processTestResult(const TestResult& result);
    void updateExecutionProgress();
    QString formatDuration(qint64 milliseconds) const;
    QString escapeXml(const QString& text) const;
    QString escapeHtml(const QString& text) const;
};

/**
 * @brief Test execution task for parallel execution
 */
class TestExecutionTask : public QObject, public QRunnable {
    Q_OBJECT
public:
    TestExecutionTask(EnhancedTestRunner* runner, const EnhancedTestRunner::TestExecutable& test);
    void run() override;

signals:
    void testCompleted(const EnhancedTestRunner::TestResult& result);

private:
    EnhancedTestRunner* m_runner;
    EnhancedTestRunner::TestExecutable m_test;
};

/**
 * @brief Command-line interface for the enhanced test runner
 */
class TestRunnerCLI {
public:
    static int main(int argc, char* argv[]);
    
private:
    static void printUsage();
    static void printAvailableTests(const EnhancedTestRunner& runner);
    static void printTestCategories();
    static bool parseArguments(int argc, char* argv[], 
                              QString& configFile,
                              QStringList& categories,
                              QStringList& tags,
                              QStringList& tests,
                              QString& reportPath,
                              bool& verbose,
                              bool& parallel);
};