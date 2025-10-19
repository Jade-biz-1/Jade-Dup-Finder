#include "enhanced_test_runner.h"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QThread>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QXmlStreamWriter>
#include <QTextStream>
#include <QDateTime>

EnhancedTestRunner::EnhancedTestRunner(QObject* parent)
    : QObject(parent)
    , m_threadPool(new QThreadPool(this))
{
    // Load default configuration
    TestConfig::instance().loadConfiguration();
    m_globalConfig = TestConfig::instance().globalConfig();
    
    // Set up thread pool
    m_threadPool->setMaxThreadCount(m_globalConfig.maxParallelTests);
}

EnhancedTestRunner::~EnhancedTestRunner() = default;

void EnhancedTestRunner::loadConfiguration(const QString& configFile) {
    TestConfig::instance().loadConfiguration(configFile);
    m_globalConfig = TestConfig::instance().globalConfig();
    m_threadPool->setMaxThreadCount(m_globalConfig.maxParallelTests);
}

void EnhancedTestRunner::setGlobalConfig(const TestConfig::GlobalConfig& config) {
    m_globalConfig = config;
    TestConfig::instance().globalConfig() = config;
    m_threadPool->setMaxThreadCount(config.maxParallelTests);
}

TestConfig::GlobalConfig EnhancedTestRunner::getGlobalConfig() const {
    return m_globalConfig;
}

void EnhancedTestRunner::discoverTests(const QString& testDirectory) {
    QString testDir = testDirectory.isEmpty() ? QCoreApplication::applicationDirPath() : testDirectory;
    
    QDir dir(testDir);
    QStringList filters;
    filters << "*test*" << "*Test*";
    
    QFileInfoList executables = dir.entryInfoList(filters, QDir::Files | QDir::Executable);
    
    for (const QFileInfo& fileInfo : executables) {
        QString testName = fileInfo.baseName();
        registerTestExecutable(testName, fileInfo.absoluteFilePath());
    }
    
    qDebug() << "Discovered" << m_availableTests.size() << "tests in" << testDir;
}

void EnhancedTestRunner::registerTestExecutable(const QString& testName, const QString& executablePath) {
    TestExecutable test;
    test.name = testName;
    test.executablePath = executablePath;
    test.config = TestConfig::instance().getTestSuiteConfig(testName);
    
    // If no configuration exists, create a default one
    if (test.config.name.isEmpty()) {
        test.config.name = testName;
        test.config.category = TestConfig::Category::Unit; // Default category
        test.config.priority = TestConfig::Priority::Medium;
        test.config.timeoutSeconds = m_globalConfig.defaultTimeoutSeconds;
        test.config.enabledByDefault = true;
        test.config.executionMode = m_globalConfig.defaultExecutionMode;
    }
    
    m_availableTests[testName] = test;
}

void EnhancedTestRunner::registerTestClass(const QString& testName, TestBase* testInstance) {
    TestExecutable test;
    test.name = testName;
    test.testInstance = testInstance;
    test.config = testInstance->getTestConfig();
    
    m_availableTests[testName] = test;
}

QStringList EnhancedTestRunner::getAvailableTests() const {
    return m_availableTests.keys();
}

void EnhancedTestRunner::setEnabledCategories(const QStringList& categories) {
    m_enabledCategories = categories;
}

void EnhancedTestRunner::setEnabledTags(const QStringList& tags) {
    m_enabledTags = tags;
}

void EnhancedTestRunner::setDisabledTests(const QStringList& tests) {
    m_disabledTests = tests;
}

QStringList EnhancedTestRunner::getFilteredTests() const {
    QStringList filtered;
    
    for (auto it = m_availableTests.begin(); it != m_availableTests.end(); ++it) {
        if (shouldRunTest(it.key())) {
            filtered.append(it.key());
        }
    }
    
    return filtered;
}

bool EnhancedTestRunner::runAllTests() {
    QStringList testsToRun = getFilteredTests();
    return runSpecificTests(testsToRun);
}

bool EnhancedTestRunner::runTestsByCategory(TestConfig::Category category) {
    QStringList testsToRun;
    
    for (auto it = m_availableTests.begin(); it != m_availableTests.end(); ++it) {
        if (it.value().config.category == category && shouldRunTest(it.key())) {
            testsToRun.append(it.key());
        }
    }
    
    return runSpecificTests(testsToRun);
}

bool EnhancedTestRunner::runTestsByTag(const QString& tag) {
    QStringList testsToRun;
    
    for (auto it = m_availableTests.begin(); it != m_availableTests.end(); ++it) {
        if (it.value().config.tags.contains(tag) && shouldRunTest(it.key())) {
            testsToRun.append(it.key());
        }
    }
    
    return runSpecificTests(testsToRun);
}

bool EnhancedTestRunner::runSpecificTests(const QStringList& testNames) {
    if (testNames.isEmpty()) {
        qWarning() << "No tests to run";
        return true;
    }
    
    // Reset execution state
    m_executionSummary = ExecutionSummary();
    m_runningTests = 0;
    m_completedTests = 0;
    m_totalTests = testNames.size();
    
    emit executionStarted(m_totalTests);
    m_executionTimer.start();
    
    qDebug() << "Starting execution of" << m_totalTests << "tests";
    
    // Execute tests based on execution mode
    if (m_globalConfig.defaultExecutionMode == TestConfig::ExecutionMode::Sequential) {
        // Sequential execution
        for (const QString& testName : testNames) {
            if (!m_availableTests.contains(testName)) {
                qWarning() << "Test not found:" << testName;
                continue;
            }
            
            TestResult result = executeTest(testName);
            processTestResult(result);
            
            if (m_globalConfig.stopOnFirstFailure && !result.passed) {
                qDebug() << "Stopping execution due to test failure:" << testName;
                break;
            }
        }
    } else {
        // Parallel execution (simplified implementation)
        for (const QString& testName : testNames) {
            if (!m_availableTests.contains(testName)) {
                qWarning() << "Test not found:" << testName;
                continue;
            }
            
            TestResult result = executeTest(testName);
            processTestResult(result);
        }
    }
    
    // Finalize execution
    m_executionSummary.totalExecutionTimeMs = m_executionTimer.elapsed();
    
    qDebug() << "Test execution completed:"
             << m_executionSummary.passedTests << "passed,"
             << m_executionSummary.failedTests << "failed,"
             << m_executionSummary.skippedTests << "skipped";
    
    emit executionFinished(m_executionSummary);
    
    return m_executionSummary.failedTests == 0;
}

bool EnhancedTestRunner::runSingleTest(const QString& testName) {
    return runSpecificTests(QStringList() << testName);
}

EnhancedTestRunner::ExecutionSummary EnhancedTestRunner::getExecutionSummary() const {
    return m_executionSummary;
}

void EnhancedTestRunner::setMaxParallelTests(int maxTests) {
    m_globalConfig.maxParallelTests = maxTests;
    m_threadPool->setMaxThreadCount(maxTests);
}

void EnhancedTestRunner::setExecutionMode(TestConfig::ExecutionMode mode) {
    m_globalConfig.defaultExecutionMode = mode;
}

bool EnhancedTestRunner::shouldRunTest(const QString& testName) const {
    // Check if test is explicitly disabled
    if (m_disabledTests.contains(testName)) {
        return false;
    }
    
    if (!m_availableTests.contains(testName)) {
        return false;
    }
    
    const TestExecutable& test = m_availableTests[testName];
    
    // Check category filtering
    if (!m_enabledCategories.isEmpty()) {
        QString categoryStr = TestConfig::categoryToString(test.config.category);
        if (!m_enabledCategories.contains(categoryStr)) {
            return false;
        }
    }
    
    // Check tag filtering
    if (!m_enabledTags.isEmpty()) {
        bool hasMatchingTag = false;
        for (const QString& tag : test.config.tags) {
            if (m_enabledTags.contains(tag)) {
                hasMatchingTag = true;
                break;
            }
        }
        if (!hasMatchingTag) {
            return false;
        }
    }
    
    return test.config.enabledByDefault;
}

EnhancedTestRunner::TestResult EnhancedTestRunner::executeTest(const QString& testName) {
    const TestExecutable& test = m_availableTests[testName];
    
    emit testStarted(testName);
    
    TestResult result;
    result.testName = testName;
    result.category = test.config.category;
    
    QElapsedTimer timer;
    timer.start();
    
    if (test.testInstance) {
        result = executeTestClass(test);
    } else {
        result = executeTestProcess(test);
    }
    
    result.executionTimeMs = timer.elapsed();
    
    emit testFinished(testName, result);
    
    return result;
}

EnhancedTestRunner::TestResult EnhancedTestRunner::executeTestProcess(const TestExecutable& test) {
    TestResult result;
    result.testName = test.name;
    result.category = test.config.category;
    
    QProcess process;
    process.setProgram(test.executablePath);
    
    QElapsedTimer timer;
    timer.start();
    
    process.start();
    
    if (!process.waitForStarted(5000)) {
        result.passed = false;
        result.errorMessage = "Failed to start test process";
        result.exitCode = -1;
        return result;
    }
    
    if (!process.waitForFinished(test.config.timeoutSeconds * 1000)) {
        process.kill();
        result.passed = false;
        result.errorMessage = "Test timed out";
        result.exitCode = -2;
        return result;
    }
    
    result.executionTimeMs = timer.elapsed();
    result.exitCode = process.exitCode();
    result.passed = (result.exitCode == 0);
    
    if (!result.passed) {
        result.errorMessage = process.readAllStandardError();
    }
    
    return result;
}

EnhancedTestRunner::TestResult EnhancedTestRunner::executeTestClass(const TestExecutable& test) {
    TestResult result;
    result.testName = test.name;
    result.category = test.config.category;
    
    // For in-process test execution, we would use QTest::qExec
    // This is a simplified implementation
    if (test.testInstance) {
        QElapsedTimer timer;
        timer.start();
        
        int exitCode = QTest::qExec(test.testInstance);
        
        result.executionTimeMs = timer.elapsed();
        result.exitCode = exitCode;
        result.passed = (exitCode == 0);
        
        if (!result.passed) {
            result.errorMessage = "Test failed (see detailed output)";
        }
    }
    
    return result;
}

void EnhancedTestRunner::processTestResult(const TestResult& result) {
    QMutexLocker locker(&m_resultsMutex);
    
    m_executionSummary.results.append(result);
    m_executionSummary.totalTests++;
    
    if (result.passed) {
        m_executionSummary.passedTests++;
    } else {
        m_executionSummary.failedTests++;
        m_executionSummary.failedTestNames.append(result.testName);
    }
    
    m_completedTests++;
    emit progressUpdated(m_completedTests, m_totalTests);
}

void EnhancedTestRunner::generateReport(const QString& outputPath) {
    QString reportPath = outputPath;
    if (reportPath.isEmpty()) {
        QString reportDir = m_globalConfig.reportOutputDirectory;
        QDir().mkpath(reportDir);
        reportPath = QDir(reportDir).absoluteFilePath(
            QString("test_report_%1.json")
            .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"))
        );
    }
    
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["platform"] = TestConfig::instance().getPlatformName();
    report["totalTests"] = m_executionSummary.totalTests;
    report["passedTests"] = m_executionSummary.passedTests;
    report["failedTests"] = m_executionSummary.failedTests;
    report["skippedTests"] = m_executionSummary.skippedTests;
    report["executionTimeMs"] = m_executionSummary.totalExecutionTimeMs;
    report["successRate"] = m_executionSummary.totalTests > 0 ? 
        (double)m_executionSummary.passedTests / m_executionSummary.totalTests * 100.0 : 0.0;
    
    QJsonArray results;
    for (const TestResult& result : m_executionSummary.results) {
        QJsonObject resultObj;
        resultObj["name"] = result.testName;
        resultObj["category"] = TestConfig::categoryToString(result.category);
        resultObj["passed"] = result.passed;
        resultObj["executionTimeMs"] = result.executionTimeMs;
        resultObj["exitCode"] = result.exitCode;
        if (!result.errorMessage.isEmpty()) {
            resultObj["errorMessage"] = result.errorMessage;
        }
        results.append(resultObj);
    }
    report["results"] = results;
    
    QFile file(reportPath);
    if (file.open(QIODevice::WriteOnly)) {
        QJsonDocument doc(report);
        file.write(doc.toJson());
        qDebug() << "Test report generated:" << reportPath;
    } else {
        qWarning() << "Failed to write test report:" << reportPath;
    }
}

QString EnhancedTestRunner::formatDuration(qint64 milliseconds) const {
    if (milliseconds < 1000) {
        return QString("%1ms").arg(milliseconds);
    } else if (milliseconds < 60000) {
        return QString("%1.%2s").arg(milliseconds / 1000).arg((milliseconds % 1000) / 100);
    } else {
        int minutes = milliseconds / 60000;
        int seconds = (milliseconds % 60000) / 1000;
        return QString("%1m %2s").arg(minutes).arg(seconds);
    }
}