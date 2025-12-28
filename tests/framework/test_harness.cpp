#include "test_harness.h"
#include "test_environment.h"
#include "test_reporting.h"
#include <QCoreApplication>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QStandardPaths>
#include <QDebug>
#include <QThreadPool>
#include <QtConcurrent>
#include <algorithm>

// TestSuite Implementation
TestSuite::TestSuite(const QString& name, TestCategory category, QObject* parent)
    : QObject(parent)
    , m_name(name)
    , m_category(category)
{
}

void TestSuite::clearResults() {
    m_results = TestResults();
}

void TestSuite::addTest(const QString& testName) {
    if (!m_testNames.contains(testName)) {
        m_testNames.append(testName);
    }
}

void TestSuite::recordTestResult(const QString& testName, bool passed, const QString& error) {
    m_results.totalTests++;
    
    if (passed) {
        m_results.passedTests++;
        emit testCompleted(testName, true);
    } else {
        m_results.failedTests++;
        
        TestFailure failure;
        failure.testName = testName;
        failure.category = TestHarness::categoryToString(m_category);
        failure.errorMessage = error;
        failure.timestamp = QDateTime::currentMSecsSinceEpoch();
        m_results.failures.append(failure);
        
        emit testFailed(testName, error);
        emit testCompleted(testName, false);
    }
}

void TestSuite::recordTestWarning(const QString& testName, const QString& warning) {
    TestWarning testWarning;
    testWarning.testName = testName;
    testWarning.message = warning;
    testWarning.category = TestHarness::categoryToString(m_category);
    testWarning.timestamp = QDateTime::currentMSecsSinceEpoch();
    m_results.warnings.append(testWarning);
}

void TestSuite::addMetric(const QString& name, const QVariant& value) {
    m_results.metrics[name] = value;
}

// TestHarness Implementation
TestHarness::TestHarness(QObject* parent)
    : QObject(parent)
{
    // Set default configuration
    m_config.enabledCategories << "Unit" << "Integration" << "UI";
    m_config.outputDirectory = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/cloneclean_tests";
    
    // Ensure output directory exists
    QDir().mkpath(m_config.outputDirectory);
}

TestHarness::~TestHarness() {
    // Clean up worker threads
    for (QThread* thread : m_workerThreads) {
        if (thread->isRunning()) {
            thread->quit();
            thread->wait(5000);
        }
        delete thread;
    }
}

void TestHarness::loadConfiguration(const QString& configFile) {
    QFile file(configFile);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open test configuration file:" << configFile;
        return;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError) {
        qWarning() << "Failed to parse test configuration:" << error.errorString();
        return;
    }
    
    QJsonObject config = doc.object();
    
    // Load configuration values
    if (config.contains("enabledCategories")) {
        m_config.enabledCategories.clear();
        QJsonArray categories = config["enabledCategories"].toArray();
        for (const QJsonValue& value : categories) {
            m_config.enabledCategories << value.toString();
        }
    }
    
    if (config.contains("disabledTests")) {
        m_config.disabledTests.clear();
        QJsonArray tests = config["disabledTests"].toArray();
        for (const QJsonValue& value : tests) {
            m_config.disabledTests << value.toString();
        }
    }
    
    m_config.timeoutSeconds = config.value("timeoutSeconds").toInt(m_config.timeoutSeconds);
    m_config.parallelExecution = config.value("parallelExecution").toBool(m_config.parallelExecution);
    m_config.maxParallelThreads = config.value("maxParallelThreads").toInt(m_config.maxParallelThreads);
    m_config.outputDirectory = config.value("outputDirectory").toString(m_config.outputDirectory);
    m_config.generateHtmlReport = config.value("generateHtmlReport").toBool(m_config.generateHtmlReport);
    m_config.generateJunitXml = config.value("generateJunitXml").toBool(m_config.generateJunitXml);
    m_config.captureScreenshots = config.value("captureScreenshots").toBool(m_config.captureScreenshots);
    m_config.visualThreshold = config.value("visualThreshold").toDouble(m_config.visualThreshold);
    m_config.stopOnFirstFailure = config.value("stopOnFirstFailure").toBool(m_config.stopOnFirstFailure);
    m_config.verboseOutput = config.value("verboseOutput").toBool(m_config.verboseOutput);
    m_config.logLevel = config.value("logLevel").toString(m_config.logLevel);
}

void TestHarness::setConfiguration(const TestSuiteConfig& config) {
    m_config = config;
    
    // Ensure output directory exists
    QDir().mkpath(m_config.outputDirectory);
}

void TestHarness::registerTestSuite(std::shared_ptr<TestSuite> suite) {
    if (!suite) {
        qWarning() << "Cannot register null test suite";
        return;
    }
    
    QString suiteName = suite->name();
    if (m_testSuites.contains(suiteName)) {
        qWarning() << "Test suite already registered:" << suiteName;
        return;
    }
    
    m_testSuites[suiteName] = suite;
    
    // Connect signals
    connect(suite.get(), &TestSuite::testStarted, this, [this, suiteName](const QString& testName) {
        emit testStarted(suiteName, testName);
    });
    
    connect(suite.get(), &TestSuite::testCompleted, this, [this, suiteName](const QString& testName, bool passed) {
        emit testCompleted(suiteName, testName, passed);
    });
    
    connect(suite.get(), &TestSuite::testFailed, this, [this, suiteName](const QString& testName, const QString& error) {
        Q_UNUSED(suiteName)
        Q_UNUSED(testName)
        Q_UNUSED(error)
        // Error already handled by testCompleted signal
    });
    
    connect(suite.get(), &TestSuite::suiteCompleted, this, &TestHarness::onSuiteCompleted);
    
    qDebug() << "Registered test suite:" << suiteName << "Category:" << categoryToString(suite->category());
}

void TestHarness::unregisterTestSuite(const QString& suiteName) {
    if (m_testSuites.contains(suiteName)) {
        m_testSuites.remove(suiteName);
        qDebug() << "Unregistered test suite:" << suiteName;
    }
}

QStringList TestHarness::getRegisteredSuites() const {
    return m_testSuites.keys();
}

void TestHarness::setTestEnvironment(std::shared_ptr<TestEnvironment> env) {
    m_testEnvironment = env;
}

void TestHarness::setReportGenerator(std::shared_ptr<TestReporting> reporter) {
    m_reportGenerator = reporter;
}

bool TestHarness::runTestSuite(const QString& suiteName) {
    if (!m_testSuites.contains(suiteName)) {
        qWarning() << "Test suite not found:" << suiteName;
        return false;
    }
    
    auto suite = m_testSuites[suiteName];
    return executeTestSuite(suite);
}

bool TestHarness::runTestCategory(TestCategory category) {
    QList<std::shared_ptr<TestSuite>> suitesToRun;
    
    for (auto it = m_testSuites.begin(); it != m_testSuites.end(); ++it) {
        if (it.value()->category() == category || category == TestCategory::All) {
            QString categoryStr = categoryToString(it.value()->category());
            if (m_config.enabledCategories.contains(categoryStr) || m_config.enabledCategories.contains("All")) {
                suitesToRun.append(it.value());
            }
        }
    }
    
    if (suitesToRun.isEmpty()) {
        qWarning() << "No test suites found for category:" << categoryToString(category);
        return false;
    }
    
    if (m_config.parallelExecution && suitesToRun.size() > 1) {
        return executeTestSuiteParallel(suitesToRun);
    } else {
        bool allPassed = true;
        for (auto suite : suitesToRun) {
            if (!executeTestSuite(suite)) {
                allPassed = false;
                if (m_config.stopOnFirstFailure) {
                    break;
                }
            }
        }
        return allPassed;
    }
}

bool TestHarness::runSpecificTest(const QString& suiteName, const QString& testName) {
    if (!m_testSuites.contains(suiteName)) {
        qWarning() << "Test suite not found:" << suiteName;
        return false;
    }
    
    auto suite = m_testSuites[suiteName];
    if (!suite->testNames().contains(testName)) {
        qWarning() << "Test not found:" << testName << "in suite:" << suiteName;
        return false;
    }
    
    setupTestEnvironment();
    
    emit executionStarted();
    emit suiteStarted(suiteName);
    emit testStarted(suiteName, testName);
    
    m_executionTimer.start();
    
    suite->setUp();
    bool result = suite->runTest(testName);
    suite->tearDown();
    
    TestResults suiteResults = suite->getResults();
    suiteResults.executionTimeMs = m_executionTimer.elapsed();
    m_suiteResults[suiteName] = suiteResults;
    
    emit suiteCompleted(suiteName, suiteResults);
    
    aggregateResults();
    
    emit executionCompleted(m_overallResults);
    
    cleanupTestEnvironment();
    
    return result;
}

bool TestHarness::runAllTests() {
    return runTestCategory(TestCategory::All);
}

void TestHarness::setParallelExecution(bool enabled, int maxThreads) {
    m_config.parallelExecution = enabled;
    if (maxThreads > 0) {
        m_config.maxParallelThreads = maxThreads;
    }
}

void TestHarness::setTimeout(int seconds) {
    m_config.timeoutSeconds = seconds;
}

void TestHarness::stopExecution() {
    m_stopRequested = true;
}

TestResults TestHarness::getResults(const QString& suiteName) const {
    return m_suiteResults.value(suiteName, TestResults());
}

void TestHarness::generateReport(const QString& outputPath) {
    if (!m_reportGenerator) {
        qWarning() << "No report generator configured";
        return;
    }
    
    QString reportPath = outputPath.isEmpty() ? m_config.outputDirectory : outputPath;
    m_reportGenerator->generateReport(m_overallResults, m_suiteResults, reportPath);
}

QString TestHarness::categoryToString(TestCategory category) {
    switch (category) {
        case TestCategory::Unit: return "Unit";
        case TestCategory::Integration: return "Integration";
        case TestCategory::UI: return "UI";
        case TestCategory::EndToEnd: return "EndToEnd";
        case TestCategory::Performance: return "Performance";
        case TestCategory::Accessibility: return "Accessibility";
        case TestCategory::CrossPlatform: return "CrossPlatform";
        case TestCategory::Security: return "Security";
        case TestCategory::Visual: return "Visual";
        case TestCategory::All: return "All";
    }
    return "Unknown";
}

TestCategory TestHarness::stringToCategory(const QString& categoryStr) {
    if (categoryStr == "Unit") return TestCategory::Unit;
    if (categoryStr == "Integration") return TestCategory::Integration;
    if (categoryStr == "UI") return TestCategory::UI;
    if (categoryStr == "EndToEnd") return TestCategory::EndToEnd;
    if (categoryStr == "Performance") return TestCategory::Performance;
    if (categoryStr == "Accessibility") return TestCategory::Accessibility;
    if (categoryStr == "CrossPlatform") return TestCategory::CrossPlatform;
    if (categoryStr == "Security") return TestCategory::Security;
    if (categoryStr == "Visual") return TestCategory::Visual;
    if (categoryStr == "All") return TestCategory::All;
    return TestCategory::Unit; // Default
}

// Private methods
bool TestHarness::executeTestSuite(std::shared_ptr<TestSuite> suite) {
    if (!suite) {
        return false;
    }
    
    QString suiteName = suite->name();
    
    setupTestEnvironment();
    
    emit suiteStarted(suiteName);
    
    QElapsedTimer suiteTimer;
    suiteTimer.start();
    
    suite->clearResults();
    suite->setUp();
    
    bool result = suite->runAllTests();
    
    suite->tearDown();
    
    TestResults suiteResults = suite->getResults();
    suiteResults.executionTimeMs = suiteTimer.elapsed();
    m_suiteResults[suiteName] = suiteResults;
    
    emit suiteCompleted(suiteName, suiteResults);
    
    cleanupTestEnvironment();
    
    return result;
}

bool TestHarness::executeTestSuiteParallel(const QList<std::shared_ptr<TestSuite>>& suites) {
    if (suites.isEmpty()) {
        return true;
    }
    
    emit executionStarted();
    m_executionTimer.start();
    
    // Set up thread pool
    QThreadPool* threadPool = QThreadPool::globalInstance();
    int originalMaxThreadCount = threadPool->maxThreadCount();
    threadPool->setMaxThreadCount(qMin(m_config.maxParallelThreads, suites.size()));
    
    // Execute suites in parallel
    QList<QFuture<bool>> futures;
    for (auto suite : suites) {
        QFuture<bool> future = QtConcurrent::run([this, suite]() {
            return executeTestSuite(suite);
        });
        futures.append(future);
    }
    
    // Wait for all to complete
    bool allPassed = true;
    for (auto& future : futures) {
        future.waitForFinished();
        if (!future.result()) {
            allPassed = false;
        }
    }
    
    // Restore thread pool
    threadPool->setMaxThreadCount(originalMaxThreadCount);
    
    aggregateResults();
    
    m_overallResults.executionTimeMs = m_executionTimer.elapsed();
    emit executionCompleted(m_overallResults);
    
    return allPassed;
}

void TestHarness::aggregateResults() {
    m_overallResults = TestResults();
    
    for (auto it = m_suiteResults.begin(); it != m_suiteResults.end(); ++it) {
        const TestResults& suiteResult = it.value();
        
        m_overallResults.totalTests += suiteResult.totalTests;
        m_overallResults.passedTests += suiteResult.passedTests;
        m_overallResults.failedTests += suiteResult.failedTests;
        m_overallResults.skippedTests += suiteResult.skippedTests;
        
        m_overallResults.failures.append(suiteResult.failures);
        m_overallResults.warnings.append(suiteResult.warnings);
        
        // Merge metrics
        for (auto metricIt = suiteResult.metrics.begin(); metricIt != suiteResult.metrics.end(); ++metricIt) {
            m_overallResults.metrics[it.key() + "." + metricIt.key()] = metricIt.value();
        }
    }
}

void TestHarness::setupTestEnvironment() {
    if (m_testEnvironment) {
        m_testEnvironment->setupTestEnvironment();
    }
}

void TestHarness::cleanupTestEnvironment() {
    if (m_testEnvironment) {
        m_testEnvironment->cleanupTestEnvironment();
    }
}

// Slot implementations
void TestHarness::onSuiteCompleted(bool allPassed) {
    Q_UNUSED(allPassed)
    // Suite completion is handled in executeTestSuite
}

void TestHarness::onTestStarted(const QString& testName) {
    Q_UNUSED(testName)
    // Test start is handled by suite-specific signals
}

void TestHarness::onTestCompleted(const QString& testName, bool passed) {
    Q_UNUSED(testName)
    Q_UNUSED(passed)
    // Test completion is handled by suite-specific signals
}

void TestHarness::onTestFailed(const QString& testName, const QString& error) {
    Q_UNUSED(testName)
    Q_UNUSED(error)
    // Test failure is handled by suite-specific signals
}

#include "test_harness.moc"