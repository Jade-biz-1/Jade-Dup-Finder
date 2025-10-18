#include "test_base.h"
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QProcessEnvironment>
#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QRegularExpression>

TestBase::TestBase(QObject* parent)
    : QObject(parent)
{
    setupTestEnvironment();
}

void TestBase::initTestCase() {
    registerTestSuite();
    globalSetUp();
    
    logTestInfo(QString("Starting test suite: %1").arg(testSuiteName()));
    logTestInfo(QString("Category: %1, Priority: %2")
                .arg(TestConfig::categoryToString(testCategory()))
                .arg(TestConfig::priorityToString(testPriority())));
    
    if (!testTags().isEmpty()) {
        logTestInfo(QString("Tags: %1").arg(testTags().join(", ")));
    }
    
    TestExecutionCoordinator::instance().testSuiteStarted(testSuiteName());
}

void TestBase::cleanupTestCase() {
    globalTearDown();
    cleanupTestEnvironment();
    
    // Generate execution statistics
    TestExecutionStats stats;
    stats.testSuiteName = testSuiteName();
    stats.category = testCategory();
    // Note: Actual test counts would be populated by the test framework
    
    TestExecutionCoordinator::instance().testSuiteFinished(testSuiteName(), stats);
    logTestInfo(QString("Finished test suite: %1").arg(testSuiteName()));
}

void TestBase::init() {
    // Per-test initialization
    logTestStep(QString("Initializing test: %1").arg(QTest::currentTestFunction()));
}

void TestBase::cleanup() {
    // Per-test cleanup
    cleanupTestData();
    logTestStep(QString("Cleaning up test: %1").arg(QTest::currentTestFunction()));
}

void TestBase::logTestStep(const QString& step) {
    if (TestConfig::instance().globalConfig().verboseOutput) {
        qDebug() << QString("[STEP] %1::%2: %3")
                    .arg(testSuiteName())
                    .arg(QTest::currentTestFunction() ? QTest::currentTestFunction() : "setup")
                    .arg(step);
    }
}

void TestBase::logTestInfo(const QString& info) {
    qDebug() << QString("[INFO] %1: %2").arg(testSuiteName()).arg(info);
}

void TestBase::logTestWarning(const QString& warning) {
    qWarning() << QString("[WARN] %1: %2").arg(testSuiteName()).arg(warning);
}

void TestBase::logTestSuccess(const QString& function, const QString& condition) {
    if (TestConfig::instance().globalConfig().verboseOutput) {
        qDebug() << QString("[PASS] %1::%2: %3")
                    .arg(testSuiteName())
                    .arg(function)
                    .arg(condition);
    }
}

void TestBase::logTestFailure(const QString& function, int line, const QString& condition, const QString& message) {
    qCritical() << QString("[FAIL] %1::%2:%3: %4 - %5")
                   .arg(testSuiteName())
                   .arg(function)
                   .arg(line)
                   .arg(condition)
                   .arg(message);
}

void TestBase::startPerformanceMeasurement(const QString& operationName) {
    m_performanceTimers[operationName].start();
    logTestStep(QString("Started performance measurement: %1").arg(operationName));
}

qint64 TestBase::stopPerformanceMeasurement(const QString& operationName) {
    if (!m_performanceTimers.contains(operationName)) {
        logTestWarning(QString("Performance timer not found: %1").arg(operationName));
        return -1;
    }
    
    qint64 elapsed = m_performanceTimers[operationName].elapsed();
    m_performanceTimers.remove(operationName);
    
    recordPerformanceMetric(operationName, elapsed);
    logTestStep(QString("Stopped performance measurement: %1 (%2ms)").arg(operationName).arg(elapsed));
    
    return elapsed;
}

void TestBase::recordPerformanceMetric(const QString& metricName, qint64 value, const QString& unit) {
    m_performanceMetrics[metricName] = value;
    logTestInfo(QString("Performance metric: %1 = %2 %3").arg(metricName).arg(value).arg(unit));
}

QString TestBase::createTestDirectory(const QString& suffix) {
    QString baseName = QString("test_%1_%2")
                       .arg(testSuiteName().toLower())
                       .arg(QDateTime::currentMSecsSinceEpoch());
    
    if (!suffix.isEmpty()) {
        baseName += "_" + suffix;
    }
    
    QString dirPath = QDir(m_testDataDirectory).absoluteFilePath(baseName);
    
    if (QDir().mkpath(dirPath)) {
        m_createdDirectories.append(dirPath);
        logTestStep(QString("Created test directory: %1").arg(dirPath));
        return dirPath;
    }
    
    logTestWarning(QString("Failed to create test directory: %1").arg(dirPath));
    return QString();
}

QString TestBase::createTestFile(const QString& fileName, const QString& content) {
    QString filePath = QDir(m_testDataDirectory).absoluteFilePath(fileName);
    
    // Ensure parent directory exists
    QFileInfo fileInfo(filePath);
    QDir().mkpath(fileInfo.absolutePath());
    
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        stream << content;
        file.close();
        
        m_createdFiles.append(filePath);
        logTestStep(QString("Created test file: %1 (%2 bytes)").arg(filePath).arg(content.length()));
        return filePath;
    }
    
    logTestWarning(QString("Failed to create test file: %1").arg(filePath));
    return QString();
}

void TestBase::cleanupTestData() {
    // Remove created files
    for (const QString& filePath : m_createdFiles) {
        if (QFile::exists(filePath)) {
            if (QFile::remove(filePath)) {
                logTestStep(QString("Removed test file: %1").arg(filePath));
            } else {
                logTestWarning(QString("Failed to remove test file: %1").arg(filePath));
            }
        }
    }
    m_createdFiles.clear();
    
    // Remove created directories
    for (const QString& dirPath : m_createdDirectories) {
        if (QDir(dirPath).exists()) {
            if (QDir(dirPath).removeRecursively()) {
                logTestStep(QString("Removed test directory: %1").arg(dirPath));
            } else {
                logTestWarning(QString("Failed to remove test directory: %1").arg(dirPath));
            }
        }
    }
    m_createdDirectories.clear();
}

bool TestBase::isRunningInCI() const {
    return TestConfig::instance().isRunningInCI();
}

QString TestBase::getPlatformName() const {
    return TestConfig::instance().getPlatformName();
}

void TestBase::skipIfPlatformNot(const QStringList& supportedPlatforms) {
    QString currentPlatform = getPlatformName();
    if (!supportedPlatforms.contains(currentPlatform)) {
        QString message = QString("Test not supported on %1. Supported platforms: %2")
                         .arg(currentPlatform)
                         .arg(supportedPlatforms.join(", "));
        QSKIP(message.toUtf8().constData());
    }
}

void TestBase::skipIfCI(const QString& reason) {
    if (isRunningInCI()) {
        QSKIP(reason.toUtf8().constData());
    }
}

void TestBase::skipIfNotCI(const QString& reason) {
    if (!isRunningInCI()) {
        QSKIP(reason.toUtf8().constData());
    }
}

TestConfig::TestSuiteConfig TestBase::getTestConfig() const {
    return TestConfig::instance().getTestSuiteConfig(testSuiteName());
}

bool TestBase::shouldRunTest() const {
    return TestConfig::instance().shouldRunTest(testSuiteName());
}

void TestBase::setupTestEnvironment() {
    // Create test data directory
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    m_testDataDirectory = QDir(tempDir).absoluteFilePath(
        QString("dupfinder_test_%1_%2")
        .arg(QCoreApplication::applicationPid())
        .arg(QDateTime::currentMSecsSinceEpoch())
    );
    
    if (!QDir().mkpath(m_testDataDirectory)) {
        qWarning() << "Failed to create test data directory:" << m_testDataDirectory;
    }
}

void TestBase::cleanupTestEnvironment() {
    cleanupTestData();
    
    // Remove main test data directory if empty
    QDir testDir(m_testDataDirectory);
    if (testDir.exists() && testDir.isEmpty()) {
        if (testDir.removeRecursively()) {
            logTestStep(QString("Removed test data directory: %1").arg(m_testDataDirectory));
        }
    }
}

void TestBase::registerTestSuite() {
    TestConfig::TestSuiteConfig config;
    config.name = testSuiteName();
    config.category = testCategory();
    config.priority = testPriority();
    config.tags = testTags();
    config.timeoutSeconds = TestConfig::instance().globalConfig().defaultTimeoutSeconds;
    config.enabledByDefault = true;
    config.executionMode = TestConfig::instance().globalConfig().defaultExecutionMode;
    
    TestConfig::instance().registerTestSuite(testSuiteName(), config);
}

// TestNaming namespace implementation
namespace TestNaming {

QString generateTestMethodName(const QString& component, const QString& scenario, const QString& expectedResult) {
    return QString("test_%1_%2_%3")
           .arg(component.toLower())
           .arg(scenario.toLower())
           .arg(expectedResult.toLower());
}

QString generateTestDataName(const QString& testCase, const QString& variant) {
    QString name = testCase.toLower().replace(" ", "_");
    if (!variant.isEmpty()) {
        name += "_" + variant.toLower();
    }
    return name;
}

bool validateTestMethodName(const QString& methodName) {
    // Check if method name follows the pattern: test_<component>_<scenario>_<expectedResult>
    QRegularExpression pattern("^test_[a-z][a-z0-9_]*_[a-z][a-z0-9_]*_[a-z][a-z0-9_]*$");
    return pattern.match(methodName).hasMatch();
}

} // namespace TestNaming

// TestExecutionCoordinator implementation
TestExecutionCoordinator& TestExecutionCoordinator::instance() {
    static TestExecutionCoordinator instance;
    return instance;
}

void TestExecutionCoordinator::registerTestExecution(const TestExecutionStats& stats) {
    m_executionStats.append(stats);
}

QList<TestExecutionStats> TestExecutionCoordinator::getExecutionStats() const {
    return m_executionStats;
}

void TestExecutionCoordinator::generateExecutionReport(const QString& outputPath) {
    QString reportPath = outputPath;
    if (reportPath.isEmpty()) {
        QString reportDir = TestConfig::instance().globalConfig().reportOutputDirectory;
        QDir().mkpath(reportDir);
        reportPath = QDir(reportDir).absoluteFilePath(
            QString("test_execution_report_%1.json")
            .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"))
        );
    }
    
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["platform"] = TestConfig::instance().getPlatformName();
    report["totalSuites"] = m_executionStats.size();
    
    int totalTests = 0, totalPassed = 0, totalFailed = 0, totalSkipped = 0;
    qint64 totalExecutionTime = 0;
    
    QJsonArray suites;
    for (const TestExecutionStats& stats : m_executionStats) {
        QJsonObject suite;
        suite["name"] = stats.testSuiteName;
        suite["category"] = TestConfig::categoryToString(stats.category);
        suite["totalTests"] = stats.totalTests;
        suite["passedTests"] = stats.passedTests;
        suite["failedTests"] = stats.failedTests;
        suite["skippedTests"] = stats.skippedTests;
        suite["executionTimeMs"] = stats.executionTimeMs;
        
        QJsonArray failures;
        for (const QString& failure : stats.failureMessages) {
            failures.append(failure);
        }
        suite["failures"] = failures;
        
        QJsonObject metrics;
        for (auto it = stats.performanceMetrics.begin(); it != stats.performanceMetrics.end(); ++it) {
            metrics[it.key()] = it.value();
        }
        suite["performanceMetrics"] = metrics;
        
        suites.append(suite);
        
        totalTests += stats.totalTests;
        totalPassed += stats.passedTests;
        totalFailed += stats.failedTests;
        totalSkipped += stats.skippedTests;
        totalExecutionTime += stats.executionTimeMs;
    }
    
    report["suites"] = suites;
    report["summary"] = QJsonObject{
        {"totalTests", totalTests},
        {"passedTests", totalPassed},
        {"failedTests", totalFailed},
        {"skippedTests", totalSkipped},
        {"totalExecutionTimeMs", totalExecutionTime},
        {"successRate", totalTests > 0 ? (double)totalPassed / totalTests * 100.0 : 0.0}
    };
    
    QFile file(reportPath);
    if (file.open(QIODevice::WriteOnly)) {
        QJsonDocument doc(report);
        file.write(doc.toJson());
        qDebug() << "Test execution report generated:" << reportPath;
    } else {
        qWarning() << "Failed to write test execution report:" << reportPath;
    }
}

void TestExecutionCoordinator::clearStats() {
    m_executionStats.clear();
}