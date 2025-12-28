#include "test_harness.h"
#include "test_environment.h"
#include "test_reporting.h"
#include "test_utilities.h"
#include <QCoreApplication>
#include <QDebug>

/**
 * @brief Example test suite demonstrating the testing framework usage
 */
class ExampleTestSuite : public TestSuite {
    Q_OBJECT

public:
    ExampleTestSuite(QObject* parent = nullptr) 
        : TestSuite("ExampleTests", TestCategory::Unit, parent) {
        setupTests();
    }

private:
    void setupTests() {
        REGISTER_TEST(testBasicAssertion);
        REGISTER_TEST(testStringOperations);
        REGISTER_TEST(testPerformanceMeasurement);
        REGISTER_TEST(testFileOperations);
        REGISTER_TEST(testFailureExample);
    }

    bool runAllTests() override {
        bool allPassed = true;
        
        for (const QString& testName : testNames()) {
            emit testStarted(testName);
            
            bool passed = runTest(testName);
            if (!passed) {
                allPassed = false;
            }
        }
        
        emit suiteCompleted(allPassed);
        return allPassed;
    }

    bool runTest(const QString& testName) override {
        try {
            if (testName == "testBasicAssertion") {
                return test_testBasicAssertion();
            } else if (testName == "testStringOperations") {
                return test_testStringOperations();
            } else if (testName == "testPerformanceMeasurement") {
                return test_testPerformanceMeasurement();
            } else if (testName == "testFileOperations") {
                return test_testFileOperations();
            } else if (testName == "testFailureExample") {
                return test_testFailureExample();
            }
            
            recordTestResult(testName, false, "Unknown test: " + testName);
            return false;
        } catch (const std::exception& e) {
            recordTestResult(testName, false, QString::fromStdString(e.what()));
            return false;
        }
    }

    // Test method declarations
    TEST_METHOD(testBasicAssertion);
    TEST_METHOD(testStringOperations);
    TEST_METHOD(testPerformanceMeasurement);
    TEST_METHOD(testFileOperations);
    TEST_METHOD(testFailureExample);
};

// Test implementations
void ExampleTestSuite::testBasicAssertion() {
    LOG_TEST_STEP("Testing basic assertions");
    
    // Test basic comparisons
    TEST_COMPARE(2 + 2, 4);
    TEST_COMPARE(QString("hello"), QString("hello"));
    TEST_VERIFY(true);
    TEST_VERIFY(!false);
    
    // Test ranges
    TEST_VERIFY(TestUtilities::validateRange(5.0, 1.0, 10.0));
    TEST_VERIFY(!TestUtilities::validateRange(15.0, 1.0, 10.0));
    
    recordTestResult("testBasicAssertion", true);
}

void ExampleTestSuite::testStringOperations() {
    LOG_TEST_STEP("Testing string operations");
    
    // Generate random strings
    QString randomStr = TestUtilities::generateRandomString(10);
    TEST_VERIFY(randomStr.length() == 10);
    
    QStringList stringList = TestUtilities::generateRandomStringList(5, 3, 8);
    TEST_COMPARE(stringList.size(), 5);
    
    for (const QString& str : stringList) {
        TEST_VERIFY(str.length() >= 3 && str.length() <= 8);
    }
    
    // Test validation
    TEST_VERIFY(TestUtilities::validateStringLength("hello", 3, 10));
    TEST_VERIFY(!TestUtilities::validateStringLength("hi", 3, 10));
    
    recordTestResult("testStringOperations", true);
}

void ExampleTestSuite::testPerformanceMeasurement() {
    LOG_TEST_STEP("Testing performance measurement");
    
    {
        PERFORMANCE_MEASURE(sleepTest);
        
        // Simulate some work
        QThread::msleep(100);
        
        // Performance measurement will be automatically stopped when scope ends
    }
    
    // Manual performance measurement
    TestUtilities::startPerformanceMeasurement("manualTest");
    
    // Do some work
    QString largeString;
    for (int i = 0; i < 10000; ++i) {
        largeString += QString::number(i);
    }
    
    qint64 elapsed = TestUtilities::stopPerformanceMeasurement("manualTest");
    TEST_VERIFY(elapsed >= 0);
    
    addMetric("performanceTestElapsed", elapsed);
    recordTestResult("testPerformanceMeasurement", true);
}

void ExampleTestSuite::testFileOperations() {
    LOG_TEST_STEP("Testing file operations");
    
    {
        TEMP_FILE_GUARD("Hello, World!");
        
        TEST_VERIFY(tempFile.isValid());
        TEST_VERIFY(QFile::exists(tempFile.path()));
        
        QString content = TestUtilities::readFileContent(tempFile.path());
        TEST_COMPARE(content, QString("Hello, World!"));
        
        qint64 size = TestUtilities::getFileSize(tempFile.path());
        TEST_VERIFY(size > 0);
        
        // File will be automatically cleaned up when tempFile goes out of scope
    }
    
    // Test file creation with specific size
    QString largeTempFile = TestUtilities::createTempFile();
    TEST_VERIFY(TestUtilities::createFileWithSize(largeTempFile, 1024));
    TEST_COMPARE(TestUtilities::getFileSize(largeTempFile), 1024LL);
    
    QFile::remove(largeTempFile);
    
    recordTestResult("testFileOperations", true);
}

void ExampleTestSuite::testFailureExample() {
    LOG_TEST_STEP("Testing failure handling");
    
    // This test is designed to fail to demonstrate failure reporting
    TEST_ASSERT(false, "This is an intentional failure for demonstration");
    
    // This line should not be reached
    recordTestResult("testFailureExample", true);
}

/**
 * @brief Example main function showing how to use the test framework
 */
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    qDebug() << "========================================";
    qDebug() << "CloneClean Test Framework Example";
    qDebug() << "========================================";
    
    // Create test harness
    TestHarness harness;
    
    // Create test environment
    auto testEnv = std::make_shared<TestEnvironment>();
    harness.setTestEnvironment(testEnv);
    
    // Create report generator
    auto reporter = std::make_shared<TestReporting>();
    ReportConfig reportConfig;
    reportConfig.formats = {ReportFormat::Console, ReportFormat::HTML, ReportFormat::JSON};
    reportConfig.outputDirectory = "example_test_results";
    reporter->setReportConfig(reportConfig);
    harness.setReportGenerator(reporter);
    
    // Register test suite
    auto exampleSuite = std::make_shared<ExampleTestSuite>();
    harness.registerTestSuite(exampleSuite);
    
    // Configure test execution
    TestSuiteConfig config;
    config.enabledCategories = {"Unit"};
    config.parallelExecution = false; // Keep simple for example
    config.verboseOutput = true;
    config.generateHtmlReport = true;
    harness.setConfiguration(config);
    
    // Run tests
    qDebug() << "Running example test suite...";
    bool success = harness.runAllTests();
    
    // Generate reports
    harness.generateReport();
    
    // Print results
    TestResults results = harness.getResults();
    qDebug() << "\n" << results.summary();
    
    if (results.hasFailures()) {
        qDebug() << "\nFailures:";
        for (const TestFailure& failure : results.failures) {
            qDebug() << QString("  - %1: %2").arg(failure.testName).arg(failure.errorMessage);
        }
    }
    
    qDebug() << "\nReports generated in:" << reportConfig.outputDirectory;
    qDebug() << "========================================";
    
    return success ? 0 : 1;
}

#include "example_test_suite.moc"