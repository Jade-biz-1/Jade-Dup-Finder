#include "test_base.h"
#include <QTest>
#include <QDebug>

/**
 * @brief Example test class demonstrating the enhanced test framework
 * 
 * This example shows how to use the new standardized test base class,
 * configuration system, and naming conventions.
 */
DECLARE_TEST_CLASS(ExampleEnhancedTest, Unit, High, "example", "framework", "demo")

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        logTestInfo("Setting up example test suite");
    }

    void cleanupTestCase() {
        logTestInfo("Cleaning up example test suite");
        TestBase::cleanupTestCase();
    }

    // Example test using standardized naming convention
    // Format: test_<component>_<scenario>_<expectedResult>
    TEST_METHOD(test_framework_basicFunctionality_returnsExpectedResults) {
        logTestStep("Testing basic framework functionality");
        
        // Test basic assertions with enhanced logging
        TEST_VERIFY_WITH_MSG(true, "Basic verification should pass");
        TEST_COMPARE_WITH_MSG(2 + 2, 4, "Basic arithmetic should work");
        TEST_COMPARE_WITH_MSG(QString("test"), QString("test"), "String comparison should work");
        
        logTestStep("Basic functionality test completed successfully");
    }

    TEST_METHOD(test_configuration_loadDefaults_setsCorrectValues) {
        logTestStep("Testing configuration system");
        
        // Test configuration access
        TestConfig::TestSuiteConfig config = getTestConfig();
        TEST_VERIFY_WITH_MSG(!config.name.isEmpty(), "Test suite should have a name");
        TEST_COMPARE_WITH_MSG(config.category, TestConfig::Category::Unit, "Test should be categorized as Unit");
        TEST_COMPARE_WITH_MSG(config.priority, TestConfig::Priority::High, "Test should have High priority");
        
        // Test tags
        QStringList expectedTags = {"example", "framework", "demo"};
        for (const QString& tag : expectedTags) {
            TEST_VERIFY_WITH_MSG(config.tags.contains(tag), QString("Test should have tag: %1").arg(tag));
        }
        
        logTestStep("Configuration test completed successfully");
    }

    TEST_METHOD(test_testData_createAndCleanup_worksCorrectly) {
        logTestStep("Testing test data management");
        
        // Test directory creation
        QString testDir = createTestDirectory("example_test");
        TEST_VERIFY_WITH_MSG(!testDir.isEmpty(), "Test directory should be created");
        TEST_VERIFY_WITH_MSG(QDir(testDir).exists(), "Test directory should exist");
        
        // Test file creation
        QString testFile = createTestFile("example.txt", "Hello, Test World!");
        TEST_VERIFY_WITH_MSG(!testFile.isEmpty(), "Test file should be created");
        TEST_VERIFY_WITH_MSG(QFile::exists(testFile), "Test file should exist");
        
        // Verify file content
        QFile file(testFile);
        TEST_VERIFY_WITH_MSG(file.open(QIODevice::ReadOnly), "Test file should be readable");
        QString content = file.readAll();
        TEST_COMPARE_WITH_MSG(content, QString("Hello, Test World!"), "File content should match");
        file.close();
        
        // Cleanup is automatic in TestBase::cleanup()
        logTestStep("Test data management test completed successfully");
    }

    TEST_METHOD(test_performance_measurement_recordsAccurateTimings) {
        logTestStep("Testing performance measurement");
        
        // Start performance measurement
        startPerformanceMeasurement("example_operation");
        
        // Simulate some work
        QThread::msleep(50); // 50ms delay
        
        // Stop measurement
        qint64 elapsed = stopPerformanceMeasurement("example_operation");
        
        // Verify timing (should be at least 50ms, but allow some variance)
        TEST_VERIFY_WITH_MSG(elapsed >= 45, QString("Operation should take at least 45ms, got %1ms").arg(elapsed));
        TEST_VERIFY_WITH_MSG(elapsed <= 200, QString("Operation should take less than 200ms, got %1ms").arg(elapsed));
        
        // Record additional metric
        recordPerformanceMetric("custom_metric", 100, "units");
        
        logTestStep("Performance measurement test completed successfully");
    }

    TEST_METHOD(test_platform_detection_identifiesCorrectPlatform) {
        logTestStep("Testing platform detection");
        
        QString platform = getPlatformName();
        TEST_VERIFY_WITH_MSG(!platform.isEmpty(), "Platform name should not be empty");
        
        QStringList validPlatforms = {"Windows", "macOS", "Linux", "Unknown"};
        TEST_VERIFY_WITH_MSG(validPlatforms.contains(platform), 
                           QString("Platform should be one of: %1, got: %2")
                           .arg(validPlatforms.join(", ")).arg(platform));
        
        logTestStep(QString("Detected platform: %1").arg(platform));
        logTestStep("Platform detection test completed successfully");
    }

    // Example of conditional test execution
    TEST_METHOD(test_conditional_ciEnvironment_behavesCorrectly) {
        logTestStep("Testing conditional execution based on CI environment");
        
        if (isRunningInCI()) {
            logTestInfo("Running in CI environment");
            // CI-specific test logic
            TEST_VERIFY_WITH_MSG(true, "CI environment detected correctly");
        } else {
            logTestInfo("Running in local development environment");
            // Local development test logic
            TEST_VERIFY_WITH_MSG(true, "Local environment detected correctly");
        }
        
        logTestStep("Conditional execution test completed successfully");
    }

    // Example of platform-specific test
    TEST_METHOD(test_platform_specificFeatures_workAsExpected) {
        logTestStep("Testing platform-specific features");
        
        QString platform = getPlatformName();
        
        if (platform == "Windows") {
            logTestInfo("Testing Windows-specific features");
            // Windows-specific test logic
        } else if (platform == "macOS") {
            logTestInfo("Testing macOS-specific features");
            // macOS-specific test logic
        } else if (platform == "Linux") {
            logTestInfo("Testing Linux-specific features");
            // Linux-specific test logic
        } else {
            logTestWarning("Unknown platform, skipping platform-specific tests");
            QSKIP("Platform-specific tests not available for this platform");
        }
        
        TEST_VERIFY_WITH_MSG(true, "Platform-specific test completed");
        logTestStep("Platform-specific test completed successfully");
    }

END_TEST_CLASS()

/**
 * @brief Main function for running the example enhanced test
 */
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    qDebug() << "========================================";
    qDebug() << "Enhanced Test Framework Example";
    qDebug() << "========================================";
    
    // Load test configuration
    TestConfig::instance().loadConfiguration();
    
    // Create and run the test
    ExampleEnhancedTest test;
    
    if (test.shouldRunTest()) {
        int result = QTest::qExec(&test, argc, argv);
        
        if (result == 0) {
            qDebug() << "✅ Enhanced test framework example PASSED";
        } else {
            qDebug() << "❌ Enhanced test framework example FAILED";
        }
        
        return result;
    } else {
        qDebug() << "⏭️  Enhanced test framework example SKIPPED (disabled by configuration)";
        return 0;
    }
}

#include "example_enhanced_test.moc"