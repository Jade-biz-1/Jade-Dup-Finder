#include "workflow_testing.h"
#include "user_scenario_testing.h"
#include "cross_platform_testing.h"
#include "framework/test_environment.h"
#include "ui_automation.h"
#include <QApplication>
#include <QTest>
#include <QDebug>
#include <memory>

/**
 * @brief Example End-to-End Testing Suite
 * 
 * This example demonstrates how to use the comprehensive End-to-End Testing Framework
 * including workflow testing, user scenario testing, and cross-platform validation.
 */
class ExampleEndToEndTesting : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Workflow Testing Examples
    void testBasicWorkflowExecution();
    void testScanToDeleteWorkflow();
    void testFirstTimeUserWorkflow();
    void testPowerUserWorkflow();
    void testSafetyFocusedWorkflow();
    void testErrorRecoveryWorkflow();
    
    // User Scenario Testing Examples
    void testFirstTimeUserScenario();
    void testCasualUserScenario();
    void testPowerUserScenario();
    void testSafetyFocusedUserScenario();
    void testAccessibilityUserScenario();
    void testPhotoLibraryCleanupScenario();
    
    // Cross-Platform Testing Examples
    void testFileOperationsCrossPlatform();
    void testPathHandlingCrossPlatform();
    void testDisplayScalingCrossPlatform();
    void testFileSystemCompatibility();
    void testOSIntegration();

private:
    std::shared_ptr<TestEnvironment> m_testEnvironment;
    std::shared_ptr<UIAutomation> m_uiAutomation;
    std::shared_ptr<WorkflowTesting> m_workflowTesting;
    std::shared_ptr<UserScenarioTesting> m_scenarioTesting;
    std::shared_ptr<CrossPlatformTesting> m_crossPlatformTesting;
};

void ExampleEndToEndTesting::initTestCase() {
    qDebug() << "Initializing End-to-End Testing Framework";
    
    // Initialize core components
    m_testEnvironment = std::make_shared<TestEnvironment>();
    m_uiAutomation = std::make_shared<UIAutomation>();
    m_workflowTesting = std::make_shared<WorkflowTesting>();
    m_scenarioTesting = std::make_shared<UserScenarioTesting>();
    m_crossPlatformTesting = std::make_shared<CrossPlatformTesting>();
    
    // Setup component relationships
    m_workflowTesting->setTestEnvironment(m_testEnvironment);
    m_workflowTesting->setUIAutomation(m_uiAutomation);
    m_scenarioTesting->setWorkflowTesting(m_workflowTesting);
    m_crossPlatformTesting->setWorkflowTesting(m_workflowTesting);
    
    // Configure testing environment
    m_workflowTesting->setDefaultTimeout(60000); // 1 minute
    m_workflowTesting->enableDetailedLogging(true);
    m_workflowTesting->enableAutomaticScreenshots(true);
    
    m_scenarioTesting->enableUserExperienceMetrics(true);
    m_scenarioTesting->enableAccessibilityTesting(true);
    
    // Setup test environment
    QVERIFY(m_testEnvironment->setupTestEnvironment());
    
    qDebug() << "End-to-End Testing Framework initialized successfully";
}

void ExampleEndToEndTesting::cleanupTestCase() {
    qDebug() << "Cleaning up End-to-End Testing Framework";
    
    if (m_testEnvironment) {
        m_testEnvironment->cleanupTestEnvironment();
    }
    
    qDebug() << "End-to-End Testing Framework cleanup completed";
}

void ExampleEndToEndTesting::testBasicWorkflowExecution() {
    qDebug() << "Testing basic workflow execution";
    
    // Create a simple test workflow
    UserWorkflow workflow;
    workflow.id = "basic_test_workflow";
    workflow.name = "Basic Test Workflow";
    workflow.description = "Simple workflow for testing basic functionality";
    
    // Add basic workflow steps
    WorkflowStep launchStep;
    launchStep.id = "launch_app";
    launchStep.name = "Launch Application";
    launchStep.type = WorkflowStepType::Setup;
    launchStep.parameters["action"] = "launch_application";
    workflow.steps.append(launchStep);
    
    WorkflowStep validateStep;
    validateStep.id = "validate_launch";
    validateStep.name = "Validate Application Launch";
    validateStep.type = WorkflowStepType::Validation;
    validateStep.parameters["action"] = "validate_main_window";
    workflow.steps.append(validateStep);
    
    // Register and execute workflow
    m_workflowTesting->registerWorkflow(workflow);
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow.id);
    
    // Validate results
    QVERIFY(result.success);
    QCOMPARE(result.totalSteps, 2);
    QCOMPARE(result.completedSteps, 2);
    QCOMPARE(result.failedSteps, 0);
    QVERIFY(result.executionTimeMs > 0);
    
    qDebug() << "Basic workflow execution test completed successfully";
}

void ExampleEndToEndTesting::testScanToDeleteWorkflow() {
    qDebug() << "Testing scan-to-delete workflow";
    
    // Create test data with duplicates
    QString testDataDir = m_testEnvironment->createTestDirectory("scan_test_data");
    QVERIFY(!testDataDir.isEmpty());
    
    // Create duplicate files for testing
    TestFileSpec file1;
    file1.fileName = "original.txt";
    file1.content = "This is test content for duplicate detection";
    QVERIFY(m_testEnvironment->createTestFile(testDataDir + "/original.txt", file1));
    
    TestFileSpec file2;
    file2.fileName = "duplicate.txt";
    file2.content = "This is test content for duplicate detection"; // Same content
    QVERIFY(m_testEnvironment->createTestFile(testDataDir + "/duplicate.txt", file2));
    
    // Execute scan-to-delete workflow
    UserWorkflow workflow = m_workflowTesting->createScanToDeleteWorkflow();
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
    
    // Validate workflow execution
    QVERIFY(result.success);
    QVERIFY(result.completedSteps > 0);
    QVERIFY(result.executionTimeMs > 0);
    
    // Validate that duplicates were found and processed
    // (This would require actual application integration)
    
    qDebug() << "Scan-to-delete workflow test completed";
}

void ExampleEndToEndTesting::testFirstTimeUserWorkflow() {
    qDebug() << "Testing first-time user workflow";
    
    // Clear application settings to simulate first-time user
    m_testEnvironment->clearApplicationSettings();
    
    // Execute first-time user workflow
    UserWorkflow workflow = m_workflowTesting->createFirstTimeUserWorkflow();
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
    
    // Validate workflow execution
    QVERIFY(result.success);
    QVERIFY(result.completedSteps >= result.totalSteps * 0.8); // Allow some optional steps to be skipped
    
    qDebug() << "First-time user workflow test completed";
}

void ExampleEndToEndTesting::testPowerUserWorkflow() {
    qDebug() << "Testing power user workflow";
    
    // Setup power user configuration
    QMap<QString, QVariant> powerUserConfig;
    powerUserConfig["enable_advanced_features"] = true;
    powerUserConfig["show_detailed_options"] = true;
    powerUserConfig["allow_batch_operations"] = true;
    m_testEnvironment->setConfigValue("user_profile", "power_user");
    
    // Execute power user workflow
    UserWorkflow workflow = m_workflowTesting->createPowerUserWorkflow();
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
    
    // Validate workflow execution
    QVERIFY(result.success);
    QVERIFY(result.executionTimeMs > 0);
    
    qDebug() << "Power user workflow test completed";
}

void ExampleEndToEndTesting::testSafetyFocusedWorkflow() {
    qDebug() << "Testing safety-focused workflow";
    
    // Setup safety-focused configuration
    m_testEnvironment->setConfigValue("enable_backups", true);
    m_testEnvironment->setConfigValue("confirm_all_operations", true);
    m_testEnvironment->setConfigValue("use_recycle_bin", true);
    
    // Execute safety-focused workflow
    UserWorkflow workflow = m_workflowTesting->createSafetyFocusedWorkflow();
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
    
    // Validate workflow execution
    QVERIFY(result.success);
    QCOMPARE(result.failedSteps, 0); // Safety workflows should not fail
    
    qDebug() << "Safety-focused workflow test completed";
}

void ExampleEndToEndTesting::testErrorRecoveryWorkflow() {
    qDebug() << "Testing error recovery workflow";
    
    // Create a workflow that will encounter errors
    UserWorkflow workflow;
    workflow.id = "error_recovery_test";
    workflow.name = "Error Recovery Test";
    workflow.description = "Test error handling and recovery";
    workflow.allowPartialFailure = true;
    
    // Add steps that might fail
    WorkflowStep errorStep;
    errorStep.id = "error_step";
    errorStep.name = "Step That May Fail";
    errorStep.type = WorkflowStepType::Custom;
    errorStep.retryOnFailure = true;
    errorStep.maxRetries = 3;
    errorStep.customAction = [](const QMap<QString, QVariant>&) -> bool {
        // Simulate intermittent failure
        return QRandomGenerator::global()->bounded(2) == 1;
    };
    workflow.steps.append(errorStep);
    
    // Execute workflow with error recovery
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
    
    // Validate that error recovery was attempted
    QVERIFY(result.totalSteps > 0);
    // Result may succeed or fail, but should handle errors gracefully
    
    qDebug() << "Error recovery workflow test completed";
}

void ExampleEndToEndTesting::testFirstTimeUserScenario() {
    qDebug() << "Testing first-time user scenario";
    
    // Create and register first-time user scenario
    UserScenario scenario = m_scenarioTesting->createFirstTimeUserScenario();
    m_scenarioTesting->registerScenario(scenario);
    
    // Execute scenario
    ScenarioResult result = m_scenarioTesting->executeScenario(scenario.id);
    
    // Validate scenario execution
    QVERIFY(result.success);
    QCOMPARE(result.persona, UserPersona::FirstTimeUser);
    QVERIFY(result.satisfactionScore >= 6); // Minimum acceptable satisfaction
    QVERIFY(result.completedGoals.size() > result.failedGoals.size());
    
    // Validate user experience metrics
    QVERIFY(result.totalExecutionTimeMs > 0);
    QVERIFY(result.userActionsCount > 0);
    QVERIFY(result.errorEncountered <= 2); // Allow minimal errors for first-time users
    
    qDebug() << "First-time user scenario test completed";
}

void ExampleEndToEndTesting::testCasualUserScenario() {
    qDebug() << "Testing casual user scenario";
    
    // Create and execute casual user scenario
    UserScenario scenario = m_scenarioTesting->createCasualUserScenario();
    ScenarioResult result = m_scenarioTesting->executeScenario(scenario);
    
    // Validate scenario execution
    QVERIFY(result.success);
    QCOMPARE(result.persona, UserPersona::CasualUser);
    QVERIFY(result.totalExecutionTimeMs <= 300000); // Should complete within 5 minutes
    QVERIFY(result.userActionsCount <= 20); // Should be efficient
    
    qDebug() << "Casual user scenario test completed";
}

void ExampleEndToEndTesting::testAccessibilityUserScenario() {
    qDebug() << "Testing accessibility user scenario";
    
    // Create accessibility user scenario
    UserScenario scenario = m_scenarioTesting->createAccessibilityUserScenario();
    ScenarioResult result = m_scenarioTesting->executeScenario(scenario);
    
    // Validate accessibility compliance
    QVERIFY(result.success);
    QCOMPARE(result.persona, UserPersona::AccessibilityUser);
    QVERIFY(result.accessibilityIssues.isEmpty()); // Should have no accessibility issues
    
    qDebug() << "Accessibility user scenario test completed";
}

void ExampleEndToEndTesting::testPhotoLibraryCleanupScenario() {
    qDebug() << "Testing photo library cleanup scenario";
    
    // Create test photo library with duplicates
    QString photoDir = m_testEnvironment->createTestDirectory("photo_library");
    QVERIFY(!photoDir.isEmpty());
    
    // Create sample photo files (simulated)
    QVERIFY(m_testEnvironment->createPhotoLibraryDataset(photoDir, 50, 20)); // 50 photos, 20% duplicates
    
    // Execute photo library cleanup scenario
    UserScenario scenario = m_scenarioTesting->createPhotoLibraryCleanupScenario();
    ScenarioResult result = m_scenarioTesting->executeScenario(scenario);
    
    // Validate scenario execution
    QVERIFY(result.success);
    QVERIFY(result.completedGoals.contains("identify_duplicate_photos"));
    QVERIFY(result.completedGoals.contains("preserve_highest_quality_versions"));
    
    qDebug() << "Photo library cleanup scenario test completed";
}

void ExampleEndToEndTesting::testFileOperationsCrossPlatform() {
    qDebug() << "Testing file operations cross-platform";
    
    // Create cross-platform file operations test
    CrossPlatformTest test = m_crossPlatformTesting->createFileOperationTest();
    m_crossPlatformTesting->registerCrossPlatformTest(test);
    
    // Execute cross-platform test
    CrossPlatformResult result = m_crossPlatformTesting->executeCrossPlatformTest(test.id);
    
    // Validate cross-platform execution
    QVERIFY(result.success);
    QVERIFY(result.platformResults.contains(result.currentPlatform));
    
    // Check for expected platform differences
    Platform currentPlatform = m_crossPlatformTesting->getCurrentPlatform();
    if (result.platformDifferences.contains(currentPlatform)) {
        QStringList differences = result.platformDifferences[currentPlatform];
        qDebug() << "Platform differences detected:" << differences;
    }
    
    qDebug() << "File operations cross-platform test completed";
}

void ExampleEndToEndTesting::testPathHandlingCrossPlatform() {
    qDebug() << "Testing path handling cross-platform";
    
    // Test path normalization across platforms
    QString testPath = "test/path/with/separators";
    Platform currentPlatform = m_crossPlatformTesting->getCurrentPlatform();
    
    QString normalizedPath = CrossPlatformTesting::normalizePath(testPath, currentPlatform);
    QVERIFY(!normalizedPath.isEmpty());
    
    // Validate path is valid for current platform
    QVERIFY(CrossPlatformTesting::isPathValid(normalizedPath, currentPlatform));
    
    qDebug() << "Path handling cross-platform test completed";
}

void ExampleEndToEndTesting::testDisplayScalingCrossPlatform() {
    qDebug() << "Testing display scaling cross-platform";
    
    // Create display scaling test
    CrossPlatformTest test = m_crossPlatformTesting->createDisplayScalingTest();
    CrossPlatformResult result = m_crossPlatformTesting->executeCrossPlatformTest(test);
    
    // Validate display scaling
    QVERIFY(result.success);
    
    // Test specific scaling factors
    QVERIFY(m_crossPlatformTesting->testDisplayScaling(1.0)); // 100% scaling
    QVERIFY(m_crossPlatformTesting->testDisplayScaling(1.5)); // 150% scaling
    
    qDebug() << "Display scaling cross-platform test completed";
}

void ExampleEndToEndTesting::testFileSystemCompatibility() {
    qDebug() << "Testing file system compatibility";
    
    QString testPath = m_testEnvironment->getTestDataDirectory();
    
    // Test file system features
    bool compatibility = m_crossPlatformTesting->testFileSystemCompatibility(testPath);
    QVERIFY(compatibility);
    
    // Test specific file system features
    QVERIFY(m_crossPlatformTesting->testCaseSensitivity(testPath) || 
            m_crossPlatformTesting->getCurrentPlatform() == Platform::Windows);
    
    qDebug() << "File system compatibility test completed";
}

void ExampleEndToEndTesting::testOSIntegration() {
    qDebug() << "Testing OS integration";
    
    // Test file manager integration
    bool fileManagerIntegration = m_crossPlatformTesting->testFileManagerIntegration();
    QVERIFY(fileManagerIntegration);
    
    // Test trash integration
    bool trashIntegration = m_crossPlatformTesting->testTrashIntegration();
    QVERIFY(trashIntegration);
    
    qDebug() << "OS integration test completed";
}

// Example of how to run the complete test suite
void runCompleteEndToEndTestSuite() {
    qDebug() << "Running Complete End-to-End Test Suite";
    
    // Initialize testing components
    auto testEnvironment = std::make_shared<TestEnvironment>();
    auto uiAutomation = std::make_shared<UIAutomation>();
    auto workflowTesting = std::make_shared<WorkflowTesting>();
    auto scenarioTesting = std::make_shared<UserScenarioTesting>();
    auto crossPlatformTesting = std::make_shared<CrossPlatformTesting>();
    
    // Setup relationships
    workflowTesting->setTestEnvironment(testEnvironment);
    workflowTesting->setUIAutomation(uiAutomation);
    scenarioTesting->setWorkflowTesting(workflowTesting);
    crossPlatformTesting->setWorkflowTesting(workflowTesting);
    
    // Setup test environment
    testEnvironment->setupTestEnvironment();
    
    // Register predefined workflows
    workflowTesting->registerWorkflow(workflowTesting->createScanToDeleteWorkflow());
    workflowTesting->registerWorkflow(workflowTesting->createFirstTimeUserWorkflow());
    workflowTesting->registerWorkflow(workflowTesting->createPowerUserWorkflow());
    workflowTesting->registerWorkflow(workflowTesting->createSafetyFocusedWorkflow());
    
    // Register predefined scenarios
    scenarioTesting->registerScenario(scenarioTesting->createFirstTimeUserScenario());
    scenarioTesting->registerScenario(scenarioTesting->createCasualUserScenario());
    scenarioTesting->registerScenario(scenarioTesting->createPowerUserScenario());
    scenarioTesting->registerScenario(scenarioTesting->createAccessibilityUserScenario());
    
    // Register cross-platform tests
    crossPlatformTesting->registerCrossPlatformTest(crossPlatformTesting->createFileOperationTest());
    crossPlatformTesting->registerCrossPlatformTest(crossPlatformTesting->createPathHandlingTest());
    crossPlatformTesting->registerCrossPlatformTest(crossPlatformTesting->createDisplayScalingTest());
    
    // Execute workflow tests
    qDebug() << "Executing workflow tests...";
    QStringList workflows = workflowTesting->getRegisteredWorkflows();
    for (const QString& workflowId : workflows) {
        WorkflowResult result = workflowTesting->executeWorkflow(workflowId);
        qDebug() << "Workflow" << workflowId << "result:" << (result.success ? "PASS" : "FAIL");
    }
    
    // Execute scenario tests
    qDebug() << "Executing scenario tests...";
    QStringList scenarios = scenarioTesting->getRegisteredScenarios();
    for (const QString& scenarioId : scenarios) {
        ScenarioResult result = scenarioTesting->executeScenario(scenarioId);
        qDebug() << "Scenario" << scenarioId << "result:" << (result.success ? "PASS" : "FAIL");
    }
    
    // Execute cross-platform tests
    qDebug() << "Executing cross-platform tests...";
    QStringList crossPlatformTests = crossPlatformTesting->getRegisteredTests();
    for (const QString& testId : crossPlatformTests) {
        CrossPlatformResult result = crossPlatformTesting->executeCrossPlatformTest(testId);
        qDebug() << "Cross-platform test" << testId << "result:" << (result.success ? "PASS" : "FAIL");
    }
    
    // Cleanup
    testEnvironment->cleanupTestEnvironment();
    
    qDebug() << "Complete End-to-End Test Suite finished";
}

QTEST_MAIN(ExampleEndToEndTesting)
#include "example_end_to_end_testing.moc"