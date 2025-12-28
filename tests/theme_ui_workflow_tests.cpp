#include <QtTest/QtTest>
#include <QApplication>
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QElapsedTimer>

#include "workflow_testing.h"
#include "user_scenario_testing.h"
#include "ui_automation.h"
#include "visual_testing.h"
#include "theme_accessibility_testing.h"
#include "../include/ui_theme_test_integration.h"
#include "../include/theme_manager.h"

/**
 * @brief Comprehensive end-to-end UI operation validation with theme integration
 * 
 * This test class implements task 11.1 and 11.2 from the UI/UX architect review fixes:
 * - Create complete workflow tests using WorkflowTesting
 * - Add cross-theme interaction validation
 * 
 * It integrates with the existing testing framework to provide comprehensive
 * theme-aware workflow validation across all supported themes.
 */
class ThemeUIWorkflowTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

    // Task 11.1: Create complete workflow tests using WorkflowTesting
    void testScanToDeleteWorkflowAcrossThemes();
    void testResultsViewingAndSelectionWorkflow();
    void testSettingsAndPreferencesWorkflowWithThemes();
    void testFileOperationWorkflowValidation();
    void testErrorRecoveryWorkflowAcrossThemes();

    // Task 11.2: Add cross-theme interaction validation
    void testUserInteractionsInLightTheme();
    void testUserInteractionsInDarkTheme();
    void testUIStateMaintenanceThroughoutWorkflows();
    void testConsistentUIBehaviorAcrossWorkflowSteps();
    void testThemeTransitionDuringWorkflows();

    // Comprehensive workflow validation
    void testCompleteUserJourneyAcrossAllThemes();
    void testWorkflowPerformanceAcrossThemes();
    void testAccessibilityWorkflowCompliance();

private:
    // Helper methods for workflow creation and validation
    UserWorkflow createScanToDeleteWorkflowWithThemes();
    UserWorkflow createResultsViewingWorkflow();
    UserWorkflow createSettingsConfigurationWorkflow();
    UserWorkflow createFileOperationWorkflow();
    UserWorkflow createErrorRecoveryWorkflow();

    // Theme-aware workflow validation helpers
    bool validateWorkflowInTheme(const UserWorkflow& workflow, ThemeManager::Theme theme);
    bool validateUIStateConsistency(const UserWorkflow& workflow, ThemeManager::Theme theme);
    bool validateWorkflowAccessibility(const UserWorkflow& workflow, ThemeManager::Theme theme);
    bool validateWorkflowPerformance(const UserWorkflow& workflow, ThemeManager::Theme theme, double maxTimeMs);

    // Cross-theme interaction validation
    bool testInteractionConsistency(const QStringList& interactions, const QList<ThemeManager::Theme>& themes);
    bool validateThemeTransitionDuringWorkflow(const UserWorkflow& workflow);
    bool testWorkflowStepConsistency(const UserWorkflow& workflow, const QList<ThemeManager::Theme>& themes);

    // Workflow step creation helpers
    WorkflowStep createThemeAwareUIStep(const QString& stepId, const QString& action, 
                                       const QMap<QString, QVariant>& parameters);
    WorkflowStep createThemeValidationStep(const QString& stepId, ThemeManager::Theme expectedTheme);
    WorkflowStep createAccessibilityValidationStep(const QString& stepId, const QString& componentName);

    // Test data and validation
    void setupTestEnvironment();
    void createTestWorkflows();
    void validateTestResults(const WorkflowResult& result, const QString& workflowName);

private:
    // Core testing framework components
    std::shared_ptr<WorkflowTesting> m_workflowTesting;
    std::shared_ptr<UserScenarioTesting> m_scenarioTesting;
    std::shared_ptr<UIAutomation> m_uiAutomation;
    std::shared_ptr<VisualTesting> m_visualTesting;
    std::shared_ptr<ThemeAccessibilityTesting> m_themeAccessibilityTesting;
    
    // Theme integration
    UIThemeTestIntegration* m_themeIntegration;
    ThemeManager* m_themeManager;
    
    // Test environment
    QMainWindow* m_testMainWindow;
    QWidget* m_testWidget;
    
    // Test configuration
    QList<ThemeManager::Theme> m_supportedThemes;
    QMap<QString, UserWorkflow> m_testWorkflows;
    QElapsedTimer m_performanceTimer;
    
    // Test results tracking
    QMap<QString, QMap<ThemeManager::Theme, WorkflowResult>> m_workflowResults;
    QStringList m_failedTests;
    double m_maxAcceptableWorkflowTime;
};

void ThemeUIWorkflowTests::initTestCase() {
    qDebug() << "Initializing comprehensive theme UI workflow tests...";
    
    // Initialize core testing framework components
    m_workflowTesting = std::make_shared<WorkflowTesting>(this);
    m_scenarioTesting = std::make_shared<UserScenarioTesting>(this);
    m_uiAutomation = std::make_shared<UIAutomation>(this);
    m_visualTesting = std::make_shared<VisualTesting>(this);
    m_themeAccessibilityTesting = std::make_shared<ThemeAccessibilityTesting>(this);
    
    // Connect frameworks
    m_scenarioTesting->setWorkflowTesting(m_workflowTesting);
    m_themeAccessibilityTesting->setUIAutomation(m_uiAutomation.get());
    m_themeAccessibilityTesting->setVisualTesting(m_visualTesting.get());
    
    // Initialize theme integration
    m_themeIntegration = new UIThemeTestIntegration(this);
    m_themeIntegration->setUIAutomation(m_uiAutomation.get());
    m_themeIntegration->setVisualTesting(m_visualTesting.get());
    m_themeIntegration->setThemeAccessibilityTesting(m_themeAccessibilityTesting.get());
    
    // Get theme manager instance
    m_themeManager = ThemeManager::instance();
    
    // Setup supported themes for testing
    m_supportedThemes = {
        ThemeManager::Light,
        ThemeManager::Dark
    };
    
    // Configure test parameters
    m_maxAcceptableWorkflowTime = 30000.0; // 30 seconds max per workflow
    
    // Setup test environment
    setupTestEnvironment();
    
    // Create test workflows
    createTestWorkflows();
    
    qDebug() << "Theme UI workflow tests initialization completed";
}

void ThemeUIWorkflowTests::cleanupTestCase() {
    // Restore original theme
    m_themeManager->setTheme(ThemeManager::Light);
    
    // Clean up test environment
    delete m_testMainWindow;
    delete m_themeIntegration;
    
    qDebug() << "Theme UI workflow tests cleanup completed";
}

void ThemeUIWorkflowTests::init() {
    // Reset to light theme before each test
    m_themeManager->setTheme(ThemeManager::Light);
    QTest::qWait(200);
    
    // Clear previous test results
    m_failedTests.clear();
}

void ThemeUIWorkflowTests::cleanup() {
    // Brief pause between tests
    QTest::qWait(100);
}

void ThemeUIWorkflowTests::setupTestEnvironment() {
    // Create test main window
    m_testMainWindow = new QMainWindow();
    m_testMainWindow->setObjectName("mainWindow");
    m_testMainWindow->setWindowTitle("CloneClean - Theme Workflow Test");
    m_testMainWindow->resize(800, 600);
    
    // Create test widget with common UI components
    m_testWidget = new QWidget();
    m_testWidget->setObjectName("centralWidget");
    
    auto* layout = new QVBoxLayout(m_testWidget);
    
    // Scan controls
    auto* scanButton = new QPushButton("Start Scan");
    scanButton->setObjectName("scanButton");
    layout->addWidget(scanButton);
    
    // Progress indicator
    auto* progressBar = new QProgressBar();
    progressBar->setObjectName("progressBar");
    progressBar->setVisible(false);
    layout->addWidget(progressBar);
    
    // Results area
    auto* resultsLabel = new QLabel("No results yet");
    resultsLabel->setObjectName("resultsLabel");
    layout->addWidget(resultsLabel);
    
    // File selection controls
    auto* selectAllButton = new QPushButton("Select All");
    selectAllButton->setObjectName("selectAllButton");
    selectAllButton->setEnabled(false);
    layout->addWidget(selectAllButton);
    
    // Action buttons
    auto* buttonLayout = new QHBoxLayout();
    auto* deleteButton = new QPushButton("Delete Selected");
    deleteButton->setObjectName("deleteButton");
    deleteButton->setEnabled(false);
    
    auto* settingsButton = new QPushButton("Settings");
    settingsButton->setObjectName("settingsButton");
    
    buttonLayout->addWidget(deleteButton);
    buttonLayout->addWidget(settingsButton);
    layout->addLayout(buttonLayout);
    
    m_testMainWindow->setCentralWidget(m_testWidget);
    m_testMainWindow->show();
    
    // Register with theme manager
    m_themeManager->registerComponent(m_testWidget, ThemeManager::ComponentType::Widget);
    m_themeManager->applyToWidget(m_testWidget);
    
    QTest::qWait(300); // Allow UI to stabilize
}

void ThemeUIWorkflowTests::createTestWorkflows() {
    // Create comprehensive test workflows
    m_testWorkflows["scan_to_delete"] = createScanToDeleteWorkflowWithThemes();
    m_testWorkflows["results_viewing"] = createResultsViewingWorkflow();
    m_testWorkflows["settings_config"] = createSettingsConfigurationWorkflow();
    m_testWorkflows["file_operations"] = createFileOperationWorkflow();
    m_testWorkflows["error_recovery"] = createErrorRecoveryWorkflow();
    
    // Register workflows with framework
    for (auto it = m_testWorkflows.begin(); it != m_testWorkflows.end(); ++it) {
        m_workflowTesting->registerWorkflow(it.value());
    }
}

UserWorkflow ThemeUIWorkflowTests::createScanToDeleteWorkflowWithThemes() {
    UserWorkflow workflow;
    workflow.id = "theme_aware_scan_to_delete";
    workflow.name = "Theme-Aware Scan to Delete Workflow";
    workflow.description = "Complete scan-to-delete workflow with theme validation at each step";
    workflow.category = "core_functionality";
    workflow.tags = {"scan", "delete", "themes", "comprehensive"};
    workflow.author = "ThemeUIWorkflowTests";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Validate initial theme state
    WorkflowStep themeValidationStep = createThemeValidationStep("validate_initial_theme", ThemeManager::Light);
    workflow.steps.append(themeValidationStep);
    
    // Step 2: Click scan button
    WorkflowStep scanStep = createThemeAwareUIStep("click_scan_button", "click", 
        {{"selector", "scanButton"}, {"validate_theme", true}});
    workflow.steps.append(scanStep);
    
    // Step 3: Validate progress indication
    WorkflowStep progressStep = createThemeAwareUIStep("validate_progress", "validate_visibility", 
        {{"selector", "progressBar"}, {"should_be_visible", true}});
    workflow.steps.append(progressStep);
    
    // Step 4: Wait for scan completion
    WorkflowStep waitStep;
    waitStep.id = "wait_scan_complete";
    waitStep.name = "Wait for Scan Completion";
    waitStep.type = WorkflowStepType::Wait;
    waitStep.parameters["condition"] = "scan_completed";
    waitStep.parameters["timeout_ms"] = 10000;
    workflow.steps.append(waitStep);
    
    // Step 5: Validate results display
    WorkflowStep resultsStep = createThemeAwareUIStep("validate_results", "validate_content", 
        {{"selector", "resultsLabel"}, {"expected_content", "scan_results"}});
    workflow.steps.append(resultsStep);
    
    // Step 6: Test theme switching during workflow
    WorkflowStep themeSwitchStep;
    themeSwitchStep.id = "switch_to_dark_theme";
    themeSwitchStep.name = "Switch to Dark Theme";
    themeSwitchStep.type = WorkflowStepType::Custom;
    themeSwitchStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        return m_themeManager->setTheme(ThemeManager::Dark);
    };
    workflow.steps.append(themeSwitchStep);
    
    // Step 7: Validate UI after theme switch
    WorkflowStep postThemeValidationStep = createThemeValidationStep("validate_dark_theme", ThemeManager::Dark);
    workflow.steps.append(postThemeValidationStep);
    
    // Step 8: Select files for deletion
    WorkflowStep selectStep = createThemeAwareUIStep("select_files", "click", 
        {{"selector", "selectAllButton"}, {"validate_theme", true}});
    workflow.steps.append(selectStep);
    
    // Step 9: Delete selected files
    WorkflowStep deleteStep = createThemeAwareUIStep("delete_files", "click", 
        {{"selector", "deleteButton"}, {"validate_theme", true}});
    workflow.steps.append(deleteStep);
    
    // Step 10: Final validation
    WorkflowStep finalValidationStep = createAccessibilityValidationStep("final_accessibility_check", "deleteButton");
    workflow.steps.append(finalValidationStep);
    
    return workflow;
}

UserWorkflow ThemeUIWorkflowTests::createResultsViewingWorkflow() {
    UserWorkflow workflow;
    workflow.id = "theme_aware_results_viewing";
    workflow.name = "Theme-Aware Results Viewing Workflow";
    workflow.description = "Comprehensive results viewing and file selection workflow with theme validation";
    workflow.category = "results_management";
    workflow.tags = {"results", "viewing", "selection", "themes"};
    workflow.author = "ThemeUIWorkflowTests";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Validate results display in light theme
    WorkflowStep lightThemeResultsStep = createThemeAwareUIStep("validate_light_results", "validate_visibility", 
        {{"selector", "resultsLabel"}, {"theme", static_cast<int>(ThemeManager::Light)}});
    workflow.steps.append(lightThemeResultsStep);
    
    // Step 2: Switch to dark theme
    WorkflowStep switchToDarkStep;
    switchToDarkStep.id = "switch_to_dark_for_results";
    switchToDarkStep.name = "Switch to Dark Theme for Results";
    switchToDarkStep.type = WorkflowStepType::Custom;
    switchToDarkStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        return m_themeManager->setTheme(ThemeManager::Dark);
    };
    workflow.steps.append(switchToDarkStep);
    
    // Step 3: Validate results display in dark theme
    WorkflowStep darkThemeResultsStep = createThemeAwareUIStep("validate_dark_results", "validate_visibility", 
        {{"selector", "resultsLabel"}, {"theme", static_cast<int>(ThemeManager::Dark)}});
    workflow.steps.append(darkThemeResultsStep);
    
    // Step 4: Test file selection in dark theme
    WorkflowStep selectionStep = createThemeAwareUIStep("test_selection_dark", "click", 
        {{"selector", "selectAllButton"}, {"validate_theme", true}});
    workflow.steps.append(selectionStep);
    
    // Step 5: Switch back to light theme
    WorkflowStep switchToLightStep;
    switchToLightStep.id = "switch_back_to_light";
    switchToLightStep.name = "Switch Back to Light Theme";
    switchToLightStep.type = WorkflowStepType::Custom;
    switchToLightStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        return m_themeManager->setTheme(ThemeManager::Light);
    };
    workflow.steps.append(switchToLightStep);
    
    // Step 6: Validate selection state maintained after theme switch
    WorkflowStep selectionStateStep = createThemeAwareUIStep("validate_selection_state", "validate_state", 
        {{"selector", "selectAllButton"}, {"expected_state", "enabled"}});
    workflow.steps.append(selectionStateStep);
    
    return workflow;
}

UserWorkflow ThemeUIWorkflowTests::createSettingsConfigurationWorkflow() {
    UserWorkflow workflow;
    workflow.id = "theme_aware_settings_config";
    workflow.name = "Theme-Aware Settings Configuration Workflow";
    workflow.description = "Settings and preferences workflow with theme integration testing";
    workflow.category = "configuration";
    workflow.tags = {"settings", "preferences", "themes", "configuration"};
    workflow.author = "ThemeUIWorkflowTests";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Open settings in light theme
    WorkflowStep openSettingsStep = createThemeAwareUIStep("open_settings_light", "click", 
        {{"selector", "settingsButton"}, {"validate_theme", true}});
    workflow.steps.append(openSettingsStep);
    
    // Step 2: Validate settings dialog visibility
    WorkflowStep validateSettingsStep = createThemeAwareUIStep("validate_settings_dialog", "validate_visibility", 
        {{"selector", "settingsDialog"}, {"should_be_visible", true}});
    workflow.steps.append(validateSettingsStep);
    
    // Step 3: Switch theme while settings are open
    WorkflowStep themeSwitchInSettingsStep;
    themeSwitchInSettingsStep.id = "switch_theme_in_settings";
    themeSwitchInSettingsStep.name = "Switch Theme While Settings Open";
    themeSwitchInSettingsStep.type = WorkflowStepType::Custom;
    themeSwitchInSettingsStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        return m_themeManager->setTheme(ThemeManager::Dark);
    };
    workflow.steps.append(themeSwitchInSettingsStep);
    
    // Step 4: Validate settings dialog still accessible in new theme
    WorkflowStep validateSettingsAfterThemeStep = createThemeAwareUIStep("validate_settings_after_theme", "validate_accessibility", 
        {{"selector", "settingsDialog"}, {"theme", static_cast<int>(ThemeManager::Dark)}});
    workflow.steps.append(validateSettingsAfterThemeStep);
    
    return workflow;
}

UserWorkflow ThemeUIWorkflowTests::createFileOperationWorkflow() {
    UserWorkflow workflow;
    workflow.id = "theme_aware_file_operations";
    workflow.name = "Theme-Aware File Operations Workflow";
    workflow.description = "File operation workflow with comprehensive theme validation";
    workflow.category = "file_operations";
    workflow.tags = {"files", "operations", "themes", "validation"};
    workflow.author = "ThemeUIWorkflowTests";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Validate delete button state in light theme
    WorkflowStep validateDeleteLightStep = createThemeAwareUIStep("validate_delete_light", "validate_state", 
        {{"selector", "deleteButton"}, {"expected_state", "disabled"}, {"theme", static_cast<int>(ThemeManager::Light)}});
    workflow.steps.append(validateDeleteLightStep);
    
    // Step 2: Enable delete button by selecting files
    WorkflowStep enableDeleteStep = createThemeAwareUIStep("enable_delete", "click", 
        {{"selector", "selectAllButton"}, {"validate_theme", true}});
    workflow.steps.append(enableDeleteStep);
    
    // Step 3: Switch to dark theme
    WorkflowStep switchToDarkForDeleteStep;
    switchToDarkForDeleteStep.id = "switch_to_dark_for_delete";
    switchToDarkForDeleteStep.name = "Switch to Dark Theme for Delete";
    switchToDarkForDeleteStep.type = WorkflowStepType::Custom;
    switchToDarkForDeleteStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        return m_themeManager->setTheme(ThemeManager::Dark);
    };
    workflow.steps.append(switchToDarkForDeleteStep);
    
    // Step 4: Validate delete button still enabled in dark theme
    WorkflowStep validateDeleteDarkStep = createThemeAwareUIStep("validate_delete_dark", "validate_state", 
        {{"selector", "deleteButton"}, {"expected_state", "enabled"}, {"theme", static_cast<int>(ThemeManager::Dark)}});
    workflow.steps.append(validateDeleteDarkStep);
    
    // Step 5: Perform delete operation
    WorkflowStep performDeleteStep = createThemeAwareUIStep("perform_delete", "click", 
        {{"selector", "deleteButton"}, {"validate_theme", true}});
    workflow.steps.append(performDeleteStep);
    
    return workflow;
}

UserWorkflow ThemeUIWorkflowTests::createErrorRecoveryWorkflow() {
    UserWorkflow workflow;
    workflow.id = "theme_aware_error_recovery";
    workflow.name = "Theme-Aware Error Recovery Workflow";
    workflow.description = "Error handling and recovery workflow with theme validation";
    workflow.category = "error_handling";
    workflow.tags = {"error", "recovery", "themes", "resilience"};
    workflow.author = "ThemeUIWorkflowTests";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Simulate error condition
    WorkflowStep simulateErrorStep;
    simulateErrorStep.id = "simulate_error";
    simulateErrorStep.name = "Simulate Error Condition";
    simulateErrorStep.type = WorkflowStepType::Custom;
    simulateErrorStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        // Simulate an error by temporarily disabling a component
        QWidget* scanButton = m_testWidget->findChild<QWidget*>("scanButton");
        if (scanButton) {
            scanButton->setEnabled(false);
            return true;
        }
        return false;
    };
    workflow.steps.append(simulateErrorStep);
    
    // Step 2: Switch theme during error state
    WorkflowStep themeSwitchDuringErrorStep;
    themeSwitchDuringErrorStep.id = "switch_theme_during_error";
    themeSwitchDuringErrorStep.name = "Switch Theme During Error";
    themeSwitchDuringErrorStep.type = WorkflowStepType::Custom;
    themeSwitchDuringErrorStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        return m_themeManager->setTheme(ThemeManager::Dark);
    };
    workflow.steps.append(themeSwitchDuringErrorStep);
    
    // Step 3: Validate error state maintained after theme switch
    WorkflowStep validateErrorStateStep = createThemeAwareUIStep("validate_error_state", "validate_state", 
        {{"selector", "scanButton"}, {"expected_state", "disabled"}, {"theme", static_cast<int>(ThemeManager::Dark)}});
    workflow.steps.append(validateErrorStateStep);
    
    // Step 4: Recover from error
    WorkflowStep recoverErrorStep;
    recoverErrorStep.id = "recover_error";
    recoverErrorStep.name = "Recover from Error";
    recoverErrorStep.type = WorkflowStepType::Custom;
    recoverErrorStep.customAction = [this](const QMap<QString, QVariant>& params) -> bool {
        Q_UNUSED(params)
        // Recover by re-enabling the component
        QWidget* scanButton = m_testWidget->findChild<QWidget*>("scanButton");
        if (scanButton) {
            scanButton->setEnabled(true);
            return true;
        }
        return false;
    };
    workflow.steps.append(recoverErrorStep);
    
    // Step 5: Validate recovery in current theme
    WorkflowStep validateRecoveryStep = createThemeAwareUIStep("validate_recovery", "validate_state", 
        {{"selector", "scanButton"}, {"expected_state", "enabled"}, {"theme", static_cast<int>(ThemeManager::Dark)}});
    workflow.steps.append(validateRecoveryStep);
    
    return workflow;
}

// Implementation of test methods

void ThemeUIWorkflowTests::testScanToDeleteWorkflowAcrossThemes() {
    qDebug() << "Testing scan-to-delete workflow across all themes...";
    
    const UserWorkflow& workflow = m_testWorkflows["scan_to_delete"];
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        qDebug() << "Testing scan-to-delete workflow in theme:" << static_cast<int>(theme);
        
        // Set initial theme
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Execute workflow
        WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
        
        // Validate results
        validateTestResults(result, "scan_to_delete");
        QVERIFY2(result.success, QString("Scan-to-delete workflow failed in theme %1").arg(static_cast<int>(theme)).toUtf8());
        
        // Store results for analysis
        m_workflowResults["scan_to_delete"][theme] = result;
        
        // Validate theme-specific requirements
        QVERIFY(validateWorkflowInTheme(workflow, theme));
        QVERIFY(validateUIStateConsistency(workflow, theme));
        QVERIFY(validateWorkflowAccessibility(workflow, theme));
        QVERIFY(validateWorkflowPerformance(workflow, theme, m_maxAcceptableWorkflowTime));
    }
    
    qDebug() << "Scan-to-delete workflow testing completed successfully";
}

void ThemeUIWorkflowTests::testResultsViewingAndSelectionWorkflow() {
    qDebug() << "Testing results viewing and selection workflow...";
    
    const UserWorkflow& workflow = m_testWorkflows["results_viewing"];
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        qDebug() << "Testing results viewing workflow in theme:" << static_cast<int>(theme);
        
        // Set initial theme
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Execute workflow
        WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
        
        // Validate results
        validateTestResults(result, "results_viewing");
        QVERIFY2(result.success, QString("Results viewing workflow failed in theme %1").arg(static_cast<int>(theme)).toUtf8());
        
        // Store results
        m_workflowResults["results_viewing"][theme] = result;
        
        // Validate theme-specific requirements
        QVERIFY(validateWorkflowInTheme(workflow, theme));
        QVERIFY(validateUIStateConsistency(workflow, theme));
    }
    
    qDebug() << "Results viewing and selection workflow testing completed successfully";
}

void ThemeUIWorkflowTests::testSettingsAndPreferencesWorkflowWithThemes() {
    qDebug() << "Testing settings and preferences workflow with theme integration...";
    
    const UserWorkflow& workflow = m_testWorkflows["settings_config"];
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        qDebug() << "Testing settings workflow in theme:" << static_cast<int>(theme);
        
        // Set initial theme
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Execute workflow
        WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
        
        // Validate results
        validateTestResults(result, "settings_config");
        QVERIFY2(result.success, QString("Settings workflow failed in theme %1").arg(static_cast<int>(theme)).toUtf8());
        
        // Store results
        m_workflowResults["settings_config"][theme] = result;
        
        // Validate theme integration
        QVERIFY(validateWorkflowInTheme(workflow, theme));
    }
    
    qDebug() << "Settings and preferences workflow testing completed successfully";
}

void ThemeUIWorkflowTests::testFileOperationWorkflowValidation() {
    qDebug() << "Testing file operation workflow validation...";
    
    const UserWorkflow& workflow = m_testWorkflows["file_operations"];
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        qDebug() << "Testing file operations workflow in theme:" << static_cast<int>(theme);
        
        // Set initial theme
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Execute workflow
        WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
        
        // Validate results
        validateTestResults(result, "file_operations");
        QVERIFY2(result.success, QString("File operations workflow failed in theme %1").arg(static_cast<int>(theme)).toUtf8());
        
        // Store results
        m_workflowResults["file_operations"][theme] = result;
        
        // Validate file operation specific requirements
        QVERIFY(validateWorkflowInTheme(workflow, theme));
        QVERIFY(validateUIStateConsistency(workflow, theme));
    }
    
    qDebug() << "File operation workflow validation completed successfully";
}

void ThemeUIWorkflowTests::testErrorRecoveryWorkflowAcrossThemes() {
    qDebug() << "Testing error recovery workflow across themes...";
    
    const UserWorkflow& workflow = m_testWorkflows["error_recovery"];
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        qDebug() << "Testing error recovery workflow in theme:" << static_cast<int>(theme);
        
        // Set initial theme
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Execute workflow
        WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
        
        // Validate results
        validateTestResults(result, "error_recovery");
        QVERIFY2(result.success, QString("Error recovery workflow failed in theme %1").arg(static_cast<int>(theme)).toUtf8());
        
        // Store results
        m_workflowResults["error_recovery"][theme] = result;
        
        // Validate error recovery specific requirements
        QVERIFY(validateWorkflowInTheme(workflow, theme));
    }
    
    qDebug() << "Error recovery workflow testing completed successfully";
}

// Task 11.2 implementation: Cross-theme interaction validation

void ThemeUIWorkflowTests::testUserInteractionsInLightTheme() {
    qDebug() << "Testing user interactions in light theme...";
    
    // Set light theme
    QVERIFY(m_themeManager->setTheme(ThemeManager::Light));
    QTest::qWait(300);
    
    // Test basic interactions
    QStringList interactions = {"click_scan", "click_settings", "click_select_all", "click_delete"};
    QList<ThemeManager::Theme> lightThemeOnly = {ThemeManager::Light};
    
    QVERIFY(testInteractionConsistency(interactions, lightThemeOnly));
    
    // Validate all components are accessible in light theme
    QList<UIThemeTestIntegration::ThemeAwareSelector> selectors = UIThemeTestIntegration::createCommonUISelectors();
    QVERIFY(m_themeIntegration->validateCurrentThemeCompliance(selectors));
    
    qDebug() << "Light theme interaction testing completed successfully";
}

void ThemeUIWorkflowTests::testUserInteractionsInDarkTheme() {
    qDebug() << "Testing user interactions in dark theme...";
    
    // Set dark theme
    QVERIFY(m_themeManager->setTheme(ThemeManager::Dark));
    QTest::qWait(300);
    
    // Test basic interactions
    QStringList interactions = {"click_scan", "click_settings", "click_select_all", "click_delete"};
    QList<ThemeManager::Theme> darkThemeOnly = {ThemeManager::Dark};
    
    QVERIFY(testInteractionConsistency(interactions, darkThemeOnly));
    
    // Validate all components are accessible in dark theme
    QList<UIThemeTestIntegration::ThemeAwareSelector> selectors = UIThemeTestIntegration::createCommonUISelectors();
    QVERIFY(m_themeIntegration->validateCurrentThemeCompliance(selectors));
    
    qDebug() << "Dark theme interaction testing completed successfully";
}

void ThemeUIWorkflowTests::testUIStateMaintenanceThroughoutWorkflows() {
    qDebug() << "Testing UI state maintenance throughout workflows...";
    
    for (const auto& workflowPair : m_testWorkflows.toStdMap()) {
        const UserWorkflow& workflow = workflowPair.second;
        
        for (ThemeManager::Theme theme : m_supportedThemes) {
            qDebug() << "Testing UI state maintenance for workflow" << workflow.name << "in theme" << static_cast<int>(theme);
            
            QVERIFY(validateUIStateConsistency(workflow, theme));
        }
    }
    
    qDebug() << "UI state maintenance testing completed successfully";
}

void ThemeUIWorkflowTests::testConsistentUIBehaviorAcrossWorkflowSteps() {
    qDebug() << "Testing consistent UI behavior across workflow steps...";
    
    for (const auto& workflowPair : m_testWorkflows.toStdMap()) {
        const UserWorkflow& workflow = workflowPair.second;
        
        QVERIFY(testWorkflowStepConsistency(workflow, m_supportedThemes));
    }
    
    qDebug() << "Consistent UI behavior testing completed successfully";
}

void ThemeUIWorkflowTests::testThemeTransitionDuringWorkflows() {
    qDebug() << "Testing theme transitions during workflows...";
    
    for (const auto& workflowPair : m_testWorkflows.toStdMap()) {
        const UserWorkflow& workflow = workflowPair.second;
        
        QVERIFY(validateThemeTransitionDuringWorkflow(workflow));
    }
    
    qDebug() << "Theme transition testing completed successfully";
}

// Comprehensive validation methods

void ThemeUIWorkflowTests::testCompleteUserJourneyAcrossAllThemes() {
    qDebug() << "Testing complete user journey across all themes...";
    
    // Create a comprehensive user scenario that combines multiple workflows
    UserScenario comprehensiveScenario = m_scenarioTesting->createFirstTimeUserScenario();
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        qDebug() << "Testing complete user journey in theme:" << static_cast<int>(theme);
        
        // Set theme
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Execute comprehensive scenario
        ScenarioResult result = m_scenarioTesting->executeScenario(comprehensiveScenario);
        
        QVERIFY2(result.success, QString("Complete user journey failed in theme %1").arg(static_cast<int>(theme)).toUtf8());
        QVERIFY(result.satisfactionScore >= 7); // Minimum satisfaction score
        QVERIFY(result.usabilityIssues.size() <= 3); // Maximum usability issues
    }
    
    qDebug() << "Complete user journey testing completed successfully";
}

void ThemeUIWorkflowTests::testWorkflowPerformanceAcrossThemes() {
    qDebug() << "Testing workflow performance across themes...";
    
    for (const auto& workflowPair : m_testWorkflows.toStdMap()) {
        const UserWorkflow& workflow = workflowPair.second;
        
        for (ThemeManager::Theme theme : m_supportedThemes) {
            QVERIFY(validateWorkflowPerformance(workflow, theme, m_maxAcceptableWorkflowTime));
        }
    }
    
    qDebug() << "Workflow performance testing completed successfully";
}

void ThemeUIWorkflowTests::testAccessibilityWorkflowCompliance() {
    qDebug() << "Testing accessibility workflow compliance...";
    
    for (const auto& workflowPair : m_testWorkflows.toStdMap()) {
        const UserWorkflow& workflow = workflowPair.second;
        
        for (ThemeManager::Theme theme : m_supportedThemes) {
            QVERIFY(validateWorkflowAccessibility(workflow, theme));
        }
    }
    
    qDebug() << "Accessibility workflow compliance testing completed successfully";
}

// Helper method implementations

WorkflowStep ThemeUIWorkflowTests::createThemeAwareUIStep(const QString& stepId, const QString& action, 
                                                         const QMap<QString, QVariant>& parameters) {
    WorkflowStep step;
    step.id = stepId;
    step.name = QString("Theme-Aware %1").arg(action);
    step.type = WorkflowStepType::UIAction;
    step.parameters = parameters;
    step.parameters["action"] = action;
    step.parameters["theme_aware"] = true;
    
    // Add theme validation to the step
    step.validator = [this, parameters]() -> bool {
        if (parameters.value("validate_theme", false).toBool()) {
            QString selector = parameters.value("selector").toString();
            if (!selector.isEmpty()) {
                auto themeSelector = UIThemeTestIntegration::createSelector(selector);
                return m_themeIntegration->validateWidgetThemeCompliance(themeSelector);
            }
        }
        return true;
    };
    
    return step;
}

WorkflowStep ThemeUIWorkflowTests::createThemeValidationStep(const QString& stepId, ThemeManager::Theme expectedTheme) {
    WorkflowStep step;
    step.id = stepId;
    step.name = QString("Validate Theme: %1").arg(static_cast<int>(expectedTheme));
    step.type = WorkflowStepType::Validation;
    step.parameters["expected_theme"] = static_cast<int>(expectedTheme);
    
    step.validator = [this, expectedTheme]() -> bool {
        return m_themeManager->currentTheme() == expectedTheme;
    };
    
    return step;
}

WorkflowStep ThemeUIWorkflowTests::createAccessibilityValidationStep(const QString& stepId, const QString& componentName) {
    WorkflowStep step;
    step.id = stepId;
    step.name = QString("Validate Accessibility: %1").arg(componentName);
    step.type = WorkflowStepType::Validation;
    step.parameters["component"] = componentName;
    
    step.validator = [this, componentName]() -> bool {
        auto selector = UIThemeTestIntegration::createAccessibilitySelector(componentName);
        return m_themeIntegration->validateWidgetThemeCompliance(selector);
    };
    
    return step;
}

bool ThemeUIWorkflowTests::validateWorkflowInTheme(const UserWorkflow& workflow, ThemeManager::Theme theme) {
    // Set the theme
    if (!m_themeManager->setTheme(theme)) {
        return false;
    }
    
    QTest::qWait(200); // Allow theme to apply
    
    // Validate that all UI components are properly themed
    QList<UIThemeTestIntegration::ThemeAwareSelector> selectors = UIThemeTestIntegration::createCommonUISelectors();
    return m_themeIntegration->validateCurrentThemeCompliance(selectors);
}

bool ThemeUIWorkflowTests::validateUIStateConsistency(const UserWorkflow& workflow, ThemeManager::Theme theme) {
    // This would validate that UI state is maintained consistently throughout the workflow
    // For now, we'll do a basic validation
    return validateWorkflowInTheme(workflow, theme);
}

bool ThemeUIWorkflowTests::validateWorkflowAccessibility(const UserWorkflow& workflow, ThemeManager::Theme theme) {
    // Set the theme
    if (!m_themeManager->setTheme(theme)) {
        return false;
    }
    
    QTest::qWait(200);
    
    // Validate accessibility compliance
    QList<UIThemeTestIntegration::ThemeAwareSelector> selectors = UIThemeTestIntegration::createCommonUISelectors();
    return m_themeIntegration->testAccessibilityAcrossThemes(selectors);
}

bool ThemeUIWorkflowTests::validateWorkflowPerformance(const UserWorkflow& workflow, ThemeManager::Theme theme, double maxTimeMs) {
    m_performanceTimer.start();
    
    // Set theme and measure time
    if (!m_themeManager->setTheme(theme)) {
        return false;
    }
    
    // Execute workflow and measure performance
    WorkflowResult result = m_workflowTesting->executeWorkflow(workflow);
    
    double elapsedTime = m_performanceTimer.elapsed();
    
    return result.success && elapsedTime <= maxTimeMs;
}

bool ThemeUIWorkflowTests::testInteractionConsistency(const QStringList& interactions, const QList<ThemeManager::Theme>& themes) {
    for (ThemeManager::Theme theme : themes) {
        if (!m_themeManager->setTheme(theme)) {
            return false;
        }
        
        QTest::qWait(200);
        
        // Test each interaction
        for (const QString& interaction : interactions) {
            // This would test specific interactions - simplified for now
            if (!m_themeIntegration->validateCurrentThemeCompliance(UIThemeTestIntegration::createCommonUISelectors())) {
                return false;
            }
        }
    }
    
    return true;
}

bool ThemeUIWorkflowTests::validateThemeTransitionDuringWorkflow(const UserWorkflow& workflow) {
    // Start in light theme
    if (!m_themeManager->setTheme(ThemeManager::Light)) {
        return false;
    }
    
    // Execute first half of workflow
    QList<WorkflowStep> firstHalf = workflow.steps.mid(0, workflow.steps.size() / 2);
    
    for (const WorkflowStep& step : firstHalf) {
        QMap<QString, QVariant> context;
        if (!m_workflowTesting->executeWorkflowStep(step, context)) {
            return false;
        }
    }
    
    // Switch to dark theme
    if (!m_themeManager->setTheme(ThemeManager::Dark)) {
        return false;
    }
    
    QTest::qWait(300);
    
    // Execute second half of workflow
    QList<WorkflowStep> secondHalf = workflow.steps.mid(workflow.steps.size() / 2);
    
    for (const WorkflowStep& step : secondHalf) {
        QMap<QString, QVariant> context;
        if (!m_workflowTesting->executeWorkflowStep(step, context)) {
            return false;
        }
    }
    
    return true;
}

bool ThemeUIWorkflowTests::testWorkflowStepConsistency(const UserWorkflow& workflow, const QList<ThemeManager::Theme>& themes) {
    for (ThemeManager::Theme theme : themes) {
        if (!validateWorkflowInTheme(workflow, theme)) {
            return false;
        }
    }
    
    return true;
}

void ThemeUIWorkflowTests::validateTestResults(const WorkflowResult& result, const QString& workflowName) {
    if (!result.success) {
        m_failedTests.append(QString("%1: %2").arg(workflowName, result.validationErrors.join(", ")));
    }
    
    // Log performance metrics
    qDebug() << "Workflow" << workflowName << "completed in" << result.executionTimeMs << "ms";
    qDebug() << "Steps completed:" << result.completedSteps << "/" << result.totalSteps;
    
    if (result.failedSteps > 0) {
        qDebug() << "Failed steps:" << result.failedStepIds.join(", ");
    }
}

QTEST_MAIN(ThemeUIWorkflowTests)
#include "theme_ui_workflow_tests.moc"