#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QWidget>
#include <memory>

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
 * This class implements tasks 11.1 and 11.2 from the UI/UX architect review fixes:
 * 
 * Task 11.1: Create complete workflow tests using WorkflowTesting
 * - Implement scan-to-delete workflow testing across all themes
 * - Add results viewing and file selection workflow validation
 * - Create settings and preferences workflow testing with theme integration
 * 
 * Task 11.2: Add cross-theme interaction validation
 * - Test all user interactions work correctly in both light and dark themes
 * - Validate UI state maintenance throughout complete user workflows
 * - Ensure consistent UI behavior across all workflow steps
 * 
 * The class integrates with the existing comprehensive testing framework including:
 * - WorkflowTesting: For complete user journey validation
 * - UserScenarioTesting: For persona-based testing scenarios
 * - UIAutomation: For automated UI interactions
 * - VisualTesting: For visual regression testing
 * - ThemeAccessibilityTesting: For accessibility compliance
 * - UIThemeTestIntegration: For theme-aware testing capabilities
 */
class ThemeUIWorkflowTests : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Workflow validation result for theme-specific testing
     */
    struct ThemeWorkflowResult {
        QString workflowName;                           ///< Name of the workflow tested
        ThemeManager::Theme theme;                      ///< Theme used for testing
        bool success = false;                           ///< Overall success status
        qint64 executionTimeMs = 0;                    ///< Total execution time
        int completedSteps = 0;                        ///< Number of completed steps
        int failedSteps = 0;                           ///< Number of failed steps
        QStringList themeViolations;                   ///< Theme compliance violations
        QStringList accessibilityIssues;              ///< Accessibility issues found
        QStringList usabilityIssues;                  ///< Usability issues identified
        double performanceScore = 0.0;                ///< Performance score (0-100)
        double accessibilityScore = 0.0;              ///< Accessibility score (0-100)
        QMap<QString, QVariant> metrics;              ///< Additional metrics
    };

    /**
     * @brief Cross-theme interaction validation result
     */
    struct CrossThemeInteractionResult {
        QString interactionName;                       ///< Name of interaction tested
        QList<ThemeManager::Theme> testedThemes;      ///< Themes tested
        bool consistentBehavior = false;              ///< Whether behavior is consistent across themes
        QMap<ThemeManager::Theme, bool> themeResults; ///< Results per theme
        QStringList inconsistencies;                  ///< Identified inconsistencies
        double consistencyScore = 0.0;               ///< Consistency score (0-100)
    };

    /**
     * @brief Comprehensive test report for all theme workflow validation
     */
    struct ComprehensiveTestReport {
        QDateTime testStartTime;                       ///< Test execution start time
        QDateTime testEndTime;                         ///< Test execution end time
        int totalWorkflowsTested = 0;                 ///< Total number of workflows tested
        int successfulWorkflows = 0;                  ///< Number of successful workflows
        int failedWorkflows = 0;                      ///< Number of failed workflows
        QList<ThemeWorkflowResult> workflowResults;   ///< Detailed workflow results
        QList<CrossThemeInteractionResult> interactionResults; ///< Cross-theme interaction results
        double overallSuccessRate = 0.0;             ///< Overall success rate (0-100)
        double averagePerformanceScore = 0.0;        ///< Average performance score
        double averageAccessibilityScore = 0.0;     ///< Average accessibility score
        QStringList criticalIssues;                  ///< Critical issues requiring immediate attention
        QStringList recommendations;                  ///< Recommendations for improvement
        QMap<QString, QVariant> summaryMetrics;      ///< Summary metrics
    };

    explicit ThemeUIWorkflowTests(QObject* parent = nullptr);
    ~ThemeUIWorkflowTests();

    // Core workflow testing methods (Task 11.1)
    bool testScanToDeleteWorkflowAcrossThemes();
    bool testResultsViewingAndSelectionWorkflow();
    bool testSettingsAndPreferencesWorkflowWithThemes();
    bool testFileOperationWorkflowValidation();
    bool testErrorRecoveryWorkflowAcrossThemes();

    // Cross-theme interaction validation methods (Task 11.2)
    bool testUserInteractionsInLightTheme();
    bool testUserInteractionsInDarkTheme();
    bool testUIStateMaintenanceThroughoutWorkflows();
    bool testConsistentUIBehaviorAcrossWorkflowSteps();
    bool testThemeTransitionDuringWorkflows();

    // Comprehensive validation methods
    bool testCompleteUserJourneyAcrossAllThemes();
    bool testWorkflowPerformanceAcrossThemes();
    bool testAccessibilityWorkflowCompliance();

    // Workflow creation and management
    UserWorkflow createScanToDeleteWorkflowWithThemes();
    UserWorkflow createResultsViewingWorkflow();
    UserWorkflow createSettingsConfigurationWorkflow();
    UserWorkflow createFileOperationWorkflow();
    UserWorkflow createErrorRecoveryWorkflow();

    // Theme-aware workflow validation
    bool validateWorkflowInTheme(const UserWorkflow& workflow, ThemeManager::Theme theme);
    bool validateUIStateConsistency(const UserWorkflow& workflow, ThemeManager::Theme theme);
    bool validateWorkflowAccessibility(const UserWorkflow& workflow, ThemeManager::Theme theme);
    bool validateWorkflowPerformance(const UserWorkflow& workflow, ThemeManager::Theme theme, double maxTimeMs);

    // Cross-theme interaction validation
    bool testInteractionConsistency(const QStringList& interactions, const QList<ThemeManager::Theme>& themes);
    bool validateThemeTransitionDuringWorkflow(const UserWorkflow& workflow);
    bool testWorkflowStepConsistency(const UserWorkflow& workflow, const QList<ThemeManager::Theme>& themes);

    // Reporting and analysis
    ComprehensiveTestReport generateComprehensiveReport();
    ThemeWorkflowResult analyzeWorkflowResult(const WorkflowResult& result, const QString& workflowName, ThemeManager::Theme theme);
    CrossThemeInteractionResult analyzeInteractionConsistency(const QString& interactionName, const QList<ThemeManager::Theme>& themes);

    // Configuration and setup
    void setMaxAcceptableWorkflowTime(double timeMs);
    void setSupportedThemes(const QList<ThemeManager::Theme>& themes);
    void enableDetailedLogging(bool enable);
    void setTestEnvironment(QMainWindow* mainWindow, QWidget* testWidget);

    // Utility methods
    WorkflowStep createThemeAwareUIStep(const QString& stepId, const QString& action, 
                                       const QMap<QString, QVariant>& parameters);
    WorkflowStep createThemeValidationStep(const QString& stepId, ThemeManager::Theme expectedTheme);
    WorkflowStep createAccessibilityValidationStep(const QString& stepId, const QString& componentName);

signals:
    void workflowTestStarted(const QString& workflowName, ThemeManager::Theme theme);
    void workflowTestCompleted(const QString& workflowName, ThemeManager::Theme theme, bool success);
    void themeTransitionDetected(ThemeManager::Theme fromTheme, ThemeManager::Theme toTheme);
    void accessibilityIssueFound(const QString& workflowName, const QString& issue);
    void performanceIssueDetected(const QString& workflowName, double executionTimeMs);
    void crossThemeInconsistencyFound(const QString& interaction, const QStringList& details);
    void comprehensiveTestCompleted(const ComprehensiveTestReport& report);

private slots:
    void onWorkflowStarted(const QString& workflowId);
    void onWorkflowCompleted(const QString& workflowId, const WorkflowResult& result);
    void onThemeChanged(ThemeManager::Theme theme, const QString& themeName);

private:
    // Framework integration setup
    void initializeTestingFrameworks();
    void connectFrameworkSignals();
    void setupTestEnvironment();
    void createTestWorkflows();

    // Test execution helpers
    void executeWorkflowInAllThemes(const UserWorkflow& workflow);
    void validateWorkflowResults(const QMap<ThemeManager::Theme, WorkflowResult>& results);
    void recordWorkflowMetrics(const WorkflowResult& result, ThemeManager::Theme theme);

    // Theme transition testing
    void testThemeTransitionAtStep(const UserWorkflow& workflow, int stepIndex);
    void validateUIStateAfterThemeTransition(ThemeManager::Theme fromTheme, ThemeManager::Theme toTheme);

    // Performance and accessibility validation
    void measureWorkflowPerformance(const UserWorkflow& workflow, ThemeManager::Theme theme);
    void validateAccessibilityCompliance(const UserWorkflow& workflow, ThemeManager::Theme theme);
    void checkContrastRatios(ThemeManager::Theme theme);
    void validateKeyboardNavigation(ThemeManager::Theme theme);

    // Result analysis and reporting
    void analyzeTestResults();
    void generatePerformanceMetrics();
    void identifyCommonIssues();
    void createRecommendations();

    // Validation helpers
    void validateTestResults(const WorkflowResult& result, const QString& workflowName);
    bool isWorkflowStepThemeAware(const WorkflowStep& step);
    bool validateStepThemeCompliance(const WorkflowStep& step, ThemeManager::Theme theme);

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
    double m_maxAcceptableWorkflowTime;
    bool m_detailedLogging;
    
    // Performance tracking
    QElapsedTimer m_performanceTimer;
    QMap<QString, QMap<ThemeManager::Theme, qint64>> m_workflowExecutionTimes;
    
    // Test results tracking
    QMap<QString, QMap<ThemeManager::Theme, WorkflowResult>> m_workflowResults;
    QMap<QString, QMap<ThemeManager::Theme, ThemeWorkflowResult>> m_themeWorkflowResults;
    QList<CrossThemeInteractionResult> m_interactionResults;
    QStringList m_failedTests;
    QStringList m_criticalIssues;
    
    // Metrics and analysis
    ComprehensiveTestReport m_comprehensiveReport;
    QMap<ThemeManager::Theme, QStringList> m_themeSpecificIssues;
    QMap<QString, double> m_performanceMetrics;
    QMap<QString, double> m_accessibilityScores;
};

/**
 * @brief Convenience macros for theme workflow testing
 */
#define THEME_WORKFLOW_VERIFY(workflow, theme) \
    do { \
        if (!validateWorkflowInTheme(workflow, theme)) { \
            QFAIL(QString("Workflow validation failed in theme: %1").arg(static_cast<int>(theme)).toUtf8().constData()); \
        } \
    } while(0)

#define CROSS_THEME_INTERACTION_VERIFY(interaction, themes) \
    do { \
        if (!testInteractionConsistency(QStringList{interaction}, themes)) { \
            QFAIL(QString("Cross-theme interaction consistency failed: %1").arg(interaction).toUtf8().constData()); \
        } \
    } while(0)

#define WORKFLOW_PERFORMANCE_VERIFY(workflow, theme, maxTime) \
    do { \
        if (!validateWorkflowPerformance(workflow, theme, maxTime)) { \
            QFAIL(QString("Workflow performance validation failed in theme: %1").arg(static_cast<int>(theme)).toUtf8().constData()); \
        } \
    } while(0)

#define ACCESSIBILITY_WORKFLOW_VERIFY(workflow, theme) \
    do { \
        if (!validateWorkflowAccessibility(workflow, theme)) { \
            QFAIL(QString("Workflow accessibility validation failed in theme: %1").arg(static_cast<int>(theme)).toUtf8().constData()); \
        } \
    } while(0)