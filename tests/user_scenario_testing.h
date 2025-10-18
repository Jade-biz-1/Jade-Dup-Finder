#pragma once

#include "workflow_testing.h"
#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>
#include <functional>
#include <memory>

/**
 * @brief User persona types for scenario testing
 */
enum class UserPersona {
    FirstTimeUser,      ///< New user experiencing the application for the first time
    CasualUser,         ///< Occasional user with basic needs
    PowerUser,          ///< Advanced user utilizing complex features
    SafetyFocusedUser,  ///< User prioritizing data safety and backups
    BatchUser,          ///< User processing large amounts of data
    MobileUser,         ///< User on mobile/touch devices
    AccessibilityUser   ///< User requiring accessibility features
};

/**
 * @brief Scenario complexity levels
 */
enum class ScenarioComplexity {
    Simple,             ///< Basic single-task scenarios
    Intermediate,       ///< Multi-step scenarios with some complexity
    Advanced,           ///< Complex scenarios with multiple features
    Expert              ///< Highly complex scenarios with edge cases
};

/**
 * @brief User scenario specification
 */
struct UserScenario {
    QString id;                                         ///< Unique scenario identifier
    QString name;                                       ///< Scenario name
    QString description;                                ///< Detailed scenario description
    UserPersona persona;                                ///< Target user persona
    ScenarioComplexity complexity;                      ///< Scenario complexity level
    QStringList goals;                                  ///< User goals for this scenario
    QStringList prerequisites;                          ///< Required setup/knowledge
    UserWorkflow workflow;                              ///< Workflow implementation
    QMap<QString, QVariant> contextData;               ///< Scenario-specific context
    QStringList successCriteria;                       ///< Success measurement criteria
    QStringList failureModes;                          ///< Known failure modes
    int estimatedDurationMs = 300000;                   ///< Estimated execution time (5 min default)
    QString category;                                   ///< Scenario category
    QStringList tags;                                   ///< Scenario tags
    QString author;                                     ///< Scenario author
    QString version;                                    ///< Scenario version
    QDateTime created;                                  ///< Creation timestamp
    QDateTime lastModified;                             ///< Last modification timestamp
};

/**
 * @brief Error recovery scenario specification
 */
struct ErrorRecoveryScenario {
    QString id;                                         ///< Scenario identifier
    QString name;                                       ///< Scenario name
    QString description;                                ///< Scenario description
    QString baseScenarioId;                            ///< Base scenario to inject errors into
    QList<ErrorType> errorTypes;                       ///< Types of errors to simulate
    QMap<QString, QVariant> errorParameters;           ///< Error-specific parameters
    QStringList recoverySteps;                         ///< Expected recovery steps
    QStringList recoveryValidation;                     ///< Recovery validation criteria
    bool shouldFullyRecover = true;                     ///< Whether full recovery is expected
    int recoveryTimeoutMs = 60000;                      ///< Maximum recovery time
    QString expectedFinalState;                         ///< Expected state after recovery
};

/**
 * @brief Edge case scenario specification
 */
struct EdgeCaseScenario {
    QString id;                                         ///< Scenario identifier
    QString name;                                       ///< Scenario name
    QString description;                                ///< Scenario description
    QString edgeCondition;                              ///< Description of edge condition
    QMap<QString, QVariant> edgeParameters;            ///< Parameters that create edge case
    QString expectedBehavior;                           ///< Expected application behavior
    QStringList validationCriteria;                     ///< Criteria for validating behavior
    bool shouldSucceed = true;                          ///< Whether scenario should succeed
    QString alternativeOutcome;                         ///< Alternative acceptable outcome
};

/**
 * @brief Scenario execution result with user experience metrics
 */
struct ScenarioResult {
    QString scenarioId;                                 ///< Scenario identifier
    UserPersona persona;                                ///< User persona tested
    bool success = false;                               ///< Overall success
    WorkflowResult workflowResult;                      ///< Underlying workflow result
    qint64 totalExecutionTimeMs = 0;                    ///< Total execution time
    qint64 userInteractionTimeMs = 0;                   ///< Time spent on user interactions
    qint64 waitTimeMs = 0;                             ///< Time spent waiting
    int userActionsCount = 0;                          ///< Number of user actions performed
    int errorEncountered = 0;                          ///< Number of errors encountered
    int recoveryAttempts = 0;                          ///< Number of recovery attempts
    QStringList completedGoals;                        ///< Goals successfully completed
    QStringList failedGoals;                           ///< Goals that failed
    QMap<QString, QVariant> userExperienceMetrics;     ///< UX-specific metrics
    QStringList usabilityIssues;                       ///< Identified usability issues
    QStringList accessibilityIssues;                   ///< Identified accessibility issues
    QString userFeedback;                              ///< Simulated user feedback
    int satisfactionScore = 0;                         ///< User satisfaction score (1-10)
};

/**
 * @brief Comprehensive user scenario testing framework
 * 
 * Extends workflow testing with user-centric scenario validation,
 * including persona-based testing, error recovery, and edge cases.
 */
class UserScenarioTesting : public QObject {
    Q_OBJECT

public:
    explicit UserScenarioTesting(QObject* parent = nullptr);
    ~UserScenarioTesting();

    // Workflow testing integration
    void setWorkflowTesting(std::shared_ptr<WorkflowTesting> workflowTesting);
    std::shared_ptr<WorkflowTesting> getWorkflowTesting() const { return m_workflowTesting; }

    // Scenario management
    void registerScenario(const UserScenario& scenario);
    void unregisterScenario(const QString& scenarioId);
    QStringList getRegisteredScenarios() const;
    UserScenario getScenario(const QString& scenarioId) const;
    QStringList getScenariosByPersona(UserPersona persona) const;
    QStringList getScenariosByComplexity(ScenarioComplexity complexity) const;

    // Predefined scenario creation
    UserScenario createFirstTimeUserScenario();
    UserScenario createCasualUserScenario();
    UserScenario createPowerUserScenario();
    UserScenario createSafetyFocusedUserScenario();
    UserScenario createBatchUserScenario();
    UserScenario createAccessibilityUserScenario();
    UserScenario createMobileUserScenario();

    // Advanced scenario creation
    UserScenario createPhotoLibraryCleanupScenario();
    UserScenario createDocumentOrganizationScenario();
    UserScenario createSystemMaintenanceScenario();
    UserScenario createDataMigrationScenario();
    UserScenario createEmergencyRecoveryScenario();

    // Scenario execution
    ScenarioResult executeScenario(const QString& scenarioId);
    ScenarioResult executeScenario(const UserScenario& scenario);
    QList<ScenarioResult> executeScenarioSuite(const QStringList& scenarioIds);
    QList<ScenarioResult> executePersonaScenarios(UserPersona persona);

    // Error recovery testing
    void registerErrorRecoveryScenario(const ErrorRecoveryScenario& scenario);
    ScenarioResult testErrorRecovery(const QString& scenarioId);
    QList<ScenarioResult> testAllErrorRecoveryScenarios();

    // Edge case testing
    void registerEdgeCaseScenario(const EdgeCaseScenario& scenario);
    ScenarioResult testEdgeCase(const QString& scenarioId);
    QList<ScenarioResult> testAllEdgeCases();

    // User experience validation
    bool validateUserExperience(const ScenarioResult& result);
    QStringList identifyUsabilityIssues(const ScenarioResult& result);
    QStringList identifyAccessibilityIssues(const ScenarioResult& result);
    int calculateSatisfactionScore(const ScenarioResult& result);

    // Scenario analysis and reporting
    QMap<QString, QVariant> analyzeScenarioPerformance(const ScenarioResult& result);
    QMap<QString, QVariant> compareScenarioResults(const QList<ScenarioResult>& results);
    bool generateScenarioReport(const ScenarioResult& result, const QString& outputPath);
    bool generatePersonaReport(UserPersona persona, const QList<ScenarioResult>& results, const QString& outputPath);

    // Configuration and settings
    void setPersonaConfiguration(UserPersona persona, const QMap<QString, QVariant>& config);
    QMap<QString, QVariant> getPersonaConfiguration(UserPersona persona) const;
    void setScenarioTimeout(int timeoutMs);
    void enableUserExperienceMetrics(bool enable);
    void enableAccessibilityTesting(bool enable);
    void setErrorRecoveryTimeout(int timeoutMs);

    // Utility functions
    static QString personaToString(UserPersona persona);
    static UserPersona stringToPersona(const QString& personaStr);
    static QString complexityToString(ScenarioComplexity complexity);
    static ScenarioComplexity stringToComplexity(const QString& complexityStr);

signals:
    void scenarioStarted(const QString& scenarioId, UserPersona persona);
    void scenarioCompleted(const QString& scenarioId, const ScenarioResult& result);
    void goalCompleted(const QString& scenarioId, const QString& goal);
    void goalFailed(const QString& scenarioId, const QString& goal, const QString& reason);
    void usabilityIssueDetected(const QString& scenarioId, const QString& issue);
    void accessibilityIssueDetected(const QString& scenarioId, const QString& issue);
    void errorRecoveryAttempted(const QString& scenarioId, ErrorType errorType, bool success);
    void edgeCaseEncountered(const QString& scenarioId, const QString& edgeCondition);

private:
    // Scenario execution helpers
    bool prepareScenarioExecution(const UserScenario& scenario);
    bool finalizeScenarioExecution(const UserScenario& scenario, ScenarioResult& result);
    bool executeScenarioGoals(const UserScenario& scenario, ScenarioResult& result);
    bool validateScenarioSuccess(const UserScenario& scenario, const ScenarioResult& result);

    // User experience measurement
    void measureUserInteractionTime(const UserScenario& scenario, ScenarioResult& result);
    void measureWaitTime(const UserScenario& scenario, ScenarioResult& result);
    void countUserActions(const UserScenario& scenario, ScenarioResult& result);
    void detectUsabilityIssues(const UserScenario& scenario, ScenarioResult& result);
    void detectAccessibilityIssues(const UserScenario& scenario, ScenarioResult& result);

    // Error recovery helpers
    bool injectError(const ErrorRecoveryScenario& scenario, const QString& stepId);
    bool validateRecovery(const ErrorRecoveryScenario& scenario, const ScenarioResult& result);
    QStringList captureRecoverySteps(const ErrorRecoveryScenario& scenario);

    // Edge case helpers
    bool setupEdgeCondition(const EdgeCaseScenario& scenario);
    bool validateEdgeBehavior(const EdgeCaseScenario& scenario, const ScenarioResult& result);
    void cleanupEdgeCondition(const EdgeCaseScenario& scenario);

    // Persona-specific helpers
    QMap<QString, QVariant> getPersonaDefaults(UserPersona persona);
    bool configureForPersona(UserPersona persona);
    bool validatePersonaExperience(UserPersona persona, const ScenarioResult& result);

    // Reporting helpers
    QString generateScenarioSummary(const ScenarioResult& result);
    QMap<QString, QVariant> extractUserExperienceMetrics(const ScenarioResult& result);
    QString generatePersonaSummary(UserPersona persona, const QList<ScenarioResult>& results);

private:
    // Core components
    std::shared_ptr<WorkflowTesting> m_workflowTesting;

    // Scenario management
    QMap<QString, UserScenario> m_registeredScenarios;
    QMap<QString, ErrorRecoveryScenario> m_errorRecoveryScenarios;
    QMap<QString, EdgeCaseScenario> m_edgeCaseScenarios;

    // Persona configurations
    QMap<UserPersona, QMap<QString, QVariant>> m_personaConfigurations;

    // Execution state
    QString m_currentScenarioId;
    UserPersona m_currentPersona;
    QElapsedTimer m_scenarioTimer;
    QElapsedTimer m_interactionTimer;
    QElapsedTimer m_waitTimer;

    // Configuration
    int m_scenarioTimeoutMs;
    int m_errorRecoveryTimeoutMs;
    bool m_userExperienceMetricsEnabled;
    bool m_accessibilityTestingEnabled;

    // Metrics tracking
    QMap<QString, qint64> m_interactionTimes;
    QMap<QString, qint64> m_waitTimes;
    QMap<QString, int> m_actionCounts;
    QStringList m_detectedIssues;
};

/**
 * @brief Convenience macros for scenario testing
 */
#define SCENARIO_GOAL(goal) \
    do { \
        if (!(goal)) { \
            emit goalFailed(m_currentScenarioId, #goal, "Goal condition not met"); \
            return false; \
        } else { \
            emit goalCompleted(m_currentScenarioId, #goal); \
        } \
    } while(0)

#define SCENARIO_ASSERT_UX(condition, issue) \
    do { \
        if (!(condition)) { \
            emit usabilityIssueDetected(m_currentScenarioId, issue); \
        } \
    } while(0)

#define SCENARIO_ASSERT_A11Y(condition, issue) \
    do { \
        if (!(condition)) { \
            emit accessibilityIssueDetected(m_currentScenarioId, issue); \
        } \
    } while(0)

#define MEASURE_INTERACTION_TIME(action) \
    QElapsedTimer interactionTimer; \
    interactionTimer.start(); \
    action; \
    m_interactionTimes[#action] += interactionTimer.elapsed();

#define MEASURE_WAIT_TIME(condition) \
    QElapsedTimer waitTimer; \
    waitTimer.start(); \
    while (!(condition)) { \
        QThread::msleep(100); \
    } \
    m_waitTimes[#condition] += waitTimer.elapsed();