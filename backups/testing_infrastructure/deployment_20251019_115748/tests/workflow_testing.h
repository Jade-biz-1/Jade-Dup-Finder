#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <functional>
#include <memory>

// Forward declarations
class TestEnvironment;
class UIAutomation;

/**
 * @brief Types of workflow steps that can be executed
 */
enum class WorkflowStepType {
    UIAction,           ///< User interface interaction
    FileOperation,      ///< File system operation
    Validation,         ///< State validation check
    Wait,              ///< Wait for condition
    Setup,             ///< Environment setup
    Cleanup,           ///< Environment cleanup
    Custom             ///< Custom action function
};

/**
 * @brief Error types that can be simulated during workflow execution
 */
enum class ErrorType {
    FileSystemError,    ///< File system access errors
    PermissionError,    ///< Permission denied errors
    NetworkError,       ///< Network connectivity errors
    MemoryError,        ///< Out of memory errors
    ApplicationCrash,   ///< Application crash simulation
    UserCancellation,   ///< User cancellation simulation
    TimeoutError,       ///< Operation timeout errors
    DataCorruption,     ///< Data corruption simulation
    DiskFull,          ///< Disk space exhaustion
    Custom             ///< Custom error condition
};

/**
 * @brief Individual step in a user workflow
 */
struct WorkflowStep {
    QString id;                                         ///< Unique step identifier
    QString name;                                       ///< Human-readable step name
    QString description;                                ///< Step description
    WorkflowStepType type;                             ///< Type of step
    QMap<QString, QVariant> parameters;                ///< Step parameters
    QStringList preconditions;                         ///< Required preconditions
    QStringList postconditions;                        ///< Expected postconditions
    int timeoutMs = 30000;                             ///< Step timeout
    bool optional = false;                             ///< Whether step is optional
    bool retryOnFailure = false;                       ///< Retry step on failure
    int maxRetries = 3;                                ///< Maximum retry attempts
    std::function<bool(const QMap<QString, QVariant>&)> customAction; ///< Custom action function
    std::function<bool()> validator;                   ///< Custom validation function
};

/**
 * @brief Workflow validation specification
 */
struct WorkflowValidation {
    QStringList requiredFiles;                         ///< Files that must exist
    QStringList forbiddenFiles;                        ///< Files that must not exist
    QMap<QString, QVariant> requiredProperties;       ///< Required application properties
    QMap<QString, QVariant> forbiddenProperties;      ///< Forbidden application properties
    QStringList requiredUIElements;                    ///< UI elements that must be visible
    QStringList forbiddenUIElements;                   ///< UI elements that must not be visible
    std::function<bool()> customValidator;             ///< Custom validation function
};

/**
 * @brief Application state specification for validation
 */
struct ApplicationState {
    QMap<QString, QVariant> settings;                 ///< Application settings
    QStringList openWindows;                          ///< List of open window titles
    QString activeWindow;                              ///< Currently active window
    QMap<QString, QVariant> windowStates;             ///< Window-specific states
    QStringList enabledActions;                        ///< Enabled menu/toolbar actions
    QStringList disabledActions;                       ///< Disabled menu/toolbar actions
    QMap<QString, QVariant> customStates;             ///< Custom application states
};

/**
 * @brief File system state specification for validation
 */
struct FileSystemState {
    QStringList existingFiles;                         ///< Files that should exist
    QStringList missingFiles;                          ///< Files that should not exist
    QMap<QString, qint64> fileSizes;                  ///< Expected file sizes
    QMap<QString, QDateTime> fileTimestamps;          ///< Expected file timestamps
    QMap<QString, QString> fileHashes;                ///< Expected file hashes
    QString backupLocation;                            ///< Backup directory location
    QStringList backupFiles;                           ///< Files in backup
};

/**
 * @brief Complete user workflow definition
 */
struct UserWorkflow {
    QString id;                                        ///< Unique workflow identifier
    QString name;                                      ///< Workflow name
    QString description;                               ///< Workflow description
    QString category;                                  ///< Workflow category
    QStringList tags;                                  ///< Workflow tags
    QList<WorkflowStep> steps;                        ///< Workflow steps
    WorkflowValidation validation;                     ///< Final validation
    QMap<QString, QVariant> initialState;             ///< Required initial state
    QMap<QString, QVariant> expectedFinalState;       ///< Expected final state
    int totalTimeoutMs = 300000;                       ///< Total workflow timeout (5 minutes)
    bool allowPartialFailure = false;                 ///< Allow some steps to fail
    QString author;                                    ///< Workflow author
    QString version;                                   ///< Workflow version
    QDateTime created;                                 ///< Creation timestamp
    QDateTime lastModified;                            ///< Last modification timestamp
};

/**
 * @brief Error scenario for testing error handling and recovery
 */
struct ErrorScenario {
    QString id;                                        ///< Scenario identifier
    QString name;                                      ///< Scenario name
    QString description;                               ///< Scenario description
    ErrorType errorType;                               ///< Type of error to simulate
    QString triggerStepId;                             ///< Step that triggers the error
    QMap<QString, QVariant> errorParameters;          ///< Error-specific parameters
    QStringList expectedRecoveryActions;               ///< Expected recovery actions
    QString expectedFinalState;                        ///< Expected state after recovery
    bool shouldRecover = true;                         ///< Whether recovery is expected
    int recoveryTimeoutMs = 60000;                     ///< Recovery timeout
};

/**
 * @brief Workflow execution result
 */
struct WorkflowResult {
    QString workflowId;                                ///< Workflow identifier
    bool success = false;                              ///< Overall success
    int totalSteps = 0;                                ///< Total number of steps
    int completedSteps = 0;                            ///< Successfully completed steps
    int failedSteps = 0;                               ///< Failed steps
    int skippedSteps = 0;                              ///< Skipped steps
    qint64 executionTimeMs = 0;                        ///< Total execution time
    QMap<QString, qint64> stepExecutionTimes;          ///< Individual step times
    QStringList failedStepIds;                         ///< IDs of failed steps
    QMap<QString, QString> stepErrors;                 ///< Error messages for failed steps
    QMap<QString, QVariant> finalState;               ///< Final application state
    QStringList validationErrors;                      ///< Validation error messages
    QMap<QString, QVariant> metrics;                   ///< Additional metrics
    QString screenshotPath;                            ///< Final screenshot path
    QString logPath;                                   ///< Execution log path
};

/**
 * @brief Comprehensive workflow testing framework
 * 
 * Provides complete user journey validation through workflow definition,
 * execution, and verification. Supports error simulation and recovery testing.
 */
class WorkflowTesting : public QObject {
    Q_OBJECT

public:
    explicit WorkflowTesting(QObject* parent = nullptr);
    ~WorkflowTesting();

    // Environment and automation setup
    void setTestEnvironment(std::shared_ptr<TestEnvironment> environment);
    void setUIAutomation(std::shared_ptr<UIAutomation> automation);
    std::shared_ptr<TestEnvironment> getTestEnvironment() const { return m_testEnvironment; }
    std::shared_ptr<UIAutomation> getUIAutomation() const { return m_uiAutomation; }

    // Workflow definition and management
    bool loadWorkflow(const QString& filePath);
    bool saveWorkflow(const UserWorkflow& workflow, const QString& filePath);
    void registerWorkflow(const UserWorkflow& workflow);
    void unregisterWorkflow(const QString& workflowId);
    QStringList getRegisteredWorkflows() const;
    UserWorkflow getWorkflow(const QString& workflowId) const;

    // Predefined workflow creation
    UserWorkflow createScanToDeleteWorkflow();
    UserWorkflow createFirstTimeUserWorkflow();
    UserWorkflow createPowerUserWorkflow();
    UserWorkflow createSafetyFocusedWorkflow();
    UserWorkflow createSettingsConfigurationWorkflow();
    UserWorkflow createBatchOperationWorkflow();
    UserWorkflow createErrorRecoveryWorkflow();

    // Workflow execution
    WorkflowResult executeWorkflow(const QString& workflowId);
    WorkflowResult executeWorkflow(const UserWorkflow& workflow);
    WorkflowResult executeWorkflowSteps(const UserWorkflow& workflow, const QStringList& stepIds);
    bool executeWorkflowStep(const WorkflowStep& step, QMap<QString, QVariant>& context);

    // Scenario testing
    WorkflowResult executeScenario(const QString& scenarioName, const QMap<QString, QVariant>& parameters = {});
    bool registerScenario(const QString& scenarioName, const UserWorkflow& workflow);
    QStringList getAvailableScenarios() const;

    // State validation
    bool validateWorkflowState(const WorkflowValidation& validation);
    bool validateApplicationState(const ApplicationState& expectedState);
    bool validateFileSystemState(const FileSystemState& expectedState);
    ApplicationState captureApplicationState();
    FileSystemState captureFileSystemState(const QString& basePath);

    // Error simulation and recovery testing
    bool simulateError(ErrorType errorType, const QString& context, const QMap<QString, QVariant>& parameters = {});
    WorkflowResult testErrorRecovery(const ErrorScenario& scenario);
    bool registerErrorScenario(const ErrorScenario& scenario);
    QStringList getAvailableErrorScenarios() const;

    // Workflow step execution helpers
    bool executeUIAction(const QMap<QString, QVariant>& parameters);
    bool executeFileOperation(const QMap<QString, QVariant>& parameters);
    bool executeValidation(const QMap<QString, QVariant>& parameters);
    bool executeWait(const QMap<QString, QVariant>& parameters);
    bool executeSetup(const QMap<QString, QVariant>& parameters);
    bool executeCleanup(const QMap<QString, QVariant>& parameters);

    // Workflow analysis and reporting
    QStringList analyzeWorkflow(const UserWorkflow& workflow);
    QMap<QString, QVariant> generateWorkflowMetrics(const WorkflowResult& result);
    bool generateWorkflowReport(const WorkflowResult& result, const QString& outputPath);
    QStringList getWorkflowDependencies(const QString& workflowId);

    // Configuration and settings
    void setDefaultTimeout(int timeoutMs);
    void setScreenshotDirectory(const QString& directory);
    void setLogDirectory(const QString& directory);
    void enableDetailedLogging(bool enable);
    void enableAutomaticScreenshots(bool enable);
    void setRetryAttempts(int attempts);
    void setParallelExecution(bool enable);

    // Utility functions
    static UserWorkflow loadWorkflowFromJson(const QJsonObject& json);
    static QJsonObject saveWorkflowToJson(const UserWorkflow& workflow);
    static WorkflowStep createUIActionStep(const QString& id, const QString& action, const QMap<QString, QVariant>& parameters);
    static WorkflowStep createValidationStep(const QString& id, const QString& description, std::function<bool()> validator);
    static WorkflowStep createWaitStep(const QString& id, const QString& condition, int timeoutMs = 10000);

signals:
    void workflowStarted(const QString& workflowId);
    void workflowCompleted(const QString& workflowId, const WorkflowResult& result);
    void stepStarted(const QString& workflowId, const QString& stepId);
    void stepCompleted(const QString& workflowId, const QString& stepId, bool success);
    void stepFailed(const QString& workflowId, const QString& stepId, const QString& error);
    void validationFailed(const QString& workflowId, const QString& validation, const QString& error);
    void errorSimulated(ErrorType errorType, const QString& context);
    void recoveryAttempted(const QString& scenarioId, bool success);

private slots:
    void onStepTimeout();
    void onWorkflowTimeout();

private:
    // Workflow execution helpers
    bool prepareWorkflowExecution(const UserWorkflow& workflow);
    bool finalizeWorkflowExecution(const UserWorkflow& workflow, WorkflowResult& result);
    bool checkPreconditions(const QStringList& preconditions, const QMap<QString, QVariant>& context);
    bool verifyPostconditions(const QStringList& postconditions, const QMap<QString, QVariant>& context);
    void updateExecutionContext(const QString& stepId, const QMap<QString, QVariant>& stepResult, QMap<QString, QVariant>& context);

    // Error simulation helpers
    bool simulateFileSystemError(const QMap<QString, QVariant>& parameters);
    bool simulatePermissionError(const QMap<QString, QVariant>& parameters);
    bool simulateNetworkError(const QMap<QString, QVariant>& parameters);
    bool simulateMemoryError(const QMap<QString, QVariant>& parameters);
    bool simulateApplicationCrash(const QMap<QString, QVariant>& parameters);
    bool simulateUserCancellation(const QMap<QString, QVariant>& parameters);
    bool simulateTimeoutError(const QMap<QString, QVariant>& parameters);
    bool simulateDataCorruption(const QMap<QString, QVariant>& parameters);
    bool simulateDiskFull(const QMap<QString, QVariant>& parameters);

    // State capture helpers
    QStringList captureOpenWindows();
    QString captureActiveWindow();
    QMap<QString, QVariant> captureWindowStates();
    QStringList captureEnabledActions();
    QStringList captureDisabledActions();
    QMap<QString, QVariant> captureApplicationSettings();

    // File system helpers
    QStringList findExistingFiles(const QString& basePath);
    QMap<QString, qint64> getFileSizes(const QStringList& files);
    QMap<QString, QDateTime> getFileTimestamps(const QStringList& files);
    QMap<QString, QString> calculateFileHashes(const QStringList& files);
    QString findBackupLocation();
    QStringList findBackupFiles(const QString& backupLocation);

    // Validation helpers
    bool validateFileExists(const QString& filePath);
    bool validateFileNotExists(const QString& filePath);
    bool validateFileSize(const QString& filePath, qint64 expectedSize);
    bool validateFileHash(const QString& filePath, const QString& expectedHash);
    bool validateUIElement(const QString& elementSelector, bool shouldExist);
    bool validateApplicationProperty(const QString& propertyName, const QVariant& expectedValue);

    // Logging and reporting helpers
    void logWorkflowStart(const QString& workflowId);
    void logWorkflowEnd(const QString& workflowId, const WorkflowResult& result);
    void logStepStart(const QString& stepId, const WorkflowStep& step);
    void logStepEnd(const QString& stepId, bool success, const QString& error = QString());
    void logValidationResult(const QString& validation, bool success, const QString& error = QString());
    QString captureScreenshot(const QString& prefix);
    QString generateExecutionLog(const WorkflowResult& result);

private:
    // Core components
    std::shared_ptr<TestEnvironment> m_testEnvironment;
    std::shared_ptr<UIAutomation> m_uiAutomation;

    // Workflow management
    QMap<QString, UserWorkflow> m_registeredWorkflows;
    QMap<QString, UserWorkflow> m_registeredScenarios;
    QMap<QString, ErrorScenario> m_errorScenarios;

    // Execution state
    QString m_currentWorkflowId;
    QString m_currentStepId;
    QMap<QString, QVariant> m_executionContext;
    QElapsedTimer m_executionTimer;
    QElapsedTimer m_stepTimer;

    // Configuration
    int m_defaultTimeoutMs;
    QString m_screenshotDirectory;
    QString m_logDirectory;
    bool m_detailedLogging;
    bool m_automaticScreenshots;
    int m_retryAttempts;
    bool m_parallelExecution;

    // Error simulation state
    QMap<ErrorType, bool> m_activeErrors;
    QMap<QString, QVariant> m_errorStates;
};

/**
 * @brief Convenience macros for workflow testing
 */
#define WORKFLOW_STEP(id, name, type) \
    WorkflowStep { id, name, "", type, {}, {}, {}, 30000, false, false, 3, nullptr, nullptr }

#define UI_ACTION_STEP(id, action, params) \
    WorkflowTesting::createUIActionStep(id, action, params)

#define VALIDATION_STEP(id, desc, validator) \
    WorkflowTesting::createValidationStep(id, desc, validator)

#define WAIT_STEP(id, condition, timeout) \
    WorkflowTesting::createWaitStep(id, condition, timeout)

#define WORKFLOW_ASSERT(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(QString("Workflow assertion failed: %1").arg(message).toStdString()); \
    }

#define WORKFLOW_VERIFY_FILE_EXISTS(path) \
    WORKFLOW_ASSERT(QFile::exists(path), QString("File does not exist: %1").arg(path))

#define WORKFLOW_VERIFY_FILE_NOT_EXISTS(path) \
    WORKFLOW_ASSERT(!QFile::exists(path), QString("File should not exist: %1").arg(path))