#include "workflow_testing.h"
#include "framework/test_environment.h"
#include "ui_automation.h"
#include <QApplication>
#include <QWidget>
#include <QMainWindow>
#include <QDialog>
#include <QTimer>
#include <QEventLoop>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QProcess>
#include <stdexcept>

WorkflowTesting::WorkflowTesting(QObject* parent)
    : QObject(parent)
    , m_defaultTimeoutMs(30000)
    , m_detailedLogging(true)
    , m_automaticScreenshots(true)
    , m_retryAttempts(3)
    , m_parallelExecution(false)
{
    // Set default directories
    m_screenshotDirectory = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/workflow_screenshots";
    m_logDirectory = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/workflow_logs";
    
    // Create directories if they don't exist
    QDir().mkpath(m_screenshotDirectory);
    QDir().mkpath(m_logDirectory);
}

WorkflowTesting::~WorkflowTesting() = default;

void WorkflowTesting::setTestEnvironment(std::shared_ptr<TestEnvironment> environment) {
    m_testEnvironment = environment;
}

void WorkflowTesting::setUIAutomation(std::shared_ptr<UIAutomation> automation) {
    m_uiAutomation = automation;
}

bool WorkflowTesting::loadWorkflow(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open workflow file:" << filePath;
        return false;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError) {
        qWarning() << "Failed to parse workflow JSON:" << error.errorString();
        return false;
    }
    
    UserWorkflow workflow = loadWorkflowFromJson(doc.object());
    registerWorkflow(workflow);
    return true;
}

bool WorkflowTesting::saveWorkflow(const UserWorkflow& workflow, const QString& filePath) {
    QJsonObject json = saveWorkflowToJson(workflow);
    QJsonDocument doc(json);
    
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to open file for writing:" << filePath;
        return false;
    }
    
    file.write(doc.toJson());
    return true;
}

void WorkflowTesting::registerWorkflow(const UserWorkflow& workflow) {
    m_registeredWorkflows[workflow.id] = workflow;
}

void WorkflowTesting::unregisterWorkflow(const QString& workflowId) {
    m_registeredWorkflows.remove(workflowId);
}

QStringList WorkflowTesting::getRegisteredWorkflows() const {
    return m_registeredWorkflows.keys();
}

UserWorkflow WorkflowTesting::getWorkflow(const QString& workflowId) const {
    return m_registeredWorkflows.value(workflowId);
}

UserWorkflow WorkflowTesting::createScanToDeleteWorkflow() {
    UserWorkflow workflow;
    workflow.id = "scan_to_delete_workflow";
    workflow.name = "Complete Scan to Delete Workflow";
    workflow.description = "Tests the complete user journey from scanning for duplicates to deleting them";
    workflow.category = "core_functionality";
    workflow.tags = {"scan", "delete", "duplicates", "core"};
    workflow.author = "WorkflowTesting";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Launch application
    WorkflowStep launchStep;
    launchStep.id = "launch_app";
    launchStep.name = "Launch CloneClean Application";
    launchStep.description = "Start the CloneClean application";
    launchStep.type = WorkflowStepType::Setup;
    launchStep.parameters["action"] = "launch_application";
    launchStep.postconditions = {"main_window_visible"};
    workflow.steps.append(launchStep);
    
    // Step 2: Select scan directory
    WorkflowStep selectDirStep;
    selectDirStep.id = "select_directory";
    selectDirStep.name = "Select Scan Directory";
    selectDirStep.description = "Choose directory to scan for duplicates";
    selectDirStep.type = WorkflowStepType::UIAction;
    selectDirStep.parameters["action"] = "click";
    selectDirStep.parameters["selector"] = "browse_button";
    selectDirStep.parameters["dialog_action"] = "select_directory";
    selectDirStep.parameters["directory_path"] = "test_data";
    selectDirStep.preconditions = {"main_window_visible"};
    selectDirStep.postconditions = {"directory_selected"};
    workflow.steps.append(selectDirStep);
    
    // Step 3: Configure scan settings
    WorkflowStep configureStep;
    configureStep.id = "configure_scan";
    configureStep.name = "Configure Scan Settings";
    configureStep.description = "Set up scan parameters and filters";
    configureStep.type = WorkflowStepType::UIAction;
    configureStep.parameters["action"] = "configure_settings";
    configureStep.parameters["min_file_size"] = 1024;
    configureStep.parameters["file_types"] = QStringList{"jpg", "png", "pdf", "txt"};
    configureStep.parameters["include_subdirectories"] = true;
    configureStep.preconditions = {"directory_selected"};
    configureStep.postconditions = {"scan_configured"};
    workflow.steps.append(configureStep);
    
    // Step 4: Start scan
    WorkflowStep startScanStep;
    startScanStep.id = "start_scan";
    startScanStep.name = "Start Duplicate Scan";
    startScanStep.description = "Begin scanning for duplicate files";
    startScanStep.type = WorkflowStepType::UIAction;
    startScanStep.parameters["action"] = "click";
    startScanStep.parameters["selector"] = "start_scan_button";
    startScanStep.preconditions = {"scan_configured"};
    startScanStep.postconditions = {"scan_started"};
    startScanStep.timeoutMs = 60000; // Allow more time for scanning
    workflow.steps.append(startScanStep);
    
    // Step 5: Wait for scan completion
    WorkflowStep waitScanStep;
    waitScanStep.id = "wait_scan_complete";
    waitScanStep.name = "Wait for Scan Completion";
    waitScanStep.description = "Wait for the duplicate scan to complete";
    waitScanStep.type = WorkflowStepType::Wait;
    waitScanStep.parameters["condition"] = "scan_completed";
    waitScanStep.parameters["indicator"] = "progress_dialog_closed";
    waitScanStep.preconditions = {"scan_started"};
    waitScanStep.postconditions = {"scan_completed", "results_available"};
    waitScanStep.timeoutMs = 120000; // Allow up to 2 minutes for scan
    workflow.steps.append(waitScanStep);
    
    // Step 6: Review results
    WorkflowStep reviewStep;
    reviewStep.id = "review_results";
    reviewStep.name = "Review Scan Results";
    reviewStep.description = "Examine the duplicate files found";
    reviewStep.type = WorkflowStepType::Validation;
    reviewStep.parameters["action"] = "validate_results";
    reviewStep.parameters["min_duplicates"] = 1;
    reviewStep.parameters["results_window_visible"] = true;
    reviewStep.preconditions = {"results_available"};
    reviewStep.postconditions = {"results_reviewed"};
    workflow.steps.append(reviewStep);
    
    // Step 7: Select files for deletion
    WorkflowStep selectFilesStep;
    selectFilesStep.id = "select_files";
    selectFilesStep.name = "Select Files for Deletion";
    selectFilesStep.description = "Choose which duplicate files to delete";
    selectFilesStep.type = WorkflowStepType::UIAction;
    selectFilesStep.parameters["action"] = "select_duplicates";
    selectFilesStep.parameters["selection_strategy"] = "keep_newest";
    selectFilesStep.parameters["auto_select"] = true;
    selectFilesStep.preconditions = {"results_reviewed"};
    selectFilesStep.postconditions = {"files_selected"};
    workflow.steps.append(selectFilesStep);
    
    // Step 8: Delete selected files
    WorkflowStep deleteStep;
    deleteStep.id = "delete_files";
    deleteStep.name = "Delete Selected Files";
    deleteStep.description = "Delete the selected duplicate files";
    deleteStep.type = WorkflowStepType::UIAction;
    deleteStep.parameters["action"] = "delete_selected";
    deleteStep.parameters["confirm_deletion"] = true;
    deleteStep.parameters["use_recycle_bin"] = true;
    deleteStep.preconditions = {"files_selected"};
    deleteStep.postconditions = {"files_deleted"};
    deleteStep.timeoutMs = 60000; // Allow time for deletion
    workflow.steps.append(deleteStep);
    
    // Step 9: Verify deletion
    WorkflowStep verifyStep;
    verifyStep.id = "verify_deletion";
    verifyStep.name = "Verify File Deletion";
    verifyStep.description = "Confirm that selected files were deleted";
    verifyStep.type = WorkflowStepType::Validation;
    verifyStep.parameters["action"] = "verify_files_deleted";
    verifyStep.parameters["check_recycle_bin"] = true;
    verifyStep.preconditions = {"files_deleted"};
    verifyStep.postconditions = {"deletion_verified"};
    workflow.steps.append(verifyStep);
    
    // Final validation
    workflow.validation.requiredUIElements = {"main_window"};
    workflow.validation.forbiddenUIElements = {"error_dialog", "crash_dialog"};
    workflow.validation.customValidator = [this]() -> bool {
        // Custom validation logic
        return validateApplicationState(captureApplicationState());
    };
    
    return workflow;
}

UserWorkflow WorkflowTesting::createFirstTimeUserWorkflow() {
    UserWorkflow workflow;
    workflow.id = "first_time_user_workflow";
    workflow.name = "First-Time User Experience";
    workflow.description = "Tests the complete first-time user experience including setup and first scan";
    workflow.category = "user_experience";
    workflow.tags = {"first_time", "setup", "onboarding"};
    workflow.author = "WorkflowTesting";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: First launch with clean settings
    WorkflowStep firstLaunchStep;
    firstLaunchStep.id = "first_launch";
    firstLaunchStep.name = "First Application Launch";
    firstLaunchStep.description = "Launch application with clean settings";
    firstLaunchStep.type = WorkflowStepType::Setup;
    firstLaunchStep.parameters["action"] = "clean_launch";
    firstLaunchStep.parameters["clear_settings"] = true;
    firstLaunchStep.postconditions = {"app_launched", "welcome_shown"};
    workflow.steps.append(firstLaunchStep);
    
    // Step 2: Welcome dialog interaction
    WorkflowStep welcomeStep;
    welcomeStep.id = "welcome_dialog";
    welcomeStep.name = "Welcome Dialog Interaction";
    welcomeStep.description = "Interact with welcome dialog";
    welcomeStep.type = WorkflowStepType::UIAction;
    welcomeStep.parameters["action"] = "handle_welcome";
    welcomeStep.parameters["show_tips"] = true;
    welcomeStep.parameters["accept_dialog"] = true;
    welcomeStep.preconditions = {"welcome_shown"};
    welcomeStep.postconditions = {"welcome_completed"};
    workflow.steps.append(welcomeStep);
    
    // Step 3: Explore main interface
    WorkflowStep exploreStep;
    exploreStep.id = "explore_interface";
    exploreStep.name = "Explore Main Interface";
    exploreStep.description = "Familiarize with main interface elements";
    exploreStep.type = WorkflowStepType::UIAction;
    exploreStep.parameters["action"] = "explore_ui";
    exploreStep.parameters["check_menus"] = true;
    exploreStep.parameters["check_toolbars"] = true;
    exploreStep.parameters["check_help"] = true;
    exploreStep.preconditions = {"welcome_completed"};
    exploreStep.postconditions = {"interface_explored"};
    workflow.steps.append(exploreStep);
    
    // Step 4: First scan setup
    WorkflowStep setupScanStep;
    setupScanStep.id = "setup_first_scan";
    setupScanStep.name = "Setup First Scan";
    setupScanStep.description = "Configure and run first duplicate scan";
    setupScanStep.type = WorkflowStepType::UIAction;
    setupScanStep.parameters["action"] = "setup_scan";
    setupScanStep.parameters["use_default_settings"] = true;
    setupScanStep.parameters["select_sample_directory"] = true;
    setupScanStep.preconditions = {"interface_explored"};
    setupScanStep.postconditions = {"first_scan_setup"};
    workflow.steps.append(setupScanStep);
    
    // Step 5: Run first scan
    WorkflowStep runScanStep;
    runScanStep.id = "run_first_scan";
    runScanStep.name = "Run First Scan";
    runScanStep.description = "Execute the first duplicate scan";
    runScanStep.type = WorkflowStepType::UIAction;
    runScanStep.parameters["action"] = "run_scan";
    runScanStep.parameters["monitor_progress"] = true;
    runScanStep.preconditions = {"first_scan_setup"};
    runScanStep.postconditions = {"first_scan_completed"};
    runScanStep.timeoutMs = 90000;
    workflow.steps.append(runScanStep);
    
    // Step 6: Review first results
    WorkflowStep reviewFirstStep;
    reviewFirstStep.id = "review_first_results";
    reviewFirstStep.name = "Review First Results";
    reviewFirstStep.description = "Examine first scan results";
    reviewFirstStep.type = WorkflowStepType::UIAction;
    reviewFirstStep.parameters["action"] = "review_results";
    reviewFirstStep.parameters["explore_features"] = true;
    reviewFirstStep.parameters["try_grouping"] = true;
    reviewFirstStep.preconditions = {"first_scan_completed"};
    reviewFirstStep.postconditions = {"first_results_reviewed"};
    workflow.steps.append(reviewFirstStep);
    
    return workflow;
}

UserWorkflow WorkflowTesting::createPowerUserWorkflow() {
    UserWorkflow workflow;
    workflow.id = "power_user_workflow";
    workflow.name = "Power User Advanced Features";
    workflow.description = "Tests advanced features used by power users";
    workflow.category = "advanced_features";
    workflow.tags = {"power_user", "advanced", "batch", "automation"};
    workflow.author = "WorkflowTesting";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Launch with existing configuration
    WorkflowStep launchStep;
    launchStep.id = "launch_configured";
    launchStep.name = "Launch with Configuration";
    launchStep.description = "Launch application with existing power user configuration";
    launchStep.type = WorkflowStepType::Setup;
    launchStep.parameters["action"] = "launch_with_config";
    launchStep.parameters["config_profile"] = "power_user";
    launchStep.postconditions = {"app_launched", "config_loaded"};
    workflow.steps.append(launchStep);
    
    // Step 2: Configure advanced filters
    WorkflowStep filtersStep;
    filtersStep.id = "advanced_filters";
    filtersStep.name = "Configure Advanced Filters";
    filtersStep.description = "Set up complex filtering rules";
    filtersStep.type = WorkflowStepType::UIAction;
    filtersStep.parameters["action"] = "configure_advanced_filters";
    filtersStep.parameters["size_filters"] = true;
    filtersStep.parameters["date_filters"] = true;
    filtersStep.parameters["type_filters"] = true;
    filtersStep.parameters["custom_patterns"] = true;
    filtersStep.preconditions = {"config_loaded"};
    filtersStep.postconditions = {"advanced_filters_set"};
    workflow.steps.append(filtersStep);
    
    // Step 3: Batch scan multiple directories
    WorkflowStep batchScanStep;
    batchScanStep.id = "batch_scan";
    batchScanStep.name = "Batch Scan Multiple Directories";
    batchScanStep.description = "Scan multiple directories in batch mode";
    batchScanStep.type = WorkflowStepType::UIAction;
    batchScanStep.parameters["action"] = "batch_scan";
    batchScanStep.parameters["directories"] = QStringList{"dir1", "dir2", "dir3"};
    batchScanStep.parameters["parallel_processing"] = true;
    batchScanStep.preconditions = {"advanced_filters_set"};
    batchScanStep.postconditions = {"batch_scan_completed"};
    batchScanStep.timeoutMs = 180000; // 3 minutes for batch scan
    workflow.steps.append(batchScanStep);
    
    // Step 4: Advanced result analysis
    WorkflowStep analysisStep;
    analysisStep.id = "advanced_analysis";
    analysisStep.name = "Advanced Result Analysis";
    analysisStep.description = "Perform detailed analysis of scan results";
    analysisStep.type = WorkflowStepType::UIAction;
    analysisStep.parameters["action"] = "analyze_results";
    analysisStep.parameters["generate_statistics"] = true;
    analysisStep.parameters["create_reports"] = true;
    analysisStep.parameters["export_data"] = true;
    analysisStep.preconditions = {"batch_scan_completed"};
    analysisStep.postconditions = {"analysis_completed"};
    workflow.steps.append(analysisStep);
    
    // Step 5: Automated selection rules
    WorkflowStep autoSelectStep;
    autoSelectStep.id = "automated_selection";
    autoSelectStep.name = "Apply Automated Selection Rules";
    autoSelectStep.description = "Use automated rules for file selection";
    autoSelectStep.type = WorkflowStepType::UIAction;
    autoSelectStep.parameters["action"] = "apply_selection_rules";
    autoSelectStep.parameters["rules"] = QStringList{"keep_newest", "keep_largest", "prefer_path"};
    autoSelectStep.parameters["custom_rules"] = true;
    autoSelectStep.preconditions = {"analysis_completed"};
    autoSelectStep.postconditions = {"auto_selection_applied"};
    workflow.steps.append(autoSelectStep);
    
    // Step 6: Batch operations
    WorkflowStep batchOpsStep;
    batchOpsStep.id = "batch_operations";
    batchOpsStep.name = "Execute Batch Operations";
    batchOpsStep.description = "Perform batch file operations";
    batchOpsStep.type = WorkflowStepType::UIAction;
    batchOpsStep.parameters["action"] = "batch_operations";
    batchOpsStep.parameters["operations"] = QStringList{"delete", "move", "copy"};
    batchOpsStep.parameters["confirm_batch"] = true;
    batchOpsStep.preconditions = {"auto_selection_applied"};
    batchOpsStep.postconditions = {"batch_operations_completed"};
    batchOpsStep.timeoutMs = 120000; // 2 minutes for batch operations
    workflow.steps.append(batchOpsStep);
    
    return workflow;
}

UserWorkflow WorkflowTesting::createSafetyFocusedWorkflow() {
    UserWorkflow workflow;
    workflow.id = "safety_focused_workflow";
    workflow.name = "Safety-Focused Operations with Backup/Restore";
    workflow.description = "Tests safety features including backup creation and restore operations";
    workflow.category = "safety_features";
    workflow.tags = {"safety", "backup", "restore", "protection"};
    workflow.author = "WorkflowTesting";
    workflow.version = "1.0";
    workflow.created = QDateTime::currentDateTime();
    workflow.lastModified = QDateTime::currentDateTime();
    
    // Step 1: Launch with safety settings
    WorkflowStep launchStep;
    launchStep.id = "launch_safety_mode";
    launchStep.name = "Launch in Safety Mode";
    launchStep.description = "Launch application with maximum safety settings";
    launchStep.type = WorkflowStepType::Setup;
    launchStep.parameters["action"] = "launch_safety_mode";
    launchStep.parameters["enable_backups"] = true;
    launchStep.parameters["confirm_all_operations"] = true;
    launchStep.postconditions = {"app_launched", "safety_mode_enabled"};
    workflow.steps.append(launchStep);
    
    // Step 2: Configure backup settings
    WorkflowStep backupConfigStep;
    backupConfigStep.id = "configure_backup";
    backupConfigStep.name = "Configure Backup Settings";
    backupConfigStep.description = "Set up automatic backup configuration";
    backupConfigStep.type = WorkflowStepType::UIAction;
    backupConfigStep.parameters["action"] = "configure_backup";
    backupConfigStep.parameters["backup_location"] = "backup_test_dir";
    backupConfigStep.parameters["auto_backup"] = true;
    backupConfigStep.parameters["backup_before_delete"] = true;
    backupConfigStep.preconditions = {"safety_mode_enabled"};
    backupConfigStep.postconditions = {"backup_configured"};
    workflow.steps.append(backupConfigStep);
    
    // Step 3: Create test backup
    WorkflowStep createBackupStep;
    createBackupStep.id = "create_backup";
    createBackupStep.name = "Create Test Backup";
    createBackupStep.description = "Create a backup of test files";
    createBackupStep.type = WorkflowStepType::UIAction;
    createBackupStep.parameters["action"] = "create_backup";
    createBackupStep.parameters["backup_name"] = "safety_test_backup";
    createBackupStep.parameters["include_metadata"] = true;
    createBackupStep.preconditions = {"backup_configured"};
    createBackupStep.postconditions = {"backup_created"};
    workflow.steps.append(createBackupStep);
    
    // Step 4: Perform scan with safety checks
    WorkflowStep safeScanStep;
    safeScanStep.id = "safe_scan";
    safeScanStep.name = "Perform Safe Scan";
    safeScanStep.description = "Run scan with all safety checks enabled";
    safeScanStep.type = WorkflowStepType::UIAction;
    safeScanStep.parameters["action"] = "safe_scan";
    safeScanStep.parameters["verify_permissions"] = true;
    safeScanStep.parameters["check_system_files"] = true;
    safeScanStep.parameters["exclude_protected"] = true;
    safeScanStep.preconditions = {"backup_created"};
    safeScanStep.postconditions = {"safe_scan_completed"};
    safeScanStep.timeoutMs = 90000;
    workflow.steps.append(safeScanStep);
    
    // Step 5: Safe deletion with confirmation
    WorkflowStep safeDeletionStep;
    safeDeletionStep.id = "safe_deletion";
    safeDeletionStep.name = "Safe Deletion with Confirmation";
    safeDeletionStep.description = "Delete files with multiple confirmations";
    safeDeletionStep.type = WorkflowStepType::UIAction;
    safeDeletionStep.parameters["action"] = "safe_delete";
    safeDeletionStep.parameters["require_confirmation"] = true;
    safeDeletionStep.parameters["show_preview"] = true;
    safeDeletionStep.parameters["create_backup_before"] = true;
    safeDeletionStep.preconditions = {"safe_scan_completed"};
    safeDeletionStep.postconditions = {"safe_deletion_completed"};
    workflow.steps.append(safeDeletionStep);
    
    // Step 6: Verify backup integrity
    WorkflowStep verifyBackupStep;
    verifyBackupStep.id = "verify_backup";
    verifyBackupStep.name = "Verify Backup Integrity";
    verifyBackupStep.description = "Verify that backup was created correctly";
    verifyBackupStep.type = WorkflowStepType::Validation;
    verifyBackupStep.parameters["action"] = "verify_backup";
    verifyBackupStep.parameters["check_file_count"] = true;
    verifyBackupStep.parameters["verify_checksums"] = true;
    verifyBackupStep.preconditions = {"safe_deletion_completed"};
    verifyBackupStep.postconditions = {"backup_verified"};
    workflow.steps.append(verifyBackupStep);
    
    // Step 7: Test restore functionality
    WorkflowStep restoreStep;
    restoreStep.id = "test_restore";
    restoreStep.name = "Test Restore Functionality";
    restoreStep.description = "Test restoring files from backup";
    restoreStep.type = WorkflowStepType::UIAction;
    restoreStep.parameters["action"] = "restore_from_backup";
    restoreStep.parameters["backup_name"] = "safety_test_backup";
    restoreStep.parameters["restore_location"] = "restore_test_dir";
    restoreStep.parameters["verify_restore"] = true;
    restoreStep.preconditions = {"backup_verified"};
    restoreStep.postconditions = {"restore_completed"};
    workflow.steps.append(restoreStep);
    
    return workflow;
}

WorkflowResult WorkflowTesting::executeWorkflow(const QString& workflowId) {
    if (!m_registeredWorkflows.contains(workflowId)) {
        WorkflowResult result;
        result.workflowId = workflowId;
        result.success = false;
        result.validationErrors.append(QString("Workflow not found: %1").arg(workflowId));
        return result;
    }
    
    return executeWorkflow(m_registeredWorkflows[workflowId]);
}

WorkflowResult WorkflowTesting::executeWorkflow(const UserWorkflow& workflow) {
    WorkflowResult result;
    result.workflowId = workflow.id;
    result.totalSteps = workflow.steps.size();
    
    m_currentWorkflowId = workflow.id;
    m_executionContext.clear();
    m_executionTimer.start();
    
    logWorkflowStart(workflow.id);
    emit workflowStarted(workflow.id);
    
    if (!prepareWorkflowExecution(workflow)) {
        result.success = false;
        result.validationErrors.append("Failed to prepare workflow execution");
        return result;
    }
    
    // Execute each step
    for (const WorkflowStep& step : workflow.steps) {
        m_currentStepId = step.id;
        m_stepTimer.start();
        
        logStepStart(step.id, step);
        emit stepStarted(workflow.id, step.id);
        
        bool stepSuccess = executeWorkflowStep(step, m_executionContext);
        qint64 stepTime = m_stepTimer.elapsed();
        
        result.stepExecutionTimes[step.id] = stepTime;
        
        if (stepSuccess) {
            result.completedSteps++;
            logStepEnd(step.id, true);
            emit stepCompleted(workflow.id, step.id, true);
        } else {
            result.failedSteps++;
            result.failedStepIds.append(step.id);
            QString error = QString("Step failed: %1").arg(step.name);
            result.stepErrors[step.id] = error;
            
            logStepEnd(step.id, false, error);
            emit stepFailed(workflow.id, step.id, error);
            
            if (!workflow.allowPartialFailure && !step.optional) {
                result.success = false;
                break;
            } else {
                result.skippedSteps++;
            }
        }
        
        // Check for timeout
        if (m_executionTimer.elapsed() > workflow.totalTimeoutMs) {
            result.success = false;
            result.validationErrors.append("Workflow execution timeout");
            break;
        }
    }
    
    result.executionTimeMs = m_executionTimer.elapsed();
    
    // Final validation
    if (result.failedSteps == 0 || workflow.allowPartialFailure) {
        result.success = validateWorkflowState(workflow.validation);
        if (!result.success) {
            result.validationErrors.append("Final workflow validation failed");
        }
    }
    
    // Capture final state
    result.finalState["application_state"] = QVariant::fromValue(captureApplicationState());
    result.finalState["file_system_state"] = QVariant::fromValue(captureFileSystemState("."));
    
    // Generate artifacts
    if (m_automaticScreenshots) {
        result.screenshotPath = captureScreenshot("final_" + workflow.id);
    }
    result.logPath = generateExecutionLog(result);
    
    finalizeWorkflowExecution(workflow, result);
    
    logWorkflowEnd(workflow.id, result);
    emit workflowCompleted(workflow.id, result);
    
    return result;
}

bool WorkflowTesting::executeWorkflowStep(const WorkflowStep& step, QMap<QString, QVariant>& context) {
    // Check preconditions
    if (!checkPreconditions(step.preconditions, context)) {
        return false;
    }
    
    bool success = false;
    int attempts = 0;
    int maxAttempts = step.retryOnFailure ? step.maxRetries : 1;
    
    while (attempts < maxAttempts && !success) {
        attempts++;
        
        try {
            switch (step.type) {
                case WorkflowStepType::UIAction:
                    success = executeUIAction(step.parameters);
                    break;
                case WorkflowStepType::FileOperation:
                    success = executeFileOperation(step.parameters);
                    break;
                case WorkflowStepType::Validation:
                    success = executeValidation(step.parameters);
                    break;
                case WorkflowStepType::Wait:
                    success = executeWait(step.parameters);
                    break;
                case WorkflowStepType::Setup:
                    success = executeSetup(step.parameters);
                    break;
                case WorkflowStepType::Cleanup:
                    success = executeCleanup(step.parameters);
                    break;
                case WorkflowStepType::Custom:
                    if (step.customAction) {
                        success = step.customAction(step.parameters);
                    }
                    break;
            }
        } catch (const std::exception& e) {
            qWarning() << "Exception in workflow step" << step.id << ":" << e.what();
            success = false;
        }
        
        if (!success && attempts < maxAttempts) {
            QThread::msleep(1000); // Wait before retry
        }
    }
    
    if (success) {
        // Verify postconditions
        success = verifyPostconditions(step.postconditions, context);
        if (success) {
            updateExecutionContext(step.id, step.parameters, context);
        }
    }
    
    return success;
}

bool WorkflowTesting::executeUIAction(const QMap<QString, QVariant>& parameters) {
    if (!m_uiAutomation) {
        qWarning() << "UI automation not available";
        return false;
    }
    
    QString action = parameters.value("action").toString();
    
    if (action == "click") {
        QString selector = parameters.value("selector").toString();
        UIAutomation::WidgetSelector widgetSelector = UIAutomation::byObjectName(selector);
        return m_uiAutomation->clickWidget(widgetSelector);
    } else if (action == "type") {
        QString selector = parameters.value("selector").toString();
        QString text = parameters.value("text").toString();
        UIAutomation::WidgetSelector widgetSelector = UIAutomation::byObjectName(selector);
        return m_uiAutomation->typeText(widgetSelector, text);
    } else if (action == "configure_settings") {
        // Implement settings configuration
        return true; // Placeholder
    } else if (action == "select_duplicates") {
        // Implement duplicate selection
        return true; // Placeholder
    } else if (action == "delete_selected") {
        // Implement file deletion
        return true; // Placeholder
    }
    
    return false;
}

bool WorkflowTesting::executeFileOperation(const QMap<QString, QVariant>& parameters) {
    QString operation = parameters.value("operation").toString();
    
    if (operation == "create_file") {
        QString filePath = parameters.value("file_path").toString();
        QByteArray content = parameters.value("content").toByteArray();
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(content);
            return true;
        }
    } else if (operation == "delete_file") {
        QString filePath = parameters.value("file_path").toString();
        return QFile::remove(filePath);
    } else if (operation == "copy_file") {
        QString source = parameters.value("source").toString();
        QString destination = parameters.value("destination").toString();
        return QFile::copy(source, destination);
    }
    
    return false;
}

bool WorkflowTesting::executeValidation(const QMap<QString, QVariant>& parameters) {
    QString validationType = parameters.value("action").toString();
    
    if (validationType == "validate_results") {
        int minDuplicates = parameters.value("min_duplicates", 0).toInt();
        bool resultsWindowVisible = parameters.value("results_window_visible", false).toBool();
        
        // Implement result validation
        return true; // Placeholder
    } else if (validationType == "verify_files_deleted") {
        QStringList files = parameters.value("files").toStringList();
        bool checkRecycleBin = parameters.value("check_recycle_bin", false).toBool();
        
        // Verify files are deleted
        for (const QString& file : files) {
            if (QFile::exists(file)) {
                return false;
            }
        }
        return true;
    }
    
    return false;
}

bool WorkflowTesting::executeWait(const QMap<QString, QVariant>& parameters) {
    QString condition = parameters.value("condition").toString();
    int timeoutMs = parameters.value("timeout", 10000).toInt();
    
    if (condition == "scan_completed") {
        QString indicator = parameters.value("indicator").toString();
        
        // Wait for scan completion indicator
        QElapsedTimer timer;
        timer.start();
        
        while (timer.elapsed() < timeoutMs) {
            if (indicator == "progress_dialog_closed") {
                // Check if progress dialog is closed
                // This is a placeholder - implement actual check
                QThread::msleep(100);
                if (timer.elapsed() > 5000) { // Simulate completion after 5 seconds
                    return true;
                }
            }
            QThread::msleep(100);
        }
        return false;
    }
    
    return true;
}

bool WorkflowTesting::executeSetup(const QMap<QString, QVariant>& parameters) {
    QString action = parameters.value("action").toString();
    
    if (action == "launch_application") {
        if (m_testEnvironment) {
            return m_testEnvironment->launchApplication();
        }
    } else if (action == "clean_launch") {
        if (m_testEnvironment) {
            bool clearSettings = parameters.value("clear_settings", false).toBool();
            if (clearSettings) {
                m_testEnvironment->clearApplicationSettings();
            }
            return m_testEnvironment->launchApplication();
        }
    }
    
    return false;
}

bool WorkflowTesting::executeCleanup(const QMap<QString, QVariant>& parameters) {
    QString action = parameters.value("action").toString();
    
    if (action == "close_application") {
        if (m_testEnvironment) {
            return m_testEnvironment->closeApplication();
        }
    } else if (action == "cleanup_files") {
        QStringList files = parameters.value("files").toStringList();
        for (const QString& file : files) {
            QFile::remove(file);
        }
        return true;
    }
    
    return false;
}

bool WorkflowTesting::validateWorkflowState(const WorkflowValidation& validation) {
    // Check required files
    for (const QString& file : validation.requiredFiles) {
        if (!QFile::exists(file)) {
            return false;
        }
    }
    
    // Check forbidden files
    for (const QString& file : validation.forbiddenFiles) {
        if (QFile::exists(file)) {
            return false;
        }
    }
    
    // Check UI elements
    if (m_uiAutomation) {
        for (const QString& element : validation.requiredUIElements) {
            UIAutomation::WidgetSelector selector = UIAutomation::byObjectName(element);
            if (!m_uiAutomation->verifyWidgetExists(selector)) {
                return false;
            }
        }
        
        for (const QString& element : validation.forbiddenUIElements) {
            UIAutomation::WidgetSelector selector = UIAutomation::byObjectName(element);
            if (m_uiAutomation->verifyWidgetExists(selector)) {
                return false;
            }
        }
    }
    
    // Custom validation
    if (validation.customValidator) {
        return validation.customValidator();
    }
    
    return true;
}

ApplicationState WorkflowTesting::captureApplicationState() {
    ApplicationState state;
    
    // Capture basic application state
    state.openWindows = captureOpenWindows();
    state.activeWindow = captureActiveWindow();
    state.windowStates = captureWindowStates();
    state.enabledActions = captureEnabledActions();
    state.disabledActions = captureDisabledActions();
    state.settings = captureApplicationSettings();
    
    return state;
}

FileSystemState WorkflowTesting::captureFileSystemState(const QString& basePath) {
    FileSystemState state;
    
    state.existingFiles = findExistingFiles(basePath);
    state.fileSizes = getFileSizes(state.existingFiles);
    state.fileTimestamps = getFileTimestamps(state.existingFiles);
    state.fileHashes = calculateFileHashes(state.existingFiles);
    state.backupLocation = findBackupLocation();
    if (!state.backupLocation.isEmpty()) {
        state.backupFiles = findBackupFiles(state.backupLocation);
    }
    
    return state;
}

// Helper method implementations
bool WorkflowTesting::prepareWorkflowExecution(const UserWorkflow& workflow) {
    // Setup test environment if available
    if (m_testEnvironment && !m_testEnvironment->isSetup()) {
        if (!m_testEnvironment->setupTestEnvironment()) {
            return false;
        }
    }
    
    // Initialize execution context with initial state
    m_executionContext = workflow.initialState;
    
    return true;
}

bool WorkflowTesting::finalizeWorkflowExecution(const UserWorkflow& workflow, WorkflowResult& result) {
    // Cleanup if needed
    if (m_testEnvironment) {
        // Don't cleanup automatically - let test environment handle it
    }
    
    return true;
}

bool WorkflowTesting::checkPreconditions(const QStringList& preconditions, const QMap<QString, QVariant>& context) {
    for (const QString& condition : preconditions) {
        if (!context.contains(condition) || !context[condition].toBool()) {
            return false;
        }
    }
    return true;
}

bool WorkflowTesting::verifyPostconditions(const QStringList& postconditions, const QMap<QString, QVariant>& context) {
    // For now, assume postconditions are met if step succeeded
    // In a real implementation, this would check actual conditions
    return true;
}

void WorkflowTesting::updateExecutionContext(const QString& stepId, const QMap<QString, QVariant>& stepResult, QMap<QString, QVariant>& context) {
    // Update context with step results
    context[stepId + "_completed"] = true;
    context[stepId + "_result"] = QVariant(stepResult);
    
    // Add any specific context updates based on step type
    if (stepResult.contains("files_created")) {
        context["created_files"] = stepResult["files_created"];
    }
    if (stepResult.contains("files_deleted")) {
        context["deleted_files"] = stepResult["files_deleted"];
    }
}

// Placeholder implementations for helper methods
QStringList WorkflowTesting::captureOpenWindows() {
    QStringList windows;
    // Implementation would capture actual open windows
    return windows;
}

QString WorkflowTesting::captureActiveWindow() {
    // Implementation would capture actual active window
    return QString();
}

QMap<QString, QVariant> WorkflowTesting::captureWindowStates() {
    QMap<QString, QVariant> states;
    // Implementation would capture actual window states
    return states;
}

QStringList WorkflowTesting::captureEnabledActions() {
    QStringList actions;
    // Implementation would capture actual enabled actions
    return actions;
}

QStringList WorkflowTesting::captureDisabledActions() {
    QStringList actions;
    // Implementation would capture actual disabled actions
    return actions;
}

QMap<QString, QVariant> WorkflowTesting::captureApplicationSettings() {
    QMap<QString, QVariant> settings;
    // Implementation would capture actual application settings
    return settings;
}

QStringList WorkflowTesting::findExistingFiles(const QString& basePath) {
    QStringList files;
    QDir dir(basePath);
    if (dir.exists()) {
        QDirIterator it(basePath, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            files.append(it.next());
        }
    }
    return files;
}

QMap<QString, qint64> WorkflowTesting::getFileSizes(const QStringList& files) {
    QMap<QString, qint64> sizes;
    for (const QString& file : files) {
        QFileInfo info(file);
        sizes[file] = info.size();
    }
    return sizes;
}

QMap<QString, QDateTime> WorkflowTesting::getFileTimestamps(const QStringList& files) {
    QMap<QString, QDateTime> timestamps;
    for (const QString& file : files) {
        QFileInfo info(file);
        timestamps[file] = info.lastModified();
    }
    return timestamps;
}

QMap<QString, QString> WorkflowTesting::calculateFileHashes(const QStringList& files) {
    QMap<QString, QString> hashes;
    for (const QString& file : files) {
        QFile f(file);
        if (f.open(QIODevice::ReadOnly)) {
            QCryptographicHash hash(QCryptographicHash::Md5);
            hash.addData(f.readAll());
            hashes[file] = hash.result().toHex();
        }
    }
    return hashes;
}

QString WorkflowTesting::findBackupLocation() {
    // Implementation would find actual backup location
    return QString();
}

QStringList WorkflowTesting::findBackupFiles(const QString& backupLocation) {
    return findExistingFiles(backupLocation);
}

void WorkflowTesting::logWorkflowStart(const QString& workflowId) {
    if (m_detailedLogging) {
        qDebug() << "Starting workflow:" << workflowId;
    }
}

void WorkflowTesting::logWorkflowEnd(const QString& workflowId, const WorkflowResult& result) {
    if (m_detailedLogging) {
        qDebug() << "Completed workflow:" << workflowId << "Success:" << result.success;
    }
}

void WorkflowTesting::logStepStart(const QString& stepId, const WorkflowStep& step) {
    if (m_detailedLogging) {
        qDebug() << "Starting step:" << stepId << "-" << step.name;
    }
}

void WorkflowTesting::logStepEnd(const QString& stepId, bool success, const QString& error) {
    if (m_detailedLogging) {
        qDebug() << "Completed step:" << stepId << "Success:" << success;
        if (!error.isEmpty()) {
            qDebug() << "Error:" << error;
        }
    }
}

QString WorkflowTesting::captureScreenshot(const QString& prefix) {
    if (!m_automaticScreenshots) {
        return QString();
    }
    
    QString filename = QString("%1_%2_%3.png")
                      .arg(prefix)
                      .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"))
                      .arg(QRandomGenerator::global()->bounded(1000));
    
    QString filepath = QDir(m_screenshotDirectory).absoluteFilePath(filename);
    
    // Implementation would capture actual screenshot
    // For now, just create an empty file
    QFile file(filepath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write("screenshot placeholder");
        return filepath;
    }
    
    return QString();
}

QString WorkflowTesting::generateExecutionLog(const WorkflowResult& result) {
    QString filename = QString("workflow_%1_%2.log")
                      .arg(result.workflowId)
                      .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
    
    QString filepath = QDir(m_logDirectory).absoluteFilePath(filename);
    
    QFile file(filepath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        stream << "Workflow Execution Log\n";
        stream << "======================\n";
        stream << "Workflow ID: " << result.workflowId << "\n";
        stream << "Success: " << (result.success ? "Yes" : "No") << "\n";
        stream << "Total Steps: " << result.totalSteps << "\n";
        stream << "Completed Steps: " << result.completedSteps << "\n";
        stream << "Failed Steps: " << result.failedSteps << "\n";
        stream << "Execution Time: " << result.executionTimeMs << " ms\n";
        stream << "\nStep Execution Times:\n";
        for (auto it = result.stepExecutionTimes.begin(); it != result.stepExecutionTimes.end(); ++it) {
            stream << "  " << it.key() << ": " << it.value() << " ms\n";
        }
        if (!result.validationErrors.isEmpty()) {
            stream << "\nValidation Errors:\n";
            for (const QString& error : result.validationErrors) {
                stream << "  - " << error << "\n";
            }
        }
        return filepath;
    }
    
    return QString();
}

// Static utility methods
UserWorkflow WorkflowTesting::loadWorkflowFromJson(const QJsonObject& json) {
    UserWorkflow workflow;
    workflow.id = json["id"].toString();
    workflow.name = json["name"].toString();
    workflow.description = json["description"].toString();
    workflow.category = json["category"].toString();
    
    QJsonArray tagsArray = json["tags"].toArray();
    for (const QJsonValue& tag : tagsArray) {
        workflow.tags.append(tag.toString());
    }
    
    // Load steps
    QJsonArray stepsArray = json["steps"].toArray();
    for (const QJsonValue& stepValue : stepsArray) {
        QJsonObject stepObj = stepValue.toObject();
        WorkflowStep step;
        step.id = stepObj["id"].toString();
        step.name = stepObj["name"].toString();
        step.description = stepObj["description"].toString();
        step.type = static_cast<WorkflowStepType>(stepObj["type"].toInt());
        step.timeoutMs = stepObj["timeout"].toInt(30000);
        step.optional = stepObj["optional"].toBool(false);
        
        // Load parameters
        QJsonObject paramsObj = stepObj["parameters"].toObject();
        for (auto it = paramsObj.begin(); it != paramsObj.end(); ++it) {
            step.parameters[it.key()] = it.value().toVariant();
        }
        
        workflow.steps.append(step);
    }
    
    return workflow;
}

QJsonObject WorkflowTesting::saveWorkflowToJson(const UserWorkflow& workflow) {
    QJsonObject json;
    json["id"] = workflow.id;
    json["name"] = workflow.name;
    json["description"] = workflow.description;
    json["category"] = workflow.category;
    
    QJsonArray tagsArray;
    for (const QString& tag : workflow.tags) {
        tagsArray.append(tag);
    }
    json["tags"] = tagsArray;
    
    QJsonArray stepsArray;
    for (const WorkflowStep& step : workflow.steps) {
        QJsonObject stepObj;
        stepObj["id"] = step.id;
        stepObj["name"] = step.name;
        stepObj["description"] = step.description;
        stepObj["type"] = static_cast<int>(step.type);
        stepObj["timeout"] = step.timeoutMs;
        stepObj["optional"] = step.optional;
        
        QJsonObject paramsObj;
        for (auto it = step.parameters.begin(); it != step.parameters.end(); ++it) {
            paramsObj[it.key()] = QJsonValue::fromVariant(it.value());
        }
        stepObj["parameters"] = paramsObj;
        
        stepsArray.append(stepObj);
    }
    json["steps"] = stepsArray;
    
    return json;
}

WorkflowStep WorkflowTesting::createUIActionStep(const QString& id, const QString& action, const QMap<QString, QVariant>& parameters) {
    WorkflowStep step;
    step.id = id;
    step.name = QString("UI Action: %1").arg(action);
    step.type = WorkflowStepType::UIAction;
    step.parameters = parameters;
    step.parameters["action"] = action;
    return step;
}

WorkflowStep WorkflowTesting::createValidationStep(const QString& id, const QString& description, std::function<bool()> validator) {
    WorkflowStep step;
    step.id = id;
    step.name = QString("Validation: %1").arg(description);
    step.description = description;
    step.type = WorkflowStepType::Validation;
    step.validator = validator;
    return step;
}

WorkflowStep WorkflowTesting::createWaitStep(const QString& id, const QString& condition, int timeoutMs) {
    WorkflowStep step;
    step.id = id;
    step.name = QString("Wait: %1").arg(condition);
    step.type = WorkflowStepType::Wait;
    step.parameters["condition"] = condition;
    step.timeoutMs = timeoutMs;
    return step;
}

// Configuration methods
void WorkflowTesting::setDefaultTimeout(int timeoutMs) {
    m_defaultTimeoutMs = timeoutMs;
}

void WorkflowTesting::setScreenshotDirectory(const QString& directory) {
    m_screenshotDirectory = directory;
    QDir().mkpath(directory);
}

void WorkflowTesting::setLogDirectory(const QString& directory) {
    m_logDirectory = directory;
    QDir().mkpath(directory);
}

void WorkflowTesting::enableDetailedLogging(bool enable) {
    m_detailedLogging = enable;
}

void WorkflowTesting::enableAutomaticScreenshots(bool enable) {
    m_automaticScreenshots = enable;
}

void WorkflowTesting::setRetryAttempts(int attempts) {
    m_retryAttempts = attempts;
}

void WorkflowTesting::setParallelExecution(bool enable) {
    m_parallelExecution = enable;
}

#include "workflow_testing.moc"