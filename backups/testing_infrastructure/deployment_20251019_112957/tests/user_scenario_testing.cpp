#include "user_scenario_testing.h"
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
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QRandomGenerator>
#include <stdexcept>

UserScenarioTesting::UserScenarioTesting(QObject* parent)
    : QObject(parent)
    , m_currentPersona(UserPersona::FirstTimeUser)
    , m_scenarioTimeoutMs(300000)  // 5 minutes default
    , m_errorRecoveryTimeoutMs(60000)  // 1 minute default
    , m_userExperienceMetricsEnabled(true)
    , m_accessibilityTestingEnabled(true)
{
    // Initialize persona configurations
    setupPersonaConfigurations();
}

UserScenarioTesting::~UserScenarioTesting() = default;

void UserScenarioTesting::setWorkflowTesting(std::shared_ptr<WorkflowTesting> workflowTesting) {
    m_workflowTesting = workflowTesting;
}

void UserScenarioTesting::registerScenario(const UserScenario& scenario) {
    m_registeredScenarios[scenario.id] = scenario;
    
    // Also register the underlying workflow
    if (m_workflowTesting) {
        m_workflowTesting->registerWorkflow(scenario.workflow);
    }
}

void UserScenarioTesting::unregisterScenario(const QString& scenarioId) {
    if (m_registeredScenarios.contains(scenarioId)) {
        UserScenario scenario = m_registeredScenarios[scenarioId];
        m_registeredScenarios.remove(scenarioId);
        
        // Also unregister the underlying workflow
        if (m_workflowTesting) {
            m_workflowTesting->unregisterWorkflow(scenario.workflow.id);
        }
    }
}

QStringList UserScenarioTesting::getRegisteredScenarios() const {
    return m_registeredScenarios.keys();
}

UserScenario UserScenarioTesting::getScenario(const QString& scenarioId) const {
    return m_registeredScenarios.value(scenarioId);
}

QStringList UserScenarioTesting::getScenariosByPersona(UserPersona persona) const {
    QStringList scenarios;
    for (auto it = m_registeredScenarios.begin(); it != m_registeredScenarios.end(); ++it) {
        if (it.value().persona == persona) {
            scenarios.append(it.key());
        }
    }
    return scenarios;
}

QStringList UserScenarioTesting::getScenariosByComplexity(ScenarioComplexity complexity) const {
    QStringList scenarios;
    for (auto it = m_registeredScenarios.begin(); it != m_registeredScenarios.end(); ++it) {
        if (it.value().complexity == complexity) {
            scenarios.append(it.key());
        }
    }
    return scenarios;
}

UserScenario UserScenarioTesting::createFirstTimeUserScenario() {
    UserScenario scenario;
    scenario.id = "first_time_user_complete";
    scenario.name = "First-Time User Complete Experience";
    scenario.description = "Complete first-time user journey from installation to first successful duplicate cleanup";
    scenario.persona = UserPersona::FirstTimeUser;
    scenario.complexity = ScenarioComplexity::Simple;
    scenario.category = "user_onboarding";
    scenario.tags = {"first_time", "onboarding", "tutorial", "basic"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    // Define user goals
    scenario.goals = {
        "understand_application_purpose",
        "successfully_launch_application",
        "complete_initial_setup",
        "perform_first_scan",
        "understand_results",
        "safely_delete_duplicates",
        "feel_confident_using_app"
    };
    
    // Prerequisites for first-time user
    scenario.prerequisites = {
        "clean_system_state",
        "sample_duplicate_files_available",
        "no_previous_application_settings"
    };
    
    // Success criteria
    scenario.successCriteria = {
        "application_launches_successfully",
        "welcome_tutorial_completed",
        "first_scan_completes_without_errors",
        "duplicate_files_found_and_displayed",
        "user_successfully_deletes_duplicates",
        "no_important_files_accidentally_deleted",
        "user_understands_basic_features"
    };
    
    // Known failure modes
    scenario.failureModes = {
        "application_fails_to_launch",
        "tutorial_confuses_user",
        "scan_finds_no_duplicates",
        "user_accidentally_deletes_important_files",
        "interface_too_complex_for_beginner"
    };
    
    // Create the underlying workflow
    scenario.workflow = createFirstTimeUserWorkflow();
    
    // Context data specific to first-time users
    scenario.contextData["show_tooltips"] = true;
    scenario.contextData["enable_confirmation_dialogs"] = true;
    scenario.contextData["use_safe_defaults"] = true;
    scenario.contextData["provide_explanations"] = true;
    scenario.contextData["limit_advanced_features"] = true;
    
    return scenario;
}

UserScenario UserScenarioTesting::createCasualUserScenario() {
    UserScenario scenario;
    scenario.id = "casual_user_routine_cleanup";
    scenario.name = "Casual User Routine Cleanup";
    scenario.description = "Typical casual user performing routine duplicate file cleanup";
    scenario.persona = UserPersona::CasualUser;
    scenario.complexity = ScenarioComplexity::Simple;
    scenario.category = "routine_maintenance";
    scenario.tags = {"casual", "routine", "cleanup", "basic"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    scenario.goals = {
        "quickly_find_duplicates",
        "safely_remove_duplicates",
        "free_up_disk_space",
        "maintain_file_organization"
    };
    
    scenario.prerequisites = {
        "application_previously_used",
        "basic_familiarity_with_interface",
        "typical_file_collection_with_duplicates"
    };
    
    scenario.successCriteria = {
        "scan_completes_in_reasonable_time",
        "duplicates_identified_correctly",
        "safe_deletion_without_data_loss",
        "disk_space_freed_up",
        "process_completed_efficiently"
    };
    
    // Create workflow for casual user
    scenario.workflow = createCasualUserWorkflow();
    
    scenario.contextData["prefer_automatic_selection"] = true;
    scenario.contextData["use_default_settings"] = true;
    scenario.contextData["minimize_confirmations"] = false;
    scenario.contextData["focus_on_efficiency"] = true;
    
    return scenario;
}

UserScenario UserScenarioTesting::createPowerUserScenario() {
    UserScenario scenario;
    scenario.id = "power_user_advanced_operations";
    scenario.name = "Power User Advanced Operations";
    scenario.description = "Advanced user utilizing complex features and batch operations";
    scenario.persona = UserPersona::PowerUser;
    scenario.complexity = ScenarioComplexity::Advanced;
    scenario.category = "advanced_features";
    scenario.tags = {"power_user", "advanced", "batch", "complex"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    scenario.goals = {
        "configure_advanced_filters",
        "perform_batch_operations",
        "customize_selection_rules",
        "generate_detailed_reports",
        "automate_repetitive_tasks",
        "optimize_performance_settings"
    };
    
    scenario.prerequisites = {
        "expert_knowledge_of_application",
        "large_dataset_for_testing",
        "understanding_of_advanced_features"
    };
    
    scenario.successCriteria = {
        "complex_filters_applied_correctly",
        "batch_operations_complete_successfully",
        "custom_rules_work_as_expected",
        "reports_generated_with_accurate_data",
        "automation_saves_significant_time",
        "performance_optimized_for_large_datasets"
    };
    
    // Create workflow for power user
    scenario.workflow = createPowerUserWorkflow();
    
    scenario.contextData["enable_all_features"] = true;
    scenario.contextData["show_advanced_options"] = true;
    scenario.contextData["allow_dangerous_operations"] = true;
    scenario.contextData["provide_detailed_feedback"] = true;
    scenario.contextData["optimize_for_performance"] = true;
    
    return scenario;
}

UserScenario UserScenarioTesting::createSafetyFocusedUserScenario() {
    UserScenario scenario;
    scenario.id = "safety_focused_user_operations";
    scenario.name = "Safety-Focused User Operations";
    scenario.description = "User prioritizing data safety with comprehensive backup and verification";
    scenario.persona = UserPersona::SafetyFocusedUser;
    scenario.complexity = ScenarioComplexity::Intermediate;
    scenario.category = "safety_operations";
    scenario.tags = {"safety", "backup", "verification", "cautious"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    scenario.goals = {
        "ensure_complete_data_backup",
        "verify_backup_integrity",
        "perform_safe_duplicate_removal",
        "test_restore_functionality",
        "maintain_audit_trail",
        "minimize_risk_of_data_loss"
    };
    
    scenario.prerequisites = {
        "sufficient_backup_storage_space",
        "understanding_of_backup_procedures",
        "important_files_to_protect"
    };
    
    scenario.successCriteria = {
        "complete_backup_created_successfully",
        "backup_integrity_verified",
        "duplicates_removed_without_data_loss",
        "restore_functionality_tested_and_working",
        "audit_trail_maintained_throughout",
        "user_confidence_in_data_safety"
    };
    
    // Create workflow for safety-focused user
    scenario.workflow = createSafetyFocusedWorkflow();
    
    scenario.contextData["require_backups"] = true;
    scenario.contextData["verify_all_operations"] = true;
    scenario.contextData["enable_audit_logging"] = true;
    scenario.contextData["use_conservative_settings"] = true;
    scenario.contextData["test_restore_before_delete"] = true;
    
    return scenario;
}

UserScenario UserScenarioTesting::createBatchUserScenario() {
    UserScenario scenario;
    scenario.id = "batch_user_large_scale_operations";
    scenario.name = "Batch User Large-Scale Operations";
    scenario.description = "User processing large amounts of data with batch operations";
    scenario.persona = UserPersona::BatchUser;
    scenario.complexity = ScenarioComplexity::Advanced;
    scenario.category = "batch_processing";
    scenario.tags = {"batch", "large_scale", "automation", "efficiency"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    scenario.goals = {
        "process_multiple_directories_simultaneously",
        "apply_consistent_rules_across_datasets",
        "generate_comprehensive_reports",
        "automate_repetitive_operations",
        "optimize_processing_time",
        "handle_large_file_volumes_efficiently"
    };
    
    scenario.prerequisites = {
        "large_dataset_with_multiple_directories",
        "sufficient_system_resources",
        "understanding_of_batch_operations"
    };
    
    scenario.successCriteria = {
        "all_directories_processed_successfully",
        "consistent_results_across_datasets",
        "processing_completed_within_time_limits",
        "reports_generated_for_all_operations",
        "system_resources_used_efficiently",
        "no_data_corruption_or_loss"
    };
    
    // Create workflow for batch user
    scenario.workflow = createBatchUserWorkflow();
    
    scenario.contextData["enable_parallel_processing"] = true;
    scenario.contextData["optimize_for_throughput"] = true;
    scenario.contextData["generate_detailed_logs"] = true;
    scenario.contextData["use_batch_operations"] = true;
    scenario.contextData["monitor_system_resources"] = true;
    
    return scenario;
}

UserScenario UserScenarioTesting::createAccessibilityUserScenario() {
    UserScenario scenario;
    scenario.id = "accessibility_user_operations";
    scenario.name = "Accessibility User Operations";
    scenario.description = "User requiring accessibility features for application interaction";
    scenario.persona = UserPersona::AccessibilityUser;
    scenario.complexity = ScenarioComplexity::Intermediate;
    scenario.category = "accessibility";
    scenario.tags = {"accessibility", "keyboard_navigation", "screen_reader", "a11y"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    scenario.goals = {
        "navigate_interface_using_keyboard_only",
        "access_all_features_via_screen_reader",
        "complete_tasks_with_high_contrast_mode",
        "use_application_with_large_fonts",
        "receive_audio_feedback_for_actions",
        "maintain_efficient_workflow_with_assistive_tech"
    };
    
    scenario.prerequisites = {
        "screen_reader_software_available",
        "keyboard_navigation_enabled",
        "high_contrast_mode_supported",
        "understanding_of_accessibility_features"
    };
    
    scenario.successCriteria = {
        "all_interface_elements_keyboard_accessible",
        "screen_reader_announces_all_important_information",
        "high_contrast_mode_provides_sufficient_visibility",
        "large_fonts_display_correctly",
        "audio_feedback_provided_for_critical_actions",
        "workflow_completion_time_reasonable_with_assistive_tech"
    };
    
    // Create workflow for accessibility user
    scenario.workflow = createAccessibilityUserWorkflow();
    
    scenario.contextData["keyboard_navigation_only"] = true;
    scenario.contextData["screen_reader_enabled"] = true;
    scenario.contextData["high_contrast_mode"] = true;
    scenario.contextData["large_fonts"] = true;
    scenario.contextData["audio_feedback"] = true;
    
    return scenario;
}

UserScenario UserScenarioTesting::createPhotoLibraryCleanupScenario() {
    UserScenario scenario;
    scenario.id = "photo_library_cleanup";
    scenario.name = "Photo Library Cleanup";
    scenario.description = "User cleaning up a large photo library with many duplicates";
    scenario.persona = UserPersona::CasualUser;
    scenario.complexity = ScenarioComplexity::Intermediate;
    scenario.category = "media_management";
    scenario.tags = {"photos", "media", "cleanup", "duplicates"};
    scenario.author = "UserScenarioTesting";
    scenario.version = "1.0";
    scenario.created = QDateTime::currentDateTime();
    scenario.lastModified = QDateTime::currentDateTime();
    
    scenario.goals = {
        "identify_duplicate_photos",
        "preserve_highest_quality_versions",
        "organize_photos_by_date_or_event",
        "free_up_significant_storage_space",
        "maintain_photo_metadata",
        "avoid_deleting_unique_photos"
    };
    
    scenario.prerequisites = {
        "large_photo_collection_with_duplicates",
        "photos_in_various_formats_and_qualities",
        "sufficient_time_for_processing"
    };
    
    scenario.successCriteria = {
        "duplicate_photos_identified_accurately",
        "highest_quality_versions_preserved",
        "significant_storage_space_freed",
        "photo_metadata_preserved",
        "no_unique_photos_accidentally_deleted",
        "photo_organization_improved"
    };
    
    // Create specialized workflow for photo cleanup
    scenario.workflow = createPhotoLibraryWorkflow();
    
    scenario.contextData["file_types"] = QStringList{"jpg", "jpeg", "png", "tiff", "raw", "heic"};
    scenario.contextData["preserve_metadata"] = true;
    scenario.contextData["quality_comparison"] = true;
    scenario.contextData["date_organization"] = true;
    
    return scenario;
}

ScenarioResult UserScenarioTesting::executeScenario(const QString& scenarioId) {
    if (!m_registeredScenarios.contains(scenarioId)) {
        ScenarioResult result;
        result.scenarioId = scenarioId;
        result.success = false;
        result.userFeedback = QString("Scenario not found: %1").arg(scenarioId);
        return result;
    }
    
    return executeScenario(m_registeredScenarios[scenarioId]);
}

ScenarioResult UserScenarioTesting::executeScenario(const UserScenario& scenario) {
    ScenarioResult result;
    result.scenarioId = scenario.id;
    result.persona = scenario.persona;
    
    m_currentScenarioId = scenario.id;
    m_currentPersona = scenario.persona;
    m_scenarioTimer.start();
    
    emit scenarioStarted(scenario.id, scenario.persona);
    
    if (!prepareScenarioExecution(scenario)) {
        result.success = false;
        result.userFeedback = "Failed to prepare scenario execution";
        return result;
    }
    
    // Configure for persona
    if (!configureForPersona(scenario.persona)) {
        result.success = false;
        result.userFeedback = "Failed to configure for user persona";
        return result;
    }
    
    // Execute the underlying workflow
    if (m_workflowTesting) {
        result.workflowResult = m_workflowTesting->executeWorkflow(scenario.workflow);
        result.success = result.workflowResult.success;
    } else {
        result.success = false;
        result.userFeedback = "Workflow testing not available";
        return result;
    }
    
    result.totalExecutionTimeMs = m_scenarioTimer.elapsed();
    
    // Execute scenario-specific goals
    if (result.success) {
        result.success = executeScenarioGoals(scenario, result);
    }
    
    // Measure user experience metrics
    if (m_userExperienceMetricsEnabled) {
        measureUserInteractionTime(scenario, result);
        measureWaitTime(scenario, result);
        countUserActions(scenario, result);
    }
    
    // Detect usability and accessibility issues
    detectUsabilityIssues(scenario, result);
    if (m_accessibilityTestingEnabled) {
        detectAccessibilityIssues(scenario, result);
    }
    
    // Calculate satisfaction score
    result.satisfactionScore = calculateSatisfactionScore(result);
    
    // Validate user experience
    if (result.success) {
        result.success = validateUserExperience(result);
    }
    
    // Validate persona-specific experience
    if (result.success) {
        result.success = validatePersonaExperience(scenario.persona, result);
    }
    
    finalizeScenarioExecution(scenario, result);
    
    emit scenarioCompleted(scenario.id, result);
    
    return result;
}

QList<ScenarioResult> UserScenarioTesting::executeScenarioSuite(const QStringList& scenarioIds) {
    QList<ScenarioResult> results;
    
    for (const QString& scenarioId : scenarioIds) {
        ScenarioResult result = executeScenario(scenarioId);
        results.append(result);
        
        // Brief pause between scenarios
        QThread::msleep(1000);
    }
    
    return results;
}

QList<ScenarioResult> UserScenarioTesting::executePersonaScenarios(UserPersona persona) {
    QStringList scenarioIds = getScenariosByPersona(persona);
    return executeScenarioSuite(scenarioIds);
}

bool UserScenarioTesting::validateUserExperience(const ScenarioResult& result) {
    // Check basic UX criteria
    if (result.totalExecutionTimeMs > 600000) { // 10 minutes max
        return false;
    }
    
    if (result.errorEncountered > 3) { // Max 3 errors acceptable
        return false;
    }
    
    if (result.satisfactionScore < 6) { // Minimum satisfaction score
        return false;
    }
    
    if (result.usabilityIssues.size() > 5) { // Max 5 usability issues
        return false;
    }
    
    return true;
}

QStringList UserScenarioTesting::identifyUsabilityIssues(const ScenarioResult& result) {
    QStringList issues;
    
    // Check for common usability issues
    if (result.userInteractionTimeMs > result.totalExecutionTimeMs * 0.8) {
        issues.append("Excessive user interaction time - interface may be too complex");
    }
    
    if (result.waitTimeMs > result.totalExecutionTimeMs * 0.5) {
        issues.append("Excessive wait time - performance issues detected");
    }
    
    if (result.userActionsCount > 50) {
        issues.append("Too many user actions required - workflow could be simplified");
    }
    
    if (result.recoveryAttempts > 2) {
        issues.append("Multiple recovery attempts needed - error handling could be improved");
    }
    
    return issues;
}

QStringList UserScenarioTesting::identifyAccessibilityIssues(const ScenarioResult& result) {
    QStringList issues;
    
    // Check for accessibility issues based on scenario context
    if (m_currentPersona == UserPersona::AccessibilityUser) {
        if (result.totalExecutionTimeMs > result.workflowResult.executionTimeMs * 2) {
            issues.append("Accessibility workflow takes significantly longer than standard workflow");
        }
        
        if (result.errorEncountered > 0) {
            issues.append("Errors encountered during accessibility testing - may indicate keyboard navigation issues");
        }
    }
    
    return issues;
}

int UserScenarioTesting::calculateSatisfactionScore(const ScenarioResult& result) {
    int score = 10; // Start with perfect score
    
    // Deduct points for various issues
    score -= result.errorEncountered; // -1 per error
    score -= result.usabilityIssues.size() / 2; // -0.5 per usability issue
    score -= result.accessibilityIssues.size(); // -1 per accessibility issue
    
    // Deduct for excessive time
    if (result.totalExecutionTimeMs > 300000) { // > 5 minutes
        score -= 2;
    }
    
    // Deduct for failed goals
    score -= result.failedGoals.size() * 2; // -2 per failed goal
    
    // Ensure score is within valid range
    return qMax(1, qMin(10, score));
}

// Helper method implementations
bool UserScenarioTesting::prepareScenarioExecution(const UserScenario& scenario) {
    // Clear previous metrics
    m_interactionTimes.clear();
    m_waitTimes.clear();
    m_actionCounts.clear();
    m_detectedIssues.clear();
    
    return true;
}

bool UserScenarioTesting::finalizeScenarioExecution(const UserScenario& scenario, ScenarioResult& result) {
    // Aggregate metrics
    result.userInteractionTimeMs = 0;
    for (auto it = m_interactionTimes.begin(); it != m_interactionTimes.end(); ++it) {
        result.userInteractionTimeMs += it.value();
    }
    
    result.waitTimeMs = 0;
    for (auto it = m_waitTimes.begin(); it != m_waitTimes.end(); ++it) {
        result.waitTimeMs += it.value();
    }
    
    result.userActionsCount = 0;
    for (auto it = m_actionCounts.begin(); it != m_actionCounts.end(); ++it) {
        result.userActionsCount += it.value();
    }
    
    result.usabilityIssues = identifyUsabilityIssues(result);
    result.accessibilityIssues = identifyAccessibilityIssues(result);
    
    return true;
}

bool UserScenarioTesting::executeScenarioGoals(const UserScenario& scenario, ScenarioResult& result) {
    for (const QString& goal : scenario.goals) {
        // Simulate goal execution and validation
        bool goalAchieved = true; // Placeholder - implement actual goal validation
        
        if (goalAchieved) {
            result.completedGoals.append(goal);
            emit goalCompleted(scenario.id, goal);
        } else {
            result.failedGoals.append(goal);
            emit goalFailed(scenario.id, goal, "Goal validation failed");
        }
    }
    
    // Scenario succeeds if most goals are achieved
    return result.failedGoals.size() <= scenario.goals.size() / 3; // Allow up to 1/3 failures
}

bool UserScenarioTesting::validateScenarioSuccess(const UserScenario& scenario, const ScenarioResult& result) {
    // Check success criteria
    for (const QString& criterion : scenario.successCriteria) {
        // Implement actual criterion validation
        bool criterionMet = true; // Placeholder
        if (!criterionMet) {
            return false;
        }
    }
    
    return true;
}

void UserScenarioTesting::measureUserInteractionTime(const UserScenario& scenario, ScenarioResult& result) {
    // Measure time spent on actual user interactions vs waiting
    // This would be implemented by tracking UI automation timing
    result.userInteractionTimeMs = result.totalExecutionTimeMs * 0.6; // Placeholder
}

void UserScenarioTesting::measureWaitTime(const UserScenario& scenario, ScenarioResult& result) {
    // Measure time spent waiting for operations to complete
    result.waitTimeMs = result.totalExecutionTimeMs * 0.3; // Placeholder
}

void UserScenarioTesting::countUserActions(const UserScenario& scenario, ScenarioResult& result) {
    // Count the number of user actions performed
    result.userActionsCount = scenario.workflow.steps.size() * 2; // Placeholder
}

void UserScenarioTesting::detectUsabilityIssues(const UserScenario& scenario, ScenarioResult& result) {
    // Detect usability issues based on execution metrics
    QStringList issues = identifyUsabilityIssues(result);
    for (const QString& issue : issues) {
        emit usabilityIssueDetected(scenario.id, issue);
    }
}

void UserScenarioTesting::detectAccessibilityIssues(const UserScenario& scenario, ScenarioResult& result) {
    // Detect accessibility issues
    QStringList issues = identifyAccessibilityIssues(result);
    for (const QString& issue : issues) {
        emit accessibilityIssueDetected(scenario.id, issue);
    }
}

QMap<QString, QVariant> UserScenarioTesting::getPersonaDefaults(UserPersona persona) {
    QMap<QString, QVariant> defaults;
    
    switch (persona) {
        case UserPersona::FirstTimeUser:
            defaults["show_tooltips"] = true;
            defaults["enable_confirmations"] = true;
            defaults["use_safe_defaults"] = true;
            defaults["show_help"] = true;
            break;
            
        case UserPersona::CasualUser:
            defaults["prefer_automation"] = true;
            defaults["use_defaults"] = true;
            defaults["minimize_steps"] = true;
            break;
            
        case UserPersona::PowerUser:
            defaults["show_advanced_options"] = true;
            defaults["enable_batch_operations"] = true;
            defaults["allow_dangerous_operations"] = true;
            defaults["detailed_feedback"] = true;
            break;
            
        case UserPersona::SafetyFocusedUser:
            defaults["require_backups"] = true;
            defaults["verify_operations"] = true;
            defaults["conservative_settings"] = true;
            defaults["audit_logging"] = true;
            break;
            
        case UserPersona::BatchUser:
            defaults["parallel_processing"] = true;
            defaults["optimize_throughput"] = true;
            defaults["batch_operations"] = true;
            defaults["detailed_logging"] = true;
            break;
            
        case UserPersona::AccessibilityUser:
            defaults["keyboard_navigation"] = true;
            defaults["screen_reader_support"] = true;
            defaults["high_contrast"] = true;
            defaults["large_fonts"] = true;
            defaults["audio_feedback"] = true;
            break;
            
        case UserPersona::MobileUser:
            defaults["touch_interface"] = true;
            defaults["simplified_ui"] = true;
            defaults["gesture_support"] = true;
            defaults["mobile_optimized"] = true;
            break;
    }
    
    return defaults;
}

bool UserScenarioTesting::configureForPersona(UserPersona persona) {
    QMap<QString, QVariant> config = getPersonaDefaults(persona);
    
    // Apply persona-specific configuration
    // This would configure the application for the specific persona
    // For now, just store the configuration
    m_personaConfigurations[persona] = config;
    
    return true;
}

bool UserScenarioTesting::validatePersonaExperience(UserPersona persona, const ScenarioResult& result) {
    // Validate that the experience is appropriate for the persona
    switch (persona) {
        case UserPersona::FirstTimeUser:
            // First-time users should have guided experience
            return result.errorEncountered <= 1 && result.satisfactionScore >= 7;
            
        case UserPersona::CasualUser:
            // Casual users should have efficient experience
            return result.totalExecutionTimeMs <= 300000 && result.userActionsCount <= 20;
            
        case UserPersona::PowerUser:
            // Power users should have access to advanced features
            return result.completedGoals.size() >= result.failedGoals.size() * 3;
            
        case UserPersona::SafetyFocusedUser:
            // Safety-focused users should have no data loss
            return result.errorEncountered == 0;
            
        case UserPersona::BatchUser:
            // Batch users should have efficient processing
            return result.totalExecutionTimeMs <= 600000; // Allow more time for batch operations
            
        case UserPersona::AccessibilityUser:
            // Accessibility users should have full access
            return result.accessibilityIssues.isEmpty();
            
        case UserPersona::MobileUser:
            // Mobile users should have touch-friendly experience
            return result.userActionsCount <= 15; // Fewer actions for mobile
    }
    
    return true;
}

void UserScenarioTesting::setupPersonaConfigurations() {
    // Initialize default configurations for each persona
    for (int i = static_cast<int>(UserPersona::FirstTimeUser); 
         i <= static_cast<int>(UserPersona::AccessibilityUser); ++i) {
        UserPersona persona = static_cast<UserPersona>(i);
        m_personaConfigurations[persona] = getPersonaDefaults(persona);
    }
}

// Create helper workflows for different scenarios
UserWorkflow UserScenarioTesting::createFirstTimeUserWorkflow() {
    // This would create a workflow specifically designed for first-time users
    // For now, return a basic workflow
    UserWorkflow workflow;
    workflow.id = "first_time_user_workflow_detailed";
    workflow.name = "Detailed First-Time User Workflow";
    workflow.description = "Step-by-step workflow for first-time users";
    
    // Add workflow steps specific to first-time users
    // This is a simplified version - full implementation would have detailed steps
    
    return workflow;
}

UserWorkflow UserScenarioTesting::createCasualUserWorkflow() {
    UserWorkflow workflow;
    workflow.id = "casual_user_workflow";
    workflow.name = "Casual User Workflow";
    workflow.description = "Efficient workflow for casual users";
    return workflow;
}

UserWorkflow UserScenarioTesting::createBatchUserWorkflow() {
    UserWorkflow workflow;
    workflow.id = "batch_user_workflow";
    workflow.name = "Batch User Workflow";
    workflow.description = "Batch processing workflow";
    return workflow;
}

UserWorkflow UserScenarioTesting::createAccessibilityUserWorkflow() {
    UserWorkflow workflow;
    workflow.id = "accessibility_user_workflow";
    workflow.name = "Accessibility User Workflow";
    workflow.description = "Accessibility-focused workflow";
    return workflow;
}

UserWorkflow UserScenarioTesting::createPhotoLibraryWorkflow() {
    UserWorkflow workflow;
    workflow.id = "photo_library_workflow";
    workflow.name = "Photo Library Cleanup Workflow";
    workflow.description = "Specialized workflow for photo library cleanup";
    return workflow;
}

// Static utility methods
QString UserScenarioTesting::personaToString(UserPersona persona) {
    switch (persona) {
        case UserPersona::FirstTimeUser: return "FirstTimeUser";
        case UserPersona::CasualUser: return "CasualUser";
        case UserPersona::PowerUser: return "PowerUser";
        case UserPersona::SafetyFocusedUser: return "SafetyFocusedUser";
        case UserPersona::BatchUser: return "BatchUser";
        case UserPersona::MobileUser: return "MobileUser";
        case UserPersona::AccessibilityUser: return "AccessibilityUser";
    }
    return "Unknown";
}

UserPersona UserScenarioTesting::stringToPersona(const QString& personaStr) {
    if (personaStr == "FirstTimeUser") return UserPersona::FirstTimeUser;
    if (personaStr == "CasualUser") return UserPersona::CasualUser;
    if (personaStr == "PowerUser") return UserPersona::PowerUser;
    if (personaStr == "SafetyFocusedUser") return UserPersona::SafetyFocusedUser;
    if (personaStr == "BatchUser") return UserPersona::BatchUser;
    if (personaStr == "MobileUser") return UserPersona::MobileUser;
    if (personaStr == "AccessibilityUser") return UserPersona::AccessibilityUser;
    return UserPersona::FirstTimeUser;
}

QString UserScenarioTesting::complexityToString(ScenarioComplexity complexity) {
    switch (complexity) {
        case ScenarioComplexity::Simple: return "Simple";
        case ScenarioComplexity::Intermediate: return "Intermediate";
        case ScenarioComplexity::Advanced: return "Advanced";
        case ScenarioComplexity::Expert: return "Expert";
    }
    return "Unknown";
}

ScenarioComplexity UserScenarioTesting::stringToComplexity(const QString& complexityStr) {
    if (complexityStr == "Simple") return ScenarioComplexity::Simple;
    if (complexityStr == "Intermediate") return ScenarioComplexity::Intermediate;
    if (complexityStr == "Advanced") return ScenarioComplexity::Advanced;
    if (complexityStr == "Expert") return ScenarioComplexity::Expert;
    return ScenarioComplexity::Simple;
}

// Configuration methods
void UserScenarioTesting::setPersonaConfiguration(UserPersona persona, const QMap<QString, QVariant>& config) {
    m_personaConfigurations[persona] = config;
}

QMap<QString, QVariant> UserScenarioTesting::getPersonaConfiguration(UserPersona persona) const {
    return m_personaConfigurations.value(persona);
}

void UserScenarioTesting::setScenarioTimeout(int timeoutMs) {
    m_scenarioTimeoutMs = timeoutMs;
}

void UserScenarioTesting::enableUserExperienceMetrics(bool enable) {
    m_userExperienceMetricsEnabled = enable;
}

void UserScenarioTesting::enableAccessibilityTesting(bool enable) {
    m_accessibilityTestingEnabled = enable;
}

void UserScenarioTesting::setErrorRecoveryTimeout(int timeoutMs) {
    m_errorRecoveryTimeoutMs = timeoutMs;
}

#include "user_scenario_testing.moc"