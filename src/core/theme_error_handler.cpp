#include "theme_error_handler.h"
#include "theme_recovery_dialog.h"
#include "theme_notification_widget.h"
#include "core/logger.h"
#include <QApplication>
#include <QMessageBox>
#include <QTimer>
#include <QDebug>
#include <QStandardPaths>
#include <QDir>

// Static member initialization
ThemeErrorHandler* ThemeErrorHandler::s_instance = nullptr;
QStringList ThemeErrorHandler::s_failedComponents;
QList<ThemeErrorHandler::ErrorContext> ThemeErrorHandler::s_errorHistory;
int ThemeErrorHandler::s_maxRetryAttempts = 3;
bool ThemeErrorHandler::s_autoRecoveryEnabled = true;
bool ThemeErrorHandler::s_userNotificationEnabled = true;
QMap<ThemeErrorHandler::ErrorType, int> ThemeErrorHandler::s_errorCounts;

ThemeErrorHandler* ThemeErrorHandler::instance()
{
    if (!s_instance) {
        s_instance = new ThemeErrorHandler(qApp);
    }
    return s_instance;
}

ThemeErrorHandler::ThemeErrorHandler(QObject* parent)
    : QObject(parent)
{
    LOG_INFO(LogCategories::UI, "ThemeErrorHandler initialized");
}

void ThemeErrorHandler::handleError(ErrorType type, const QString& details, QWidget* component)
{
    // Create error context
    ErrorContext context;
    context.type = type;
    context.componentName = component ? component->metaObject()->className() : "Unknown";
    context.errorMessage = details;
    context.failedWidget = component;
    context.retryCount = 0;
    context.timestamp = QDateTime::currentDateTime();
    
    // Record the error
    recordError(context);
    
    // Log the error with appropriate severity
    QString errorTypeStr;
    QString logCategory = "UI";
    
    switch (type) {
        case ErrorType::ThemeLoadFailure:
            errorTypeStr = "Theme Load Failure";
            LOG_ERROR("UI", QString("Theme Load Failure: %1").arg(details));
            break;
        case ErrorType::StyleApplicationFailure:
            errorTypeStr = "Style Application Failure";
            LOG_ERROR("UI", QString("Style Application Failure on %1: %2")
                     .arg(context.componentName).arg(details));
            break;
        case ErrorType::ComponentRegistrationFailure:
            errorTypeStr = "Component Registration Failure";
            LOG_WARNING("UI", QString("Component Registration Failure for %1: %2")
                       .arg(context.componentName).arg(details));
            break;
        case ErrorType::ValidationFailure:
            errorTypeStr = "Validation Failure";
            LOG_WARNING("UI", QString("Theme Validation Failure: %1").arg(details));
            break;
        case ErrorType::PersistenceFailure:
            errorTypeStr = "Persistence Failure";
            LOG_ERROR("UI", QString("Theme Persistence Failure: %1").arg(details));
            break;
        case ErrorType::CustomThemeCorruption:
            errorTypeStr = "Custom Theme Corruption";
            LOG_ERROR("UI", QString("Custom Theme Corruption: %1").arg(details));
            break;
        case ErrorType::SystemThemeDetectionFailure:
            errorTypeStr = "System Theme Detection Failure";
            LOG_WARNING("UI", QString("System Theme Detection Failure: %1").arg(details));
            break;
        case ErrorType::DialogRegistrationFailure:
            errorTypeStr = "Dialog Registration Failure";
            LOG_WARNING("UI", QString("Dialog Registration Failure for %1: %2")
                       .arg(context.componentName).arg(details));
            break;
    }
    
    // Increment error count
    s_errorCounts[type]++;
    
    // Emit signal for other components to handle
    instance()->errorOccurred(type, details);
    
    // Attempt automatic recovery if enabled
    if (s_autoRecoveryEnabled && canRecover(type)) {
        LOG_INFO("UI", QString("Attempting automatic recovery for %1").arg(errorTypeStr));
        
        bool recoverySuccess = attemptRecovery(type, component, details);
        
        instance()->recoveryAttempted(type, recoverySuccess);
        
        if (recoverySuccess) {
            LOG_INFO(logCategory, QString("Automatic recovery successful for %1").arg(errorTypeStr));
        } else {
            LOG_ERROR(logCategory, QString("Automatic recovery failed for %1").arg(errorTypeStr));
            
            // If auto recovery fails and user notification is enabled, show recovery dialog for critical errors
            if (s_userNotificationEnabled && (type == ErrorType::ThemeLoadFailure || 
                                            type == ErrorType::CustomThemeCorruption ||
                                            type == ErrorType::StyleApplicationFailure)) {
                LOG_INFO("UI", "Showing recovery dialog for critical theme error");
                
                // Use QTimer to ensure we're in the main thread and UI is ready
                QTimer::singleShot(100, [type, details]() {
                    showRecoveryDialog(type, details);
                });
            } else if (s_userNotificationEnabled) {
                // For non-critical errors, show notification widget
                QTimer::singleShot(100, [type, details]() {
                    static ThemeNotificationWidget* notificationWidget = nullptr;
                    if (!notificationWidget) {
                        notificationWidget = new ThemeNotificationWidget();
                    }
                    notificationWidget->showThemeError(type, details);
                });
            }
        }
    }
}

bool ThemeErrorHandler::attemptRecovery(ErrorType type, QWidget* component, const QString& context)
{
    LOG_INFO(LogCategories::UI, QString("Attempting recovery for error type: %1").arg(static_cast<int>(type)));
    
    RecoveryAction action = determineRecoveryAction(type);
    bool recoverySuccess = false;
    
    switch (action) {
        case RecoveryAction::FallbackToDefault:
            LOG_INFO(LogCategories::UI, "Recovery action: Fallback to default theme");
            fallbackToDefaultTheme();
            recoverySuccess = true;
            break;
            
        case RecoveryAction::RetryOperation:
            LOG_INFO(LogCategories::UI, "Recovery action: Retry operation");
            if (component) {
                recoverySuccess = retryThemeApplication(component);
            } else {
                // Retry theme application for all components
                ThemeManager* themeManager = ThemeManager::instance();
                if (themeManager) {
                    try {
                        themeManager->applyToApplication();
                        recoverySuccess = true;
                    } catch (...) {
                        recoverySuccess = false;
                    }
                }
            }
            break;
            
        case RecoveryAction::SkipComponent:
            LOG_INFO(LogCategories::UI, QString("Recovery action: Skip component %1")
                     .arg(component ? component->metaObject()->className() : "Unknown"));
            if (component) {
                s_failedComponents.append(component->metaObject()->className());
            }
            recoverySuccess = true; // Consider skipping as successful recovery
            break;
            
        case RecoveryAction::NotifyUser:
            LOG_INFO(LogCategories::UI, "Recovery action: Notify user");
            if (s_userNotificationEnabled) {
                recoverySuccess = showRecoveryDialog(type, context);
            }
            break;
            
        case RecoveryAction::LogAndContinue:
            LOG_INFO(LogCategories::UI, "Recovery action: Log and continue");
            logError(QString("Non-critical error handled: %1").arg(context), type);
            recoverySuccess = true;
            break;
    }
    
    if (recoverySuccess) {
        LOG_INFO(LogCategories::UI, "Recovery attempt successful");
    } else {
        LOG_ERROR(LogCategories::UI, "Recovery attempt failed");
    }
    
    return recoverySuccess;
}

void ThemeErrorHandler::fallbackToDefaultTheme()
{
    LOG_INFO(LogCategories::UI, "Falling back to default theme");
    
    ThemeManager* themeManager = ThemeManager::instance();
    if (!themeManager) {
        LOG_ERROR(LogCategories::UI, "ThemeManager not available for fallback");
        return;
    }
    
    try {
        // Clear any failed components
        clearFailedComponents();
        
        // Reset to system default theme
        themeManager->setTheme(ThemeManager::SystemDefault);
        
        // Apply to application
        themeManager->applyToApplication();
        
        LOG_INFO(LogCategories::UI, "Successfully fell back to default theme");
        
        // Emit signal
        instance()->fallbackActivated("Theme error recovery");
        
    } catch (const std::exception& e) {
        LOG_ERROR(LogCategories::UI, QString("Failed to fallback to default theme: %1").arg(e.what()));
        
        // Last resort: try to apply basic styling
        try {
            qApp->setStyleSheet("");  // Clear all styles
            LOG_WARNING(LogCategories::UI, "Cleared all styles as last resort");
        } catch (...) {
            LOG_CRITICAL(LogCategories::UI, "Complete theme system failure - unable to apply any styling");
        }
    }
}

void ThemeErrorHandler::logError(const QString& message, ErrorType type)
{
    QString typeStr;
    switch (type) {
        case ErrorType::ThemeLoadFailure: typeStr = "THEME_LOAD"; break;
        case ErrorType::StyleApplicationFailure: typeStr = "STYLE_APP"; break;
        case ErrorType::ComponentRegistrationFailure: typeStr = "COMP_REG"; break;
        case ErrorType::ValidationFailure: typeStr = "VALIDATION"; break;
        case ErrorType::PersistenceFailure: typeStr = "PERSISTENCE"; break;
        case ErrorType::CustomThemeCorruption: typeStr = "THEME_CORRUPT"; break;
        case ErrorType::SystemThemeDetectionFailure: typeStr = "SYS_DETECT"; break;
        case ErrorType::DialogRegistrationFailure: typeStr = "DIALOG_REG"; break;
    }
    
    LOG_ERROR(LogCategories::UI, QString("[%1] %2").arg(typeStr).arg(message));
}

bool ThemeErrorHandler::retryThemeApplication(QWidget* component, int maxRetries)
{
    if (!component) {
        return false;
    }
    
    ThemeManager* themeManager = ThemeManager::instance();
    if (!themeManager) {
        return false;
    }
    
    for (int attempt = 1; attempt <= maxRetries; ++attempt) {
        LOG_INFO(LogCategories::UI, QString("Theme application retry attempt %1/%2 for %3")
                 .arg(attempt).arg(maxRetries).arg(component->metaObject()->className()));
        
        try {
            // Clear existing styles first
            component->setStyleSheet("");
            
            // Wait a bit before retry
            QTimer::singleShot(100 * attempt, [=]() {
                try {
                    themeManager->applyToWidget(component);
                    component->update();
                } catch (...) {
                    // Ignore errors in lambda
                }
            });
            
            // Apply theme
            themeManager->applyToWidget(component);
            component->update();
            
            LOG_INFO(LogCategories::UI, QString("Theme application retry successful on attempt %1").arg(attempt));
            return true;
            
        } catch (const std::exception& e) {
            LOG_WARNING(LogCategories::UI, QString("Theme application retry attempt %1 failed: %2")
                       .arg(attempt).arg(e.what()));
            
            if (attempt == maxRetries) {
                LOG_ERROR(LogCategories::UI, QString("All retry attempts failed for %1")
                         .arg(component->metaObject()->className()));
                
                // Add to failed components list
                QString componentName = component->metaObject()->className();
                if (!s_failedComponents.contains(componentName)) {
                    s_failedComponents.append(componentName);
                }
            }
        }
    }
    
    return false;
}

bool ThemeErrorHandler::recoverFromCorruptedTheme(const QString& themeName)
{
    LOG_INFO(LogCategories::UI, QString("Attempting to recover from corrupted theme: %1").arg(themeName));
    
    // Try to load a backup or default version
    ThemeManager* themeManager = ThemeManager::instance();
    if (!themeManager) {
        return false;
    }
    
    try {
        // If it's a custom theme, try to delete it and fall back
        if (!themeName.isEmpty() && themeName != "system" && themeName != "light" && themeName != "dark") {
            LOG_INFO(LogCategories::UI, QString("Deleting corrupted custom theme: %1").arg(themeName));
            themeManager->deleteCustomTheme(themeName);
        }
        
        // Fall back to system default
        fallbackToDefaultTheme();
        
        LOG_INFO(LogCategories::UI, QString("Successfully recovered from corrupted theme: %1").arg(themeName));
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR(LogCategories::UI, QString("Failed to recover from corrupted theme %1: %2")
                 .arg(themeName).arg(e.what()));
        return false;
    }
}

void ThemeErrorHandler::clearFailedComponents()
{
    LOG_INFO(LogCategories::UI, QString("Clearing %1 failed components").arg(s_failedComponents.size()));
    s_failedComponents.clear();
}

QStringList ThemeErrorHandler::getFailedComponents()
{
    return s_failedComponents;
}

QList<ThemeErrorHandler::ErrorContext> ThemeErrorHandler::getErrorHistory()
{
    return s_errorHistory;
}

int ThemeErrorHandler::getErrorCount(ErrorType type)
{
    return s_errorCounts.value(type, 0);
}

void ThemeErrorHandler::resetErrorCounters()
{
    LOG_INFO(LogCategories::UI, "Resetting theme error counters");
    s_errorCounts.clear();
    s_errorHistory.clear();
}

void ThemeErrorHandler::notifyUser(const QString& message, const QString& title)
{
    if (!s_userNotificationEnabled) {
        return;
    }
    
    LOG_INFO(LogCategories::UI, QString("Notifying user: %1").arg(title));
    
    // Use QTimer to ensure we're in the main thread
    QTimer::singleShot(0, [=]() {
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.setWindowTitle(title);
        msgBox.setText(message);
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
    });
}

bool ThemeErrorHandler::askUserForRecovery(const QString& message, const QString& title)
{
    if (!s_userNotificationEnabled) {
        return false;
    }
    
    LOG_INFO(LogCategories::UI, QString("Asking user for recovery: %1").arg(title));
    
    QMessageBox msgBox;
    msgBox.setIcon(QMessageBox::Question);
    msgBox.setWindowTitle(title);
    msgBox.setText(message);
    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::Yes);
    
    return msgBox.exec() == QMessageBox::Yes;
}

void ThemeErrorHandler::setMaxRetryAttempts(int maxRetries)
{
    s_maxRetryAttempts = qMax(1, maxRetries);
    LOG_INFO(LogCategories::UI, QString("Max retry attempts set to: %1").arg(s_maxRetryAttempts));
}

void ThemeErrorHandler::setAutoRecoveryEnabled(bool enabled)
{
    s_autoRecoveryEnabled = enabled;
    LOG_INFO(LogCategories::UI, QString("Auto recovery %1").arg(enabled ? "enabled" : "disabled"));
}

void ThemeErrorHandler::setUserNotificationEnabled(bool enabled)
{
    s_userNotificationEnabled = enabled;
    LOG_INFO(LogCategories::UI, QString("User notifications %1").arg(enabled ? "enabled" : "disabled"));
}

// Private methods

ThemeData ThemeErrorHandler::getDefaultTheme()
{
    ThemeData defaultTheme;
    defaultTheme.name = "System Default";
    defaultTheme.description = "Fallback system theme";
    defaultTheme.created = QDateTime::currentDateTime();
    defaultTheme.modified = QDateTime::currentDateTime();
    
    // Use safe default colors
    defaultTheme.colors.background = QColor(255, 255, 255);
    defaultTheme.colors.foreground = QColor(0, 0, 0);
    defaultTheme.colors.accent = QColor(0, 120, 215);
    defaultTheme.colors.border = QColor(200, 200, 200);
    defaultTheme.colors.hover = QColor(230, 230, 230);
    defaultTheme.colors.disabled = QColor(150, 150, 150);
    
    return defaultTheme;
}

bool ThemeErrorHandler::canRecover(ErrorType type)
{
    switch (type) {
        case ErrorType::ThemeLoadFailure:
        case ErrorType::StyleApplicationFailure:
        case ErrorType::CustomThemeCorruption:
        case ErrorType::SystemThemeDetectionFailure:
            return true;
        case ErrorType::ComponentRegistrationFailure:
        case ErrorType::DialogRegistrationFailure:
        case ErrorType::ValidationFailure:
        case ErrorType::PersistenceFailure:
            return s_autoRecoveryEnabled;
    }
    return false;
}

ThemeErrorHandler::RecoveryAction ThemeErrorHandler::determineRecoveryAction(ErrorType type)
{
    switch (type) {
        case ErrorType::ThemeLoadFailure:
        case ErrorType::CustomThemeCorruption:
            return RecoveryAction::FallbackToDefault;
        case ErrorType::StyleApplicationFailure:
            return RecoveryAction::RetryOperation;
        case ErrorType::ComponentRegistrationFailure:
        case ErrorType::DialogRegistrationFailure:
            return RecoveryAction::SkipComponent;
        case ErrorType::SystemThemeDetectionFailure:
            return RecoveryAction::LogAndContinue;
        case ErrorType::ValidationFailure:
            return RecoveryAction::LogAndContinue;
        case ErrorType::PersistenceFailure:
            return RecoveryAction::NotifyUser;
    }
    return RecoveryAction::LogAndContinue;
}

void ThemeErrorHandler::recordError(const ErrorContext& context)
{
    s_errorHistory.append(context);
    
    // Keep only last 100 errors to prevent memory issues
    if (s_errorHistory.size() > 100) {
        s_errorHistory.removeFirst();
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Recorded error: %1 for component %2")
             .arg(static_cast<int>(context.type)).arg(context.componentName));
}

bool ThemeErrorHandler::recoverStyleApplicationFailure(QWidget* component)
{
    return retryThemeApplication(component, s_maxRetryAttempts);
}

bool ThemeErrorHandler::recoverThemeLoadFailure(const QString& themeName)
{
    if (themeName.isEmpty()) {
        fallbackToDefaultTheme();
        return true;
    }
    
    return recoverFromCorruptedTheme(themeName);
}

bool ThemeErrorHandler::recoverComponentRegistrationFailure(QWidget* component)
{
    if (!component) {
        return false;
    }
    
    // Try to re-register the component
    ThemeManager* themeManager = ThemeManager::instance();
    if (themeManager) {
        try {
            themeManager->registerComponent(component, ThemeManager::ComponentType::Widget);
            return true;
        } catch (...) {
            return false;
        }
    }
    
    return false;
}
bool ThemeErrorHandler::showRecoveryDialog(ErrorType errorType, const QString& errorMessage)
{
    if (!s_userNotificationEnabled) {
        return false;
    }
    
    LOG_INFO(LogCategories::UI, "Showing theme recovery dialog for error type: " + QString::number(static_cast<int>(errorType)));
    
    // Create and show recovery dialog
    ThemeRecoveryDialog dialog(errorMessage, errorType);
    
    int result = dialog.exec();
    if (result != QDialog::Accepted) {
        LOG_INFO(LogCategories::UI, "User cancelled theme recovery dialog");
        return false;
    }
    
    // Apply user's recovery choice
    ThemeRecoveryDialog::RecoveryOption option = dialog.getSelectedRecoveryOption();
    QString selectedTheme = dialog.getSelectedTheme();
    bool rememberChoice = dialog.shouldRememberChoice();
    bool enableAutoRecovery = dialog.shouldEnableAutoRecovery();
    
    LOG_INFO(LogCategories::UI, QString("User selected recovery option: %1, theme: %2, remember: %3, auto-recovery: %4")
             .arg(static_cast<int>(option))
             .arg(selectedTheme)
             .arg(rememberChoice)
             .arg(enableAutoRecovery));
    
    // Update settings based on user preferences
    setAutoRecoveryEnabled(enableAutoRecovery);
    if (rememberChoice) {
        setUserNotificationEnabled(false); // Don't show dialog again for similar errors
    }
    
    // Apply the selected recovery action
    ThemeManager* themeManager = ThemeManager::instance();
    if (!themeManager) {
        LOG_ERROR(LogCategories::UI, "ThemeManager not available for recovery");
        return false;
    }
    
    try {
        switch (option) {
            case ThemeRecoveryDialog::RecoveryOption::ResetToDefault:
                LOG_INFO(LogCategories::UI, "Applying recovery: Reset to default theme");
                fallbackToDefaultTheme();
                return true;
                
            case ThemeRecoveryDialog::RecoveryOption::RetryCurrentTheme:
                LOG_INFO(LogCategories::UI, "Applying recovery: Retry current theme");
                themeManager->applyToApplication();
                return true;
                
            case ThemeRecoveryDialog::RecoveryOption::SelectDifferentTheme:
                LOG_INFO(LogCategories::UI, QString("Applying recovery: Switch to %1 theme").arg(selectedTheme));
                if (selectedTheme == "light") {
                    themeManager->setTheme(ThemeManager::Light);
                } else if (selectedTheme == "dark") {
                    themeManager->setTheme(ThemeManager::Dark);
                } else {
                    themeManager->setTheme(ThemeManager::SystemDefault);
                }
                return true;
                
            case ThemeRecoveryDialog::RecoveryOption::DisableThemeSystem:
                {
                    LOG_INFO(LogCategories::UI, "Applying recovery: Disable theme system");
                    // Clear all stylesheets and use basic Qt styling
                    qApp->setStyleSheet("");
                    QWidgetList allWidgets = QApplication::allWidgets();
                    for (QWidget* widget : allWidgets) {
                        if (widget) {
                            widget->setStyleSheet("");
                        }
                    }
                    return true;
                }
                
            case ThemeRecoveryDialog::RecoveryOption::ContactSupport:
                LOG_INFO(LogCategories::UI, "User selected contact support - no recovery action taken");
                return false; // No actual recovery performed
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR(LogCategories::UI, QString("Recovery action failed: %1").arg(e.what()));
        
        // Show error message to user
        QMessageBox::critical(nullptr, "Recovery Failed", 
            QString("The selected recovery action failed: %1\n\n"
                    "The application will attempt to use basic styling.").arg(e.what()));
        
        // Fall back to clearing all styles
        qApp->setStyleSheet("");
        return false;
    }
    
    return false;
}