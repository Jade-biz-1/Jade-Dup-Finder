#ifndef THEME_ERROR_HANDLER_H
#define THEME_ERROR_HANDLER_H

#include <QObject>
#include <QString>
#include <QWidget>
#include <QMessageBox>
#include <QTimer>
#include "theme_manager.h"

class ThemeErrorHandler : public QObject
{
    Q_OBJECT

public:
    enum class ErrorType {
        ThemeLoadFailure,
        StyleApplicationFailure,
        ComponentRegistrationFailure,
        ValidationFailure,
        PersistenceFailure,
        CustomThemeCorruption,
        SystemThemeDetectionFailure,
        DialogRegistrationFailure
    };

    enum class RecoveryAction {
        FallbackToDefault,
        RetryOperation,
        SkipComponent,
        NotifyUser,
        LogAndContinue
    };

    struct ErrorContext {
        ErrorType type;
        QString componentName;
        QString errorMessage;
        QString stackTrace;
        QWidget* failedWidget;
        QString attemptedTheme;
        int retryCount;
        QDateTime timestamp;
    };

    static ThemeErrorHandler* instance();
    
    // Error handling methods
    static void handleError(ErrorType type, const QString& details, QWidget* component = nullptr);
    static bool attemptRecovery(ErrorType type, QWidget* component = nullptr, const QString& context = QString());
    static void fallbackToDefaultTheme();
    static void logError(const QString& message, ErrorType type = ErrorType::ValidationFailure);
    
    // Recovery mechanisms
    static bool retryThemeApplication(QWidget* component, int maxRetries = 3);
    static bool recoverFromCorruptedTheme(const QString& themeName);
    static void clearFailedComponents();
    
    // Error tracking
    static QStringList getFailedComponents();
    static QList<ErrorContext> getErrorHistory();
    static int getErrorCount(ErrorType type);
    static void resetErrorCounters();
    
    // User notification
    static void notifyUser(const QString& message, const QString& title = "Theme System Error");
    static bool askUserForRecovery(const QString& message, const QString& title = "Theme Recovery");
    static bool showRecoveryDialog(ErrorType errorType, const QString& errorMessage);
    
    // Configuration
    static void setMaxRetryAttempts(int maxRetries);
    static void setAutoRecoveryEnabled(bool enabled);
    static void setUserNotificationEnabled(bool enabled);
    
signals:
    void errorOccurred(ErrorType type, const QString& message);
    void recoveryAttempted(ErrorType type, bool success);
    void fallbackActivated(const QString& reason);

private:
    explicit ThemeErrorHandler(QObject* parent = nullptr);
    ~ThemeErrorHandler() = default;
    
    // Internal methods
    static ThemeData getDefaultTheme();
    static bool canRecover(ErrorType type);
    static RecoveryAction determineRecoveryAction(ErrorType type);
    static void recordError(const ErrorContext& context);
    
    // Recovery strategies
    static bool recoverStyleApplicationFailure(QWidget* component);
    static bool recoverThemeLoadFailure(const QString& themeName);
    static bool recoverComponentRegistrationFailure(QWidget* component);
    
    static ThemeErrorHandler* s_instance;
    static QStringList s_failedComponents;
    static QList<ErrorContext> s_errorHistory;
    static int s_maxRetryAttempts;
    static bool s_autoRecoveryEnabled;
    static bool s_userNotificationEnabled;
    static QMap<ErrorType, int> s_errorCounts;
};

// Convenience macros for error handling
#define THEME_ERROR(type, message) \
    ThemeErrorHandler::handleError(type, message, nullptr)

#define THEME_ERROR_WITH_WIDGET(type, message, widget) \
    ThemeErrorHandler::handleError(type, message, widget)

#define THEME_RECOVERY_ATTEMPT(type, widget) \
    ThemeErrorHandler::attemptRecovery(type, widget)

#define THEME_LOG_ERROR(message) \
    ThemeErrorHandler::logError(message)

#endif // THEME_ERROR_HANDLER_H