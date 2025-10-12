#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <QtCore/QObject>
#include <QtCore/QSettings>
#include <QtCore/QDebug>

/**
 * @brief Application-wide configuration and settings
 */
class AppConfig : public QObject
{
    Q_OBJECT
    
public:
    static AppConfig& instance() {
        static AppConfig instance;
        return instance;
    }
    
    // Debug logging configuration
    bool isVerboseLoggingEnabled() const { return m_verboseLogging; }
    void setVerboseLogging(bool enabled) { 
        m_verboseLogging = enabled;
        saveSettings();
        emit verboseLoggingChanged(enabled);
    }
    
    bool isFileProgressLoggingEnabled() const { return m_fileProgressLogging; }
    void setFileProgressLogging(bool enabled) {
        m_fileProgressLogging = enabled;
        saveSettings();
        emit fileProgressLoggingChanged(enabled);
    }
    
    // Convenience logging macros
    void logDebug(const QString& message) const {
        if (m_verboseLogging) {
            qDebug() << "[DEBUG]" << message;
        }
    }
    
    void logInfo(const QString& message) const {
        qInfo() << "[INFO]" << message;
    }
    
    void logWarning(const QString& message) const {
        qWarning() << "[WARNING]" << message;
    }
    
    void logError(const QString& message) const {
        qCritical() << "[ERROR]" << message;
    }
    
    void logFileProgress(const QString& operation, const QString& filePath) const {
        if (m_fileProgressLogging) {
            qDebug() << "[FILE]" << operation << ":" << filePath;
        }
    }
    
signals:
    void verboseLoggingChanged(bool enabled);
    void fileProgressLoggingChanged(bool enabled);
    
private:
    AppConfig() {
        loadSettings();
    }
    
    void loadSettings() {
        QSettings settings;
        m_verboseLogging = settings.value("Debug/VerboseLogging", true).toBool();
        m_fileProgressLogging = settings.value("Debug/FileProgressLogging", true).toBool();
    }
    
    void saveSettings() {
        QSettings settings;
        settings.setValue("Debug/VerboseLogging", m_verboseLogging);
        settings.setValue("Debug/FileProgressLogging", m_fileProgressLogging);
    }
    
    bool m_verboseLogging = true;
    bool m_fileProgressLogging = true;
    
    // Prevent copying
    AppConfig(const AppConfig&) = delete;
    AppConfig& operator=(const AppConfig&) = delete;
};

// Convenience macros for logging
#define LOG_DEBUG(msg) AppConfig::instance().logDebug(msg)
#define LOG_INFO(msg) AppConfig::instance().logInfo(msg)
#define LOG_WARNING(msg) AppConfig::instance().logWarning(msg)
#define LOG_ERROR(msg) AppConfig::instance().logError(msg)
#define LOG_FILE(op, path) AppConfig::instance().logFileProgress(op, path)

#endif // APP_CONFIG_H
