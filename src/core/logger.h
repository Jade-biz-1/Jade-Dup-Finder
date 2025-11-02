#ifndef LOGGER_H
#define LOGGER_H

#include <QObject>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QMutex>
#include <QDir>


class Logger : public QObject
{
    Q_OBJECT

public:
    enum LogLevel {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
        Critical = 4
    };
    Q_ENUM(LogLevel)

    static Logger* instance();
    
    // Configuration
    void setLogLevel(LogLevel level);
    void setLogToFile(bool enabled);
    void setLogToConsole(bool enabled);
    void setLogDirectory(const QString& directory);
    void setMaxLogFiles(int maxFiles);
    void setMaxLogFileSize(qint64 maxSize);
    
    // Logging methods
    void log(LogLevel level, const QString& category, const QString& message);
    void debug(const QString& category, const QString& message);
    void info(const QString& category, const QString& message);
    void warning(const QString& category, const QString& message);
    void error(const QString& category, const QString& message);
    void critical(const QString& category, const QString& message);
    
    // Convenience macros will use these
    void logWithLocation(LogLevel level, const QString& category, const QString& message,
                        const QString& file, int line, const QString& function);
    
    // File management
    void rotateLogFiles();
    void clearOldLogs();
    
    // Statistics
    struct LogStats {
        int debugCount = 0;
        int infoCount = 0;
        int warningCount = 0;
        int errorCount = 0;
        int criticalCount = 0;
        QDateTime sessionStart;
    };
    
    LogStats getLogStats() const { return m_stats; }
    void resetStats();

signals:
    void logMessage(LogLevel level, const QString& category, const QString& message, const QDateTime& timestamp);
    void logFileRotated(const QString& newFileName);
    void logError(const QString& error);

private:
    explicit Logger(QObject* parent = nullptr);
    ~Logger();
    
    void writeToFile(const QString& formattedMessage);
    void writeToConsole(LogLevel level, const QString& formattedMessage);
    QString formatMessage(LogLevel level, const QString& category, const QString& message, const QDateTime& timestamp);
    QString levelToString(LogLevel level);
    void ensureLogDirectory();
    void openLogFile();
    void closeLogFile();
    
    static Logger* s_instance;
    
    // Configuration
    LogLevel m_logLevel;
    bool m_logToFile;
    bool m_logToConsole;
    QString m_logDirectory;
    int m_maxLogFiles;
    qint64 m_maxLogFileSize;
    
    // File handling
    QFile* m_logFile;
    QTextStream* m_logStream;
    QString m_currentLogFileName;
    
    // Thread safety
    mutable QMutex m_mutex;
    
    // Statistics
    LogStats m_stats;
};

// Convenience macros for logging with file/line information
#define LOG_DEBUG(category, message) \
    Logger::instance()->logWithLocation(Logger::Debug, category, message, __FILE__, __LINE__, Q_FUNC_INFO)

#define LOG_INFO(category, message) \
    Logger::instance()->logWithLocation(Logger::Info, category, message, __FILE__, __LINE__, Q_FUNC_INFO)

#define LOG_WARNING(category, message) \
    Logger::instance()->logWithLocation(Logger::Warning, category, message, __FILE__, __LINE__, Q_FUNC_INFO)

#define LOG_ERROR(category, message) \
    Logger::instance()->logWithLocation(Logger::Error, category, message, __FILE__, __LINE__, Q_FUNC_INFO)

#define LOG_CRITICAL(category, message) \
    Logger::instance()->logWithLocation(Logger::Critical, category, message, __FILE__, __LINE__, Q_FUNC_INFO)

// Category constants
namespace LogCategories {
    const QString SCAN = "SCAN";
    const QString HASH = "HASH";
    const QString DUPLICATE = "DUPLICATE";
    const QString FILE_OPS = "FILE_OPS";
    const QString SAFETY = "SAFETY";
    const QString UI = "UI";
    const QString EXPORT = "EXPORT";
    const QString PREVIEW = "PREVIEW";
    const QString CONFIG = "CONFIG";
    const QString PERFORMANCE = "PERFORMANCE";
    const QString SYSTEM = "SYSTEM";
}

#endif // LOGGER_H
