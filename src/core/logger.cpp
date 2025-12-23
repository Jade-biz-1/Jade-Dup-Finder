#include "logger.h"
#include <QStandardPaths>
#include <QCoreApplication>
#include <QThread>
#include <QStringConverter>
#include <iostream>

Logger* Logger::s_instance = nullptr;

Logger* Logger::instance()
{
    if (!s_instance) {
        s_instance = new Logger();
    }
    return s_instance;
}

Logger::Logger(QObject* parent)
    : QObject(parent)
    , m_logLevel(Info)
    , m_logToFile(true)
    , m_logToConsole(true)
    , m_maxLogFiles(10)
    , m_maxLogFileSize(10 * 1024 * 1024) // 10MB
    , m_logFile(nullptr)
    , m_logStream(nullptr)
{
    // Set default log directory
    QString appDataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    m_logDirectory = QDir(appDataDir).filePath("logs");
    
    // Initialize stats
    m_stats.sessionStart = QDateTime::currentDateTime();
    
    // Ensure log directory exists and open log file
    ensureLogDirectory();
    openLogFile();
    
    // Log session start
    info(LogCategories::SYSTEM, QString("Logging system initialized - Session started at %1")
         .arg(m_stats.sessionStart.toString("yyyy-MM-dd hh:mm:ss")));
}

Logger::~Logger()
{
    info(LogCategories::SYSTEM, "Logging system shutting down");
    closeLogFile();
}

void Logger::setLogLevel(LogLevel level)
{
    QMutexLocker locker(&m_mutex);
    LogLevel oldLevel = m_logLevel;
    m_logLevel = level;
    locker.unlock();
    
    // Log after releasing the lock to avoid potential deadlock
    if (oldLevel != level) {
        info(LogCategories::CONFIG, QString("Log level set to %1").arg(levelToString(level)));
    }
}

void Logger::setLogToFile(bool enabled)
{
    m_mutex.lock();
    m_logToFile = enabled;
    bool needOpen = enabled && !m_logFile;
    bool needClose = !enabled && m_logFile;
    m_mutex.unlock();
    
    if (needOpen) {
        openLogFile();
    } else if (needClose) {
        closeLogFile();
    }
    
    info(LogCategories::CONFIG, QString("File logging %1").arg(enabled ? "enabled" : "disabled"));
}

void Logger::setLogToConsole(bool enabled)
{
    m_mutex.lock();
    m_logToConsole = enabled;
    m_mutex.unlock();
    
    info(LogCategories::CONFIG, QString("Console logging %1").arg(enabled ? "enabled" : "disabled"));
}

void Logger::setLogDirectory(const QString& directory)
{
    m_mutex.lock();
    bool changed = (m_logDirectory != directory);
    if (changed) {
        closeLogFile();
        m_logDirectory = directory;
        ensureLogDirectory();
        openLogFile();
    }
    m_mutex.unlock();
    
    if (changed) {
        info(LogCategories::CONFIG, QString("Log directory changed to: %1").arg(directory));
    }
}

void Logger::setMaxLogFiles(int maxFiles)
{
    m_mutex.lock();
    m_maxLogFiles = maxFiles;
    m_mutex.unlock();
    
    info(LogCategories::CONFIG, QString("Max log files set to %1").arg(maxFiles));
}

void Logger::setMaxLogFileSize(qint64 maxSize)
{
    m_mutex.lock();
    m_maxLogFileSize = maxSize;
    m_mutex.unlock();
    
    info(LogCategories::CONFIG, QString("Max log file size set to %1 MB").arg(maxSize / (1024 * 1024)));
}

void Logger::log(LogLevel level, const QString& category, const QString& message)
{
    QMutexLocker locker(&m_mutex);
    
    // Check log level
    if (level < m_logLevel) {
        return;
    }
    
    // Update statistics
    switch (level) {
    case Debug: m_stats.debugCount++; break;
    case Info: m_stats.infoCount++; break;
    case Warning: m_stats.warningCount++; break;
    case Error: m_stats.errorCount++; break;
    case Critical: m_stats.criticalCount++; break;
    }
    
    QDateTime timestamp = QDateTime::currentDateTime();
    QString formattedMessage = formatMessage(level, category, message, timestamp);
    
    // Write to file
    if (m_logToFile) {
        writeToFile(formattedMessage);
    }
    
    // Write to console
    if (m_logToConsole) {
        writeToConsole(level, formattedMessage);
    }
    
    // Emit signal
    emit logMessage(level, category, message, timestamp);
    
    // Check if log rotation is needed
    if (m_logFile && m_logFile->size() > m_maxLogFileSize) {
        rotateLogFiles();
    }
}

void Logger::debug(const QString& category, const QString& message)
{
    log(Debug, category, message);
}

void Logger::info(const QString& category, const QString& message)
{
    log(Info, category, message);
}

void Logger::warning(const QString& category, const QString& message)
{
    log(Warning, category, message);
}

void Logger::error(const QString& category, const QString& message)
{
    log(Error, category, message);
}

void Logger::critical(const QString& category, const QString& message)
{
    log(Critical, category, message);
}

void Logger::logWithLocation(LogLevel level, const QString& category, const QString& message,
                            const QString& file, int line, const QString& function)
{
    QString fileBaseName = QFileInfo(file).fileName();
    QString enhancedMessage = QString("%1 [%2:%3 in %4]")
                             .arg(message)
                             .arg(fileBaseName)
                             .arg(line)
                             .arg(function);
    log(level, category, enhancedMessage);
}

void Logger::rotateLogFiles()
{
    closeLogFile();
    
    // Rename current log file with timestamp
    if (QFile::exists(m_currentLogFileName)) {
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QString rotatedName = QString("%1/cloneclean_%2.log")
                             .arg(m_logDirectory)
                             .arg(timestamp);
        
        if (QFile::rename(m_currentLogFileName, rotatedName)) {
            emit logFileRotated(rotatedName);
        }
    }
    
    // Clean up old log files
    clearOldLogs();
    
    // Open new log file
    openLogFile();
    
    info(LogCategories::SYSTEM, "Log file rotated");
}

void Logger::clearOldLogs()
{
    QDir logDir(m_logDirectory);
    QStringList filters;
    filters << "cloneclean_*.log";
    
    QFileInfoList logFiles = logDir.entryInfoList(filters, QDir::Files, QDir::Time | QDir::Reversed);
    
    // Remove excess log files
    while (logFiles.size() > m_maxLogFiles) {
        QFileInfo oldestFile = logFiles.takeLast();
        if (QFile::remove(oldestFile.absoluteFilePath())) {
            // Note: Can't use Logger here as we're inside Logger implementation
            // Using direct console output for log file management
            std::cout << "Logger: Removed old log file: " << oldestFile.fileName().toStdString() << std::endl;
        }
    }
}

void Logger::resetStats()
{
    QMutexLocker locker(&m_mutex);
    m_stats = LogStats();
    m_stats.sessionStart = QDateTime::currentDateTime();
}

void Logger::writeToFile(const QString& formattedMessage)
{
    if (m_logStream) {
        *m_logStream << formattedMessage << Qt::endl;
        m_logStream->flush();
    }
}

void Logger::writeToConsole(LogLevel level, const QString& formattedMessage)
{
    // Use different output streams based on log level
    if (level >= Error) {
        std::cerr << formattedMessage.toStdString() << std::endl;
    } else {
        std::cout << formattedMessage.toStdString() << std::endl;
    }
}

QString Logger::formatMessage(LogLevel level, const QString& category, const QString& message, const QDateTime& timestamp)
{
    QString threadId = QString("0x%1").arg(reinterpret_cast<quintptr>(QThread::currentThreadId()), 0, 16);
    
    return QString("[%1] [%2] [%3] [Thread:%4] %5")
           .arg(timestamp.toString("yyyy-MM-dd hh:mm:ss.zzz"))
           .arg(levelToString(level))
           .arg(category)
           .arg(threadId)
           .arg(message);
}

QString Logger::levelToString(LogLevel level)
{
    switch (level) {
    case Debug: return "DEBUG";
    case Info: return "INFO ";
    case Warning: return "WARN ";
    case Error: return "ERROR";
    case Critical: return "CRIT ";
    default: return "UNKN ";
    }
}

void Logger::ensureLogDirectory()
{
    QDir dir;
    if (!dir.exists(m_logDirectory)) {
        if (!dir.mkpath(m_logDirectory)) {
            emit logError(QString("Failed to create log directory: %1").arg(m_logDirectory));
        }
    }
}

void Logger::openLogFile()
{
    if (m_logFile) {
        closeLogFile();
    }
    
    m_currentLogFileName = QDir(m_logDirectory).filePath("cloneclean.log");
    m_logFile = new QFile(m_currentLogFileName);
    
    if (m_logFile->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        m_logStream = new QTextStream(m_logFile);
        m_logStream->setEncoding(QStringConverter::Utf8);
    } else {
        emit logError(QString("Failed to open log file: %1").arg(m_currentLogFileName));
        delete m_logFile;
        m_logFile = nullptr;
    }
}

void Logger::closeLogFile()
{
    if (m_logStream) {
        delete m_logStream;
        m_logStream = nullptr;
    }
    
    if (m_logFile) {
        m_logFile->close();
        delete m_logFile;
        m_logFile = nullptr;
    }
}
