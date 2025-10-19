#include "test_data_generator.h"
#include <QStandardPaths>
#include <QProcess>
#include <QTimer>
#include <QThread>
#include <QDebug>
#include <QUuid>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>

TestEnvironmentIsolator::TestEnvironmentIsolator() {
    // Set up base isolation path
    QString tempLocation = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    m_baseIsolationPath = QDir(tempLocation).absoluteFilePath("dupfinder_test_isolation");
    QDir().mkpath(m_baseIsolationPath);
}

TestEnvironmentIsolator::~TestEnvironmentIsolator() {
    cleanupAllEnvironments();
}

QString TestEnvironmentIsolator::createIsolatedEnvironment(const QString& testName) {
    QString environmentId = generateEnvironmentId();
    QString environmentPath = QDir(m_baseIsolationPath).absoluteFilePath(
        QString("%1_%2").arg(testName).arg(environmentId)
    );
    
    // Create the isolated environment directory
    if (QDir().mkpath(environmentPath)) {
        m_environments[environmentId] = environmentPath;
        setupEnvironmentIsolation(environmentPath);
        
        qDebug() << "Created isolated environment:" << environmentId << "at" << environmentPath;
        return environmentId;
    }
    
    qWarning() << "Failed to create isolated environment for test:" << testName;
    return QString();
}

void TestEnvironmentIsolator::destroyIsolatedEnvironment(const QString& environmentId) {
    if (!m_environments.contains(environmentId)) {
        qWarning() << "Environment not found:" << environmentId;
        return;
    }
    
    QString environmentPath = m_environments[environmentId];
    cleanupEnvironment(environmentPath);
    
    m_environments.remove(environmentId);
    m_resourceMonitors.remove(environmentId);
    
    qDebug() << "Destroyed isolated environment:" << environmentId;
}

QString TestEnvironmentIsolator::getEnvironmentPath(const QString& environmentId) {
    return m_environments.value(environmentId);
}

bool TestEnvironmentIsolator::runInIsolatedProcess(const QString& command, const QStringList& arguments, const QString& workingDirectory) {
    QProcess process;
    
    if (!workingDirectory.isEmpty()) {
        process.setWorkingDirectory(workingDirectory);
    }
    
    // Set up isolated environment variables
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("DUPFINDER_TEST_MODE", "1");
    env.insert("DUPFINDER_ISOLATED", "1");
    process.setProcessEnvironment(env);
    
    process.start(command, arguments);
    
    if (!process.waitForStarted(5000)) {
        qWarning() << "Failed to start isolated process:" << command;
        return false;
    }
    
    if (!process.waitForFinished(30000)) {
        qWarning() << "Isolated process timed out:" << command;
        process.kill();
        return false;
    }
    
    int exitCode = process.exitCode();
    if (exitCode != 0) {
        qWarning() << "Isolated process failed with exit code:" << exitCode;
        qWarning() << "Error output:" << process.readAllStandardError();
        return false;
    }
    
    return true;
}

void TestEnvironmentIsolator::startResourceMonitoring(const QString& environmentId) {
    if (!m_environments.contains(environmentId)) {
        qWarning() << "Cannot monitor unknown environment:" << environmentId;
        return;
    }
    
    QMap<QString, QVariant> monitor;
    monitor["startTime"] = QDateTime::currentDateTime();
    monitor["initialMemory"] = getCurrentMemoryUsage();
    monitor["initialDiskSpace"] = getAvailableDiskSpace(m_environments[environmentId]);
    
    m_resourceMonitors[environmentId] = monitor;
    
    qDebug() << "Started resource monitoring for environment:" << environmentId;
}

void TestEnvironmentIsolator::stopResourceMonitoring(const QString& environmentId) {
    if (!m_resourceMonitors.contains(environmentId)) {
        qWarning() << "No resource monitoring active for environment:" << environmentId;
        return;
    }
    
    QMap<QString, QVariant> monitor = m_resourceMonitors[environmentId].toMap();
    monitor["endTime"] = QDateTime::currentDateTime();
    monitor["finalMemory"] = getCurrentMemoryUsage();
    monitor["finalDiskSpace"] = getAvailableDiskSpace(m_environments[environmentId]);
    
    // Calculate usage
    qint64 memoryDelta = monitor["finalMemory"].toLongLong() - monitor["initialMemory"].toLongLong();
    qint64 diskDelta = monitor["initialDiskSpace"].toLongLong() - monitor["finalDiskSpace"].toLongLong();
    
    monitor["memoryUsed"] = memoryDelta;
    monitor["diskUsed"] = diskDelta;
    
    QDateTime startTime = monitor["startTime"].toDateTime();
    QDateTime endTime = monitor["endTime"].toDateTime();
    monitor["executionTimeMs"] = startTime.msecsTo(endTime);
    
    m_resourceMonitors[environmentId] = monitor;
    
    qDebug() << "Stopped resource monitoring for environment:" << environmentId;
    qDebug() << "Memory used:" << memoryDelta << "bytes";
    qDebug() << "Disk used:" << diskDelta << "bytes";
    qDebug() << "Execution time:" << startTime.msecsTo(endTime) << "ms";
}

QMap<QString, QVariant> TestEnvironmentIsolator::getResourceUsage(const QString& environmentId) {
    return m_resourceMonitors.value(environmentId).toMap();
}

void TestEnvironmentIsolator::scheduleCleanup(const QString& environmentId, int delaySeconds) {
    if (delaySeconds <= 0) {
        destroyIsolatedEnvironment(environmentId);
        return;
    }
    
    // Schedule delayed cleanup
    QTimer::singleShot(delaySeconds * 1000, [this, environmentId]() {
        destroyIsolatedEnvironment(environmentId);
    });
}

void TestEnvironmentIsolator::cleanupAllEnvironments() {
    QStringList environmentIds = m_environments.keys();
    for (const QString& environmentId : environmentIds) {
        destroyIsolatedEnvironment(environmentId);
    }
}

QString TestEnvironmentIsolator::generateEnvironmentId() {
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

void TestEnvironmentIsolator::setupEnvironmentIsolation(const QString& environmentPath) {
    // Create standard subdirectories for isolated environment
    QStringList subdirs = {"temp", "config", "data", "logs", "cache"};
    
    for (const QString& subdir : subdirs) {
        QString subdirPath = QDir(environmentPath).absoluteFilePath(subdir);
        QDir().mkpath(subdirPath);
    }
    
    // Create environment configuration file
    QString configPath = QDir(environmentPath).absoluteFilePath("environment.json");
    QJsonObject config;
    config["created"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    config["type"] = "isolated_test_environment";
    config["version"] = "1.0";
    
    QJsonDocument doc(config);
    QFile configFile(configPath);
    if (configFile.open(QIODevice::WriteOnly)) {
        configFile.write(doc.toJson());
        configFile.close();
    }
}

void TestEnvironmentIsolator::cleanupEnvironment(const QString& environmentPath) {
    QDir dir(environmentPath);
    if (dir.exists()) {
        if (!dir.removeRecursively()) {
            qWarning() << "Failed to completely remove environment:" << environmentPath;
        }
    }
}

qint64 TestEnvironmentIsolator::getCurrentMemoryUsage() {
    // Platform-specific memory usage implementation
    // This is a simplified version - real implementation would use platform APIs
#ifdef Q_OS_WIN
    // Windows memory usage
    return 0; // Placeholder
#elif defined(Q_OS_LINUX)
    // Linux memory usage from /proc/self/status
    QFile file("/proc/self/status");
    if (file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        while (stream.readLineInto(&line)) {
            if (line.startsWith("VmRSS:")) {
                QStringList parts = line.split(QRegularExpression("\\s+"));
                if (parts.size() >= 2) {
                    return parts[1].toLongLong() * 1024; // Convert KB to bytes
                }
            }
        }
    }
    return 0;
#elif defined(Q_OS_MAC)
    // macOS memory usage
    return 0; // Placeholder
#else
    return 0; // Unknown platform
#endif
}

qint64 TestEnvironmentIsolator::getAvailableDiskSpace(const QString& path) {
    QFileInfo info(path);
    QDir dir = info.isDir() ? QDir(path) : info.dir();
    
    // Get available disk space
    return dir.exists() ? 0 : 0; // Simplified - would use platform APIs
}