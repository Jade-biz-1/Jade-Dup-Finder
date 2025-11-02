#include "scan_history_manager.h"
#include "core/logger.h"
#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QUuid>
#include <QDirIterator>
#include <algorithm>

// Initialize static instance
ScanHistoryManager* ScanHistoryManager::s_instance = nullptr;

ScanHistoryManager::ScanHistoryManager()
{
    LOG_DEBUG(LogCategories::SYSTEM, "ScanHistoryManager created");
    ensureHistoryDirectory();
}

ScanHistoryManager::~ScanHistoryManager()
{
    LOG_DEBUG(LogCategories::SYSTEM, "ScanHistoryManager destroyed");
}

ScanHistoryManager* ScanHistoryManager::instance()
{
    if (!s_instance) {
        s_instance = new ScanHistoryManager();
    }
    return s_instance;
}

QString ScanHistoryManager::getHistoryDirectory() const
{
    QString appDataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return appDataPath + "/history";
}

QString ScanHistoryManager::getHistoryFilePath(const QString& scanId) const
{
    return getHistoryDirectory() + "/scan_" + scanId + ".json";
}

void ScanHistoryManager::ensureHistoryDirectory()
{
    QString historyDir = getHistoryDirectory();
    QDir dir;
    
    if (!dir.exists(historyDir)) {
        if (dir.mkpath(historyDir)) {
            LOG_INFO(LogCategories::SYSTEM, "Created history directory: " + historyDir);
        } else {
            LOG_ERROR(LogCategories::SYSTEM, "Failed to create history directory: " + historyDir);
        }
    }
}

void ScanHistoryManager::saveScan(const ScanRecord& record)
{
    if (!record.isValid()) {
        LOG_ERROR(LogCategories::SYSTEM, "Cannot save invalid scan record");
        return;
    }
    
    ensureHistoryDirectory();
    
    QString filePath = getHistoryFilePath(record.scanId);
    LOG_INFO(LogCategories::SYSTEM, QString("Saving scan to: %1").arg(filePath));
    
    // Serialize the record to JSON
    QJsonObject jsonObj = serializeScanRecord(record);
    QJsonDocument doc(jsonObj);
    
    // Write to file
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        LOG_ERROR(LogCategories::SYSTEM, QString("Failed to open file for writing: %1").arg(filePath));
        return;
    }
    
    qint64 bytesWritten = file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
    
    if (bytesWritten > 0) {
        LOG_INFO(LogCategories::SYSTEM, QString("Scan saved successfully: %1 bytes").arg(bytesWritten));
        emit scanSaved(record.scanId);
    } else {
        LOG_ERROR(LogCategories::SYSTEM, "Failed to write scan data to file");
    }
}

ScanHistoryManager::ScanRecord ScanHistoryManager::loadScan(const QString& scanId)
{
    QString filePath = getHistoryFilePath(scanId);
    LOG_INFO(LogCategories::SYSTEM, QString("Loading scan from: %1").arg(filePath));
    
    QFile file(filePath);
    if (!file.exists()) {
        LOG_WARNING(LogCategories::SYSTEM, QString("Scan file not found: %1").arg(filePath));
        return ScanRecord();
    }
    
    if (!file.open(QIODevice::ReadOnly)) {
        LOG_ERROR(LogCategories::SYSTEM, QString("Failed to open file for reading: %1").arg(filePath));
        return ScanRecord();
    }
    
    QByteArray data = file.readAll();
    file.close();
    
    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) {
        LOG_ERROR(LogCategories::SYSTEM, "Failed to parse JSON from scan file");
        return ScanRecord();
    }
    
    ScanRecord record = deserializeScanRecord(doc.object());
    
    if (record.isValid()) {
        LOG_INFO(LogCategories::SYSTEM, QString("Scan loaded successfully: %1 groups").arg(record.duplicateGroups));
    } else {
        LOG_ERROR(LogCategories::SYSTEM, "Loaded scan record is invalid");
    }
    
    return record;
}

QList<ScanHistoryManager::ScanRecord> ScanHistoryManager::getAllScans()
{
    QList<ScanRecord> scans;
    
    QString historyDir = getHistoryDirectory();
    QDir dir(historyDir);
    
    if (!dir.exists()) {
        LOG_INFO(LogCategories::SYSTEM, "History directory does not exist yet");
        return scans;
    }
    
    // Get all JSON files in the history directory
    QStringList filters;
    filters << "scan_*.json";
    QFileInfoList files = dir.entryInfoList(filters, QDir::Files, QDir::Time | QDir::Reversed);
    
    LOG_INFO(LogCategories::SYSTEM, QString("Found %1 scan files").arg(files.size()));
    
    // Load each scan file
    for (const QFileInfo& fileInfo : files) {
        // Extract scan ID from filename (scan_<uuid>.json)
        QString fileName = fileInfo.fileName();
        QString scanId = fileName.mid(5, fileName.length() - 10); // Remove "scan_" and ".json"
        
        ScanRecord record = loadScan(scanId);
        if (record.isValid()) {
            scans.append(record);
        } else {
            LOG_WARNING(LogCategories::SYSTEM, QString("Skipping invalid scan file: %1").arg(fileName));
        }
    }
    
    // Sort by timestamp descending (newest first)
    std::sort(scans.begin(), scans.end(), [](const ScanRecord& a, const ScanRecord& b) {
        return a.timestamp > b.timestamp;
    });
    
    LOG_INFO(LogCategories::SYSTEM, QString("Loaded %1 valid scans").arg(scans.size()));
    
    return scans;
}

void ScanHistoryManager::deleteScan(const QString& scanId)
{
    QString filePath = getHistoryFilePath(scanId);
    LOG_INFO(LogCategories::SYSTEM, QString("Deleting scan: %1").arg(filePath));
    
    QFile file(filePath);
    if (file.exists()) {
        if (file.remove()) {
            LOG_INFO(LogCategories::SYSTEM, "Scan deleted successfully");
            emit scanDeleted(scanId);
        } else {
            LOG_ERROR(LogCategories::SYSTEM, QString("Failed to delete scan file: %1").arg(file.errorString()));
        }
    } else {
        LOG_WARNING(LogCategories::SYSTEM, "Scan file does not exist");
    }
}

void ScanHistoryManager::clearOldScans(int daysToKeep)
{
    LOG_INFO(LogCategories::SYSTEM, QString("Clearing scans older than %1 days").arg(daysToKeep));
    
    QDateTime cutoffDate = QDateTime::currentDateTime().addDays(-daysToKeep);
    QList<ScanRecord> allScans = getAllScans();
    
    int deletedCount = 0;
    for (const ScanRecord& record : allScans) {
        if (record.timestamp < cutoffDate) {
            deleteScan(record.scanId);
            deletedCount++;
        }
    }
    
    LOG_INFO(LogCategories::SYSTEM, QString("Deleted %1 old scans").arg(deletedCount));
    
    if (deletedCount > 0) {
        emit historyCleared();
    }
}

QJsonObject ScanHistoryManager::serializeScanRecord(const ScanRecord& record) const
{
    QJsonObject json;
    
    // Basic metadata
    json["scanId"] = record.scanId;
    json["timestamp"] = record.timestamp.toString(Qt::ISODate);
    json["filesScanned"] = record.filesScanned;
    json["duplicateGroups"] = record.duplicateGroups;
    json["potentialSavings"] = QString::number(record.potentialSavings);
    
    // Target paths
    QJsonArray pathsArray;
    for (const QString& path : record.targetPaths) {
        pathsArray.append(path);
    }
    json["targetPaths"] = pathsArray;
    
    // Duplicate groups
    QJsonArray groupsArray;
    for (const auto& group : record.groups) {
        QJsonObject groupObj;
        groupObj["groupId"] = group.groupId;
        groupObj["fileSize"] = QString::number(group.fileSize);
        groupObj["hash"] = group.hash;
        groupObj["totalSize"] = QString::number(group.totalSize);
        groupObj["wastedSpace"] = QString::number(group.wastedSpace);
        groupObj["fileCount"] = group.fileCount;
        groupObj["recommendedAction"] = group.recommendedAction;
        groupObj["detected"] = group.detected.toString(Qt::ISODate);
        
        // Files in group
        QJsonArray filesArray;
        for (const auto& file : group.files) {
            QJsonObject fileObj;
            fileObj["filePath"] = file.filePath;
            fileObj["fileSize"] = QString::number(file.fileSize);
            fileObj["hash"] = file.hash;
            fileObj["lastModified"] = file.lastModified.toString(Qt::ISODate);
            fileObj["fileName"] = file.fileName;
            fileObj["directory"] = file.directory;
            filesArray.append(fileObj);
        }
        groupObj["files"] = filesArray;
        
        groupsArray.append(groupObj);
    }
    json["groups"] = groupsArray;
    
    // Version for future compatibility
    json["version"] = "1.0";
    
    return json;
}

ScanHistoryManager::ScanRecord ScanHistoryManager::deserializeScanRecord(const QJsonObject& json) const
{
    ScanRecord record;
    
    // Basic metadata
    record.scanId = json["scanId"].toString();
    record.timestamp = QDateTime::fromString(json["timestamp"].toString(), Qt::ISODate);
    record.filesScanned = json["filesScanned"].toInt();
    record.duplicateGroups = json["duplicateGroups"].toInt();
    record.potentialSavings = json["potentialSavings"].toString().toLongLong();
    
    // Target paths
    QJsonArray pathsArray = json["targetPaths"].toArray();
    for (const QJsonValue& pathValue : pathsArray) {
        record.targetPaths.append(pathValue.toString());
    }
    
    // Duplicate groups
    QJsonArray groupsArray = json["groups"].toArray();
    for (const QJsonValue& groupValue : groupsArray) {
        QJsonObject groupObj = groupValue.toObject();
        
        DuplicateDetector::DuplicateGroup group;
        group.groupId = groupObj["groupId"].toString();
        group.fileSize = groupObj["fileSize"].toString().toLongLong();
        group.hash = groupObj["hash"].toString();
        group.totalSize = groupObj["totalSize"].toString().toLongLong();
        group.wastedSpace = groupObj["wastedSpace"].toString().toLongLong();
        group.fileCount = groupObj["fileCount"].toInt();
        group.recommendedAction = groupObj["recommendedAction"].toString();
        group.detected = QDateTime::fromString(groupObj["detected"].toString(), Qt::ISODate);
        
        // Files in group
        QJsonArray filesArray = groupObj["files"].toArray();
        for (const QJsonValue& fileValue : filesArray) {
            QJsonObject fileObj = fileValue.toObject();
            
            DuplicateDetector::FileInfo file;
            file.filePath = fileObj["filePath"].toString();
            file.fileSize = fileObj["fileSize"].toString().toLongLong();
            file.hash = fileObj["hash"].toString();
            file.lastModified = QDateTime::fromString(fileObj["lastModified"].toString(), Qt::ISODate);
            file.fileName = fileObj["fileName"].toString();
            file.directory = fileObj["directory"].toString();
            
            group.files.append(file);
        }
        
        record.groups.append(group);
    }
    
    return record;
}
