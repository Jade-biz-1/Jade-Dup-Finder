#ifndef SCAN_HISTORY_MANAGER_H
#define SCAN_HISTORY_MANAGER_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QList>
#include <QJsonObject>
#include "duplicate_detector.h"

class ScanHistoryManager : public QObject
{
    Q_OBJECT
    
public:
    struct ScanRecord {
        QString scanId;              // UUID
        QDateTime timestamp;         // When scan was performed
        QStringList targetPaths;     // Scanned locations
        int filesScanned;            // Total files scanned
        int duplicateGroups;         // Number of duplicate groups found
        qint64 potentialSavings;     // Total size of duplicates
        QList<DuplicateDetector::DuplicateGroup> groups;  // Full results
        
        bool isValid() const { return !scanId.isEmpty(); }
    };
    
    // Singleton access
    static ScanHistoryManager* instance();
    
    // Core operations
    void saveScan(const ScanRecord& record);
    ScanRecord loadScan(const QString& scanId);
    QList<ScanRecord> getAllScans();
    void deleteScan(const QString& scanId);
    void clearOldScans(int daysToKeep = 30);
    
signals:
    void scanSaved(const QString& scanId);
    void scanDeleted(const QString& scanId);
    void historyCleared();
    
private:
    ScanHistoryManager();
    ~ScanHistoryManager();
    
    // Prevent copying
    ScanHistoryManager(const ScanHistoryManager&) = delete;
    ScanHistoryManager& operator=(const ScanHistoryManager&) = delete;
    
    // File system operations
    QString getHistoryFilePath(const QString& scanId) const;
    QString getHistoryDirectory() const;
    void ensureHistoryDirectory();
    
    // Serialization
    QJsonObject serializeScanRecord(const ScanRecord& record) const;
    ScanRecord deserializeScanRecord(const QJsonObject& json) const;
    
    static ScanHistoryManager* s_instance;
};

Q_DECLARE_METATYPE(ScanHistoryManager::ScanRecord)

#endif // SCAN_HISTORY_MANAGER_H
