#ifndef SAFETY_MANAGER_H
#define SAFETY_MANAGER_H

#include <QObject>
#include <QHash>
#include <QStringList>
#include <QDateTime>
#include <QMutex>
#include <QTimer>
#include <QDir>
#include <QFileInfo>

/**
 * @class SafetyManager
 * @brief Provides comprehensive file operation safety features including backups, undo, and protection
 * 
 * The SafetyManager ensures file operations are performed safely by:
 * - Creating backups before destructive operations
 * - Maintaining an undo history
 * - Protecting system and important files
 * - Validating operations before execution
 * - Logging all activities for audit trails
 */
class SafetyManager : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Types of safety operations
     */
    enum class SafetyLevel {
        Minimal,    ///< Basic validation only
        Standard,   ///< Standard backups and protection
        Maximum     ///< Full backups, verification, and strict protection
    };
    
    /**
     * @brief Backup strategies for different scenarios
     */
    enum class BackupStrategy {
        None,           ///< No backup created
        OriginalLocation, ///< Backup in same directory as original
        CentralLocation,  ///< Backup in designated backup directory
        Versioned,      ///< Multiple versioned backups
        Compressed      ///< Compressed backup archive
    };
    
    /**
     * @brief Protection levels for files and directories
     */
    enum class ProtectionLevel {
        None,       ///< No protection
        ReadOnly,   ///< Protect against modifications
        System,     ///< System file protection
        Critical    ///< Critical file - require confirmation
    };
    
    /**
     * @brief Operation types that can be tracked and undone
     */
    enum class OperationType {
        Delete,
        Move,
        Copy,
        Modify,
        Create
    };
    
    /**
     * @brief Structure to track individual file operations
     */
    struct SafetyOperation {
        QString operationId;
        OperationType type;
        QString sourceFile;
        QString targetFile;
        QString backupPath;
        QDateTime timestamp;
        QString reason;
        bool canUndo;
        qint64 fileSize;
        QString checksum;
        
        SafetyOperation() : canUndo(false), fileSize(0) {}
    };
    
    /**
     * @brief File protection entry
     */
    struct ProtectionEntry {
        QString pattern;        ///< File pattern or path
        ProtectionLevel level;
        QString description;
        bool isRegex;
        
        ProtectionEntry() : level(ProtectionLevel::None), isRegex(false) {}
    };
    
    /**
     * @brief Backup verification result
     */
    struct BackupVerification {
        bool isValid;
        QString backupPath;
        QString originalPath;
        qint64 originalSize;
        qint64 backupSize;
        QString originalChecksum;
        QString backupChecksum;
        QString errorMessage;
        
        BackupVerification() : isValid(false), originalSize(0), backupSize(0) {}
    };
    
    explicit SafetyManager(QObject* parent = nullptr);
    ~SafetyManager();
    
    // Configuration methods
    void setSafetyLevel(SafetyLevel level);
    SafetyLevel safetyLevel() const;
    
    void setDefaultBackupStrategy(BackupStrategy strategy);
    BackupStrategy defaultBackupStrategy() const;
    
    void setBackupDirectory(const QString& directory);
    QString backupDirectory() const;
    
    void setMaxBackupAge(int days);
    int maxBackupAge() const;
    
    void setMaxUndoOperations(int maxOperations);
    int maxUndoOperations() const;
    
    // Backup operations
    QString createBackup(const QString& filePath, BackupStrategy strategy = BackupStrategy::CentralLocation);
    bool restoreFromBackup(const QString& backupPath, const QString& targetPath = QString());
    BackupVerification verifyBackup(const QString& backupPath, const QString& originalPath);
    QStringList listBackups(const QString& originalFilePath) const;
    bool removeBackup(const QString& backupPath);
    void cleanupOldBackups();
    
    // Protection system
    void addProtectionRule(const QString& pattern, ProtectionLevel level, const QString& description = QString(), bool isRegex = false);
    void removeProtectionRule(const QString& pattern);
    ProtectionLevel getProtectionLevel(const QString& filePath) const;
    bool isProtected(const QString& filePath, OperationType operation) const;
    QList<ProtectionEntry> getProtectionRules() const;
    
    // Operation tracking and undo system
    QString registerOperation(OperationType type, const QString& sourceFile, const QString& targetFile = QString(), const QString& reason = QString());
    bool finalizeOperation(const QString& operationId, bool success, const QString& backupPath = QString());
    bool undoOperation(const QString& operationId);
    bool canUndoOperation(const QString& operationId) const;
    QList<SafetyOperation> getUndoHistory(int maxResults = 50) const;
    void clearUndoHistory();
    bool removeOperation(const QString& operationId);  // Remove operation from history
    void removeOperationsForBackup(const QString& backupPath);  // Remove operations by backup path
    
    // Validation methods
    bool validateOperation(OperationType type, const QString& sourceFile, const QString& targetFile = QString()) const;
    QString getOperationRiskAssessment(OperationType type, const QString& sourceFile, const QString& targetFile = QString()) const;
    bool requiresConfirmation(OperationType type, const QString& sourceFile) const;
    
    // System integration
    bool isSystemFile(const QString& filePath) const;
    bool isInSystemDirectory(const QString& filePath) const;
    QStringList getSystemProtectedPaths() const;
    void addSystemProtectedPath(const QString& path);
    
    // Statistics and reporting
    int getTotalOperationsTracked() const;
    qint64 getTotalBackupSize() const;
    int getActiveBackupCount() const;
    QHash<OperationType, int> getOperationStatistics() const;
    
    // Maintenance operations
    void performMaintenance();
    bool validateBackupIntegrity();
    void optimizeBackupStorage();
    
    // Backup path lookup
    QString getOriginalPathForBackup(const QString& backupPath) const;
    
    // Static utility methods
    static QString calculateFileChecksum(const QString& filePath);
    static bool compareFiles(const QString& file1, const QString& file2);
    static QString formatFileSize(qint64 bytes);
    static QString generateBackupName(const QString& originalPath, const QDateTime& timestamp = QDateTime::currentDateTime());

public slots:
    void startAutoMaintenance();
    void stopAutoMaintenance();
    void performScheduledCleanup();
    void onFileSystemChanged(const QString& path);
    
signals:
    void operationRegistered(const QString& operationId, const SafetyManager::SafetyOperation& operation);
    void backupCreated(const QString& originalPath, const QString& backupPath);
    void backupRestored(const QString& backupPath, const QString& targetPath);
    void operationUndone(const QString& operationId);
    void protectionViolation(const QString& filePath, SafetyManager::OperationType operation);
    void maintenanceCompleted();
    void backupCleanupCompleted(int removedCount, qint64 freedSpace);
    void riskAssessmentChanged(const QString& filePath, const QString& assessment);
    
private slots:
    void performPeriodicMaintenance();
    void cleanupExpiredOperations();
    
private:
    // Core functionality
    QString generateOperationId() const;
    QString createBackupPath(const QString& originalPath, BackupStrategy strategy) const;
    bool performBackupOperation(const QString& sourcePath, const QString& backupPath);
    bool performRestoreOperation(const QString& backupPath, const QString& targetPath);
    
    // Protection system internals
    void initializeDefaultProtectionRules();
    bool matchesProtectionPattern(const QString& filePath, const ProtectionEntry& entry) const;
    
    // Maintenance helpers
    void removeExpiredBackups();
    void compactOperationHistory();
    void validateAllBackups();
    
    // File system helpers
    bool ensureBackupDirectoryExists() const;
    QString getRelativeBackupPath(const QString& originalPath) const;
    
    // Member variables
    SafetyLevel m_safetyLevel;
    BackupStrategy m_defaultBackupStrategy;
    QString m_backupDirectory;
    int m_maxBackupAge;        // Days
    int m_maxUndoOperations;
    
    QHash<QString, SafetyOperation> m_operations;  // operationId -> operation
    QList<ProtectionEntry> m_protectionRules;
    QStringList m_systemProtectedPaths;
    QHash<OperationType, int> m_operationStats;
    
    mutable QMutex m_operationMutex;
    mutable QMutex m_protectionMutex;
    
    QTimer* m_maintenanceTimer;
    QTimer* m_cleanupTimer;
    
    // Constants
    static const int DEFAULT_MAX_BACKUP_AGE = 30;    // days
    static const int DEFAULT_MAX_UNDO_OPERATIONS = 100;
    static const int MAINTENANCE_INTERVAL = 3600000; // 1 hour in milliseconds
    static const int CLEANUP_INTERVAL = 86400000;    // 24 hours in milliseconds
};

// Register metatypes for signal/slot system
Q_DECLARE_METATYPE(SafetyManager::SafetyLevel)
Q_DECLARE_METATYPE(SafetyManager::BackupStrategy)
Q_DECLARE_METATYPE(SafetyManager::ProtectionLevel)
Q_DECLARE_METATYPE(SafetyManager::OperationType)
Q_DECLARE_METATYPE(SafetyManager::SafetyOperation)
Q_DECLARE_METATYPE(SafetyManager::ProtectionEntry)
Q_DECLARE_METATYPE(SafetyManager::BackupVerification)

#endif // SAFETY_MANAGER_H