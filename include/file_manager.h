#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QUuid>
#include <QTime>
#include <QTimer>
#include <QMutex>
#include <QQueue>
#include <QHash>

/**
 * @brief FileManager - Core component for safe file operations
 * 
 * This class handles:
 * - File deletion, moving, and copying
 * - Batch file operations
 * - Progress reporting for large operations
 * - Integration with SafetyManager for backups
 * - Operation validation and conflict resolution
 * 
 * All operations are designed to be recoverable and safe.
 */

class SafetyManager; // Forward declaration

class FileManager : public QObject {
    Q_OBJECT

public:
    enum class OperationType {
        Delete,
        Move,
        Copy,
        Restore,
        CreateBackup
    };

    enum class ConflictResolution {
        Skip,           // Skip conflicting files
        Overwrite,      // Overwrite existing files
        Rename,         // Rename with suffix (file_1.txt)
        Ask             // Ask user for each conflict
    };

    struct FileOperation {
        QString operationId;        // Unique identifier
        OperationType type;
        QStringList sourceFiles;
        QString targetPath;         // Directory or file path
        ConflictResolution conflictMode;
        bool createBackup;
        QString reason;             // User-provided reason
        QDateTime timestamp;
        
        FileOperation() : type(OperationType::Delete), conflictMode(ConflictResolution::Ask), createBackup(true) {
            operationId = QUuid::createUuid().toString(QUuid::WithoutBraces);
            timestamp = QDateTime::currentDateTime();
        }
    };

    struct OperationResult {
        QString operationId;
        bool success;
        QString errorMessage;
        QStringList processedFiles;    // Files successfully processed
        QStringList failedFiles;       // Files that failed
        QStringList skippedFiles;      // Files skipped due to conflicts
        qint64 totalSize;              // Total size of processed files
        QDateTime completed;
        
        OperationResult() : success(false), totalSize(0) {
            completed = QDateTime::currentDateTime();
        }
    };

    struct OperationProgress {
        QString operationId;
        int filesProcessed;
        int totalFiles;
        qint64 bytesProcessed;
        qint64 totalBytes;
        QString currentFile;
        double percentComplete;
        QTime estimatedTimeRemaining;
        
        OperationProgress() : filesProcessed(0), totalFiles(0), bytesProcessed(0), totalBytes(0), percentComplete(0.0) {}
    };

    explicit FileManager(QObject* parent = nullptr);
    ~FileManager();

    // Configuration
    void setSafetyManager(SafetyManager* safetyManager);
    SafetyManager* safetyManager() const;

    // File Operations
    QString deleteFiles(const QStringList& filePaths, const QString& reason = QString());
    QString moveFiles(const QStringList& filePaths, const QString& targetDirectory, ConflictResolution conflictMode = ConflictResolution::Ask);
    QString copyFiles(const QStringList& filePaths, const QString& targetDirectory, ConflictResolution conflictMode = ConflictResolution::Ask);
    QString restoreFiles(const QStringList& backupPaths, const QString& targetDirectory = QString());

    // Single file operations
    bool deleteFile(const QString& filePath, bool createBackup = true);
    bool moveFile(const QString& sourceFile, const QString& targetFile, ConflictResolution conflictMode = ConflictResolution::Ask);
    bool copyFile(const QString& sourceFile, const QString& targetFile, ConflictResolution conflictMode = ConflictResolution::Ask);

    // Batch operations
    QString executeBatchOperation(const FileOperation& operation);
    void cancelOperation(const QString& operationId);

    // Status and results
    bool isOperationInProgress(const QString& operationId) const;
    OperationResult getOperationResult(const QString& operationId) const;
    QList<OperationResult> getRecentOperations(int maxResults = 50) const;

    // Utility functions
    static bool validateFilePath(const QString& filePath);
    static QString generateUniqueFileName(const QString& basePath);
    static qint64 calculateDirectorySize(const QString& directoryPath);
    static bool hasWritePermission(const QString& directoryPath);

signals:
    void operationStarted(const QString& operationId, const FileOperation& operation);
    void operationProgress(const OperationProgress& progress);
    void operationCompleted(const OperationResult& result);
    void operationCancelled(const QString& operationId);
    void conflictResolutionRequired(const QString& sourceFile, const QString& targetFile, ConflictResolution suggestedResolution);
    void operationError(const QString& operationId, const QString& error);

private slots:
    void processOperationQueue();

private:
    // Internal operation methods
    bool performDelete(const QString& filePath, const QString& operationId);
    bool performMove(const QString& sourceFile, const QString& targetFile, const QString& operationId);
    bool performCopy(const QString& sourceFile, const QString& targetFile, const QString& operationId);
    bool performRestore(const QString& backupPath, const QString& operationId);
    bool performBackupCreation(const QString& sourceFile, const QString& operationId);
    
    // Conflict handling
    ConflictResolution resolveConflict(const QString& sourceFile, const QString& targetFile, ConflictResolution mode);
    QString generateConflictFreeName(const QString& targetFile);
    
    // Progress tracking
    void updateProgress(const QString& operationId, int filesProcessed, qint64 bytesProcessed, const QString& currentFile);
    void emitProgress(const QString& operationId);
    
    // Validation and safety
    bool validateOperation(const FileOperation& operation, QString& error);
    bool isSystemProtectedFile(const QString& filePath);
    
    // Member variables
    SafetyManager* m_safetyManager;
    QHash<QString, FileOperation> m_activeOperations;
    QHash<QString, OperationProgress> m_operationProgress;
    QHash<QString, OperationResult> m_operationResults;
    QQueue<FileOperation> m_operationQueue;
    QMutex m_operationMutex;
    QTimer* m_progressTimer;
    
    // Settings
    int m_maxConcurrentOperations;
    bool m_createBackupsByDefault;
    ConflictResolution m_defaultConflictResolution;
};

// Q_DECLARE_METATYPE for use with Qt's signal/slot system
Q_DECLARE_METATYPE(FileManager::FileOperation)
Q_DECLARE_METATYPE(FileManager::OperationResult)
Q_DECLARE_METATYPE(FileManager::OperationProgress)
Q_DECLARE_METATYPE(FileManager::ConflictResolution)