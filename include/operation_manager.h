#pragma once

#include <QObject>
#include <QTimer>
#include <QMutex>
#include <QDateTime>
#include <QStringList>
#include "scan_progress_dialog.h"
#include "file_operation_queue.h"

/**
 * @brief Enhanced Operation Manager for coordinating multiple operation types
 * 
 * This class manages and coordinates different types of operations:
 * - File operations (delete, move, copy) via FileOperationQueue
 * - Scan operations
 * - Hash calculation operations
 * - Any other long-running operations
 * 
 * It provides unified progress tracking and queue management for the enhanced
 * progress indication system (Task 7.1 & 7.2).
 */
class OperationManager : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Types of operations managed by this system
     */
    enum class OperationType {
        Scan,           // File scanning operations
        Hash,           // Hash calculation operations
        FileDelete,     // File deletion operations
        FileMove,       // File move operations
        FileCopy,       // File copy operations
        FileTrash,      // Move to trash operations
        Validation,     // Theme/style validation operations
        Custom          // Custom user-defined operations
    };
    Q_ENUM(OperationType)

    /**
     * @brief Enhanced operation information structure
     */
    struct Operation {
        QString operationId;
        OperationType type;
        QString description;
        ScanProgressDialog::OperationStatus status;
        int priority;
        QDateTime queuedTime;
        QDateTime startTime;
        QDateTime completedTime;
        
        // Progress tracking
        int filesTotal;
        int filesProcessed;
        qint64 bytesTotal;
        qint64 bytesProcessed;
        QString currentFile;
        QString currentFolder;
        
        // Performance metrics
        double filesPerSecond;
        double bytesPerSecond;
        double averageFileSize;
        
        // Error handling
        int errorsEncountered;
        QString lastError;
        QStringList errorLog;
        
        // Additional metadata
        QVariantMap metadata;
        
        Operation() 
            : type(OperationType::Custom)
            , status(ScanProgressDialog::OperationStatus::Initializing)
            , priority(0)
            , queuedTime(QDateTime::currentDateTime())
            , filesTotal(0)
            , filesProcessed(0)
            , bytesTotal(0)
            , bytesProcessed(0)
            , filesPerSecond(0.0)
            , bytesPerSecond(0.0)
            , averageFileSize(0.0)
            , errorsEncountered(0) {}
    };

    explicit OperationManager(QObject* parent = nullptr);
    ~OperationManager() override = default;

    /**
     * @brief Set the file operation queue for file operations
     */
    void setFileOperationQueue(FileOperationQueue* queue);

    /**
     * @brief Queue a new operation
     * @param type Operation type
     * @param description Human-readable description
     * @param priority Priority (higher = more important)
     * @param metadata Additional operation-specific data
     * @return Operation ID
     */
    QString queueOperation(OperationType type, const QString& description, 
                          int priority = 0, const QVariantMap& metadata = QVariantMap());

    /**
     * @brief Start processing the operation queue
     */
    void startProcessing();

    /**
     * @brief Pause processing
     */
    void pauseProcessing();

    /**
     * @brief Resume processing
     */
    void resumeProcessing();

    /**
     * @brief Cancel a specific operation
     */
    bool cancelOperation(const QString& operationId);

    /**
     * @brief Cancel all operations
     */
    void cancelAllOperations();

    /**
     * @brief Get current operation
     */
    Operation getCurrentOperation() const;

    /**
     * @brief Get all operations in queue
     */
    QList<Operation> getAllOperations() const;

    /**
     * @brief Get operations by status
     */
    QList<Operation> getOperationsByStatus(ScanProgressDialog::OperationStatus status) const;

    /**
     * @brief Update operation progress (called by operation implementations)
     */
    void updateOperationProgress(const QString& operationId, 
                                int filesProcessed, int filesTotal,
                                qint64 bytesProcessed, qint64 bytesTotal,
                                const QString& currentFile = QString(),
                                const QString& currentFolder = QString());

    /**
     * @brief Report an error for an operation
     */
    void reportOperationError(const QString& operationId, const QString& error);

    /**
     * @brief Mark operation as completed
     */
    void completeOperation(const QString& operationId);

    /**
     * @brief Get enhanced progress info for progress dialog
     */
    ScanProgressDialog::ProgressInfo getProgressInfo() const;

    /**
     * @brief Check if any operations are running
     */
    bool isProcessing() const;

    /**
     * @brief Get queue size
     */
    int getQueueSize() const;

signals:
    /**
     * @brief Emitted when an operation is queued
     */
    void operationQueued(const QString& operationId);

    /**
     * @brief Emitted when an operation starts
     */
    void operationStarted(const QString& operationId);

    /**
     * @brief Emitted when operation progress updates
     */
    void operationProgressUpdated(const QString& operationId);

    /**
     * @brief Emitted when an operation completes
     */
    void operationCompleted(const QString& operationId);

    /**
     * @brief Emitted when an operation fails
     */
    void operationFailed(const QString& operationId, const QString& error);

    /**
     * @brief Emitted when an operation is cancelled
     */
    void operationCancelled(const QString& operationId);

    /**
     * @brief Emitted when enhanced progress info is updated
     */
    void progressInfoUpdated(const ScanProgressDialog::ProgressInfo& info);

private slots:
    /**
     * @brief Process the next operation in queue
     */
    void processNextOperation();

    /**
     * @brief Update progress metrics
     */
    void updateProgressMetrics();

    /**
     * @brief Handle file operation queue signals
     */
    void onFileOperationQueued(const QString& operationId);
    void onFileOperationStarted(const QString& operationId);
    void onFileOperationProgress(const QString& operationId, int filesProcessed, int totalFiles, qint64 bytesProcessed, qint64 totalBytes);
    void onFileOperationCompleted(const QString& operationId);
    void onFileOperationFailed(const QString& operationId, const QString& error);

private:
    /**
     * @brief Convert FileOperationQueue operation to our Operation structure
     */
    Operation convertFileOperation(const FileOperationQueue::FileOperation& fileOp) const;

    /**
     * @brief Convert FileOperationQueue type to our OperationType
     */
    OperationType convertFileOperationType(FileOperationQueue::OperationType type) const;

    /**
     * @brief Calculate performance metrics
     */
    void calculatePerformanceMetrics(Operation& operation);

    /**
     * @brief Generate unique operation ID
     */
    QString generateOperationId() const;

    /**
     * @brief Get operation type name as string
     */
    QString getOperationTypeName(OperationType type) const;

    /**
     * @brief Check if operation type is a file operation
     */
    bool isFileOperation(OperationType type) const;

    // Member variables
    FileOperationQueue* m_fileOperationQueue;
    QList<Operation> m_operations;
    QString m_currentOperationId;
    QTimer* m_progressTimer;
    QMutex m_operationMutex;
    bool m_isProcessing;
    bool m_isPaused;
    
    // Performance tracking
    QElapsedTimer m_operationTimer;
    QList<double> m_recentFilesPerSecond;
    QList<double> m_recentBytesPerSecond;
    static const int MAX_PERFORMANCE_SAMPLES = 10;
};