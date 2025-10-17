#pragma once

#include <QObject>
#include <QQueue>
#include <QTimer>
#include <QMutex>
#include <QThread>
#include <QDateTime>
#include <QStringList>
#include <QElapsedTimer>

/**
 * @brief File Operation Queue for managing file operations (Task 22)
 * 
 * This class manages a queue of file operations (delete, move, copy)
 * and processes them sequentially with progress tracking and cancellation
 * support.
 */
class FileOperationQueue : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Types of file operations
     */
    enum class OperationType {
        Delete,
        Move,
        Copy,
        Trash
    };
    Q_ENUM(OperationType)

    /**
     * @brief Operation status
     */
    enum class OperationStatus {
        Pending,
        InProgress,
        Completed,
        Failed,
        Cancelled
    };
    Q_ENUM(OperationStatus)

    /**
     * @brief File operation structure
     */
    struct FileOperation {
        QString operationId;            // Unique operation ID
        OperationType type;             // Type of operation
        QStringList sourceFiles;       // Source file paths
        QString destinationPath;       // Destination path (for move/copy)
        OperationStatus status;        // Current status
        QString errorMessage;          // Error message if failed
        QDateTime createdAt;           // When operation was created
        QDateTime startedAt;           // When operation started
        QDateTime completedAt;         // When operation completed
        int filesProcessed;            // Number of files processed
        int totalFiles;                // Total number of files
        qint64 bytesProcessed;         // Bytes processed
        qint64 totalBytes;             // Total bytes to process
        
        FileOperation() 
            : type(OperationType::Delete)
            , status(OperationStatus::Pending)
            , createdAt(QDateTime::currentDateTime())
            , filesProcessed(0)
            , totalFiles(0)
            , bytesProcessed(0)
            , totalBytes(0) {}
    };

    explicit FileOperationQueue(QObject* parent = nullptr);
    ~FileOperationQueue() override = default;

    /**
     * @brief Queue a delete operation
     * @param files List of files to delete
     * @return Operation ID
     */
    QString queueDeleteOperation(const QStringList& files);

    /**
     * @brief Queue a move operation
     * @param files List of files to move
     * @param destination Destination directory
     * @return Operation ID
     */
    QString queueMoveOperation(const QStringList& files, const QString& destination);

    /**
     * @brief Queue a copy operation
     * @param files List of files to copy
     * @param destination Destination directory
     * @return Operation ID
     */
    QString queueCopyOperation(const QStringList& files, const QString& destination);

    /**
     * @brief Queue a trash operation
     * @param files List of files to move to trash
     * @return Operation ID
     */
    QString queueTrashOperation(const QStringList& files);

    /**
     * @brief Cancel an operation by ID
     * @param operationId The operation to cancel
     * @return True if operation was cancelled
     */
    bool cancelOperation(const QString& operationId);

    /**
     * @brief Cancel all pending operations
     */
    void cancelAllOperations();

    /**
     * @brief Get operation by ID
     * @param operationId The operation ID
     * @return The operation, or empty operation if not found
     */
    FileOperation getOperation(const QString& operationId) const;

    /**
     * @brief Get all operations
     */
    QList<FileOperation> getAllOperations() const;

    /**
     * @brief Get operations by status
     */
    QList<FileOperation> getOperationsByStatus(OperationStatus status) const;

    /**
     * @brief Check if queue is currently processing
     */
    bool isProcessing() const;

    /**
     * @brief Get current operation being processed
     */
    FileOperation getCurrentOperation() const;

    /**
     * @brief Get queue size
     */
    int getQueueSize() const;

    /**
     * @brief Clear completed operations from history
     */
    void clearCompletedOperations();

    /**
     * @brief Get detailed progress for current operation (Task 23)
     */
    struct OperationProgress {
        QString operationId;
        OperationType type;
        OperationStatus status;
        int filesProcessed;
        int totalFiles;
        qint64 bytesProcessed;
        qint64 totalBytes;
        QString currentFile;
        double percentComplete;
        qint64 elapsedTimeMs;
        qint64 estimatedTimeRemainingMs;
        double filesPerSecond;
        double bytesPerSecond;
        QString errorMessage;
        
        OperationProgress() : type(OperationType::Delete), status(OperationStatus::Pending),
                            filesProcessed(0), totalFiles(0), bytesProcessed(0), totalBytes(0),
                            percentComplete(0.0), elapsedTimeMs(0), estimatedTimeRemainingMs(-1),
                            filesPerSecond(0.0), bytesPerSecond(0.0) {}
    };

    /**
     * @brief Get current operation progress
     */
    OperationProgress getCurrentOperationProgress() const;

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
     * @brief Emitted during operation progress
     */
    void operationProgress(const QString& operationId, int filesProcessed, int totalFiles, qint64 bytesProcessed, qint64 totalBytes);

    /**
     * @brief Emitted with detailed progress information (Task 23)
     */
    void detailedOperationProgress(const QString& operationId, const FileOperation& operation, const QString& currentFile);

    /**
     * @brief Emitted when an operation completes
     */
    void operationCompleted(const QString& operationId, bool success, const QString& errorMessage);

    /**
     * @brief Emitted when an operation is cancelled
     */
    void operationCancelled(const QString& operationId);

    /**
     * @brief Emitted when queue becomes empty
     */
    void queueEmpty();

private slots:
    void processNextOperation();

private:
    QQueue<FileOperation> m_operationQueue;
    QList<FileOperation> m_operationHistory;
    FileOperation m_currentOperation;
    QString m_currentFile;  // Task 23: Track current file being processed
    QElapsedTimer m_operationTimer;  // Task 23: Track operation timing
    bool m_isProcessing;
    QTimer* m_processTimer;
    mutable QMutex m_mutex;

    QString generateOperationId() const;
    void updateOperationStatus(const QString& operationId, OperationStatus status, const QString& errorMessage = QString());
    bool executeOperation(FileOperation& operation);
    bool executeDeleteOperation(FileOperation& operation);
    bool executeMoveOperation(FileOperation& operation);
    bool executeCopyOperation(FileOperation& operation);
    bool executeTrashOperation(FileOperation& operation);
    qint64 calculateTotalBytes(const QStringList& files) const;
};