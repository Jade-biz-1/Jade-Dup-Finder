#include "file_operation_queue.h"
#include <QFileInfo>
#include <QDir>
#include <QUuid>
#include <QMutexLocker>


FileOperationQueue::FileOperationQueue(QObject* parent)
    : QObject(parent)
    , m_isProcessing(false)
    , m_processTimer(new QTimer(this))
{
    m_processTimer->setSingleShot(true);
    connect(m_processTimer, &QTimer::timeout, this, &FileOperationQueue::processNextOperation);
}

QString FileOperationQueue::queueDeleteOperation(const QStringList& files) {
    QMutexLocker locker(&m_mutex);
    
    FileOperation operation;
    operation.operationId = generateOperationId();
    operation.type = OperationType::Delete;
    operation.sourceFiles = files;
    operation.totalFiles = files.size();
    operation.totalBytes = calculateTotalBytes(files);
    
    m_operationQueue.enqueue(operation);
    
    emit operationQueued(operation.operationId);
    
    // Start processing if not already processing
    if (!m_isProcessing) {
        m_processTimer->start(0);
    }
    
    return operation.operationId;
}

QString FileOperationQueue::queueMoveOperation(const QStringList& files, const QString& destination) {
    QMutexLocker locker(&m_mutex);
    
    FileOperation operation;
    operation.operationId = generateOperationId();
    operation.type = OperationType::Move;
    operation.sourceFiles = files;
    operation.destinationPath = destination;
    operation.totalFiles = files.size();
    operation.totalBytes = calculateTotalBytes(files);
    
    m_operationQueue.enqueue(operation);
    
    emit operationQueued(operation.operationId);
    
    if (!m_isProcessing) {
        m_processTimer->start(0);
    }
    
    return operation.operationId;
}

QString FileOperationQueue::queueCopyOperation(const QStringList& files, const QString& destination) {
    QMutexLocker locker(&m_mutex);
    
    FileOperation operation;
    operation.operationId = generateOperationId();
    operation.type = OperationType::Copy;
    operation.sourceFiles = files;
    operation.destinationPath = destination;
    operation.totalFiles = files.size();
    operation.totalBytes = calculateTotalBytes(files);
    
    m_operationQueue.enqueue(operation);
    
    emit operationQueued(operation.operationId);
    
    if (!m_isProcessing) {
        m_processTimer->start(0);
    }
    
    return operation.operationId;
}

QString FileOperationQueue::queueTrashOperation(const QStringList& files) {
    QMutexLocker locker(&m_mutex);
    
    FileOperation operation;
    operation.operationId = generateOperationId();
    operation.type = OperationType::Trash;
    operation.sourceFiles = files;
    operation.totalFiles = files.size();
    operation.totalBytes = calculateTotalBytes(files);
    
    m_operationQueue.enqueue(operation);
    
    emit operationQueued(operation.operationId);
    
    if (!m_isProcessing) {
        m_processTimer->start(0);
    }
    
    return operation.operationId;
}

bool FileOperationQueue::cancelOperation(const QString& operationId) {
    QMutexLocker locker(&m_mutex);
    
    // Check if it's the current operation
    if (m_currentOperation.operationId == operationId && m_isProcessing) {
        m_currentOperation.status = OperationStatus::Cancelled;
        emit operationCancelled(operationId);
        return true;
    }
    
    // Check if it's in the queue
    for (int i = 0; i < m_operationQueue.size(); ++i) {
        if (m_operationQueue[i].operationId == operationId) {
            FileOperation operation = m_operationQueue[i];
            operation.status = OperationStatus::Cancelled;
            operation.completedAt = QDateTime::currentDateTime();
            
            // Remove from queue and add to history
            m_operationQueue.removeAt(i);
            m_operationHistory.append(operation);
            
            emit operationCancelled(operationId);
            return true;
        }
    }
    
    return false;
}

void FileOperationQueue::cancelAllOperations() {
    QMutexLocker locker(&m_mutex);
    
    // Cancel current operation
    if (m_isProcessing) {
        m_currentOperation.status = OperationStatus::Cancelled;
        emit operationCancelled(m_currentOperation.operationId);
    }
    
    // Cancel all queued operations
    while (!m_operationQueue.isEmpty()) {
        FileOperation operation = m_operationQueue.dequeue();
        operation.status = OperationStatus::Cancelled;
        operation.completedAt = QDateTime::currentDateTime();
        m_operationHistory.append(operation);
        emit operationCancelled(operation.operationId);
    }
}

FileOperationQueue::FileOperation FileOperationQueue::getOperation(const QString& operationId) const {
    QMutexLocker locker(&m_mutex);
    
    // Check current operation
    if (m_currentOperation.operationId == operationId) {
        return m_currentOperation;
    }
    
    // Check queue
    for (const FileOperation& op : m_operationQueue) {
        if (op.operationId == operationId) {
            return op;
        }
    }
    
    // Check history
    for (const FileOperation& op : m_operationHistory) {
        if (op.operationId == operationId) {
            return op;
        }
    }
    
    return FileOperation(); // Return empty operation if not found
}

QList<FileOperationQueue::FileOperation> FileOperationQueue::getAllOperations() const {
    QMutexLocker locker(&m_mutex);
    
    QList<FileOperation> allOperations;
    
    // Add current operation if processing
    if (m_isProcessing) {
        allOperations.append(m_currentOperation);
    }
    
    // Add queued operations
    for (const FileOperation& op : m_operationQueue) {
        allOperations.append(op);
    }
    
    // Add history
    allOperations.append(m_operationHistory);
    
    return allOperations;
}

QList<FileOperationQueue::FileOperation> FileOperationQueue::getOperationsByStatus(OperationStatus status) const {
    QList<FileOperation> allOperations = getAllOperations();
    QList<FileOperation> filteredOperations;
    
    for (const FileOperation& op : allOperations) {
        if (op.status == status) {
            filteredOperations.append(op);
        }
    }
    
    return filteredOperations;
}

bool FileOperationQueue::isProcessing() const {
    QMutexLocker locker(&m_mutex);
    return m_isProcessing;
}

FileOperationQueue::FileOperation FileOperationQueue::getCurrentOperation() const {
    QMutexLocker locker(&m_mutex);
    return m_currentOperation;
}

int FileOperationQueue::getQueueSize() const {
    QMutexLocker locker(&m_mutex);
    return m_operationQueue.size();
}

void FileOperationQueue::clearCompletedOperations() {
    QMutexLocker locker(&m_mutex);
    
    QList<FileOperation> remainingHistory;
    for (const FileOperation& op : m_operationHistory) {
        if (op.status != OperationStatus::Completed) {
            remainingHistory.append(op);
        }
    }
    
    m_operationHistory = remainingHistory;
}

void FileOperationQueue::processNextOperation() {
    QMutexLocker locker(&m_mutex);
    
    if (m_operationQueue.isEmpty()) {
        m_isProcessing = false;
        emit queueEmpty();
        return;
    }
    
    m_currentOperation = m_operationQueue.dequeue();
    m_currentOperation.status = OperationStatus::InProgress;
    m_currentOperation.startedAt = QDateTime::currentDateTime();
    m_isProcessing = true;
    m_currentFile.clear();
    m_operationTimer.start();  // Task 23: Start timing
    
    emit operationStarted(m_currentOperation.operationId);
    
    // Execute operation in a separate thread context
    // For now, execute synchronously (could be improved with QThread)
    bool success = executeOperation(m_currentOperation);
    
    // Update operation status
    m_currentOperation.completedAt = QDateTime::currentDateTime();
    if (m_currentOperation.status == OperationStatus::Cancelled) {
        // Already handled
    } else if (success) {
        m_currentOperation.status = OperationStatus::Completed;
    } else {
        m_currentOperation.status = OperationStatus::Failed;
    }
    
    // Move to history
    m_operationHistory.append(m_currentOperation);
    
    emit operationCompleted(m_currentOperation.operationId, success, m_currentOperation.errorMessage);
    
    // Reset current operation
    m_currentOperation = FileOperation();
    m_isProcessing = false;
    
    // Process next operation if available
    if (!m_operationQueue.isEmpty()) {
        m_processTimer->start(100); // Small delay between operations
    } else {
        emit queueEmpty();
    }
}

QString FileOperationQueue::generateOperationId() const {
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

void FileOperationQueue::updateOperationStatus(const QString& operationId, OperationStatus status, const QString& errorMessage) {
    // Update current operation if it matches
    if (m_currentOperation.operationId == operationId) {
        m_currentOperation.status = status;
        if (!errorMessage.isEmpty()) {
            m_currentOperation.errorMessage = errorMessage;
        }
    }
}

bool FileOperationQueue::executeOperation(FileOperation& operation) {
    switch (operation.type) {
        case OperationType::Delete:
            return executeDeleteOperation(operation);
        case OperationType::Move:
            return executeMoveOperation(operation);
        case OperationType::Copy:
            return executeCopyOperation(operation);
        case OperationType::Trash:
            return executeTrashOperation(operation);
    }
    return false;
}

bool FileOperationQueue::executeDeleteOperation(FileOperation& operation) {
    int processed = 0;
    qint64 bytesProcessed = 0;
    
    for (const QString& filePath : operation.sourceFiles) {
        // Check for cancellation
        if (operation.status == OperationStatus::Cancelled) {
            return false;
        }
        
        // Update current file (Task 23)
        m_currentFile = filePath;
        
        QFileInfo fileInfo(filePath);
        qint64 fileSize = fileInfo.size();
        
        // Emit detailed progress before processing file
        emit detailedOperationProgress(operation.operationId, operation, filePath);
        
        if (QFile::remove(filePath)) {
            processed++;
            bytesProcessed += fileSize;
            
            operation.filesProcessed = processed;
            operation.bytesProcessed = bytesProcessed;
            
            emit operationProgress(operation.operationId, processed, operation.totalFiles, bytesProcessed, operation.totalBytes);
            
            // Emit detailed progress after processing file
            emit detailedOperationProgress(operation.operationId, operation, QString());
        } else {
            operation.errorMessage = tr("Failed to delete file: %1").arg(filePath);
            return false;
        }
    }
    
    return true;
}

bool FileOperationQueue::executeMoveOperation(FileOperation& operation) {
    QDir destDir(operation.destinationPath);
    if (!destDir.exists()) {
        operation.errorMessage = tr("Destination directory does not exist: %1").arg(operation.destinationPath);
        return false;
    }
    
    int processed = 0;
    qint64 bytesProcessed = 0;
    
    for (const QString& filePath : operation.sourceFiles) {
        if (operation.status == OperationStatus::Cancelled) {
            return false;
        }
        
        // Update current file (Task 23)
        m_currentFile = filePath;
        
        QFileInfo fileInfo(filePath);
        QString destPath = destDir.absoluteFilePath(fileInfo.fileName());
        qint64 fileSize = fileInfo.size();
        
        // Emit detailed progress
        emit detailedOperationProgress(operation.operationId, operation, filePath);
        
        if (QFile::rename(filePath, destPath)) {
            processed++;
            bytesProcessed += fileSize;
            
            operation.filesProcessed = processed;
            operation.bytesProcessed = bytesProcessed;
            
            emit operationProgress(operation.operationId, processed, operation.totalFiles, bytesProcessed, operation.totalBytes);
            emit detailedOperationProgress(operation.operationId, operation, QString());
        } else {
            operation.errorMessage = tr("Failed to move file: %1 to %2").arg(filePath, destPath);
            return false;
        }
    }
    
    return true;
}

bool FileOperationQueue::executeCopyOperation(FileOperation& operation) {
    QDir destDir(operation.destinationPath);
    if (!destDir.exists()) {
        operation.errorMessage = tr("Destination directory does not exist: %1").arg(operation.destinationPath);
        return false;
    }
    
    int processed = 0;
    qint64 bytesProcessed = 0;
    
    for (const QString& filePath : operation.sourceFiles) {
        if (operation.status == OperationStatus::Cancelled) {
            return false;
        }
        
        // Update current file (Task 23)
        m_currentFile = filePath;
        
        QFileInfo fileInfo(filePath);
        QString destPath = destDir.absoluteFilePath(fileInfo.fileName());
        qint64 fileSize = fileInfo.size();
        
        // Emit detailed progress
        emit detailedOperationProgress(operation.operationId, operation, filePath);
        
        if (QFile::copy(filePath, destPath)) {
            processed++;
            bytesProcessed += fileSize;
            
            operation.filesProcessed = processed;
            operation.bytesProcessed = bytesProcessed;
            
            emit operationProgress(operation.operationId, processed, operation.totalFiles, bytesProcessed, operation.totalBytes);
            emit detailedOperationProgress(operation.operationId, operation, QString());
        } else {
            operation.errorMessage = tr("Failed to copy file: %1 to %2").arg(filePath, destPath);
            return false;
        }
    }
    
    return true;
}

bool FileOperationQueue::executeTrashOperation(FileOperation& operation) {
    // For now, implement as delete operation
    // In a real implementation, this would move files to system trash
    return executeDeleteOperation(operation);
}

qint64 FileOperationQueue::calculateTotalBytes(const QStringList& files) const {
    qint64 totalBytes = 0;
    
    for (const QString& filePath : files) {
        QFileInfo fileInfo(filePath);
        if (fileInfo.exists()) {
            totalBytes += fileInfo.size();
        }
    }
    
    return totalBytes;
}

FileOperationQueue::OperationProgress FileOperationQueue::getCurrentOperationProgress() const {
    QMutexLocker locker(&m_mutex);
    
    OperationProgress progress;
    
    if (!m_isProcessing) {
        return progress;
    }
    
    progress.operationId = m_currentOperation.operationId;
    progress.type = m_currentOperation.type;
    progress.status = m_currentOperation.status;
    progress.filesProcessed = m_currentOperation.filesProcessed;
    progress.totalFiles = m_currentOperation.totalFiles;
    progress.bytesProcessed = m_currentOperation.bytesProcessed;
    progress.totalBytes = m_currentOperation.totalBytes;
    progress.currentFile = m_currentFile;
    progress.errorMessage = m_currentOperation.errorMessage;
    
    // Calculate percentage
    if (progress.totalFiles > 0) {
        progress.percentComplete = (double)progress.filesProcessed / progress.totalFiles * 100.0;
    }
    
    // Calculate timing information
    progress.elapsedTimeMs = m_operationTimer.elapsed();
    
    if (progress.elapsedTimeMs > 0) {
        double elapsedSeconds = progress.elapsedTimeMs / 1000.0;
        
        // Calculate rates
        progress.filesPerSecond = progress.filesProcessed / elapsedSeconds;
        progress.bytesPerSecond = progress.bytesProcessed / elapsedSeconds;
        
        // Estimate time remaining
        if (progress.filesProcessed > 0 && progress.totalFiles > progress.filesProcessed) {
            int remainingFiles = progress.totalFiles - progress.filesProcessed;
            double estimatedSeconds = remainingFiles / progress.filesPerSecond;
            progress.estimatedTimeRemainingMs = static_cast<qint64>(estimatedSeconds * 1000);
        }
    }
    
    return progress;
}