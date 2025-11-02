#include "operation_manager.h"
#include <QUuid>
#include <QDebug>
#include <algorithm>

OperationManager::OperationManager(QObject* parent)
    : QObject(parent)
    , m_fileOperationQueue(nullptr)
    , m_progressTimer(new QTimer(this))
    , m_isProcessing(false)
    , m_isPaused(false)
{
    // Set up progress update timer
    m_progressTimer->setInterval(500); // Update every 500ms
    connect(m_progressTimer, &QTimer::timeout, this, &OperationManager::updateProgressMetrics);
    
    m_operationTimer.start();
}

void OperationManager::setFileOperationQueue(FileOperationQueue* queue) {
    if (m_fileOperationQueue) {
        // Disconnect old queue
        disconnect(m_fileOperationQueue, nullptr, this, nullptr);
    }
    
    m_fileOperationQueue = queue;
    
    if (m_fileOperationQueue) {
        // Connect to file operation queue signals
        connect(m_fileOperationQueue, &FileOperationQueue::operationQueued,
                this, &OperationManager::onFileOperationQueued);
        connect(m_fileOperationQueue, &FileOperationQueue::operationStarted,
                this, &OperationManager::onFileOperationStarted);
        connect(m_fileOperationQueue, &FileOperationQueue::operationProgress,
                this, &OperationManager::onFileOperationProgress);
        connect(m_fileOperationQueue, &FileOperationQueue::operationCompleted,
                this, &OperationManager::onFileOperationCompleted);
        connect(m_fileOperationQueue, &FileOperationQueue::operationFailed,
                this, &OperationManager::onFileOperationFailed);
    }
}

QString OperationManager::queueOperation(OperationType type, const QString& description, 
                                        int priority, const QVariantMap& metadata) {
    QMutexLocker locker(&m_operationMutex);
    
    Operation operation;
    operation.operationId = generateOperationId();
    operation.type = type;
    operation.description = description;
    operation.priority = priority;
    operation.metadata = metadata;
    operation.status = ScanProgressDialog::OperationStatus::Initializing;
    
    // Insert operation in priority order
    auto insertPos = std::upper_bound(m_operations.begin(), m_operations.end(), operation,
        [](const Operation& a, const Operation& b) {
            return a.priority > b.priority; // Higher priority first
        });
    
    m_operations.insert(insertPos, operation);
    
    emit operationQueued(operation.operationId);
    
    // Start processing if not already running
    if (!m_isProcessing && !m_isPaused) {
        startProcessing();
    }
    
    return operation.operationId;
}

void OperationManager::startProcessing() {
    if (m_isProcessing) {
        return;
    }
    
    m_isProcessing = true;
    m_isPaused = false;
    m_progressTimer->start();
    
    processNextOperation();
}

void OperationManager::pauseProcessing() {
    m_isPaused = true;
    m_progressTimer->stop();
    
    // Update current operation status
    if (!m_currentOperationId.isEmpty()) {
        QMutexLocker locker(&m_operationMutex);
        for (auto& operation : m_operations) {
            if (operation.operationId == m_currentOperationId) {
                operation.status = ScanProgressDialog::OperationStatus::Paused;
                break;
            }
        }
    }
    
    emit progressInfoUpdated(getProgressInfo());
}

void OperationManager::resumeProcessing() {
    if (!m_isPaused) {
        return;
    }
    
    m_isPaused = false;
    m_progressTimer->start();
    
    // Update current operation status
    if (!m_currentOperationId.isEmpty()) {
        QMutexLocker locker(&m_operationMutex);
        for (auto& operation : m_operations) {
            if (operation.operationId == m_currentOperationId) {
                operation.status = ScanProgressDialog::OperationStatus::Running;
                break;
            }
        }
    }
    
    processNextOperation();
}

bool OperationManager::cancelOperation(const QString& operationId) {
    QMutexLocker locker(&m_operationMutex);
    
    for (auto& operation : m_operations) {
        if (operation.operationId == operationId) {
            if (operation.status == ScanProgressDialog::OperationStatus::Initializing ||
                operation.status == ScanProgressDialog::OperationStatus::Running ||
                operation.status == ScanProgressDialog::OperationStatus::Paused) {
                
                operation.status = ScanProgressDialog::OperationStatus::Cancelled;
                operation.completedTime = QDateTime::currentDateTime();
                
                // If this is the current operation, move to next
                if (m_currentOperationId == operationId) {
                    m_currentOperationId.clear();
                    QTimer::singleShot(0, this, &OperationManager::processNextOperation);
                }
                
                emit operationCancelled(operationId);
                emit progressInfoUpdated(getProgressInfo());
                return true;
            }
        }
    }
    
    return false;
}

void OperationManager::cancelAllOperations() {
    QMutexLocker locker(&m_operationMutex);
    
    for (auto& operation : m_operations) {
        if (operation.status == ScanProgressDialog::OperationStatus::Initializing ||
            operation.status == ScanProgressDialog::OperationStatus::Running ||
            operation.status == ScanProgressDialog::OperationStatus::Paused) {
            
            operation.status = ScanProgressDialog::OperationStatus::Cancelled;
            operation.completedTime = QDateTime::currentDateTime();
            emit operationCancelled(operation.operationId);
        }
    }
    
    m_currentOperationId.clear();
    m_isProcessing = false;
    m_progressTimer->stop();
    
    emit progressInfoUpdated(getProgressInfo());
}

OperationManager::Operation OperationManager::getCurrentOperation() const {
    QMutexLocker locker(&m_operationMutex);
    
    if (m_currentOperationId.isEmpty()) {
        return Operation();
    }
    
    for (const auto& operation : m_operations) {
        if (operation.operationId == m_currentOperationId) {
            return operation;
        }
    }
    
    return Operation();
}

QList<OperationManager::Operation> OperationManager::getAllOperations() const {
    QMutexLocker locker(&m_operationMutex);
    return m_operations;
}

QList<OperationManager::Operation> OperationManager::getOperationsByStatus(ScanProgressDialog::OperationStatus status) const {
    QMutexLocker locker(&m_operationMutex);
    
    QList<Operation> result;
    for (const auto& operation : m_operations) {
        if (operation.status == status) {
            result.append(operation);
        }
    }
    
    return result;
}

void OperationManager::updateOperationProgress(const QString& operationId, 
                                              int filesProcessed, int filesTotal,
                                              qint64 bytesProcessed, qint64 bytesTotal,
                                              const QString& currentFile,
                                              const QString& currentFolder) {
    QMutexLocker locker(&m_operationMutex);
    
    for (auto& operation : m_operations) {
        if (operation.operationId == operationId) {
            operation.filesProcessed = filesProcessed;
            operation.filesTotal = filesTotal;
            operation.bytesProcessed = bytesProcessed;
            operation.bytesTotal = bytesTotal;
            operation.currentFile = currentFile;
            operation.currentFolder = currentFolder;
            
            // Calculate performance metrics
            calculatePerformanceMetrics(operation);
            
            emit operationProgressUpdated(operationId);
            emit progressInfoUpdated(getProgressInfo());
            break;
        }
    }
}

void OperationManager::reportOperationError(const QString& operationId, const QString& error) {
    QMutexLocker locker(&m_operationMutex);
    
    for (auto& operation : m_operations) {
        if (operation.operationId == operationId) {
            operation.errorsEncountered++;
            operation.lastError = error;
            operation.errorLog.append(QString("[%1] %2")
                                    .arg(QDateTime::currentDateTime().toString())
                                    .arg(error));
            
            // Limit error log size
            if (operation.errorLog.size() > 100) {
                operation.errorLog.removeFirst();
            }
            
            emit progressInfoUpdated(getProgressInfo());
            break;
        }
    }
}

void OperationManager::completeOperation(const QString& operationId) {
    QMutexLocker locker(&m_operationMutex);
    
    for (auto& operation : m_operations) {
        if (operation.operationId == operationId) {
            operation.status = ScanProgressDialog::OperationStatus::Completed;
            operation.completedTime = QDateTime::currentDateTime();
            
            emit operationCompleted(operationId);
            
            // If this is the current operation, move to next
            if (m_currentOperationId == operationId) {
                m_currentOperationId.clear();
                QTimer::singleShot(0, this, &OperationManager::processNextOperation);
            }
            
            emit progressInfoUpdated(getProgressInfo());
            break;
        }
    }
}

ScanProgressDialog::ProgressInfo OperationManager::getProgressInfo() const {
    QMutexLocker locker(&m_operationMutex);
    
    ScanProgressDialog::ProgressInfo info;
    
    // Get current operation
    Operation currentOp = getCurrentOperation();
    if (!currentOp.operationId.isEmpty()) {
        info.operationId = currentOp.operationId;
        info.operationType = getOperationTypeName(currentOp.type);
        info.status = currentOp.status;
        info.filesScanned = currentOp.filesProcessed;
        info.totalFiles = currentOp.filesTotal;
        info.bytesScanned = currentOp.bytesProcessed;
        info.totalBytes = currentOp.bytesTotal;
        info.currentFolder = currentOp.currentFolder;
        info.currentFile = currentOp.currentFile;
        info.filesPerSecond = currentOp.filesPerSecond;
        info.bytesPerSecond = currentOp.bytesPerSecond;
        info.averageFileSize = currentOp.averageFileSize;
        info.errorsEncountered = currentOp.errorsEncountered;
        info.lastError = currentOp.lastError;
        
        // Calculate timing
        if (!currentOp.startTime.isNull()) {
            info.secondsElapsed = currentOp.startTime.secsTo(QDateTime::currentDateTime());
        }
        
        // Calculate ETA
        if (currentOp.filesPerSecond > 0 && currentOp.filesTotal > 0) {
            int remainingFiles = currentOp.filesTotal - currentOp.filesProcessed;
            info.secondsRemaining = static_cast<int>(remainingFiles / currentOp.filesPerSecond);
        }
    }
    
    // Convert operations to QueuedOperation format
    for (const auto& operation : m_operations) {
        ScanProgressDialog::QueuedOperation queuedOp;
        queuedOp.operationId = operation.operationId;
        queuedOp.operationType = getOperationTypeName(operation.type);
        queuedOp.description = operation.description;
        queuedOp.status = operation.status;
        queuedOp.priority = operation.priority;
        queuedOp.queuedTime = operation.queuedTime;
        queuedOp.startTime = operation.startTime;
        queuedOp.completedTime = operation.completedTime;
        queuedOp.filesTotal = operation.filesTotal;
        queuedOp.filesProcessed = operation.filesProcessed;
        queuedOp.bytesTotal = operation.bytesTotal;
        queuedOp.bytesProcessed = operation.bytesProcessed;
        queuedOp.errorsEncountered = operation.errorsEncountered;
        
        info.operationQueue.append(queuedOp);
    }
    
    info.totalOperationsInQueue = m_operations.size();
    
    // Find current operation index
    for (int i = 0; i < m_operations.size(); ++i) {
        if (m_operations[i].operationId == m_currentOperationId) {
            info.currentOperationIndex = i;
            break;
        }
    }
    
    return info;
}

bool OperationManager::isProcessing() const {
    return m_isProcessing && !m_isPaused;
}

int OperationManager::getQueueSize() const {
    QMutexLocker locker(&m_operationMutex);
    
    int count = 0;
    for (const auto& operation : m_operations) {
        if (operation.status == ScanProgressDialog::OperationStatus::Initializing ||
            operation.status == ScanProgressDialog::OperationStatus::Running ||
            operation.status == ScanProgressDialog::OperationStatus::Paused) {
            count++;
        }
    }
    
    return count;
}

void OperationManager::processNextOperation() {
    if (m_isPaused) {
        return;
    }
    
    QMutexLocker locker(&m_operationMutex);
    
    // Find next operation to process
    for (auto& operation : m_operations) {
        if (operation.status == ScanProgressDialog::OperationStatus::Initializing) {
            m_currentOperationId = operation.operationId;
            operation.status = ScanProgressDialog::OperationStatus::Running;
            operation.startTime = QDateTime::currentDateTime();
            
            emit operationStarted(operation.operationId);
            emit progressInfoUpdated(getProgressInfo());
            
            // For file operations, delegate to FileOperationQueue
            if (isFileOperation(operation.type) && m_fileOperationQueue) {
                // File operations are handled by FileOperationQueue
                // The signals will update our progress
            } else {
                // For other operations, they need to be handled by external systems
                // This is a framework for custom operations
            }
            
            return;
        }
    }
    
    // No more operations to process
    m_isProcessing = false;
    m_progressTimer->stop();
    m_currentOperationId.clear();
}

void OperationManager::updateProgressMetrics() {
    emit progressInfoUpdated(getProgressInfo());
}

// File operation queue signal handlers
void OperationManager::onFileOperationQueued(const QString& operationId) {
    // File operations are automatically converted when they're queued
    // This is handled by the FileOperationQueue integration
}

void OperationManager::onFileOperationStarted(const QString& operationId) {
    // Update our operation status
    QMutexLocker locker(&m_operationMutex);
    for (auto& operation : m_operations) {
        if (operation.operationId == operationId) {
            operation.status = ScanProgressDialog::OperationStatus::Running;
            operation.startTime = QDateTime::currentDateTime();
            emit operationStarted(operationId);
            break;
        }
    }
}

void OperationManager::onFileOperationProgress(const QString& operationId, int filesProcessed, int totalFiles, qint64 bytesProcessed, qint64 totalBytes) {
    updateOperationProgress(operationId, filesProcessed, totalFiles, bytesProcessed, totalBytes);
}

void OperationManager::onFileOperationCompleted(const QString& operationId) {
    completeOperation(operationId);
}

void OperationManager::onFileOperationFailed(const QString& operationId, const QString& error) {
    QMutexLocker locker(&m_operationMutex);
    for (auto& operation : m_operations) {
        if (operation.operationId == operationId) {
            operation.status = ScanProgressDialog::OperationStatus::Error;
            operation.completedTime = QDateTime::currentDateTime();
            operation.lastError = error;
            operation.errorsEncountered++;
            
            emit operationFailed(operationId, error);
            
            // Move to next operation
            if (m_currentOperationId == operationId) {
                m_currentOperationId.clear();
                QTimer::singleShot(0, this, &OperationManager::processNextOperation);
            }
            
            emit progressInfoUpdated(getProgressInfo());
            break;
        }
    }
}

// Helper methods
OperationManager::Operation OperationManager::convertFileOperation(const FileOperationQueue::FileOperation& fileOp) const {
    Operation operation;
    operation.operationId = fileOp.operationId;
    operation.type = convertFileOperationType(fileOp.type);
    operation.description = QString("File %1 operation").arg(getOperationTypeName(operation.type));
    operation.queuedTime = fileOp.createdAt;
    operation.startTime = fileOp.startedAt;
    operation.completedTime = fileOp.completedAt;
    operation.filesTotal = fileOp.totalFiles;
    operation.filesProcessed = fileOp.filesProcessed;
    operation.bytesTotal = fileOp.totalBytes;
    operation.bytesProcessed = fileOp.bytesProcessed;
    
    // Convert status
    switch (fileOp.status) {
        case FileOperationQueue::OperationStatus::Pending:
            operation.status = ScanProgressDialog::OperationStatus::Initializing;
            break;
        case FileOperationQueue::OperationStatus::InProgress:
            operation.status = ScanProgressDialog::OperationStatus::Running;
            break;
        case FileOperationQueue::OperationStatus::Completed:
            operation.status = ScanProgressDialog::OperationStatus::Completed;
            break;
        case FileOperationQueue::OperationStatus::Failed:
            operation.status = ScanProgressDialog::OperationStatus::Error;
            operation.lastError = fileOp.errorMessage;
            operation.errorsEncountered = 1;
            break;
        case FileOperationQueue::OperationStatus::Cancelled:
            operation.status = ScanProgressDialog::OperationStatus::Cancelled;
            break;
    }
    
    return operation;
}

OperationManager::OperationType OperationManager::convertFileOperationType(FileOperationQueue::OperationType type) const {
    switch (type) {
        case FileOperationQueue::OperationType::Delete:
            return OperationType::FileDelete;
        case FileOperationQueue::OperationType::Move:
            return OperationType::FileMove;
        case FileOperationQueue::OperationType::Copy:
            return OperationType::FileCopy;
        case FileOperationQueue::OperationType::Trash:
            return OperationType::FileTrash;
        default:
            return OperationType::Custom;
    }
}

void OperationManager::calculatePerformanceMetrics(Operation& operation) {
    if (operation.startTime.isNull()) {
        return;
    }
    
    qint64 elapsedMs = operation.startTime.msecsTo(QDateTime::currentDateTime());
    if (elapsedMs > 0) {
        operation.filesPerSecond = (operation.filesProcessed * 1000.0) / elapsedMs;
        operation.bytesPerSecond = (operation.bytesProcessed * 1000.0) / elapsedMs;
        
        if (operation.filesProcessed > 0) {
            operation.averageFileSize = static_cast<double>(operation.bytesProcessed) / operation.filesProcessed;
        }
    }
}

QString OperationManager::generateOperationId() const {
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

QString OperationManager::getOperationTypeName(OperationType type) const {
    switch (type) {
        case OperationType::Scan:
            return "Scan";
        case OperationType::Hash:
            return "Hash";
        case OperationType::FileDelete:
            return "Delete";
        case OperationType::FileMove:
            return "Move";
        case OperationType::FileCopy:
            return "Copy";
        case OperationType::FileTrash:
            return "Trash";
        case OperationType::Validation:
            return "Validation";
        case OperationType::Custom:
            return "Custom";
        default:
            return "Unknown";
    }
}

bool OperationManager::isFileOperation(OperationType type) const {
    return type == OperationType::FileDelete ||
           type == OperationType::FileMove ||
           type == OperationType::FileCopy ||
           type == OperationType::FileTrash;
}