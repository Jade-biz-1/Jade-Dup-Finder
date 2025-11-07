#include "file_manager.h"
#include "safety_manager.h"
#include "logger.h"

#include <QFile>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QMutexLocker>
#include <QCoreApplication>
#include <QTimer>
#include <QUuid>
#include <algorithm>

// FileManager Implementation - Core file operations with safety integration

FileManager::FileManager(QObject* parent)
    : QObject(parent)
    , m_safetyManager(nullptr)
    , m_operationQueue(new FileOperationQueue(this))  // Task 29: Initialize FileOperationQueue
    , m_maxConcurrentOperations(4)
    , m_createBackupsByDefault(true)
    , m_defaultConflictResolution(ConflictResolution::Ask)
{
    Logger::instance()->debug(LogCategories::FILE_OPS, "FileManager created");
    
    // Initialize progress timer
    m_progressTimer = new QTimer(this);
    connect(m_progressTimer, &QTimer::timeout, this, &FileManager::processOperationQueue);
    m_progressTimer->start(100); // Process queue every 100ms
    
    // Register meta types for signal/slot system
    qRegisterMetaType<FileManager::FileOperation>("FileManager::FileOperation");
    qRegisterMetaType<FileManager::OperationResult>("FileManager::OperationResult");
    qRegisterMetaType<FileManager::OperationProgress>("FileManager::OperationProgress");
    qRegisterMetaType<FileManager::ConflictResolution>("FileManager::ConflictResolution");
    
    // Connect FileOperationQueue signals (Task 29)
    connect(m_operationQueue, &FileOperationQueue::operationStarted,
            this, [this](const QString& operationId) {
                // Create a basic FileOperation for the signal
                FileOperation operation;
                operation.operationId = operationId;
                emit operationStarted(operationId, operation);
            });
    
    connect(m_operationQueue, &FileOperationQueue::operationCompleted,
            this, [this](const QString& operationId, bool success, const QString& errorMessage) {
                // Create OperationResult from FileOperationQueue result
                OperationResult result;
                result.operationId = operationId;
                result.success = success;
                result.errorMessage = errorMessage;
                result.completed = QDateTime::currentDateTime();
                
                emit operationCompleted(result);
            });
    
    connect(m_operationQueue, &FileOperationQueue::operationCancelled,
            this, [this](const QString& operationId) {
                OperationResult result;
                result.operationId = operationId;
                result.success = false;
                result.errorMessage = "Operation was cancelled";
                result.completed = QDateTime::currentDateTime();
                
                emit operationCompleted(result);
            });
    
    LOG_DEBUG(LogCategories::FILE_OPS, "FileManager initialized with safe file operations");
}

FileManager::~FileManager()
{
    // Cancel all active operations
    QMutexLocker locker(&m_operationMutex);
    for (auto it = m_activeOperations.begin(); it != m_activeOperations.end(); ++it) {
        emit operationCancelled(it.key());
    }
    m_activeOperations.clear();
    
    LOG_DEBUG(LogCategories::FILE_OPS, "FileManager destroyed");
}

void FileManager::setSafetyManager(SafetyManager* safetyManager)
{
    m_safetyManager = safetyManager;
    LOG_INFO(LogCategories::FILE_OPS, QString("SafetyManager integration %1").arg(safetyManager ? "enabled" : "disabled"));
}

SafetyManager* FileManager::safetyManager() const
{
    return m_safetyManager;
}

QString FileManager::deleteFiles(const QStringList& filePaths, const QString& reason)
{
    FileOperation operation;
    operation.type = OperationType::Delete;
    operation.sourceFiles = filePaths;
    operation.reason = reason;
    operation.createBackup = m_createBackupsByDefault;
    
    return executeBatchOperation(operation);
}

QString FileManager::moveFiles(const QStringList& filePaths, const QString& targetDirectory, ConflictResolution conflictMode)
{
    FileOperation operation;
    operation.type = OperationType::Move;
    operation.sourceFiles = filePaths;
    operation.targetPath = targetDirectory;
    operation.conflictMode = conflictMode;
    operation.createBackup = m_createBackupsByDefault;
    
    return executeBatchOperation(operation);
}

QString FileManager::copyFiles(const QStringList& filePaths, const QString& targetDirectory, ConflictResolution conflictMode)
{
    FileOperation operation;
    operation.type = OperationType::Copy;
    operation.sourceFiles = filePaths;
    operation.targetPath = targetDirectory;
    operation.conflictMode = conflictMode;
    operation.createBackup = false; // Copies don't need backup by default
    
    return executeBatchOperation(operation);
}

QString FileManager::restoreFiles(const QStringList& backupPaths, const QString& targetDirectory)
{
    FileOperation operation;
    operation.type = OperationType::Restore;
    operation.sourceFiles = backupPaths;
    operation.targetPath = targetDirectory;
    operation.conflictMode = ConflictResolution::Ask;
    operation.createBackup = false; // Restores don't need backup
    
    return executeBatchOperation(operation);
}

bool FileManager::deleteFile(const QString& filePath, bool createBackup)
{
    if (!QFile::exists(filePath)) {
        LOG_WARNING(LogCategories::FILE_OPS, QString("File does not exist: %1").arg(filePath));
        return false;
    }
    
    QString operationId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    
    // Create backup if requested and SafetyManager available
    if (createBackup && m_safetyManager) {
        // Note: Backup creation is handled in performDeleteOperation
        LOG_DEBUG(LogCategories::FILE_OPS, QString("Creating backup for: %1").arg(filePath));
    }
    
    return performDelete(filePath, operationId);
}

bool FileManager::moveFile(const QString& sourceFile, const QString& targetFile, ConflictResolution conflictMode)
{
    if (!QFile::exists(sourceFile)) {
        LOG_WARNING(LogCategories::FILE_OPS, QString("Source file does not exist: %1").arg(sourceFile));
        return false;
    }
    
    QString operationId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    
    // Handle conflicts
    if (QFile::exists(targetFile)) {
        ConflictResolution resolution = resolveConflict(sourceFile, targetFile, conflictMode);
        if (resolution == ConflictResolution::Skip) {
            LOG_INFO(LogCategories::FILE_OPS, QString("Skipping file due to conflict: %1").arg(sourceFile));
            return false;
        }
    }
    
    return performMove(sourceFile, targetFile, operationId);
}

bool FileManager::copyFile(const QString& sourceFile, const QString& targetFile, ConflictResolution conflictMode)
{
    if (!QFile::exists(sourceFile)) {
        LOG_WARNING(LogCategories::FILE_OPS, QString("Source file does not exist: %1").arg(sourceFile));
        return false;
    }
    
    QString operationId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    
    // Handle conflicts
    if (QFile::exists(targetFile)) {
        ConflictResolution resolution = resolveConflict(sourceFile, targetFile, conflictMode);
        if (resolution == ConflictResolution::Skip) {
            LOG_INFO(LogCategories::FILE_OPS, QString("Skipping file due to conflict: %1").arg(sourceFile));
            return false;
        }
    }
    
    return performCopy(sourceFile, targetFile, operationId);
}

QString FileManager::executeBatchOperation(const FileOperation& operation)
{
    QString error;
    if (!validateOperation(operation, error)) {
        Logger::instance()->warning(LogCategories::FILE_OPS, QString("Invalid operation: %1").arg(error));
        emit operationError(operation.operationId, error);
        return QString();
    }
    
    // Add to active operations
    {
        QMutexLocker locker(&m_operationMutex);
        m_activeOperations[operation.operationId] = operation;
        
        // Initialize progress tracking
        OperationProgress progress;
        progress.operationId = operation.operationId;
        progress.totalFiles = static_cast<int>(operation.sourceFiles.size());
        m_operationProgress[operation.operationId] = progress;
    }
    
    // Add to processing queue
    m_legacyOperationQueue.enqueue(operation);
    
    emit operationStarted(operation.operationId, operation);
    Logger::instance()->info(LogCategories::FILE_OPS, QString("Started operation %1 with %2 files").arg(operation.operationId).arg(operation.sourceFiles.size()));
    
    return operation.operationId;
}

void FileManager::cancelOperation(const QString& operationId)
{
    QMutexLocker locker(&m_operationMutex);
    
    if (m_activeOperations.contains(operationId)) {
        m_activeOperations.remove(operationId);
        m_operationProgress.remove(operationId);
        
        // Remove from queue if not yet processed
        QQueue<FileOperation> newQueue;
        while (!m_legacyOperationQueue.isEmpty()) {
            FileOperation op = m_legacyOperationQueue.dequeue();
            if (op.operationId != operationId) {
                newQueue.enqueue(op);
            }
        }
        m_legacyOperationQueue = newQueue;
        
        emit operationCancelled(operationId);
        LOG_INFO(LogCategories::FILE_OPS, QString("Cancelled operation: %1").arg(operationId));
    }
}

bool FileManager::isOperationInProgress(const QString& operationId) const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_operationMutex));
    return m_activeOperations.contains(operationId);
}

FileManager::OperationResult FileManager::getOperationResult(const QString& operationId) const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_operationMutex));
    return m_operationResults.value(operationId, OperationResult());
}

QList<FileManager::OperationResult> FileManager::getRecentOperations(int maxResults) const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_operationMutex));
    
    QList<OperationResult> results;
    for (auto it = m_operationResults.begin(); it != m_operationResults.end(); ++it) {
        results.append(it.value());
    }
    
    // Sort by completion time (most recent first)
    std::sort(results.begin(), results.end(), [](const OperationResult& a, const OperationResult& b) {
        return a.completed > b.completed;
    });
    
    if (results.size() > maxResults) {
        results = results.mid(0, maxResults);
    }
    
    return results;
}

void FileManager::processOperationQueue()
{
    if (m_legacyOperationQueue.isEmpty()) {
        return;
    }
    
    // Limit concurrent operations
    int activeCount = 0;
    {
        QMutexLocker locker(&m_operationMutex);
        activeCount = static_cast<int>(m_activeOperations.size());
    }
    
    if (activeCount >= m_maxConcurrentOperations) {
        return; // Too many operations running
    }
    
    FileOperation operation = m_legacyOperationQueue.dequeue();
    
    // Process the operation
    OperationResult result;
    result.operationId = operation.operationId;
    result.success = true;
    
    for (int i = 0; i < operation.sourceFiles.size(); ++i) {
        const QString& sourceFile = operation.sourceFiles[i];
        
        // Update progress
        updateProgress(operation.operationId, i, QFileInfo(sourceFile).size(), sourceFile);
        
        bool fileSuccess = false;
        
        switch (operation.type) {
            case OperationType::Delete:
                fileSuccess = performDelete(sourceFile, operation.operationId);
                break;
                
            case OperationType::Move: {
                QString targetFile = QDir(operation.targetPath).filePath(QFileInfo(sourceFile).fileName());
                fileSuccess = performMove(sourceFile, targetFile, operation.operationId);
                break;
            }
            
            case OperationType::Copy: {
                QString targetFile = QDir(operation.targetPath).filePath(QFileInfo(sourceFile).fileName());
                fileSuccess = performCopy(sourceFile, targetFile, operation.operationId);
                break;
            }
            
            case OperationType::Restore:
                fileSuccess = performRestore(sourceFile, operation.operationId);
                break;
                
            case OperationType::CreateBackup:
                fileSuccess = performBackupCreation(sourceFile, operation.operationId);
                break;
        }
        
        if (fileSuccess) {
            result.processedFiles.append(sourceFile);
            result.totalSize += QFileInfo(sourceFile).size();
        } else {
            result.failedFiles.append(sourceFile);
            result.success = false;
        }
        
        // Process events to keep UI responsive
        QCoreApplication::processEvents();
    }

    // Final progress update
    updateProgress(operation.operationId, static_cast<int>(operation.sourceFiles.size()), result.totalSize, QString());
    
    // Store result and emit completion
    {
        QMutexLocker locker(&m_operationMutex);
        m_operationResults[operation.operationId] = result;
        m_activeOperations.remove(operation.operationId);
        m_operationProgress.remove(operation.operationId);
    }
    
    emit operationCompleted(result);
    LOG_INFO(LogCategories::FILE_OPS, QString("Completed operation %1 - Success: %2, Failed: %3")
             .arg(operation.operationId)
             .arg(result.processedFiles.size())
             .arg(result.failedFiles.size()));
}

bool FileManager::performDelete(const QString& filePath, const QString& operationId)
{
    Q_UNUSED(operationId)
    
    // Check if file is system protected
    if (isSystemProtectedFile(filePath)) {
        qWarning() << "FileManager: Attempted to delete protected file:" << filePath;
        return false;
    }
    
    // Check with SafetyManager if file is protected
    if (m_safetyManager && m_safetyManager->isProtected(filePath, SafetyManager::OperationType::Delete)) {
        qWarning() << "FileManager: File is protected by SafetyManager:" << filePath;
        return false;
    }
    
    QString safetyOperationId;
    QString backupPath;
    
    // Register operation with SafetyManager and create backup before deletion
    if (m_safetyManager) {
        // Register the delete operation
        safetyOperationId = m_safetyManager->registerOperation(
            SafetyManager::OperationType::Delete, 
            filePath, 
            QString(), // No target for delete
            QString("File deleted via FileManager: %1").arg(operationId)
        );
        
        if (m_createBackupsByDefault) {
            LOG_DEBUG(LogCategories::FILE_OPS, QString("Creating backup before delete: %1").arg(filePath));
            backupPath = m_safetyManager->createBackup(filePath, SafetyManager::BackupStrategy::CentralLocation);
            if (backupPath.isEmpty()) {
                LOG_WARNING(LogCategories::FILE_OPS, QString("Failed to create backup for: %1").arg(filePath));
                LOG_WARNING(LogCategories::FILE_OPS, "Aborting delete operation for safety");
                // Mark operation as failed
                if (!safetyOperationId.isEmpty()) {
                    m_safetyManager->finalizeOperation(safetyOperationId, false);
                }
                return false;
            }
            LOG_INFO(LogCategories::FILE_OPS, QString("Backup created successfully at: %1").arg(backupPath));
        }
    }
    
    // Perform the deletion
    QFile file(filePath);
    if (!file.remove()) {
        qWarning() << "FileManager: Failed to delete file:" << filePath << file.errorString();
        // Mark operation as failed in SafetyManager
        if (m_safetyManager && !safetyOperationId.isEmpty()) {
            m_safetyManager->finalizeOperation(safetyOperationId, false);
        }
        return false;
    }
    
    LOG_INFO(LogCategories::FILE_OPS, QString("Deleted file: %1").arg(filePath));
    
    // Mark operation as successful in SafetyManager
    if (m_safetyManager && !safetyOperationId.isEmpty()) {
        m_safetyManager->finalizeOperation(safetyOperationId, true, backupPath);
        LOG_DEBUG(LogCategories::FILE_OPS, "Operation registered successfully with SafetyManager");
    }
    
    // Update SafetyManager undo history
    if (m_safetyManager) {
        // The backup was already created, so the undo history is already updated
        LOG_DEBUG(LogCategories::FILE_OPS, "Undo history updated in SafetyManager");
    }
    
    return true;
}

bool FileManager::performMove(const QString& sourceFile, const QString& targetFile, const QString& operationId)
{
    Q_UNUSED(operationId)
    
    // Check if file is system protected
    if (isSystemProtectedFile(sourceFile)) {
        qWarning() << "FileManager: Attempted to move protected file:" << sourceFile;
        return false;
    }
    
    // Check with SafetyManager if file is protected
    if (m_safetyManager && m_safetyManager->isProtected(sourceFile, SafetyManager::OperationType::Move)) {
        qWarning() << "FileManager: File is protected by SafetyManager:" << sourceFile;
        return false;
    }
    
    // Register operation with SafetyManager and create backup before moving
    if (m_safetyManager) {
        // Register the move operation
        QString safetyOperationId = m_safetyManager->registerOperation(
            SafetyManager::OperationType::Move, 
            sourceFile, 
            targetFile,
            QString("File moved via FileManager: %1").arg(operationId)
        );
        
        if (m_createBackupsByDefault) {
            LOG_DEBUG(LogCategories::FILE_OPS, QString("Creating backup before move: %1").arg(sourceFile));
            QString backupPath = m_safetyManager->createBackup(sourceFile, SafetyManager::BackupStrategy::CentralLocation);
            if (backupPath.isEmpty()) {
                LOG_WARNING(LogCategories::FILE_OPS, QString("Failed to create backup for: %1").arg(sourceFile));
                LOG_WARNING(LogCategories::FILE_OPS, "Aborting move operation for safety");
                return false;
            }
            LOG_INFO(LogCategories::FILE_OPS, QString("Backup created successfully at: %1").arg(backupPath));
            
            // Finalize the operation with backup path
            m_safetyManager->finalizeOperation(safetyOperationId, true, backupPath);
        } else {
            // Finalize operation without backup
            m_safetyManager->finalizeOperation(safetyOperationId, true);
        }
    }
    
    // Ensure target directory exists
    QFileInfo targetInfo(targetFile);
    QDir targetDir = targetInfo.dir();
    if (!targetDir.exists()) {
        if (!targetDir.mkpath(".")) {
            qWarning() << "FileManager: Failed to create target directory:" << targetDir.path();
            return false;
        }
    }
    
    // Perform the move
    QFile file(sourceFile);
    if (!file.rename(targetFile)) {
        qWarning() << "FileManager: Failed to move file:" << sourceFile << "to" << targetFile << file.errorString();
        return false;
    }
    
    LOG_INFO(LogCategories::FILE_OPS, QString("Moved file: %1 -> %2").arg(sourceFile, targetFile));
    
    // Update SafetyManager undo history
    if (m_safetyManager) {
        // The backup was already created, so the undo history is already updated
        LOG_DEBUG(LogCategories::FILE_OPS, "Undo history updated in SafetyManager");
    }
    
    return true;
}

bool FileManager::performCopy(const QString& sourceFile, const QString& targetFile, const QString& operationId)
{
    Q_UNUSED(operationId)
    
    // Ensure target directory exists
    QFileInfo targetInfo(targetFile);
    QDir targetDir = targetInfo.dir();
    if (!targetDir.exists()) {
        if (!targetDir.mkpath(".")) {
            qWarning() << "FileManager: Failed to create target directory:" << targetDir.path();
            return false;
        }
    }
    
    QFile file(sourceFile);
    if (!file.copy(targetFile)) {
        qWarning() << "FileManager: Failed to copy file:" << sourceFile << "to" << targetFile << file.errorString();
        return false;
    }
    
    LOG_INFO(LogCategories::FILE_OPS, QString("Copied file: %1 -> %2").arg(sourceFile, targetFile));
    return true;
}

bool FileManager::performRestore(const QString& backupPath, const QString& operationId)
{
    LOG_INFO(LogCategories::FILE_OPS, QString("Restoring from backup: %1").arg(backupPath));
    
    // Verify backup exists
    if (!QFile::exists(backupPath)) {
        qWarning() << "FileManager: Backup file does not exist:" << backupPath;
        return false;
    }
    
    // Get backup information from SafetyManager
    if (!m_safetyManager) {
        qWarning() << "FileManager: SafetyManager not available for restore";
        return false;
    }
    
    // Get target path from operation if available
    QString targetPath;
    {
        QMutexLocker locker(&m_operationMutex);
        if (m_activeOperations.contains(operationId)) {
            const FileOperation& operation = m_activeOperations[operationId];
            
            // If targetPath is a directory, construct full path with original filename
            if (!operation.targetPath.isEmpty()) {
                QFileInfo targetInfo(operation.targetPath);
                if (targetInfo.isDir() || operation.targetPath.endsWith('/')) {
                    // Extract original filename from backup path or SafetyManager
                    QString originalPath = m_safetyManager->getOriginalPathForBackup(backupPath);
                    if (!originalPath.isEmpty()) {
                        QFileInfo originalInfo(originalPath);
                        targetPath = QDir(operation.targetPath).filePath(originalInfo.fileName());
                    } else {
                        // Fallback: extract filename from backup name
                        QString backupFilename = QFileInfo(backupPath).fileName();
                        // Remove timestamp and .backup extension
                        // Format: filename.YYYYMMDD_HHMMSS.backup
                        qsizetype lastDot = backupFilename.lastIndexOf('.');
                        if (lastDot > 0) {
                            backupFilename = backupFilename.left(lastDot); // Remove .backup
                            qsizetype secondLastDot = backupFilename.lastIndexOf('.');
                            if (secondLastDot > 0) {
                                backupFilename = backupFilename.left(secondLastDot); // Remove timestamp
                            }
                        }
                        targetPath = QDir(operation.targetPath).filePath(backupFilename);
                    }
                } else {
                    // targetPath is a full file path
                    targetPath = operation.targetPath;
                }
            }
        }
    }
    
    // Use SafetyManager's restore functionality
    bool success = m_safetyManager->restoreFromBackup(backupPath, targetPath);
    
    if (success) {
        LOG_INFO(LogCategories::FILE_OPS, QString("Successfully restored from backup: %1 to: %2").arg(backupPath, targetPath));
    } else {
        LOG_ERROR(LogCategories::FILE_OPS, QString("Failed to restore from backup: %1").arg(backupPath));
    }
    
    return success;
}

bool FileManager::performBackupCreation(const QString& sourceFile, const QString& operationId)
{
    Q_UNUSED(operationId)
    
    LOG_DEBUG(LogCategories::FILE_OPS, QString("Creating backup for: %1").arg(sourceFile));
    
    // Verify source file exists
    if (!QFile::exists(sourceFile)) {
        qWarning() << "FileManager: Source file does not exist:" << sourceFile;
        return false;
    }
    
    // Check if SafetyManager is available
    if (!m_safetyManager) {
        qWarning() << "FileManager: SafetyManager not available for backup creation";
        return false;
    }
    
    // Create backup using SafetyManager
    QString backupPath = m_safetyManager->createBackup(sourceFile, SafetyManager::BackupStrategy::CentralLocation);
    
    if (backupPath.isEmpty()) {
        qWarning() << "FileManager: Failed to create backup for:" << sourceFile;
        return false;
    }
    
    LOG_INFO(LogCategories::FILE_OPS, QString("Backup created successfully at: %1").arg(backupPath));
    return true;
}

FileManager::ConflictResolution FileManager::resolveConflict(const QString& sourceFile, const QString& targetFile, ConflictResolution mode)
{
    switch (mode) {
        case ConflictResolution::Skip:
            return ConflictResolution::Skip;
            
        case ConflictResolution::Overwrite:
            return ConflictResolution::Overwrite;
            
        case ConflictResolution::Rename:
            // TODO: Generate unique name and update target
            return ConflictResolution::Rename;
            
        case ConflictResolution::Ask:
            // Emit signal for UI to handle
            emit conflictResolutionRequired(sourceFile, targetFile, ConflictResolution::Skip);
            return ConflictResolution::Skip; // Default to skip for now
    }
    
    return ConflictResolution::Skip;
}

QString FileManager::generateConflictFreeName(const QString& targetFile)
{
    QFileInfo info(targetFile);
    QString baseName = info.completeBaseName();
    QString suffix = info.suffix();
    QString dir = info.absolutePath();
    
    int counter = 1;
    QString newName;
    
    do {
        if (suffix.isEmpty()) {
            newName = QDir(dir).filePath(QString("%1_%2").arg(baseName).arg(counter));
        } else {
            newName = QDir(dir).filePath(QString("%1_%2.%3").arg(baseName).arg(counter).arg(suffix));
        }
        counter++;
    } while (QFile::exists(newName) && counter < 1000);
    
    return newName;
}

void FileManager::updateProgress(const QString& operationId, int filesProcessed, qint64 bytesProcessed, const QString& currentFile)
{
    QMutexLocker locker(&m_operationMutex);
    
    if (!m_operationProgress.contains(operationId)) {
        return;
    }
    
    OperationProgress& progress = m_operationProgress[operationId];
    progress.filesProcessed = filesProcessed;
    progress.bytesProcessed += bytesProcessed;
    progress.currentFile = currentFile;
    
    if (progress.totalFiles > 0) {
        progress.percentComplete = (static_cast<double>(progress.filesProcessed) / progress.totalFiles) * 100.0;
    }
    
    // Emit progress update
    locker.unlock();
    emit operationProgress(progress);
}

void FileManager::emitProgress(const QString& operationId)
{
    QMutexLocker locker(&m_operationMutex);
    
    if (m_operationProgress.contains(operationId)) {
        emit operationProgress(m_operationProgress[operationId]);
    }
}

bool FileManager::validateOperation(const FileOperation& operation, QString& error)
{
    if (operation.sourceFiles.isEmpty()) {
        error = "No source files specified";
        return false;
    }
    
    // Check if source files exist (except for restore operations which check backup existence separately)
    if (operation.type != OperationType::Restore) {
        for (const QString& filePath : operation.sourceFiles) {
            if (!QFile::exists(filePath)) {
                error = QString("Source file does not exist: %1").arg(filePath);
                return false;
            }
        }
    }
    
    // Check target directory for move/copy operations
    if (operation.type == OperationType::Move || operation.type == OperationType::Copy) {
        if (operation.targetPath.isEmpty()) {
            error = "No target directory specified";
            return false;
        }
        
        QDir targetDir(operation.targetPath);
        if (!targetDir.exists() && !targetDir.mkpath(".")) {
            error = QString("Cannot create target directory: %1").arg(operation.targetPath);
            return false;
        }
    }
    
    return true;
}

bool FileManager::isSystemProtectedFile(const QString& filePath)
{
    // Basic protection for common system paths
    if (filePath.startsWith("/bin/") || 
        filePath.startsWith("/sbin/") ||
        filePath.startsWith("/usr/bin/") ||
        filePath.startsWith("/usr/sbin/") ||
        filePath.startsWith("/system/") ||
        filePath.startsWith("/boot/")) {
        return true;
    }
    
    // Note: SafetyManager protection is checked in operation methods (performDeleteOperation, etc.)
    return false;
}

// Static utility functions

bool FileManager::validateFilePath(const QString& filePath)
{
    if (filePath.isEmpty()) {
        return false;
    }
    
    // Basic validation - check for invalid characters
    QChar invalidChars[] = {'<', '>', ':', '\"', '|', '?', '*'};
    for (const QChar& ch : invalidChars) {
        if (filePath.contains(ch)) {
            return false;
        }
    }
    
    return true;
}

QString FileManager::generateUniqueFileName(const QString& basePath)
{
    QFileInfo info(basePath);
    
    if (!QFile::exists(basePath)) {
        return basePath; // Already unique
    }
    
    QString dir = info.absolutePath();
    QString baseName = info.completeBaseName();
    QString suffix = info.suffix();
    
    int counter = 1;
    QString uniquePath;
    
    do {
        if (suffix.isEmpty()) {
            uniquePath = QDir(dir).filePath(QString("%1_%2").arg(baseName).arg(counter));
        } else {
            uniquePath = QDir(dir).filePath(QString("%1_%2.%3").arg(baseName).arg(counter).arg(suffix));
        }
        counter++;
    } while (QFile::exists(uniquePath) && counter < 10000);
    
    return uniquePath;
}

qint64 FileManager::calculateDirectorySize(const QString& directoryPath)
{
    QDir dir(directoryPath);
    qint64 totalSize = 0;
    
    QFileInfoList fileList = dir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot, QDir::DirsFirst);
    
    for (const QFileInfo& fileInfo : fileList) {
        if (fileInfo.isDir()) {
            totalSize += calculateDirectorySize(fileInfo.absoluteFilePath());
        } else {
            totalSize += fileInfo.size();
        }
    }
    
    return totalSize;
}

bool FileManager::hasWritePermission(const QString& directoryPath)
{
    QFileInfo dirInfo(directoryPath);
    return dirInfo.isWritable();
}

// FileOperationQueue integration methods (Task 29)

FileOperationQueue* FileManager::operationQueue() const
{
    return m_operationQueue;
}

QString FileManager::queueDeleteOperation(const QStringList& files)
{
    if (!m_operationQueue) {
        return QString();
    }
    
    // Queue the operation using FileOperationQueue
    QString operationId = m_operationQueue->queueDeleteOperation(files);
    
    // Connect to progress signals to emit FileManager's progress signals
    connect(m_operationQueue, &FileOperationQueue::operationProgress,
            this, [this](const QString& opId, int filesProcessed, int totalFiles, qint64 bytesProcessed, qint64 totalBytes) {
                // Convert to FileManager's progress format
                OperationProgress progress;
                progress.operationId = opId;
                progress.filesProcessed = filesProcessed;
                progress.totalFiles = totalFiles;
                progress.bytesProcessed = bytesProcessed;
                progress.totalBytes = totalBytes;
                progress.percentComplete = totalFiles > 0 ? (double)filesProcessed / totalFiles * 100.0 : 0.0;
                
                emit operationProgress(progress);
            });
    
    return operationId;
}

QString FileManager::queueMoveOperation(const QStringList& files, const QString& destination)
{
    if (!m_operationQueue) {
        return QString();
    }
    
    return m_operationQueue->queueMoveOperation(files, destination);
}

QString FileManager::queueCopyOperation(const QStringList& files, const QString& destination)
{
    if (!m_operationQueue) {
        return QString();
    }
    
    return m_operationQueue->queueCopyOperation(files, destination);
}