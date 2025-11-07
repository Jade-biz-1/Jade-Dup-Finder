#include "safety_manager.h"
#include "logger.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QFile>
#include <QMutexLocker>
#include <QStandardPaths>
#include <QUuid>
#include <QRegularExpression>
#include <QDirIterator>
#include <algorithm>

// SafetyManager Implementation - Comprehensive file operation safety

SafetyManager::SafetyManager(QObject* parent)
    : QObject(parent)
    , m_safetyLevel(SafetyLevel::Standard)
    , m_defaultBackupStrategy(BackupStrategy::CentralLocation)
    , m_maxBackupAge(DEFAULT_MAX_BACKUP_AGE)
    , m_maxUndoOperations(DEFAULT_MAX_UNDO_OPERATIONS)
{
    Logger::instance()->debug(LogCategories::SAFETY, "SafetyManager created");
    
    // Initialize backup directory
    m_backupDirectory = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/backups";
    
    // Setup maintenance timers
    m_maintenanceTimer = new QTimer(this);
    connect(m_maintenanceTimer, &QTimer::timeout, this, &SafetyManager::performPeriodicMaintenance);
    
    m_cleanupTimer = new QTimer(this);
    connect(m_cleanupTimer, &QTimer::timeout, this, &SafetyManager::cleanupExpiredOperations);
    
    // Initialize protection rules
    initializeDefaultProtectionRules();
    
    // Register metatypes for signal/slot system
    qRegisterMetaType<SafetyManager::SafetyLevel>("SafetyManager::SafetyLevel");
    qRegisterMetaType<SafetyManager::BackupStrategy>("SafetyManager::BackupStrategy");
    qRegisterMetaType<SafetyManager::ProtectionLevel>("SafetyManager::ProtectionLevel");
    qRegisterMetaType<SafetyManager::OperationType>("SafetyManager::OperationType");
    qRegisterMetaType<SafetyManager::SafetyOperation>("SafetyManager::SafetyOperation");
    qRegisterMetaType<SafetyManager::ProtectionEntry>("SafetyManager::ProtectionEntry");
    qRegisterMetaType<SafetyManager::BackupVerification>("SafetyManager::BackupVerification");
    
    LOG_INFO(LogCategories::SAFETY, QString("SafetyManager initialized - Level: %1, Backup directory: %2")
             .arg(static_cast<int>(m_safetyLevel))
             .arg(m_backupDirectory));
}

SafetyManager::~SafetyManager()
{
    stopAutoMaintenance();
    LOG_DEBUG(LogCategories::SAFETY, "SafetyManager destroyed");
}

// Configuration methods

void SafetyManager::setSafetyLevel(SafetyLevel level)
{
    m_safetyLevel = level;
    LOG_INFO(LogCategories::SAFETY, QString("Safety level set to %1").arg(static_cast<int>(level)));
}

SafetyManager::SafetyLevel SafetyManager::safetyLevel() const
{
    return m_safetyLevel;
}

void SafetyManager::setDefaultBackupStrategy(BackupStrategy strategy)
{
    m_defaultBackupStrategy = strategy;
    LOG_DEBUG(LogCategories::SAFETY, QString("Default backup strategy set to %1").arg(static_cast<int>(strategy)));
}

SafetyManager::BackupStrategy SafetyManager::defaultBackupStrategy() const
{
    return m_defaultBackupStrategy;
}

void SafetyManager::setBackupDirectory(const QString& directory)
{
    m_backupDirectory = directory;
    ensureBackupDirectoryExists();
    LOG_INFO(LogCategories::SAFETY, QString("Backup directory set to %1").arg(directory));
}

QString SafetyManager::backupDirectory() const
{
    return m_backupDirectory;
}

void SafetyManager::setMaxBackupAge(int days)
{
    m_maxBackupAge = qMax(1, days);
    LOG_DEBUG(LogCategories::SAFETY, QString("Max backup age set to %1 days").arg(m_maxBackupAge));
}

int SafetyManager::maxBackupAge() const
{
    return m_maxBackupAge;
}

void SafetyManager::setMaxUndoOperations(int maxOperations)
{
    m_maxUndoOperations = qMax(10, maxOperations);
    LOG_DEBUG(LogCategories::SAFETY, QString("Max undo operations set to %1").arg(m_maxUndoOperations));
}

int SafetyManager::maxUndoOperations() const
{
    return m_maxUndoOperations;
}

// Backup operations

QString SafetyManager::createBackup(const QString& filePath, BackupStrategy strategy)
{
    if (!QFile::exists(filePath)) {
        Logger::instance()->warning(LogCategories::SAFETY, QString("Cannot backup non-existent file: %1").arg(filePath));
        return QString();
    }
    
    if (strategy == BackupStrategy::None) {
        Logger::instance()->debug(LogCategories::SAFETY, QString("Backup creation disabled for: %1").arg(filePath));
        return QString();
    }
    
    QString backupPath = createBackupPath(filePath, strategy);
    if (backupPath.isEmpty()) {
        Logger::instance()->warning(LogCategories::SAFETY, QString("Failed to generate backup path for: %1").arg(filePath));
        return QString();
    }
    
    if (performBackupOperation(filePath, backupPath)) {
        Logger::instance()->info(LogCategories::SAFETY, QString("Created backup: %1 -> %2").arg(filePath, backupPath));
        emit backupCreated(filePath, backupPath);
        return backupPath;
    }
    
    qWarning() << "SafetyManager: Failed to create backup for:" << filePath;
    return QString();
}

bool SafetyManager::restoreFromBackup(const QString& backupPath, const QString& targetPath)
{
    if (!QFile::exists(backupPath)) {
        qWarning() << "SafetyManager: Backup file does not exist:" << backupPath;
        return false;
    }
    
    QString actualTargetPath = targetPath;
    
    // If no target path specified, look up original path from undo history
    if (actualTargetPath.isEmpty()) {
        actualTargetPath = getOriginalPathForBackup(backupPath);
        
        // If still empty, try to extract from backup filename (fallback)
        if (actualTargetPath.isEmpty()) {
            actualTargetPath = backupPath.left(backupPath.lastIndexOf(".backup"));
        }
    }
        
    if (performRestoreOperation(backupPath, actualTargetPath)) {
        LOG_INFO(LogCategories::SAFETY, QString("Restored from backup: %1 -> %2").arg(backupPath, actualTargetPath));
        
        // Remove the operation from history since it's been restored
        removeOperationsForBackup(backupPath);
        
        emit backupRestored(backupPath, actualTargetPath);
        return true;
    }
    
    qWarning() << "SafetyManager: Failed to restore from backup:" << backupPath;
    return false;
}

SafetyManager::BackupVerification SafetyManager::verifyBackup(const QString& backupPath, const QString& originalPath)
{
    BackupVerification verification;
    verification.backupPath = backupPath;
    verification.originalPath = originalPath;
    
    if (!QFile::exists(backupPath)) {
        verification.errorMessage = "Backup file does not exist";
        return verification;
    }
    
    if (!originalPath.isEmpty() && QFile::exists(originalPath)) {
        QFileInfo backupInfo(backupPath);
        QFileInfo originalInfo(originalPath);
        
        verification.backupSize = backupInfo.size();
        verification.originalSize = originalInfo.size();
        
        if (verification.backupSize != verification.originalSize) {
            verification.errorMessage = "File sizes do not match";
            return verification;
        }
        
        verification.backupChecksum = calculateFileChecksum(backupPath);
        verification.originalChecksum = calculateFileChecksum(originalPath);
        
        if (verification.backupChecksum != verification.originalChecksum) {
            verification.errorMessage = "File checksums do not match";
            return verification;
        }
    }
    
    verification.isValid = true;
    return verification;
}

QStringList SafetyManager::listBackups(const QString& originalFilePath) const
{
    QStringList backups;
    
    if (!ensureBackupDirectoryExists()) {
        return backups;
    }
    
    QFileInfo fileInfo(originalFilePath);
    QString baseName = fileInfo.fileName();
    
    QDir backupDir(m_backupDirectory);
    QStringList filters;
    filters << baseName + ".*backup*";
    
    QFileInfoList backupFiles = backupDir.entryInfoList(filters, QDir::Files, QDir::Time);
    
    for (const QFileInfo& info : backupFiles) {
        backups.append(info.absoluteFilePath());
    }
    
    return backups;
}

bool SafetyManager::removeBackup(const QString& backupPath)
{
    if (!QFile::exists(backupPath)) {
        LOG_DEBUG(LogCategories::SAFETY, QString("Backup file does not exist: %1").arg(backupPath));
        return true; // Already removed
    }
    
    QFile file(backupPath);
    if (file.remove()) {
        LOG_DEBUG(LogCategories::SAFETY, QString("Removed backup: %1").arg(backupPath));
        return true;
    }
    
    qWarning() << "SafetyManager: Failed to remove backup:" << backupPath << file.errorString();
    return false;
}

void SafetyManager::cleanupOldBackups()
{
    QDateTime cutoffDate = QDateTime::currentDateTime().addDays(-m_maxBackupAge);
    
    if (!ensureBackupDirectoryExists()) {
        return;
    }
    
    int removedCount = 0;
    qint64 freedSpace = 0;
    
    QDirIterator iterator(m_backupDirectory, QStringList() << "*.backup*", QDir::Files, QDirIterator::Subdirectories);
    
    while (iterator.hasNext()) {
        iterator.next();
        QFileInfo fileInfo = iterator.fileInfo();
        
        if (fileInfo.lastModified() < cutoffDate) {
            qint64 fileSize = fileInfo.size();
            if (QFile::remove(fileInfo.absoluteFilePath())) {
                removedCount++;
                freedSpace += fileSize;
                LOG_DEBUG(LogCategories::SAFETY, QString("Removed old backup: %1").arg(fileInfo.fileName()));
            }
        }
    }
    
    if (removedCount > 0) {
        LOG_INFO(LogCategories::SAFETY, QString("Cleaned up %1 old backups, freed %2").arg(removedCount).arg(formatFileSize(freedSpace)));
        emit backupCleanupCompleted(removedCount, freedSpace);
    }
}

// Protection system

void SafetyManager::addProtectionRule(const QString& pattern, ProtectionLevel level, const QString& description, bool isRegex)
{
    QMutexLocker locker(&m_protectionMutex);
    
    ProtectionEntry entry;
    entry.pattern = pattern;
    entry.level = level;
    entry.description = description;
    entry.isRegex = isRegex;
    
    // Remove any existing rule with the same pattern
    for (auto it = m_protectionRules.begin(); it != m_protectionRules.end(); ++it) {
        if (it->pattern == pattern) {
            m_protectionRules.erase(it);
            break;
        }
    }
    
    m_protectionRules.append(entry);
    LOG_DEBUG(LogCategories::SAFETY, QString("Added protection rule: %1, Level: %2").arg(pattern).arg(static_cast<int>(level)));
}

void SafetyManager::removeProtectionRule(const QString& pattern)
{
    QMutexLocker locker(&m_protectionMutex);
    
    for (auto it = m_protectionRules.begin(); it != m_protectionRules.end(); ++it) {
        if (it->pattern == pattern) {
            m_protectionRules.erase(it);
            LOG_DEBUG(LogCategories::SAFETY, QString("Removed protection rule: %1").arg(pattern));
            return;
        }
    }
}

SafetyManager::ProtectionLevel SafetyManager::getProtectionLevel(const QString& filePath) const
{
    QMutexLocker locker(&m_protectionMutex);
    
    ProtectionLevel highestLevel = ProtectionLevel::None;
    
    for (const ProtectionEntry& entry : m_protectionRules) {
        if (matchesProtectionPattern(filePath, entry)) {
            if (entry.level > highestLevel) {
                highestLevel = entry.level;
            }
        }
    }
    
    return highestLevel;
}

bool SafetyManager::isProtected(const QString& filePath, OperationType operation) const
{
    ProtectionLevel level = getProtectionLevel(filePath);
    
    switch (level) {
        case ProtectionLevel::None:
            return false;
            
        case ProtectionLevel::ReadOnly:
            return (operation == OperationType::Delete || operation == OperationType::Modify);
            
        case ProtectionLevel::System:
            return (operation == OperationType::Delete || 
                   operation == OperationType::Move || 
                   operation == OperationType::Modify);
            
        case ProtectionLevel::Critical:
            return true; // All operations require confirmation
    }
    
    return false;
}

QList<SafetyManager::ProtectionEntry> SafetyManager::getProtectionRules() const
{
    QMutexLocker locker(&m_protectionMutex);
    return m_protectionRules;
}

// Operation tracking and undo system

QString SafetyManager::registerOperation(OperationType type, const QString& sourceFile, const QString& targetFile, const QString& reason)
{
    QString operationId = generateOperationId();
    
    Logger::instance()->info(LogCategories::SAFETY, QString("Registering operation %1: %2 on %3").arg(operationId).arg(static_cast<int>(type)).arg(sourceFile));
    
    SafetyOperation operation;
    operation.operationId = operationId;
    operation.type = type;
    operation.sourceFile = sourceFile;
    operation.targetFile = targetFile;
    operation.reason = reason;
    operation.timestamp = QDateTime::currentDateTime();
    operation.canUndo = (type != OperationType::Delete); // Deletes can be undone if backup exists
    
    if (QFile::exists(sourceFile)) {
        QFileInfo fileInfo(sourceFile);
        operation.fileSize = fileInfo.size();
        operation.checksum = calculateFileChecksum(sourceFile);
    }
    
    {
        QMutexLocker locker(&m_operationMutex);
        m_operations[operationId] = operation;
        
        // Update statistics
        m_operationStats[type]++;
        
        // Cleanup old operations if we exceed the limit
        if (m_operations.size() > m_maxUndoOperations) {
            compactOperationHistory();
        }
    }
    
    LOG_DEBUG(LogCategories::SAFETY, QString("Registered operation %1, Type: %2").arg(operationId).arg(static_cast<int>(type)));
    emit operationRegistered(operationId, operation);
    
    return operationId;
}

bool SafetyManager::finalizeOperation(const QString& operationId, bool success, const QString& backupPath)
{
    QMutexLocker locker(&m_operationMutex);
    
    if (!m_operations.contains(operationId)) {
        qWarning() << "SafetyManager: Unknown operation ID:" << operationId;
        return false;
    }
    
    SafetyOperation& operation = m_operations[operationId];
    operation.backupPath = backupPath;
    
    // Update undo capability based on success and backup availability
    if (operation.type == OperationType::Delete) {
        operation.canUndo = success && !backupPath.isEmpty();
    }
    
    LOG_DEBUG(LogCategories::SAFETY, QString("Finalized operation %1, Success: %2").arg(operationId).arg(success));
    return true;
}

bool SafetyManager::undoOperation(const QString& operationId)
{
    QMutexLocker locker(&m_operationMutex);
    
    if (!m_operations.contains(operationId)) {
        qWarning() << "SafetyManager: Cannot undo unknown operation:" << operationId;
        return false;
    }
    
    const SafetyOperation& operation = m_operations[operationId];
    
    if (!operation.canUndo) {
        qWarning() << "SafetyManager: Operation cannot be undone:" << operationId;
        return false;
    }
    
    bool success = false;
    
    switch (operation.type) {
        case OperationType::Delete:
            if (!operation.backupPath.isEmpty()) {
                success = performRestoreOperation(operation.backupPath, operation.sourceFile);
            }
            break;
            
        case OperationType::Move:
            if (QFile::exists(operation.targetFile)) {
                QFile file(operation.targetFile);
                success = file.rename(operation.sourceFile);
            }
            break;
            
        case OperationType::Copy:
            if (QFile::exists(operation.targetFile)) {
                success = QFile::remove(operation.targetFile);
            }
            break;
            
        case OperationType::Create:
            if (QFile::exists(operation.targetFile)) {
                success = QFile::remove(operation.targetFile);
            }
            break;
            
        case OperationType::Modify:
            if (!operation.backupPath.isEmpty()) {
                success = performRestoreOperation(operation.backupPath, operation.sourceFile);
            }
            break;
    }
    
    if (success) {
        LOG_INFO(LogCategories::SAFETY, QString("Successfully undone operation: %1").arg(operationId));
        emit operationUndone(operationId);
    } else {
        qWarning() << "SafetyManager: Failed to undo operation:" << operationId;
    }
    
    return success;
}

bool SafetyManager::canUndoOperation(const QString& operationId) const
{
    QMutexLocker locker(&m_operationMutex);
    
    if (!m_operations.contains(operationId)) {
        return false;
    }
    
    return m_operations[operationId].canUndo;
}

QList<SafetyManager::SafetyOperation> SafetyManager::getUndoHistory(int maxResults) const
{
    QMutexLocker locker(&m_operationMutex);
    
    QList<SafetyOperation> operations;
    for (const SafetyOperation& op : m_operations) {
        operations.append(op);
    }
    
    // Sort by timestamp (most recent first)
    std::sort(operations.begin(), operations.end(), 
        [](const SafetyOperation& a, const SafetyOperation& b) {
            return a.timestamp > b.timestamp;
        });
    
    if (operations.size() > maxResults) {
        operations = operations.mid(0, maxResults);
    }
    
    return operations;
}

void SafetyManager::clearUndoHistory()
{
    QMutexLocker locker(&m_operationMutex);
    m_operations.clear();
    m_operationStats.clear();
    LOG_DEBUG(LogCategories::SAFETY, "Cleared undo history");
}

bool SafetyManager::removeOperation(const QString& operationId)
{
    QMutexLocker locker(&m_operationMutex);
    
    if (!m_operations.contains(operationId)) {
        return false;
    }
    
    m_operations.remove(operationId);
    LOG_DEBUG(LogCategories::SAFETY, QString("Removed operation from history: %1").arg(operationId));
    return true;
}

void SafetyManager::removeOperationsForBackup(const QString& backupPath)
{
    QMutexLocker locker(&m_operationMutex);
    
    QStringList operationsToRemove;
    
    // Find all operations that use this backup path
    for (auto it = m_operations.begin(); it != m_operations.end(); ++it) {
        if (it.value().backupPath == backupPath) {
            operationsToRemove.append(it.key());
        }
    }
    
    // Remove the operations
    for (const QString& operationId : operationsToRemove) {
        m_operations.remove(operationId);
        LOG_DEBUG(LogCategories::SAFETY, QString("Removed operation %1 for backup: %2").arg(operationId, backupPath));
    }
    
    if (!operationsToRemove.isEmpty()) {
        LOG_INFO(LogCategories::SAFETY, QString("Removed %1 operations for restored backup: %2")
                 .arg(operationsToRemove.size()).arg(backupPath));
    }
}

// Validation methods

bool SafetyManager::validateOperation(OperationType type, const QString& sourceFile, const QString& targetFile) const
{
    // Check if source file exists for operations that require it
    if ((type == OperationType::Delete || type == OperationType::Move || type == OperationType::Copy || type == OperationType::Modify) && 
        !QFile::exists(sourceFile)) {
        return false;
    }
    
    // Check protection level
    if (isProtected(sourceFile, type)) {
        return false;
    }
    
    // Check target path for operations that create files
    if ((type == OperationType::Move || type == OperationType::Copy || type == OperationType::Create) && 
        !targetFile.isEmpty()) {
        QFileInfo targetInfo(targetFile);
        QDir targetDir = targetInfo.dir();
        if (!targetDir.exists() && !targetDir.mkpath(".")) {
            return false;
        }
    }
    
    return true;
}

QString SafetyManager::getOperationRiskAssessment(OperationType type, const QString& sourceFile, const QString& targetFile) const
{
    Q_UNUSED(targetFile)
    
    QStringList risks;
    
    // Check system file status
    if (isSystemFile(sourceFile)) {
        risks << "System file";
    }
    
    // Check protection level
    ProtectionLevel protection = getProtectionLevel(sourceFile);
    if (protection != ProtectionLevel::None) {
        risks << QString("Protected (%1)").arg(static_cast<int>(protection));
    }
    
    // Check file size for large files
    if (QFile::exists(sourceFile)) {
        QFileInfo fileInfo(sourceFile);
        if (fileInfo.size() > 100 * 1024 * 1024) { // 100MB
            risks << "Large file";
        }
    }
    
    // Check operation type risk
    switch (type) {
        case OperationType::Delete:
            risks << "Destructive operation";
            break;
        case OperationType::Move:
            risks << "File relocation";
            break;
        case OperationType::Modify:
            risks << "Content modification";
            break;
        default:
            break;
    }
    
    return risks.isEmpty() ? "Low risk" : risks.join(", ");
}

bool SafetyManager::requiresConfirmation(OperationType type, const QString& sourceFile) const
{
    // Always require confirmation for critical files
    if (getProtectionLevel(sourceFile) == ProtectionLevel::Critical) {
        return true;
    }
    
    // Require confirmation for system files and destructive operations
    if (isSystemFile(sourceFile) && (type == OperationType::Delete || type == OperationType::Modify)) {
        return true;
    }
    
    // Safety level considerations
    if (m_safetyLevel == SafetyLevel::Maximum) {
        return (type == OperationType::Delete || type == OperationType::Move);
    }
    
    return false;
}

// System integration

bool SafetyManager::isSystemFile(const QString& filePath) const
{
    return isInSystemDirectory(filePath) || getProtectionLevel(filePath) == ProtectionLevel::System;
}

bool SafetyManager::isInSystemDirectory(const QString& filePath) const
{
    QMutexLocker locker(&m_protectionMutex);
    
    for (const QString& systemPath : m_systemProtectedPaths) {
        if (filePath.startsWith(systemPath)) {
            return true;
        }
    }
    
    return false;
}

QStringList SafetyManager::getSystemProtectedPaths() const
{
    QMutexLocker locker(&m_protectionMutex);
    return m_systemProtectedPaths;
}

void SafetyManager::addSystemProtectedPath(const QString& path)
{
    QMutexLocker locker(&m_protectionMutex);
    
    if (!m_systemProtectedPaths.contains(path)) {
        m_systemProtectedPaths.append(path);
        LOG_DEBUG(LogCategories::SAFETY, QString("Added system protected path: %1").arg(path));
    }
}

// Statistics and reporting

int SafetyManager::getTotalOperationsTracked() const
{
    QMutexLocker locker(&m_operationMutex);
    return static_cast<int>(m_operations.size());
}

qint64 SafetyManager::getTotalBackupSize() const
{
    qint64 totalSize = 0;
    
    if (!ensureBackupDirectoryExists()) {
        return totalSize;
    }
    
    QDirIterator iterator(m_backupDirectory, QStringList() << "*.backup*", QDir::Files, QDirIterator::Subdirectories);
    
    while (iterator.hasNext()) {
        iterator.next();
        totalSize += iterator.fileInfo().size();
    }
    
    return totalSize;
}

int SafetyManager::getActiveBackupCount() const
{
    int count = 0;
    
    if (!ensureBackupDirectoryExists()) {
        return count;
    }
    
    QDirIterator iterator(m_backupDirectory, QStringList() << "*.backup*", QDir::Files, QDirIterator::Subdirectories);
    
    while (iterator.hasNext()) {
        iterator.next();
        count++;
    }
    
    return count;
}

QHash<SafetyManager::OperationType, int> SafetyManager::getOperationStatistics() const
{
    QMutexLocker locker(&m_operationMutex);
    return m_operationStats;
}

// Maintenance operations

void SafetyManager::performMaintenance()
{
    LOG_DEBUG(LogCategories::SAFETY, "Starting maintenance");
    
    cleanupOldBackups();
    validateAllBackups();
    compactOperationHistory();
    
    LOG_DEBUG(LogCategories::SAFETY, "Maintenance completed");
    emit maintenanceCompleted();
}

bool SafetyManager::validateBackupIntegrity()
{
    LOG_INFO(LogCategories::SAFETY, "Starting backup integrity validation");
    
    QMutexLocker locker(&m_operationMutex);
    
    bool allValid = true;
    int validCount = 0;
    int invalidCount = 0;
    int missingCount = 0;
    
    // Get undo history
    QList<SafetyOperation> undoHistory = getUndoHistory();
    
    // Iterate through all operations in undo history
    for (const auto& op : undoHistory) {
        if (op.backupPath.isEmpty()) {
            continue; // No backup for this operation
        }
        
        // Check if backup file exists
        if (!QFile::exists(op.backupPath)) {
            qWarning() << "SafetyManager: Backup file missing:" << op.backupPath;
            missingCount++;
            allValid = false;
            continue;
        }
        
        // Verify file size matches
        QFileInfo backupInfo(op.backupPath);
        if (backupInfo.size() != op.fileSize) {
            qWarning() << "SafetyManager: Backup size mismatch for:" << op.backupPath
                      << "Expected:" << op.fileSize << "Actual:" << backupInfo.size();
            invalidCount++;
            allValid = false;
            continue;
        }
        
        // If checksum is available, verify it
        if (!op.checksum.isEmpty()) {
            // Calculate checksum of backup
            QFile backupFile(op.backupPath);
            if (backupFile.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(QCryptographicHash::Sha256);
                if (hash.addData(&backupFile)) {
                    QString backupChecksum = hash.result().toHex();
                    if (backupChecksum != op.checksum) {
                        qWarning() << "SafetyManager: Backup checksum mismatch for:" << op.backupPath;
                        invalidCount++;
                        allValid = false;
                        continue;
                    }
                }
                backupFile.close();
            }
        }
        
        validCount++;
    }
    
    LOG_INFO(LogCategories::SAFETY, QString("Backup validation complete - Valid: %1, Invalid: %2, Missing: %3")
             .arg(validCount).arg(invalidCount).arg(missingCount));
    
    return allValid;
}

void SafetyManager::optimizeBackupStorage()
{
    LOG_DEBUG(LogCategories::SAFETY, "Starting backup storage optimization");
    
    QMutexLocker locker(&m_operationMutex);
    
    QDateTime cutoffDate = QDateTime::currentDateTime().addDays(-m_maxBackupAge);
    int removedCount = 0;
    qint64 spaceFreed = 0;
    
    // Get undo history
    QList<SafetyOperation> undoHistory = getUndoHistory();
    
    // Find old backups
    QList<SafetyOperation> oldBackups;
    for (const auto& op : undoHistory) {
        if (!op.backupPath.isEmpty() && op.timestamp < cutoffDate) {
            oldBackups.append(op);
        }
    }
    
    if (oldBackups.isEmpty()) {
        LOG_DEBUG(LogCategories::SAFETY, "No old backups to remove");
        return;
    }
    
    LOG_DEBUG(LogCategories::SAFETY, QString("Found %1 old backups").arg(oldBackups.size()));
    
    // Remove old backups
    for (const auto& op : oldBackups) {
        QFileInfo backupInfo(op.backupPath);
        qint64 fileSize = backupInfo.size();
        
        if (QFile::remove(op.backupPath)) {
            LOG_DEBUG(LogCategories::SAFETY, QString("Removed old backup: %1").arg(op.backupPath));
            removedCount++;
            spaceFreed += fileSize;
            
            // Note: We can't directly remove from undo history here since it's returned by value
            // The actual removal would need to be done through a dedicated method
        } else {
            qWarning() << "SafetyManager: Failed to remove backup:" << op.backupPath;
        }
    }
    
    LOG_INFO(LogCategories::SAFETY, QString("Backup optimization complete - Backups removed: %1, Space freed: %2")
             .arg(removedCount).arg(formatFileSize(spaceFreed)));
}

// Auto-maintenance slots

void SafetyManager::startAutoMaintenance()
{
    m_maintenanceTimer->start(MAINTENANCE_INTERVAL);
    m_cleanupTimer->start(CLEANUP_INTERVAL);
    LOG_DEBUG(LogCategories::SAFETY, "Auto-maintenance started");
}

void SafetyManager::stopAutoMaintenance()
{
    m_maintenanceTimer->stop();
    m_cleanupTimer->stop();
    LOG_DEBUG(LogCategories::SAFETY, "Auto-maintenance stopped");
}

void SafetyManager::performScheduledCleanup()
{
    cleanupOldBackups();
}

void SafetyManager::onFileSystemChanged(const QString& path)
{
    Q_UNUSED(path)
    // TODO: Implement file system change handling
}

void SafetyManager::performPeriodicMaintenance()
{
    performMaintenance();
}

void SafetyManager::cleanupExpiredOperations()
{
    QMutexLocker locker(&m_operationMutex);
    
    QDateTime cutoffDate = QDateTime::currentDateTime().addDays(-30); // Keep operations for 30 days
    
    auto it = m_operations.begin();
    while (it != m_operations.end()) {
        if (it.value().timestamp < cutoffDate) {
            it = m_operations.erase(it);
        } else {
            ++it;
        }
    }
}

// Private helper methods

QString SafetyManager::generateOperationId() const
{
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

QString SafetyManager::createBackupPath(const QString& originalPath, BackupStrategy strategy) const
{
    QFileInfo fileInfo(originalPath);
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString backupName = QString("%1.%2.backup").arg(fileInfo.fileName()).arg(timestamp);
    
    switch (strategy) {
        case BackupStrategy::OriginalLocation:
            return QDir(fileInfo.absolutePath()).filePath(backupName);
            
        case BackupStrategy::CentralLocation:
            if (ensureBackupDirectoryExists()) {
                return QDir(m_backupDirectory).filePath(backupName);
            }
            break;
            
        case BackupStrategy::Versioned:
            // TODO: Implement versioned backup naming
            if (ensureBackupDirectoryExists()) {
                return QDir(m_backupDirectory).filePath(backupName);
            }
            break;
            
        case BackupStrategy::Compressed:
            // TODO: Implement compressed backup naming
            if (ensureBackupDirectoryExists()) {
                return QDir(m_backupDirectory).filePath(backupName + ".gz");
            }
            break;
            
        case BackupStrategy::None:
            break;
    }
    
    return QString();
}

bool SafetyManager::performBackupOperation(const QString& sourcePath, const QString& backupPath)
{
    QFile sourceFile(sourcePath);
    
    // Ensure backup directory exists
    QFileInfo backupInfo(backupPath);
    QDir backupDir = backupInfo.dir();
    if (!backupDir.exists() && !backupDir.mkpath(".")) {
        return false;
    }
    
    return sourceFile.copy(backupPath);
}

bool SafetyManager::performRestoreOperation(const QString& backupPath, const QString& targetPath)
{
    QFile backupFile(backupPath);
    
    // Remove existing target file if it exists
    if (QFile::exists(targetPath)) {
        QFile::remove(targetPath);
    }
    
    // Ensure target directory exists
    QFileInfo targetInfo(targetPath);
    QDir targetDir = targetInfo.dir();
    if (!targetDir.exists() && !targetDir.mkpath(".")) {
        return false;
    }
    
    return backupFile.copy(targetPath);
}

void SafetyManager::initializeDefaultProtectionRules()
{
    // System directories
    addSystemProtectedPath("/bin");
    addSystemProtectedPath("/sbin");
    addSystemProtectedPath("/usr/bin");
    addSystemProtectedPath("/usr/sbin");
    addSystemProtectedPath("/system");
    addSystemProtectedPath("/boot");
    
    // Common protected file patterns
    addProtectionRule("*.sys", ProtectionLevel::System, "System files", false);
    addProtectionRule("*.dll", ProtectionLevel::System, "Dynamic libraries", false);
    addProtectionRule("*.exe", ProtectionLevel::ReadOnly, "Executable files", false);
    addProtectionRule("/etc/*", ProtectionLevel::System, "Configuration files", true);
    addProtectionRule("*.conf", ProtectionLevel::ReadOnly, "Configuration files", false);
    
    LOG_DEBUG(LogCategories::SAFETY, "Initialized default protection rules");
}

bool SafetyManager::matchesProtectionPattern(const QString& filePath, const ProtectionEntry& entry) const
{
    if (entry.isRegex) {
        QRegularExpression regex(entry.pattern);
        return regex.match(filePath).hasMatch();
    } else {
        // Simple wildcard matching
        QRegularExpression regex(QRegularExpression::wildcardToRegularExpression(entry.pattern));
        return regex.match(filePath).hasMatch();
    }
}

void SafetyManager::removeExpiredBackups()
{
    cleanupOldBackups();
}

void SafetyManager::compactOperationHistory()
{
    // Keep only the most recent operations up to the limit
    if (m_operations.size() <= m_maxUndoOperations) {
        return;
    }
    
    QList<SafetyOperation> operations;
    for (const SafetyOperation& op : m_operations) {
        operations.append(op);
    }
    
    // Sort by timestamp (most recent first)
    std::sort(operations.begin(), operations.end(), 
        [](const SafetyOperation& a, const SafetyOperation& b) {
            return a.timestamp > b.timestamp;
        });
    
    // Keep only the most recent ones
    operations = operations.mid(0, m_maxUndoOperations);
    
    // Rebuild the hash
    m_operations.clear();
    for (const SafetyOperation& op : operations) {
        m_operations[op.operationId] = op;
    }
    
    LOG_DEBUG(LogCategories::SAFETY, QString("Compacted operation history to %1 entries").arg(m_operations.size()));
}

void SafetyManager::validateAllBackups()
{
    // TODO: Implement comprehensive backup validation
    LOG_DEBUG(LogCategories::SAFETY, "Backup validation not yet implemented");
}

bool SafetyManager::ensureBackupDirectoryExists() const
{
    QDir dir(m_backupDirectory);
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {
            qWarning() << "SafetyManager: Failed to create backup directory:" << m_backupDirectory;
            return false;
        }
    }
    return true;
}

QString SafetyManager::getRelativeBackupPath(const QString& originalPath) const
{
    // TODO: Implement relative backup path generation
    Q_UNUSED(originalPath)
    return QString();
}

// Static utility methods

QString SafetyManager::calculateFileChecksum(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return QString();
    }
    
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (hash.addData(&file)) {
        return QString(hash.result().toHex());
    }
    
    return QString();
}

bool SafetyManager::compareFiles(const QString& file1, const QString& file2)
{
    return calculateFileChecksum(file1) == calculateFileChecksum(file2);
}

QString SafetyManager::formatFileSize(qint64 bytes)
{
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = bytes;
    
    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }
    
    return QString("%1 %2").arg(QString::number(size, 'f', 2)).arg(units[unitIndex]);
}

QString SafetyManager::getOriginalPathForBackup(const QString& backupPath) const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_operationMutex));
    
    // Search through operations to find the one with this backup path
    for (auto it = m_operations.constBegin(); it != m_operations.constEnd(); ++it) {
        const SafetyOperation& op = it.value();
        if (op.backupPath == backupPath) {
            return op.sourceFile;
        }
    }
    
    return QString(); // Not found
}

QString SafetyManager::generateBackupName(const QString& originalPath, const QDateTime& timestamp)
{
    QFileInfo fileInfo(originalPath);
    QString timeStr = timestamp.toString("yyyyMMdd_hhmmss");
    return QString("%1.%2.backup").arg(fileInfo.fileName()).arg(timeStr);
}