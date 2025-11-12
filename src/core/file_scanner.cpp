#include "file_scanner.h"
#include "app_config.h"
#include "logger.h"

#include <QtCore/QFileInfo>
#include <QtCore/QDir>
#include <QtCore/QDirIterator>
#include <QtCore/QTimer>
#include <QtCore/QRegularExpression>
#include <QtCore/QHash>
#include <QtCore/QThread>
#include <QtCore/QElapsedTimer>
#include <QtCore/QCoreApplication>

// FileScanner Implementation - Phase 1 Basic Version

FileScanner::FileScanner(QObject* parent)
    : QObject(parent)
    , m_isScanning(false)
    , m_cancelRequested(false)
    , m_isPaused(false)
    , m_filesProcessed(0)
    , m_totalBytesScanned(0)

{
    LOG_DEBUG(LogCategories::SCAN, "FileScanner initialized");
}

void FileScanner::startScan(const ScanOptions& options)
{
    if (m_isScanning) {
        LOG_WARNING(LogCategories::SCAN, "Scan already in progress");
        return;
    }
    
    m_currentOptions = options;
    m_scannedFiles.clear();
    m_scanQueue.clear();
    m_filesProcessed = 0;
    m_totalBytesScanned = 0;
    m_cancelRequested = false;
    m_isPaused = false;  // Task 9: Reset pause state
    m_isScanning = true;
    m_patternCache.clear();  // Clear pattern cache for new scan
    m_scanErrors.clear();     // Clear error list for new scan
    
    // Initialize statistics
    m_statistics = ScanStatistics();
    m_scanStartTime = QDateTime::currentDateTime();
    
    // Initialize progress tracking (Task 7)
    m_currentFolder.clear();
    m_currentFile.clear();
    m_elapsedTimer.start();
    
    // Clear metadata cache if not enabled
    if (!options.enableMetadataCache) {
        m_metadataCache.clear();
    }
    
    // Reserve capacity if estimate provided (optimization for large scans)
    if (!options.streamingMode && options.estimatedFileCount > 0) {
        m_scannedFiles.reserve(options.estimatedFileCount);
        LOG_DEBUG(LogCategories::SCAN, QString("Reserved capacity for %1 files").arg(options.estimatedFileCount));
    }
    
    LOG_INFO(LogCategories::SCAN, QString("FileScanner: Starting scan of %1 paths %2")
             .arg(options.targetPaths.size())
             .arg(options.streamingMode ? "(streaming mode)" : ""));
    LOG_DEBUG(LogCategories::SCAN, QString("  - Target paths: %1").arg(options.targetPaths.join(", ")));
    LOG_DEBUG(LogCategories::SCAN, QString("  - Min file size: %1 bytes").arg(options.minimumFileSize));
    LOG_DEBUG(LogCategories::SCAN, QString("  - Include hidden: %1").arg(options.includeHiddenFiles ? "Yes" : "No"));
    if (!options.excludePatterns.isEmpty()) {
        LOG_DEBUG(LogCategories::SCAN, QString("  - Exclude patterns: %1").arg(options.excludePatterns.join(", ")));
    }
    
    // Add target paths to scan queue
    m_scanQueue = options.targetPaths;
    
    emit scanStarted();
    
    // Start processing asynchronously
    QTimer::singleShot(0, this, &FileScanner::processScanQueue);
}

void FileScanner::cancelScan()
{
    if (!m_isScanning) {
        return;
    }
    
    m_cancelRequested = true;
    LOG_INFO(LogCategories::SCAN, "Scan cancellation requested");
}

bool FileScanner::isScanning() const
{
    return m_isScanning;
}

void FileScanner::pauseScan()
{
    if (!m_isScanning || m_isPaused) {
        return;
    }
    
    m_isPaused = true;
    LOG_INFO(LogCategories::SCAN, "FileScanner: Scan paused");
    emit scanPaused();
}

void FileScanner::resumeScan()
{
    if (!m_isScanning || !m_isPaused) {
        return;
    }
    
    m_isPaused = false;
    LOG_INFO(LogCategories::SCAN, "FileScanner: Scan resumed");
    emit scanResumed();
    
    // Continue processing the scan queue
    QTimer::singleShot(0, this, &FileScanner::processScanQueue);
}

bool FileScanner::isPaused() const
{
    return m_isPaused;
}

QVector<FileScanner::FileInfo> FileScanner::getScannedFiles() const
{
    return m_scannedFiles;
}

int FileScanner::getTotalFilesFound() const
{
    return static_cast<int>(m_scannedFiles.size());
}

qint64 FileScanner::getTotalBytesScanned() const
{
    return m_totalBytesScanned;
}

QList<FileScanner::ScanErrorInfo> FileScanner::getScanErrors() const
{
    return m_scanErrors;
}

int FileScanner::getTotalErrorsEncountered() const
{
    return static_cast<int>(m_scanErrors.size());
}

FileScanner::ScanStatistics FileScanner::getScanStatistics() const
{
    return m_statistics;
}

void FileScanner::processScanQueue()
{
    // Check if paused - if so, don't process anything
    if (m_isPaused) {
        return;
    }
    
    if (m_cancelRequested || m_scanQueue.isEmpty()) {
        // Scan completed or cancelled
        m_isScanning = false;
        m_scanEndTime = QDateTime::currentDateTime();
        

        
        if (m_cancelRequested) {
            LOG_INFO(LogCategories::SCAN, "Scan cancelled");
            emit scanCancelled();
        } else {
            LOG_INFO(LogCategories::SCAN, QString("Scan completed - found %1 files").arg(m_scannedFiles.size()));
            
            // Finalize statistics
            m_statistics.totalFilesScanned = m_filesProcessed;
            m_statistics.totalBytesScanned = m_totalBytesScanned;
            m_statistics.errorsEncountered = static_cast<int>(m_scanErrors.size());
            
            // Calculate scan duration
            m_statistics.scanDurationMs = m_scanStartTime.msecsTo(m_scanEndTime);
            
            // Calculate files per second
            if (m_statistics.scanDurationMs > 0) {
                double durationSeconds = m_statistics.scanDurationMs / 1000.0;
                m_statistics.filesPerSecond = m_statistics.totalFilesScanned / durationSeconds;
            }
            
            // Emit statistics
            emit scanStatistics(m_statistics);
            
            LOG_INFO(LogCategories::SCAN, QString("Statistics - Files: %1, Directories: %2, Bytes: %3, Duration: %4ms, Rate: %5 files/sec")
                     .arg(m_statistics.totalFilesScanned)
                     .arg(m_statistics.totalDirectoriesScanned)
                     .arg(m_statistics.totalBytesScanned)
                     .arg(m_statistics.scanDurationMs)
                     .arg(m_statistics.filesPerSecond));
            
            // Emit error summary if there were errors
            if (!m_scanErrors.isEmpty()) {
                emit scanErrorSummary(static_cast<int>(m_scanErrors.size()), m_scanErrors);
            }
            
            LOG_DEBUG(LogCategories::SCAN, "About to emit scanCompleted signal");
            emit scanCompleted();
            LOG_DEBUG(LogCategories::SCAN, "scanCompleted signal emitted");
        }
        return;
    }
    
    QString currentPath = m_scanQueue.takeFirst();
    LOG_DEBUG(LogCategories::SCAN, QString("Processing path: %1").arg(currentPath));
    
    // Scan the current directory
    scanDirectory(currentPath);
    
    // Continue processing queue asynchronously
    // PERFORMANCE FIX: Remove artificial delay - process immediately
    QTimer::singleShot(0, this, &FileScanner::processScanQueue);
}

bool FileScanner::shouldIncludeFile(const QFileInfo& fileInfo) const
{
    // Check file size constraints
    qint64 fileSize = fileInfo.size();
    
    if (fileSize < m_currentOptions.minimumFileSize) {
        m_statistics.filesFilteredBySize++;
        return false;
    }
    
    if (m_currentOptions.maximumFileSize > 0 && fileSize > m_currentOptions.maximumFileSize) {
        m_statistics.filesFilteredBySize++;
        return false;
    }
    
    // Check hidden files
    if (!m_currentOptions.includeHiddenFiles && fileInfo.isHidden()) {
        m_statistics.filesFilteredByHidden++;
        return false;
    }
    
    // Pattern matching
    QString fileName = fileInfo.fileName();
    
    // If include patterns are specified, file must match at least one
    if (!m_currentOptions.includePatterns.isEmpty()) {
        if (!matchesIncludePatterns(fileName)) {
            m_statistics.filesFilteredByPattern++;
            return false;
        }
    }
    
    // If exclude patterns are specified, file must not match any
    if (!m_currentOptions.excludePatterns.isEmpty()) {
        if (matchesExcludePatterns(fileName)) {
            m_statistics.filesFilteredByPattern++;
            return false;
        }
    }
    
    return true;
}

bool FileScanner::shouldScanDirectory(const QDir& directory) const
{
    QString dirPath = directory.absolutePath();
    
    // Skip hidden directories if not requested
    if (!m_currentOptions.includeHiddenFiles && directory.dirName().startsWith('.')) {
        m_statistics.directoriesSkipped++;
        return false;
    }
    
    // Skip system directories if not requested
    if (!m_currentOptions.scanSystemDirectories) {
        // Skip common system directories
        if (dirPath.startsWith("/sys") || dirPath.startsWith("/proc") || 
            dirPath.startsWith("/dev") || dirPath.startsWith("/run")) {
            m_statistics.directoriesSkipped++;
            return false;
        }
    }
    
    return true;
}

void FileScanner::scanDirectory(const QString& directoryPath)
{
    LOG_FILE("Scanning directory", directoryPath);
    
    // Update current folder for progress tracking (Task 7)
    m_currentFolder = directoryPath;
    
    QDir dir(directoryPath);
    if (!dir.exists()) {
        LOG_WARNING(LogCategories::SCAN, QString("FileScanner: Directory does not exist: %1").arg(directoryPath));
        
        // Try to retry in case it's a transient network issue
        if (retryOperation(directoryPath, 2)) {
            LOG_INFO(LogCategories::SCAN, QString("FileScanner: Retry successful for: %1").arg(directoryPath));
            dir = QDir(directoryPath);  // Refresh directory object
        } else {
            recordError(ScanError::FileSystemError, directoryPath, "Directory does not exist");
            return;  // Continue with next directory in queue
        }
    }
    
    // Check if directory is readable
    QFileInfo dirInfo(directoryPath);
    if (!dirInfo.isReadable()) {
        LOG_WARNING(LogCategories::SCAN, QString("Permission denied for directory: %1").arg(directoryPath));
        recordError(ScanError::PermissionDenied, directoryPath, "Permission denied");
        return;  // Continue with next directory in queue
    }
    
    if (!shouldScanDirectory(dir)) {
        LOG_DEBUG(LogCategories::SCAN, QString("Skipping directory: %1").arg(directoryPath));
        return;
    }
    
    // Track that we're scanning this directory
    m_statistics.totalDirectoriesScanned++;
    
    // Use QDirIterator for recursive scanning
    QDirIterator iterator(directoryPath, QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot,
                         QDirIterator::Subdirectories);

    int filesProcessedSinceYield = 0;
    int iterationsProcessedSinceYield = 0;
    // CRITICAL FIX: Balance between performance and responsiveness
    // For large directory scans, we need to yield more frequently to prevent hanging
    const int EVENT_YIELD_INTERVAL = 100;  // Process events every 100 files
    const int ITERATION_YIELD_INTERVAL = 500;  // Process events every 500 iterations

    while (iterator.hasNext() && !m_cancelRequested && !m_isPaused) {
        QString filePath;
        iterationsProcessedSinceYield++;

        try {
            filePath = iterator.next();
        } catch (...) {
            qWarning() << "FileScanner: Exception during directory iteration";
            recordError(ScanError::FileSystemError, directoryPath, "Exception during directory iteration");
            continue;
        }

        QFileInfo fileInfo(filePath);

        // Check for file system errors
        if (!fileInfo.exists()) {
            // File may have been deleted during scan
            continue;
        }

        // Check if file is readable
        if (fileInfo.isFile() && !fileInfo.isReadable()) {
            recordError(ScanError::PermissionDenied, filePath, "Permission denied");
            continue;
        }

        // Check for path length issues
        if (filePath.length() > 4096) {  // Common path length limit
            recordError(ScanError::PathTooLong, filePath, "Path exceeds maximum length");
            continue;
        }

        if (fileInfo.isFile() && shouldIncludeFile(fileInfo)) {
            LOG_FILE("Processing file", filePath);

            // Update current file for progress tracking (Task 7)
            m_currentFile = filePath;

            // Create FileInfo structure
            FileInfo info;
            info.filePath = fileInfo.absoluteFilePath();
            info.fileName = fileInfo.fileName();
            info.directory = fileInfo.absolutePath();

            // Try to use cached metadata if enabled
            if (m_currentOptions.enableMetadataCache) {
                CachedFileInfo cachedInfo;
                if (getCachedMetadata(info.filePath, cachedInfo)) {
                    // Check if file has been modified since cached
                    QDateTime currentModified = fileInfo.lastModified();
                    if (cachedInfo.lastModified == currentModified) {
                        // Use cached data
                        info.fileSize = cachedInfo.fileSize;
                        info.lastModified = cachedInfo.lastModified;
                    } else {
                        // File modified, update cache
                        info.fileSize = fileInfo.size();
                        info.lastModified = currentModified;
                        cacheMetadata(info.filePath, info.fileSize, info.lastModified);
                    }
                } else {
                    // Not in cache, read and cache
                    info.fileSize = fileInfo.size();
                    info.lastModified = fileInfo.lastModified();
                    cacheMetadata(info.filePath, info.fileSize, info.lastModified);
                }
            } else {
                // No caching, read directly
                info.fileSize = fileInfo.size();
                info.lastModified = fileInfo.lastModified();
            }

            // In streaming mode, emit files immediately without storing
            if (m_currentOptions.streamingMode) {
                emit fileFound(info);
            } else {
                // Store files in memory for later retrieval
                m_scannedFiles.append(info);
            }

            m_totalBytesScanned += info.fileSize;
            m_filesProcessed++;
            filesProcessedSinceYield++;

            // BALANCED FIX: Emit progress frequently enough for UI updates
            // Update every 100 files for better user feedback
            int batchSize = m_currentOptions.progressBatchSize > 0 ? m_currentOptions.progressBatchSize : 100;
            if (m_filesProcessed % batchSize == 0) {
                emit scanProgress(m_filesProcessed, -1, filePath);
                // In non-streaming mode, also emit fileFound periodically
                if (!m_currentOptions.streamingMode) {
                    emit fileFound(info);
                }

                // Emit detailed progress (Task 7)
                emitDetailedProgress();
            }
        }

        // CRITICAL FIX: Yield to event loop more frequently to prevent UI freezing
        // This prevents the application from becoming unresponsive during large directory scans
        // Yield based on either files processed OR total iterations to handle directories with many non-matching files
        if (filesProcessedSinceYield >= EVENT_YIELD_INTERVAL || iterationsProcessedSinceYield >= ITERATION_YIELD_INTERVAL) {
            // IMPORTANT: Process ALL events including user input to allow cancellation
            QCoreApplication::processEvents(QEventLoop::AllEvents);
            
            // Also emit progress update to keep UI informed even if no files match criteria
            if (iterationsProcessedSinceYield >= ITERATION_YIELD_INTERVAL) {
                emit scanProgress(m_filesProcessed, -1, filePath);
                emitDetailedProgress();
            }
            
            // Reset BOTH counters to ensure proper yielding
            filesProcessedSinceYield = 0;
            iterationsProcessedSinceYield = 0;
            
            // Check for cancellation or pause after processing events
            if (m_cancelRequested || m_isPaused) {
                break;
            }
        }
    }
    
    LOG_DEBUG(LogCategories::SCAN, QString("Completed directory: %1 - found %2 files so far").arg(directoryPath).arg(m_scannedFiles.size()));
}

bool FileScanner::matchesIncludePatterns(const QString& fileName) const
{
    // File must match at least one include pattern
    for (const QString& pattern : m_currentOptions.includePatterns) {
        if (matchesPattern(fileName, pattern, m_currentOptions.caseSensitivePatterns)) {
            return true;
        }
    }
    return false;
}

bool FileScanner::matchesExcludePatterns(const QString& fileName) const
{
    // File must not match any exclude pattern
    for (const QString& pattern : m_currentOptions.excludePatterns) {
        if (matchesPattern(fileName, pattern, m_currentOptions.caseSensitivePatterns)) {
            return true;
        }
    }
    return false;
}

bool FileScanner::matchesPattern(const QString& fileName, const QString& pattern, bool caseSensitive) const
{
    if (pattern.isEmpty()) {
        return false;
    }
    
    try {
        QRegularExpression regex = compilePattern(pattern, caseSensitive);
        QRegularExpressionMatch match = regex.match(fileName);
        return match.hasMatch();
    } catch (...) {
        qWarning() << "FileScanner: Invalid pattern:" << pattern;
        return false;
    }
}

QRegularExpression FileScanner::compilePattern(const QString& pattern, bool caseSensitive) const
{
    // Create cache key with case sensitivity flag
    QString cacheKey = pattern + (caseSensitive ? ":cs" : ":ci");
    
    // Check cache first
    if (m_patternCache.contains(cacheKey)) {
        return m_patternCache.value(cacheKey);
    }
    
    QRegularExpression regex;
    
    // Determine if this is a glob pattern or regex pattern
    // Glob patterns contain wildcards like *, ?, [, but not regex-specific chars
    bool isGlobPattern = pattern.contains('*') || pattern.contains('?') || 
                        (pattern.contains('[') && pattern.contains(']'));
    
    // Check if it looks like a regex (contains regex-specific characters)
    bool hasRegexChars = pattern.contains('^') || pattern.contains('$') || 
                        pattern.contains('(') || pattern.contains(')') ||
                        pattern.contains('{') || pattern.contains('}') ||
                        pattern.contains('+') || pattern.contains('|');
    
    if (isGlobPattern && !hasRegexChars) {
        // Convert glob pattern to regex
        QString regexPattern = QRegularExpression::wildcardToRegularExpression(pattern);
        regex.setPattern(regexPattern);
    } else {
        // Treat as regex pattern
        regex.setPattern(pattern);
    }
    
    // Set case sensitivity
    if (!caseSensitive) {
        regex.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
    }
    
    // Validate the pattern
    if (!regex.isValid()) {
        qWarning() << "FileScanner: Invalid regex pattern:" << pattern 
                   << "Error:" << regex.errorString();
        // Return a pattern that never matches
        regex.setPattern("(?!)");
    }
    
    // Cache the compiled pattern
    m_patternCache.insert(cacheKey, regex);
    
    return regex;
}

void FileScanner::recordError(ScanError errorType, const QString& filePath, const QString& errorMessage, const QString& systemErrorCode)
{
    ScanErrorInfo errorInfo;
    errorInfo.errorType = errorType;
    errorInfo.filePath = filePath;
    errorInfo.errorMessage = errorMessage;
    errorInfo.systemErrorCode = systemErrorCode;
    errorInfo.timestamp = QDateTime::currentDateTime();
    
    m_scanErrors.append(errorInfo);
    
    // Emit individual error signal
    emit scanError(errorType, filePath, errorMessage);
    
    qWarning() << "FileScanner Error:" << errorMessage << "- Path:" << filePath;
}

FileScanner::ScanError FileScanner::classifyFileSystemError(const QString& filePath, const QFileInfo& fileInfo) const
{
    if (!fileInfo.exists()) {
        return ScanError::FileSystemError;
    }
    
    if (!fileInfo.isReadable()) {
        return ScanError::PermissionDenied;
    }
    
    if (filePath.length() > 4096) {
        return ScanError::PathTooLong;
    }
    
    return ScanError::UnknownError;
}

bool FileScanner::isTransientError(ScanError errorType) const
{
    // Network timeouts are transient and can be retried
    return errorType == ScanError::NetworkTimeout;
}

bool FileScanner::retryOperation(const QString& directoryPath, int maxRetries)
{
    for (int attempt = 0; attempt < maxRetries; ++attempt) {
        QDir dir(directoryPath);
        if (dir.exists()) {
            QFileInfo dirInfo(directoryPath);
            if (dirInfo.isReadable()) {
                return true;  // Operation succeeded
            }
        }
        
        // Wait a bit before retrying (exponential backoff)
        QThread::msleep(static_cast<unsigned long>(100 * (attempt + 1)));
    }
    
    return false;  // All retries failed
}

bool FileScanner::getCachedMetadata(const QString& filePath, CachedFileInfo& cachedInfo) const
{
    if (m_metadataCache.contains(filePath)) {
        cachedInfo = m_metadataCache.value(filePath);
        return true;
    }
    return false;
}

void FileScanner::cacheMetadata(const QString& filePath, qint64 fileSize, const QDateTime& lastModified)
{
    CachedFileInfo cachedInfo;
    cachedInfo.filePath = filePath;
    cachedInfo.fileSize = fileSize;
    cachedInfo.lastModified = lastModified;
    cachedInfo.cachedAt = QDateTime::currentDateTime();
    
    m_metadataCache.insert(filePath, cachedInfo);
    
    // Enforce cache size limit
    enforceCacheSizeLimit();
}

void FileScanner::clearMetadataCache()
{
    m_metadataCache.clear();
}

void FileScanner::enforceCacheSizeLimit()
{
    int cacheLimit = m_currentOptions.metadataCacheSizeLimit;
    if (cacheLimit > 0 && m_metadataCache.size() > cacheLimit) {
        // Remove oldest entries (simple FIFO eviction)
        // Find and remove the oldest 10% of entries
        int entriesToRemove = static_cast<int>(m_metadataCache.size()) - cacheLimit;
        
        QList<QString> keysToRemove;
        QDateTime oldestTime = QDateTime::currentDateTime();
        
        // Find oldest entries
        for (auto it = m_metadataCache.constBegin(); it != m_metadataCache.constEnd(); ++it) {
            if (keysToRemove.size() < entriesToRemove) {
                keysToRemove.append(it.key());
                if (it.value().cachedAt < oldestTime) {
                    oldestTime = it.value().cachedAt;
                }
            } else {
                // Check if this entry is older than the newest in our removal list
                if (it.value().cachedAt < oldestTime) {
                    // Find the newest entry in removal list and replace it
                    int newestIdx = 0;
                    QDateTime newestTime = m_metadataCache.value(keysToRemove[0]).cachedAt;
                    for (int i = 1; i < keysToRemove.size(); ++i) {
                        QDateTime time = m_metadataCache.value(keysToRemove[i]).cachedAt;
                        if (time > newestTime) {
                            newestTime = time;
                            newestIdx = i;
                        }
                    }
                    keysToRemove[newestIdx] = it.key();
                    oldestTime = it.value().cachedAt;
                }
            }
        }
        
        // Remove the selected entries
        for (const QString& key : keysToRemove) {
            m_metadataCache.remove(key);
        }
    }
}

void FileScanner::emitDetailedProgress()
{
    // Calculate elapsed time
    qint64 elapsedMs = m_elapsedTimer.elapsed();
    
    // Calculate files per second
    double filesPerSecond = 0.0;
    if (elapsedMs > 0) {
        double elapsedSeconds = elapsedMs / 1000.0;
        filesPerSecond = m_filesProcessed / elapsedSeconds;
    }
    
    // Create progress structure
    ScanProgress progress;
    progress.filesScanned = m_filesProcessed;
    progress.bytesScanned = m_totalBytesScanned;
    progress.currentFolder = m_currentFolder;
    progress.currentFile = m_currentFile;
    progress.elapsedTimeMs = elapsedMs;
    progress.filesPerSecond = filesPerSecond;
    progress.directoriesScanned = m_statistics.totalDirectoriesScanned;
    progress.isPaused = m_isPaused;  // Task 9: Include pause state
    
    // Emit the detailed progress signal
    emit detailedProgress(progress);
}