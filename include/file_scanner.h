#pragma once

#include <QString>
#include <QStringList>
#include <QObject>
#include <QFileInfo>
#include <QDir>
#include <QRegularExpression>
#include <QHash>
#include <QDateTime>

/**
 * @brief File Scanner - Core component for traversing directories and finding files
 * 
 * This class handles:
 * - Recursive directory scanning
 * - File filtering by size, type, and patterns
 * - Progress reporting for large scans
 * - Respecting system file exclusions
 * 
 * Phase 1 Implementation:
 * - Basic directory traversal
 * - File size filtering
 * - Simple progress reporting
 */

class FileScanner : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Error types that can occur during scanning
     */
    enum class ScanError {
        None,
        PermissionDenied,      // Cannot read file/directory
        FileSystemError,       // General I/O error
        NetworkTimeout,        // Network drive timeout
        DiskReadError,         // Bad sectors, disk errors
        PathTooLong,          // Path exceeds system limits
        UnknownError          // Catch-all
    };
    Q_ENUM(ScanError)

    /**
     * @brief Detailed information about a scan error
     */
    struct ScanErrorInfo {
        ScanError errorType;
        QString filePath;
        QString errorMessage;
        QString systemErrorCode;  // OS-specific error code
        QDateTime timestamp;
    };
    
    /**
     * @brief Cached file metadata for performance optimization
     */
    struct CachedFileInfo {
        QString filePath;
        qint64 fileSize;
        QDateTime lastModified;
        QDateTime cachedAt;
    };
    
    /**
     * @brief Statistics collected during a scan operation
     */
    struct ScanStatistics {
        int totalFilesScanned = 0;          // Total files found and processed
        int totalDirectoriesScanned = 0;    // Total directories traversed
        qint64 totalBytesScanned = 0;       // Total bytes in scanned files
        int filesFilteredBySize = 0;        // Files excluded by size constraints
        int filesFilteredByPattern = 0;     // Files excluded by pattern matching
        int filesFilteredByHidden = 0;      // Files excluded because they're hidden
        int directoriesSkipped = 0;         // Directories skipped (system dirs, hidden, etc.)
        int errorsEncountered = 0;          // Total errors during scan
        qint64 scanDurationMs = 0;          // Scan duration in milliseconds
        double filesPerSecond = 0.0;        // Scan rate (files/second)
        qint64 peakMemoryUsage = 0;         // Peak memory usage (if available)
    };
    struct ScanOptions {
        QStringList targetPaths;           // Directories/files to scan
        qint64 minimumFileSize = 0;        // Include all files by default (0 bytes)
        qint64 maximumFileSize = -1;       // No limit by default
        QStringList includePatterns;      // File patterns to include (*.jpg, etc.)
        QStringList excludePatterns;      // File patterns to exclude
        bool caseSensitivePatterns = false; // Case-sensitive pattern matching
        bool includeHiddenFiles = false;  // Include hidden/dot files
        bool followSymlinks = false;      // Follow symbolic links
        bool scanSystemDirectories = false; // Include system directories
        
        // Performance options
        bool streamingMode = false;        // Don't store all files in memory
        int estimatedFileCount = 0;        // Hint for capacity reservation (0 = no hint)
        int progressBatchSize = 100;       // Emit progress every N files (default: 100)
        bool enableMetadataCache = false;  // Cache file metadata for repeated scans
        int metadataCacheSizeLimit = 10000; // Maximum cache entries (default: 10,000)
    };

    struct FileInfo {
        QString filePath;
        qint64 fileSize;
        QString fileName;
        QString directory;
        QDateTime lastModified;
    };

    explicit FileScanner(QObject* parent = nullptr);
    ~FileScanner() = default;

    // Main scanning interface
    void startScan(const ScanOptions& options);
    void cancelScan();
    bool isScanning() const;

    // Results access
    QVector<FileInfo> getScannedFiles() const;
    int getTotalFilesFound() const;
    qint64 getTotalBytesScanned() const;
    
    // Error access
    QList<ScanErrorInfo> getScanErrors() const;
    int getTotalErrorsEncountered() const;
    
    // Statistics access
    ScanStatistics getScanStatistics() const;
    
    // Cache management (public for testing)
    void clearMetadataCache();

signals:
    void scanStarted();
    void scanProgress(int filesProcessed, int totalFiles, const QString& currentPath);
    void scanCompleted();
    void scanCancelled();
    void errorOccurred(const QString& error);
    void fileFound(const FileInfo& fileInfo);
    
    // Enhanced error signals
    void scanError(ScanError errorType, const QString& path, const QString& description);
    void scanErrorSummary(int totalErrors, const QList<ScanErrorInfo>& errors);
    
    // Statistics signal
    void scanStatistics(const ScanStatistics& statistics);

private slots:
    void processScanQueue();

private:
    // TODO: Implement these methods
    bool shouldIncludeFile(const QFileInfo& fileInfo) const;
    bool shouldScanDirectory(const QDir& directory) const;
    void scanDirectory(const QString& directoryPath);
    
    // Pattern matching methods
    bool matchesIncludePatterns(const QString& fileName) const;
    bool matchesExcludePatterns(const QString& fileName) const;
    bool matchesPattern(const QString& fileName, const QString& pattern, bool caseSensitive) const;
    QRegularExpression compilePattern(const QString& pattern, bool caseSensitive) const;
    
    // Metadata caching methods
    bool getCachedMetadata(const QString& filePath, CachedFileInfo& cachedInfo) const;
    void cacheMetadata(const QString& filePath, qint64 fileSize, const QDateTime& lastModified);
    void enforceCacheSizeLimit();
    
    // Error handling methods
    void recordError(ScanError errorType, const QString& filePath, const QString& errorMessage, const QString& systemErrorCode = QString());
    ScanError classifyFileSystemError(const QString& filePath, const QFileInfo& fileInfo) const;
    bool isTransientError(ScanError errorType) const;
    bool retryOperation(const QString& directoryPath, int maxRetries = 2);
    
    // Member variables
    ScanOptions m_currentOptions;
    QVector<FileInfo> m_scannedFiles;  // Changed from QList to QVector for better memory locality
    QStringList m_scanQueue;
    bool m_isScanning = false;
    bool m_cancelRequested = false;
    int m_filesProcessed = 0;
    qint64 m_totalBytesScanned = 0;
    
    // Pattern cache for performance
    mutable QHash<QString, QRegularExpression> m_patternCache;
    
    // Metadata cache for performance (optional)
    mutable QHash<QString, CachedFileInfo> m_metadataCache;
    
    // Error tracking
    QList<ScanErrorInfo> m_scanErrors;
    
    // Statistics tracking
    mutable ScanStatistics m_statistics;
    QDateTime m_scanStartTime;
    QDateTime m_scanEndTime;
};