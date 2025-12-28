# CloneClean API Design Document

**Version:** 1.0  
**Created:** 2025-10-03  
**Based on:** Architecture Design v1.0  

---

## Documentation Updates

**Note:** The FileScanner component has been completed and enhanced with new features. For the most up-to-date and comprehensive FileScanner documentation, please refer to:

- [FileScanner API Documentation](API_FILESCANNER.md) - Complete API reference
- [FileScanner Usage Examples](FILESCANNER_EXAMPLES.md) - Practical usage examples
- [FileScanner Error Handling Guide](FILESCANNER_ERROR_HANDLING.md) - Error handling best practices
- [FileScanner Performance Tuning](FILESCANNER_PERFORMANCE.md) - Performance optimization guide
- [FileScanner Integration Guide](FILESCANNER_INTEGRATION.md) - Integration with other components
- [FileScanner Migration Guide](FILESCANNER_MIGRATION.md) - Migration guide for existing code

The FileScanner section below represents the original design. The actual implementation includes additional features such as pattern matching, enhanced error handling, performance optimizations, and comprehensive statistics.

---

## Overview

This document defines the public APIs and interfaces for all major components in the CloneClean application. These APIs serve as contracts between components and provide the foundation for testing, maintainability, and future extensibility.

### API Design Principles
- **Interface Segregation:** Small, focused interfaces with specific responsibilities
- **Dependency Inversion:** Depend on abstractions, not concrete implementations
- **Consistency:** Uniform naming conventions and parameter patterns
- **Qt Integration:** Leverage Qt's signal/slot mechanism for async communication
- **Type Safety:** Strong typing with clear parameter validation

---

## Core Engine APIs

### 1. File Scanner API

#### 1.1 Public Interface

```cpp
class FileScanner : public QObject {
    Q_OBJECT

public:
    // Configuration structures
    struct ScanOptions {
        QStringList targetPaths;               // Directories/files to scan
        qint64 minimumFileSize = 1024;         // Minimum file size (bytes)
        qint64 maximumFileSize = -1;           // Maximum file size (-1 = no limit)
        QStringList includePatterns;          // File patterns to include (*.jpg, etc.)
        QStringList excludePatterns;          // File patterns to exclude (*.tmp, etc.)
        bool includeHiddenFiles = false;      // Include hidden/dot files
        bool followSymlinks = false;          // Follow symbolic links
        bool scanSystemDirectories = false;   // Include system directories
        int maxDepth = -1;                    // Maximum directory depth (-1 = unlimited)
        
        // Validation
        bool isValid() const;
        QString validationError() const;
    };

    struct FileInfo {
        QString filePath;            // Absolute file path
        QString fileName;            // File name only
        QString directory;           // Parent directory path
        qint64 fileSize;            // File size in bytes
        QDateTime lastModified;      // Last modification time
        QDateTime created;          // Creation time
        QString extension;          // File extension (lowercase)
        bool isSymlink = false;     // Is symbolic link
        bool isReadable = true;     // File is readable
        
        // Utility methods
        QString displayName() const;
        QString relativePath(const QString& basePath) const;
        bool operator==(const FileInfo& other) const;
        uint qHash() const;
    };

    // Constructor
    explicit FileScanner(QObject* parent = nullptr);
    ~FileScanner() override;

    // Main interface
    void startScan(const ScanOptions& options);
    void pauseScan();
    void resumeScan();
    void cancelScan();
    
    // Status queries
    bool isScanning() const;
    bool isPaused() const;
    ScanOptions currentOptions() const;
    
    // Results access
    QList<FileInfo> getScannedFiles() const;
    int getTotalFilesFound() const;
    qint64 getTotalBytesScanned() const;
    QStringList getErrorMessages() const;
    
    // Statistics
    struct Statistics {
        int filesProcessed = 0;
        int directoriesProcessed = 0;
        qint64 totalBytes = 0;
        std::chrono::milliseconds duration{0};
        int errors = 0;
        
        double filesPerSecond() const;
        double bytesPerSecond() const;
    };
    
    Statistics getStatistics() const;

public slots:
    void setOptions(const ScanOptions& options);

signals:
    // Status signals
    void scanStarted();
    void scanPaused();
    void scanResumed();
    void scanCompleted();
    void scanCancelled();
    void scanError(const QString& error);
    
    // Progress signals  
    void scanProgress(int filesProcessed, int totalFiles, const QString& currentPath);
    void fileFound(const FileInfo& fileInfo);
    void directoryChanged(const QString& directoryPath);
    
    // Batch signals for performance
    void filesFound(const QList<FileInfo>& fileInfos);
    void progressUpdate(int percentage);
};
```

#### 1.2 Error Handling

```cpp
class FileScannerException : public std::exception {
public:
    enum class ErrorType {
        InvalidPath,
        PermissionDenied,
        DiskError,
        OutOfMemory,
        Cancelled,
        InvalidOptions
    };
    
    explicit FileScannerException(ErrorType type, const QString& message);
    
    ErrorType errorType() const { return m_type; }
    const QString& message() const { return m_message; }
    
private:
    ErrorType m_type;
    QString m_message;
};
```

### 2. Hash Calculator API

#### 2.1 Public Interface

```cpp
class HashCalculator : public QObject {
    Q_OBJECT

public:
    // Hash algorithm enumeration
    enum class Algorithm {
        MD5,        // Fast, less secure (for compatibility)
        SHA1,       // Standard SHA-1
        SHA256,     // Default - good balance of speed/security
        SHA512,     // Most secure, slower
        CRC32       // Fastest, for quick comparison only
    };

    // Result structure
    struct HashResult {
        QString filePath;           // Original file path
        QByteArray hash;           // Computed hash value
        Algorithm algorithm;        // Algorithm used
        qint64 fileSize;           // File size in bytes
        std::chrono::milliseconds duration; // Time taken
        bool success = false;       // Operation success
        QString errorMessage;       // Error description if failed
        
        // Utility methods
        QString hashString() const;        // Hex string representation
        bool isValid() const;
        QString algorithmName() const;
    };

    // Batch processing configuration
    struct BatchOptions {
        Algorithm algorithm = Algorithm::SHA256;
        int maxConcurrentJobs = 4;         // Parallel hash calculations
        qint64 chunkSize = 64 * 1024;      // Read chunk size for large files
        bool useCache = true;              // Enable hash caching
        int cacheSizeLimit = 10000;        // Maximum cache entries
        bool skipLargeFiles = false;       // Skip files larger than threshold
        qint64 largeFileThreshold = 1024 * 1024 * 1024; // 1GB default
        
        bool isValid() const;
    };

    // Constructor
    explicit HashCalculator(QObject* parent = nullptr);
    ~HashCalculator() override;

    // Single file operations
    void calculateHash(const QString& filePath, 
                      Algorithm algorithm = Algorithm::SHA256);
    HashResult calculateHashSync(const QString& filePath, 
                               Algorithm algorithm = Algorithm::SHA256);

    // Batch operations
    void calculateHashes(const QStringList& filePaths, 
                        const BatchOptions& options = BatchOptions{});
    void calculateHashesForFiles(const QList<FileScanner::FileInfo>& files,
                               const BatchOptions& options = BatchOptions{});

    // Control operations
    void pauseCalculation();
    void resumeCalculation();
    void cancelCalculation();
    void clearCache();

    // Status queries
    bool isCalculating() const;
    bool isPaused() const;
    int pendingJobs() const;
    int completedJobs() const;
    
    // Cache management
    bool isCached(const QString& filePath, Algorithm algorithm) const;
    HashResult getCachedHash(const QString& filePath, Algorithm algorithm) const;
    void setCacheSize(int maxEntries);
    int cacheSize() const;
    int cacheHitRate() const; // Percentage
    void exportCache(const QString& filePath) const;
    bool importCache(const QString& filePath);

    // Statistics
    struct Statistics {
        int totalFiles = 0;
        int completedFiles = 0;
        int errorFiles = 0;
        qint64 totalBytes = 0;
        qint64 processedBytes = 0;
        std::chrono::milliseconds totalDuration{0};
        int cacheHits = 0;
        int cacheMisses = 0;
        
        double completionPercentage() const;
        double bytesPerSecond() const;
        double filesPerSecond() const;
        std::chrono::milliseconds estimatedTimeRemaining() const;
    };
    
    Statistics getStatistics() const;

public slots:
    void setAlgorithm(Algorithm algorithm);
    void setBatchOptions(const BatchOptions& options);

signals:
    // Single file signals
    void hashCalculated(const HashResult& result);
    void hashError(const QString& filePath, const QString& error);
    
    // Batch processing signals
    void batchStarted(int totalFiles);
    void batchProgress(int completed, int total);
    void batchCompleted(const QList<HashResult>& results);
    void batchCancelled();
    void batchError(const QString& error);
    
    // Individual file progress (for large files)
    void fileProgress(const QString& filePath, int percentage);
    
    // Cache events
    void cacheUpdated(int entries);
    void cacheCleared();
};
```

#### 2.2 Hash Utilities

```cpp
namespace HashUtils {
    // Utility functions for hash operations
    QByteArray calculateMD5(const QString& filePath);
    QByteArray calculateSHA256(const QString& filePath);
    QString hashToString(const QByteArray& hash);
    QByteArray hashFromString(const QString& hashString);
    
    // Hash comparison
    bool compareHashes(const QByteArray& hash1, const QByteArray& hash2);
    bool isValidHashString(const QString& hashString, HashCalculator::Algorithm algorithm);
    
    // Performance utilities
    qint64 estimateHashTime(qint64 fileSize, HashCalculator::Algorithm algorithm);
    HashCalculator::Algorithm selectOptimalAlgorithm(qint64 fileSize);
}
```

### 3. Duplicate Detector API

#### 3.1 Public Interface

```cpp
class DuplicateDetector : public QObject {
    Q_OBJECT

public:
    // Detection modes
    enum class DetectionMode {
        Quick,      // Size + filename matching
        Deep,       // Size + hash comparison  
        Media,      // Deep + metadata comparison for media files
        Smart,      // Adaptive based on file types and size
        Custom      // User-defined criteria
    };

    // Duplicate group representation
    struct DuplicateGroup {
        QByteArray commonHash;                      // Hash shared by all files
        QList<FileScanner::FileInfo> files;        // All duplicate files
        qint64 totalSize;                          // Total size of all files
        qint64 wastedSpace;                        // Space that can be reclaimed
        FileScanner::FileInfo recommendedKeep;     // Recommended file to keep
        QString groupId;                           // Unique identifier
        DetectionMode detectionMethod;             // How duplicates were found
        double confidence;                         // Confidence score (0.0-1.0)
        
        // Utility methods
        int fileCount() const { return files.size(); }
        qint64 reclaimableSpace() const;           // Space saved if duplicates removed
        QStringList filePaths() const;
        bool contains(const QString& filePath) const;
        QString summary() const;                   // Human-readable summary
    };

    // Detection criteria for custom mode
    struct DetectionCriteria {
        bool compareSize = true;           // Compare file sizes
        bool compareHash = true;           // Compare file hashes
        bool compareFileName = false;      // Compare file names
        bool compareContent = false;       // Deep content analysis
        bool compareMetadata = false;      // Compare file metadata
        double similarityThreshold = 1.0; // Similarity threshold (1.0 = exact)
        QStringList fileExtensions;       // Limit to specific extensions
        bool caseSensitive = false;        // Case-sensitive comparisons
        
        bool isValid() const;
    };

    // Recommendation criteria
    struct RecommendationCriteria {
        enum class PreferenceType {
            NewerDate,      // Prefer newer files
            OlderDate,      // Prefer older files
            LargerSize,     // Prefer larger files
            SmallerSize,    // Prefer smaller files
            BetterPath,     // Prefer files in better locations
            BetterName,     // Prefer files with better names
            Manual          // No automatic recommendation
        };
        
        PreferenceType primaryPreference = PreferenceType::BetterPath;
        PreferenceType secondaryPreference = PreferenceType::NewerDate;
        QStringList preferredDirectories;  // Directories to prefer
        QStringList avoidDirectories;      // Directories to avoid (Downloads, temp, etc.)
        QRegularExpression preferredNamePattern;
        QRegularExpression avoidNamePattern;
        
        bool isValid() const;
    };

    // Constructor
    explicit DuplicateDetector(QObject* parent = nullptr);
    ~DuplicateDetector() override;

    // Main detection interface
    void detectDuplicates(const QList<FileScanner::FileInfo>& files,
                         DetectionMode mode = DetectionMode::Smart);
    void detectDuplicatesWithCriteria(const QList<FileScanner::FileInfo>& files,
                                    const DetectionCriteria& criteria);

    // Control operations
    void pauseDetection();
    void resumeDetection();
    void cancelDetection();

    // Configuration
    void setRecommendationCriteria(const RecommendationCriteria& criteria);
    RecommendationCriteria recommendationCriteria() const;
    
    void setHashCalculator(HashCalculator* calculator);
    HashCalculator* hashCalculator() const;

    // Status queries
    bool isDetecting() const;
    bool isPaused() const;
    DetectionMode currentMode() const;
    
    // Results access
    QList<DuplicateGroup> getDuplicateGroups() const;
    int getTotalGroups() const;
    qint64 getTotalWastedSpace() const;
    int getTotalDuplicateFiles() const;
    
    // Group management
    DuplicateGroup getGroup(const QString& groupId) const;
    QList<DuplicateGroup> getGroupsContaining(const QString& filePath) const;
    void removeFileFromGroups(const QString& filePath);
    void updateGroupRecommendation(const QString& groupId, const QString& filePath);

    // Statistics
    struct Statistics {
        int totalFiles = 0;
        int processedFiles = 0;
        int duplicateFiles = 0;
        int uniqueFiles = 0;
        int duplicateGroups = 0;
        qint64 totalSize = 0;
        qint64 duplicateSize = 0;
        qint64 reclaimableSpace = 0;
        std::chrono::milliseconds detectionTime{0};
        
        double duplicatePercentage() const;
        double spaceWastePercentage() const;
        QString summaryText() const;
    };
    
    Statistics getStatistics() const;

public slots:
    void setDetectionMode(DetectionMode mode);

signals:
    // Detection lifecycle
    void detectionStarted();
    void detectionPaused();
    void detectionResumed();
    void detectionCompleted();
    void detectionCancelled();
    void detectionError(const QString& error);
    
    // Progress signals
    void detectionProgress(int processed, int total);
    void currentFileProgress(const QString& filePath);
    void phaseChanged(const QString& phaseName);
    
    // Results signals
    void duplicateGroupFound(const DuplicateGroup& group);
    void duplicateGroupsUpdated();
    void statisticsUpdated(const Statistics& stats);
    
    // Batch results for performance
    void duplicateGroupsBatch(const QList<DuplicateGroup>& groups);
};
```

#### 3.2 Comparison Utilities

```cpp
namespace DuplicateUtils {
    // File comparison functions
    bool compareBySize(const FileScanner::FileInfo& file1, 
                      const FileScanner::FileInfo& file2);
    bool compareByHash(const FileScanner::FileInfo& file1, 
                      const FileScanner::FileInfo& file2,
                      const QHash<QString, QByteArray>& hashCache);
    double calculateSimilarity(const FileScanner::FileInfo& file1,
                             const FileScanner::FileInfo& file2,
                             const DuplicateDetector::DetectionCriteria& criteria);
    
    // Recommendation utilities
    FileScanner::FileInfo selectBestFile(
        const QList<FileScanner::FileInfo>& files,
        const DuplicateDetector::RecommendationCriteria& criteria);
    int scoreFile(const FileScanner::FileInfo& file,
                 const DuplicateDetector::RecommendationCriteria& criteria);
    
    // Path utilities
    bool isBetterPath(const QString& path1, const QString& path2);
    bool isSystemPath(const QString& path);
    bool isTempPath(const QString& path);
}
```

### 4. Safety Manager API

#### 4.1 Public Interface

```cpp
class SafetyManager : public QObject {
    Q_OBJECT

public:
    // Operation types
    enum class OperationType {
        MoveToTrash,
        Rename,
        Copy,
        Delete,           // For testing only, never used in production
        BatchOperation
    };

    // Operation record for history and undo
    struct OperationRecord {
        QString operationId;                // Unique operation identifier
        OperationType type;                 // Type of operation performed
        QDateTime timestamp;                // When operation was performed
        QStringList sourceFiles;           // Original file paths
        QStringList targetFiles;           // Target/result file paths
        QString description;                // Human-readable description
        qint64 totalSize;                  // Total size of affected files
        bool canUndo;                      // Whether operation can be undone
        bool isCompleted;                  // Whether operation completed successfully
        QString errorMessage;              // Error message if operation failed
        QVariantMap metadata;              // Additional operation-specific data
        
        // Utility methods
        QString summaryText() const;
        bool isValid() const;
        QString statusText() const;
    };

    // Safety validation result
    struct ValidationResult {
        bool isValid = false;
        QStringList errors;
        QStringList warnings;
        QStringList protectedFiles;        // Files that cannot be operated on
        qint64 totalSize = 0;
        int fileCount = 0;
        
        bool hasErrors() const { return !errors.isEmpty(); }
        bool hasWarnings() const { return !warnings.isEmpty(); }
        QString summaryText() const;
    };

    // Batch operation configuration
    struct BatchConfiguration {
        bool confirmEachFile = false;       // Confirm each individual file
        bool confirmBatch = true;           // Confirm entire batch
        bool stopOnError = false;           // Stop batch on first error
        bool createBackup = false;          // Create backup before operation
        int maxConcurrentOps = 1;           // Maximum concurrent operations
        QString backupLocation;             // Where to store backups
        
        bool isValid() const;
    };

    // Constructor
    explicit SafetyManager(QObject* parent = nullptr);
    ~SafetyManager() override;

    // File operations
    QString moveToTrash(const QStringList& filePaths);
    QString moveFileToTrash(const QString& filePath);
    QString batchMoveToTrash(const QStringList& filePaths, 
                           const BatchConfiguration& config = BatchConfiguration{});

    // Validation
    ValidationResult validateOperation(const QStringList& filePaths, 
                                     OperationType operation) const;
    bool isOperationSafe(const QString& filePath, OperationType operation) const;
    bool isSystemFile(const QString& filePath) const;
    bool isProtectedLocation(const QString& filePath) const;

    // Undo/Redo operations
    bool canUndoOperation(const QString& operationId) const;
    bool undoOperation(const QString& operationId);
    bool canRedoOperation(const QString& operationId) const;
    bool redoOperation(const QString& operationId);

    // Operation history
    QList<OperationRecord> getOperationHistory() const;
    QList<OperationRecord> getRecentOperations(int count = 10) const;
    OperationRecord getOperation(const QString& operationId) const;
    void clearHistory();
    void exportHistory(const QString& filePath) const;
    bool importHistory(const QString& filePath);

    // Configuration
    void setMaxHistorySize(int maxOperations);
    int maxHistorySize() const;
    
    void setConfirmationRequired(OperationType operation, bool required);
    bool isConfirmationRequired(OperationType operation) const;

    void setBackupEnabled(bool enabled);
    bool isBackupEnabled() const;
    
    void setBackupLocation(const QString& location);
    QString backupLocation() const;

    // Status queries
    bool hasActiveOperations() const;
    int activeOperationCount() const;
    QStringList getActiveOperationIds() const;

    // Statistics
    struct Statistics {
        int totalOperations = 0;
        int successfulOperations = 0;
        int failedOperations = 0;
        int undoneOperations = 0;
        qint64 totalBytesProcessed = 0;
        qint64 totalBytesSaved = 0;        // Space reclaimed
        std::chrono::milliseconds totalTime{0};
        
        double successRate() const;
        QString summaryText() const;
    };
    
    Statistics getStatistics() const;

public slots:
    void cancelOperation(const QString& operationId);
    void cancelAllOperations();

signals:
    // Operation lifecycle
    void operationStarted(const OperationRecord& record);
    void operationProgress(const QString& operationId, int percentage);
    void operationCompleted(const OperationRecord& record);
    void operationFailed(const OperationRecord& record);
    void operationCancelled(const QString& operationId);
    
    // Batch operations
    void batchStarted(const QString& batchId, int totalFiles);
    void batchProgress(const QString& batchId, int completed, int total);
    void batchCompleted(const QString& batchId);
    void batchFailed(const QString& batchId, const QString& error);
    
    // History events
    void historyUpdated();
    void operationUndone(const QString& operationId);
    void operationRedone(const QString& operationId);
    
    // Safety events
    void protectedFileAccess(const QString& filePath, OperationType operation);
    void validationFailed(const ValidationResult& result);
    void confirmationRequired(const QString& operationId, const QString& message);
};
```

#### 4.2 Confirmation Dialog Interface

```cpp
// Interface for confirmation dialogs
class IConfirmationDialog {
public:
    enum class Response {
        Confirm,
        Cancel,
        Skip,
        SkipAll,
        ConfirmAll
    };
    
    virtual ~IConfirmationDialog() = default;
    virtual Response showConfirmation(const QString& title,
                                    const QString& message,
                                    const QStringList& options = {}) = 0;
    virtual bool isEnabled() const = 0;
    virtual void setEnabled(bool enabled) = 0;
};
```

---

## Platform Abstraction APIs

### 1. Platform File Operations Interface

```cpp
class IPlatformFileOperations {
public:
    enum class FilePermission {
        Read = 0x01,
        Write = 0x02,
        Execute = 0x04,
        All = Read | Write | Execute
    };
    Q_DECLARE_FLAGS(FilePermissions, FilePermission)

    struct FileSystemInfo {
        QString path;
        QString fileSystemType;     // ext4, NTFS, APFS, etc.
        qint64 totalSpace;
        qint64 availableSpace;
        qint64 freeSpace;
        bool isReadOnly;
        bool supportsSymlinks;
        bool supportsHardLinks;
        bool caseSensitive;
        
        double usagePercentage() const;
        QString formattedSize() const;
    };

    virtual ~IPlatformFileOperations() = default;

    // File operations
    virtual bool moveToTrash(const QString& filePath) = 0;
    virtual bool moveToTrash(const QStringList& filePaths) = 0;
    virtual QString getTrashLocation() = 0;
    virtual bool isInTrash(const QString& filePath) = 0;
    virtual bool restoreFromTrash(const QString& filePath) = 0;
    virtual bool emptyTrash() = 0;

    // File system queries
    virtual bool exists(const QString& path) = 0;
    virtual bool isDirectory(const QString& path) = 0;
    virtual bool isFile(const QString& path) = 0;
    virtual bool isSymlink(const QString& path) = 0;
    virtual bool isReadable(const QString& path) = 0;
    virtual bool isWritable(const QString& path) = 0;
    virtual bool isExecutable(const QString& path) = 0;

    // Permission management
    virtual FilePermissions getPermissions(const QString& path) = 0;
    virtual bool setPermissions(const QString& path, FilePermissions permissions) = 0;
    virtual bool hasPermission(const QString& path, FilePermission permission) = 0;

    // System paths
    virtual bool isSystemDirectory(const QString& path) = 0;
    virtual bool isSystemFile(const QString& path) = 0;
    virtual QStringList getSystemExclusionPaths() = 0;
    virtual QStringList getUserDirectories() = 0;  // Home, Documents, Downloads, etc.

    // File system information
    virtual FileSystemInfo getFileSystemInfo(const QString& path) = 0;
    virtual QStringList getMountPoints() = 0;
    virtual QString getFileSystemType(const QString& path) = 0;
    
    // Platform-specific features
    virtual bool supportsExtendedAttributes() const = 0;
    virtual bool supportsCaseSensitiveNames() const = 0;
    virtual bool supportsLongPaths() const = 0;
    virtual qint64 getMaxPathLength() const = 0;
    virtual qint64 getMaxFileNameLength() const = 0;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(IPlatformFileOperations::FilePermissions)
```

### 2. System Integration Interface

```cpp
class ISystemIntegration {
public:
    enum class NotificationType {
        Info,
        Warning, 
        Error,
        Success
    };

    virtual ~ISystemIntegration() = default;

    // Desktop integration
    virtual bool createDesktopShortcut(const QString& name, 
                                     const QString& executablePath,
                                     const QString& iconPath = QString()) = 0;
    virtual bool removeDesktopShortcut(const QString& name) = 0;
    virtual bool addToStartMenu(const QString& name,
                               const QString& executablePath,
                               const QString& category = QString()) = 0;
    virtual bool removeFromStartMenu(const QString& name) = 0;

    // File associations
    virtual bool registerFileType(const QString& extension,
                                 const QString& description,
                                 const QString& executablePath,
                                 const QString& iconPath = QString()) = 0;
    virtual bool unregisterFileType(const QString& extension) = 0;
    virtual QStringList getRegisteredFileTypes() = 0;

    // Context menu integration
    virtual bool addContextMenuItem(const QString& menuText,
                                  const QString& command,
                                  const QString& iconPath = QString()) = 0;
    virtual bool removeContextMenuItem(const QString& menuText) = 0;

    // Notifications
    virtual bool showNotification(const QString& title,
                                const QString& message,
                                NotificationType type = NotificationType::Info,
                                int timeoutMs = 5000) = 0;
    virtual bool isNotificationSupported() const = 0;

    // System information
    virtual QString getOSVersion() const = 0;
    virtual QString getSystemArchitecture() const = 0;
    virtual qint64 getTotalMemory() const = 0;
    virtual qint64 getAvailableMemory() const = 0;
    virtual int getCpuCoreCount() const = 0;

    // Application integration
    virtual bool setAutoStart(bool enabled) = 0;
    virtual bool isAutoStartEnabled() const = 0;
    virtual bool openFileManager(const QString& path) = 0;
    virtual bool openFile(const QString& filePath) = 0;
    virtual bool openUrl(const QString& url) = 0;
};
```

### 3. Platform Factory

```cpp
class PlatformFactory {
public:
    enum class Platform {
        Linux,
        Windows, 
        MacOS,
        Unknown
    };

    static Platform getCurrentPlatform();
    static QString getPlatformName();
    static QString getPlatformVersion();

    // Factory methods
    static std::unique_ptr<IPlatformFileOperations> createFileOperations();
    static std::unique_ptr<ISystemIntegration> createSystemIntegration();
    static std::unique_ptr<IConfirmationDialog> createConfirmationDialog();

    // Platform detection
    static bool isLinux();
    static bool isWindows();
    static bool isMacOS();
    static bool isUnix();
    
    // Feature detection
    static bool supportsTrash();
    static bool supportsNotifications();
    static bool supportsDesktopIntegration();
    static bool supportsContextMenu();
};
```

---

## GUI Component APIs

### 1. Main Window Interface

```cpp
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    enum class ViewMode {
        ScanSetup,
        ScanProgress, 
        Results,
        History,
        Settings
    };

    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    // View management
    void setViewMode(ViewMode mode);
    ViewMode currentViewMode() const;

    // Integration with core components
    void setFileScanner(FileScanner* scanner);
    void setHashCalculator(HashCalculator* calculator);
    void setDuplicateDetector(DuplicateDetector* detector);
    void setSafetyManager(SafetyManager* manager);

    // Status management
    void updateScanProgress(int percentage, const QString& status);
    void showScanResults(const QList<DuplicateDetector::DuplicateGroup>& groups);
    void showError(const QString& title, const QString& message);
    void showSuccess(const QString& title, const QString& message);

public slots:
    void startScan();
    void pauseScan();
    void cancelScan();
    void showSettings();
    void showAbout();
    void exportResults();

signals:
    void scanRequested(const FileScanner::ScanOptions& options);
    void scanCancelled();
    void filesSelectedForDeletion(const QStringList& filePaths);
    void settingsChanged();
    void applicationExit();

protected:
    void closeEvent(QCloseEvent* event) override;
    void changeEvent(QEvent* event) override;

private:
    class MainWindowPrivate;
    std::unique_ptr<MainWindowPrivate> d;
};
```

### 2. Results Widget Interface

```cpp
class ResultsWidget : public QWidget {
    Q_OBJECT

public:
    enum class DisplayMode {
        GroupedView,    // Group duplicates together
        ListView,       // Flat list of all files
        TreeView       // Hierarchical tree view
    };

    enum class SortMode {
        BySize,
        ByCount,
        ByName,
        ByDate,
        ByWastedSpace
    };

    explicit ResultsWidget(QWidget* parent = nullptr);
    ~ResultsWidget() override;

    // Data management
    void setDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups);
    void clearResults();
    void refreshResults();

    // Display configuration
    void setDisplayMode(DisplayMode mode);
    DisplayMode displayMode() const;
    
    void setSortMode(SortMode mode, Qt::SortOrder order = Qt::DescendingOrder);
    SortMode sortMode() const;
    Qt::SortOrder sortOrder() const;

    // Selection management
    QStringList getSelectedFiles() const;
    QStringList getSelectedGroups() const;
    void selectRecommended();
    void selectAll();
    void selectNone();

    // Statistics
    int totalGroups() const;
    int totalFiles() const;
    qint64 totalWastedSpace() const;
    qint64 selectedSpaceSavings() const;

public slots:
    void showGroupDetails(const QString& groupId);
    void previewFile(const QString& filePath);
    void openFileLocation(const QString& filePath);
    void selectGroup(const QString& groupId, bool selected);

signals:
    void selectionChanged();
    void fileDoubleClicked(const QString& filePath);
    void groupExpanded(const QString& groupId);
    void deleteRequested(const QStringList& filePaths);
    void statisticsUpdated(int groups, int files, qint64 wastedSpace);

private:
    class ResultsWidgetPrivate;
    std::unique_ptr<ResultsWidgetPrivate> d;
};
```

---

## Configuration and Settings APIs

### 1. Configuration Manager

```cpp
class ConfigurationManager : public QObject {
    Q_OBJECT

public:
    // Setting categories
    enum class Category {
        General,
        Scanning,
        Detection,
        Safety,
        Performance,
        UI,
        Advanced
    };

    static ConfigurationManager& instance();

    // Generic setting access
    QVariant getValue(const QString& key, const QVariant& defaultValue = QVariant()) const;
    void setValue(const QString& key, const QVariant& value);
    bool contains(const QString& key) const;
    void remove(const QString& key);

    // Typed setting access
    template<typename T>
    T getValue(const QString& key, const T& defaultValue = T{}) const;
    
    template<typename T>
    void setValue(const QString& key, const T& value);

    // Category management
    QStringList getKeys(Category category) const;
    void resetCategory(Category category);
    void exportCategory(Category category, const QString& filePath) const;
    bool importCategory(Category category, const QString& filePath);

    // Validation
    bool validateSetting(const QString& key, const QVariant& value) const;
    QStringList validateAllSettings() const;  // Returns list of invalid settings

    // File operations
    void saveToFile(const QString& filePath) const;
    bool loadFromFile(const QString& filePath);
    void resetToDefaults();
    QString getConfigFilePath() const;

    // Schema management
    void registerSettingSchema(const QString& key, const QVariantMap& schema);
    QVariantMap getSettingSchema(const QString& key) const;
    bool hasSchema(const QString& key) const;

public slots:
    void sync();  // Force immediate save to disk

signals:
    void settingChanged(const QString& key, const QVariant& value);
    void categoryChanged(Category category);
    void settingsReset();
    void settingsLoaded();
    void settingsSaved();

private:
    explicit ConfigurationManager(QObject* parent = nullptr);
    ~ConfigurationManager() override;

    class ConfigurationManagerPrivate;
    std::unique_ptr<ConfigurationManagerPrivate> d;
};
```

### 2. Settings Schema Definition

```cpp
namespace SettingKeys {
    // General settings
    constexpr const char* GENERAL_LANGUAGE = "general/language";
    constexpr const char* GENERAL_THEME = "general/theme";
    constexpr const char* GENERAL_AUTO_UPDATE = "general/autoUpdate";
    constexpr const char* GENERAL_STARTUP_MODE = "general/startupMode";

    // Scanning settings
    constexpr const char* SCAN_MIN_FILE_SIZE = "scan/minimumFileSize";
    constexpr const char* SCAN_INCLUDE_HIDDEN = "scan/includeHiddenFiles";
    constexpr const char* SCAN_FOLLOW_SYMLINKS = "scan/followSymlinks";
    constexpr const char* SCAN_EXCLUDED_PATTERNS = "scan/excludedPatterns";
    constexpr const char* SCAN_MAX_DEPTH = "scan/maximumDepth";

    // Detection settings
    constexpr const char* DETECTION_MODE = "detection/defaultMode";
    constexpr const char* DETECTION_HASH_ALGORITHM = "detection/hashAlgorithm";
    constexpr const char* DETECTION_USE_CACHE = "detection/useCache";
    constexpr const char* DETECTION_CACHE_SIZE = "detection/cacheSize";

    // Safety settings
    constexpr const char* SAFETY_CONFIRM_DELETE = "safety/confirmDelete";
    constexpr const char* SAFETY_CREATE_BACKUP = "safety/createBackup";
    constexpr const char* SAFETY_BACKUP_LOCATION = "safety/backupLocation";
    constexpr const char* SAFETY_MAX_HISTORY = "safety/maxHistorySize";

    // Performance settings
    constexpr const char* PERF_MAX_THREADS = "performance/maxThreads";
    constexpr const char* PERF_MAX_MEMORY = "performance/maxMemoryUsage";
    constexpr const char* PERF_CHUNK_SIZE = "performance/chunkSize";
    constexpr const char* PERF_PROGRESS_UPDATE = "performance/progressUpdateInterval";

    // UI settings
    constexpr const char* UI_WINDOW_GEOMETRY = "ui/windowGeometry";
    constexpr const char* UI_WINDOW_STATE = "ui/windowState";
    constexpr const char* UI_SHOW_THUMBNAILS = "ui/showThumbnails";
    constexpr const char* UI_RESULT_DISPLAY_MODE = "ui/resultDisplayMode";
}

namespace DefaultValues {
    constexpr qint64 MIN_FILE_SIZE = 1024;  // 1KB
    constexpr bool INCLUDE_HIDDEN_FILES = false;
    constexpr bool FOLLOW_SYMLINKS = false;
    constexpr int MAX_THREADS = 4;
    constexpr qint64 MAX_MEMORY_USAGE = 512 * 1024 * 1024;  // 512MB
    constexpr int CACHE_SIZE = 10000;
    constexpr bool CONFIRM_DELETE = true;
    constexpr int PROGRESS_UPDATE_INTERVAL = 250;  // milliseconds
}
```

---

## Error Handling and Logging APIs

### 1. Error Management

```cpp
namespace ErrorCodes {
    enum class Category {
        FileSystem = 1000,
        Network = 2000,
        Calculation = 3000,
        UI = 4000,
        Platform = 5000,
        Configuration = 6000
    };

    enum class FileSystemError {
        AccessDenied = 1001,
        FileNotFound = 1002,
        DirectoryNotFound = 1003,
        InsufficientSpace = 1004,
        FileLocked = 1005,
        InvalidPath = 1006,
        PermissionDenied = 1007
    };

    enum class CalculationError {
        HashCalculationFailed = 3001,
        FileCorrupted = 3002,
        OutOfMemory = 3003,
        OperationCancelled = 3004,
        InvalidAlgorithm = 3005
    };
}

class ErrorManager : public QObject {
    Q_OBJECT

public:
    struct ErrorInfo {
        int code;
        QString category;
        QString message;
        QString details;
        QString component;
        QDateTime timestamp;
        QString severity;
        QString suggestedAction;
        
        QString formatMessage() const;
        bool isCritical() const;
    };

    static ErrorManager& instance();

    // Error reporting
    void reportError(int code, const QString& message, 
                    const QString& component = QString(),
                    const QString& details = QString());
    void reportWarning(const QString& message, const QString& component = QString());
    void reportInfo(const QString& message, const QString& component = QString());

    // Error retrieval
    QList<ErrorInfo> getErrors(const QString& component = QString()) const;
    QList<ErrorInfo> getRecentErrors(int count = 50) const;
    ErrorInfo getLastError() const;
    int getErrorCount() const;

    // Error management
    void clearErrors();
    void clearComponent(const QString& component);
    void exportErrors(const QString& filePath) const;

signals:
    void errorOccurred(const ErrorInfo& error);
    void warningOccurred(const ErrorInfo& warning);
    void errorsCleared();

private:
    explicit ErrorManager(QObject* parent = nullptr);
    
    class ErrorManagerPrivate;
    std::unique_ptr<ErrorManagerPrivate> d;
};
```

### 2. Logging System

```cpp
class Logger {
public:
    enum class Level {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warning = 3,
        Error = 4,
        Critical = 5
    };

    enum class Output {
        Console = 0x01,
        File = 0x02,
        SystemLog = 0x04,
        Network = 0x08
    };
    Q_DECLARE_FLAGS(OutputFlags, Output)

    struct LogEntry {
        Level level;
        QString message;
        QString category;
        QString component;
        QDateTime timestamp;
        QThread* thread;
        QString function;
        QString file;
        int line;
        
        QString formatMessage() const;
    };

    // Configuration
    static void setLevel(Level level);
    static Level getLevel();
    
    static void setOutputs(OutputFlags outputs);
    static OutputFlags getOutputs();
    
    static void setLogFile(const QString& filePath);
    static QString getLogFile();
    
    static void setMaxFileSize(qint64 bytes);
    static void setMaxFiles(int count);

    // Logging methods
    template<typename... Args>
    static void trace(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void debug(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void info(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void warning(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void error(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void critical(const QString& format, Args&&... args);

    // Category-specific logging
    static void logToCategory(const QString& category, Level level, 
                             const QString& message);

    // Log management
    static QList<LogEntry> getRecentLogs(int count = 100);
    static QList<LogEntry> getLogsForCategory(const QString& category);
    static void clearLogs();
    static void rotateLogs();

    // Utility methods
    static QString levelToString(Level level);
    static Level stringToLevel(const QString& levelStr);
    static bool isEnabled(Level level);

private:
    class LoggerPrivate;
    static std::unique_ptr<LoggerPrivate> s_d;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(Logger::OutputFlags)

// Convenience macros for logging with source location
#define LOG_TRACE(format, ...) \
    Logger::trace(format, ##__VA_ARGS__)
    
#define LOG_DEBUG(format, ...) \
    Logger::debug(format, ##__VA_ARGS__)
    
#define LOG_INFO(format, ...) \
    Logger::info(format, ##__VA_ARGS__)
    
#define LOG_WARNING(format, ...) \
    Logger::warning(format, ##__VA_ARGS__)
    
#define LOG_ERROR(format, ...) \
    Logger::error(format, ##__VA_ARGS__)
    
#define LOG_CRITICAL(format, ...) \
    Logger::critical(format, ##__VA_ARGS__)
```

---

## Testing APIs

### 1. Test Framework Interface

```cpp
class TestFramework : public QObject {
    Q_OBJECT

public:
    struct TestResult {
        QString testName;
        bool passed;
        std::chrono::milliseconds duration;
        QString errorMessage;
        QStringList details;
        
        QString summary() const;
    };

    struct TestSuite {
        QString name;
        QList<TestResult> results;
        int passedCount() const;
        int failedCount() const;
        double successRate() const;
    };

    explicit TestFramework(QObject* parent = nullptr);
    virtual ~TestFramework() = default;

    // Test execution
    virtual void runAllTests() = 0;
    virtual void runTestSuite(const QString& suiteName) = 0;
    virtual void runTest(const QString& testName) = 0;

    // Results access
    virtual QList<TestSuite> getTestSuites() const = 0;
    virtual TestSuite getTestSuite(const QString& name) const = 0;
    virtual TestResult getTestResult(const QString& testName) const = 0;

    // Test data management
    virtual void createTestData() = 0;
    virtual void cleanupTestData() = 0;
    virtual QString getTestDataPath() const = 0;

signals:
    void testStarted(const QString& testName);
    void testCompleted(const TestResult& result);
    void testSuiteCompleted(const TestSuite& suite);
    void allTestsCompleted();
};
```

### 2. Mock Interfaces

```cpp
// Mock file system for testing
class MockFileSystem : public IPlatformFileOperations {
public:
    explicit MockFileSystem();
    
    // Mock configuration
    void addFile(const QString& path, qint64 size);
    void addDirectory(const QString& path);
    void setFilePermissions(const QString& path, FilePermissions permissions);
    void simulateError(const QString& path, const QString& error);
    
    // IPlatformFileOperations implementation
    bool moveToTrash(const QString& filePath) override;
    bool exists(const QString& path) override;
    bool isDirectory(const QString& path) override;
    // ... other interface methods
    
    // Verification methods
    bool wasMovedToTrash(const QString& filePath) const;
    QStringList getTrashContents() const;
    int getOperationCount() const;

private:
    struct MockFile {
        QString path;
        qint64 size;
        FilePermissions permissions;
        bool isDir;
        bool inTrash;
    };
    
    QHash<QString, MockFile> m_files;
    QStringList m_errors;
    int m_operationCount = 0;
};
```

This comprehensive API design provides the foundation for implementing all components of the CloneClean application. Each interface is designed to be testable, maintainable, and extensible, following modern C++ and Qt best practices.