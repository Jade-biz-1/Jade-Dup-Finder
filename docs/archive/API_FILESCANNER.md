# FileScanner API Documentation

## Overview

The `FileScanner` class is a core component responsible for recursively traversing directories, filtering files based on various criteria, and providing file information to downstream components. It's designed for high-performance scanning of large directory structures with robust error handling.

## Table of Contents

- [Quick Start](#quick-start)
- [Class Overview](#class-overview)
- [Data Structures](#data-structures)
- [Methods](#methods)
- [Signals](#signals)
- [Pattern Matching](#pattern-matching)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)

## Quick Start

```cpp
#include "file_scanner.h"

// Create scanner instance
FileScanner scanner;

// Configure scan options
FileScanner::ScanOptions options;
options.targetPaths = {"/path/to/scan"};
options.minimumFileSize = 1024 * 1024;  // 1MB minimum
options.includePatterns = {"*.jpg", "*.png", "*.gif"};
options.excludePatterns = {"*.tmp", ".DS_Store"};

// Connect signals
connect(&scanner, &FileScanner::scanProgress, 
        [](int processed, int total, const QString& path) {
    qDebug() << "Progress:" << processed << "files," << path;
});

connect(&scanner, &FileScanner::scanCompleted, [&scanner]() {
    auto files = scanner.getScannedFiles();
    auto stats = scanner.getScanStatistics();
    qDebug() << "Scan complete:" << files.size() << "files found";
    qDebug() << "Scan rate:" << stats.filesPerSecond << "files/sec";
});

// Start scanning
scanner.startScan(options);
```

## Class Overview

```cpp
class FileScanner : public QObject
```

The FileScanner class inherits from QObject to provide signal/slot functionality for asynchronous scanning operations.

### Key Features

- **Recursive Directory Traversal**: Efficiently scans nested directory structures
- **Pattern-Based Filtering**: Supports glob and regex patterns for file inclusion/exclusion
- **Size-Based Filtering**: Filter files by minimum and maximum size
- **Error Handling**: Robust error detection and reporting with specific error types
- **Performance Optimization**: Memory-efficient scanning with optional caching
- **Progress Reporting**: Real-time progress updates during scanning
- **Cancellation Support**: Graceful scan cancellation at any time

## Data Structures

### ScanOptions

Configuration structure for scan operations.

```cpp
struct ScanOptions {
    QStringList targetPaths;           // Directories/files to scan
    qint64 minimumFileSize = 1024;     // Skip files smaller than this (1KB default)
    qint64 maximumFileSize = -1;       // No limit by default (-1)
    QStringList includePatterns;       // File patterns to include (*.jpg, etc.)
    QStringList excludePatterns;       // File patterns to exclude
    bool caseSensitivePatterns = false; // Case-sensitive pattern matching
    bool includeHiddenFiles = false;   // Include hidden/dot files
    bool followSymlinks = false;       // Follow symbolic links
    bool scanSystemDirectories = false; // Include system directories
    
    // Performance options
    bool streamingMode = false;        // Don't store all files in memory
    int estimatedFileCount = 0;        // Hint for capacity reservation
    int progressBatchSize = 100;       // Emit progress every N files
    bool enableMetadataCache = false;  // Cache file metadata
    int metadataCacheSizeLimit = 10000; // Maximum cache entries
};
```

**Field Details:**

- `targetPaths`: List of directories or files to scan. Can include multiple paths.
- `minimumFileSize`: Files smaller than this size (in bytes) are excluded. Default: 1KB.
- `maximumFileSize`: Files larger than this size are excluded. -1 means no limit.
- `includePatterns`: Only files matching these patterns are included. Empty means include all.
- `excludePatterns`: Files matching these patterns are excluded.
- `caseSensitivePatterns`: Whether pattern matching is case-sensitive.
- `includeHiddenFiles`: Whether to include hidden files (starting with . on Unix).
- `followSymlinks`: Whether to follow symbolic links (use with caution to avoid loops).
- `scanSystemDirectories`: Whether to scan system directories (/sys, /proc, etc.).
- `streamingMode`: If true, files are emitted but not stored in memory (for very large scans).
- `estimatedFileCount`: Hint for pre-allocating memory. Improves performance if known.
- `progressBatchSize`: How often to emit progress signals (default: every 100 files).
- `enableMetadataCache`: Enable caching of file metadata for repeated scans.
- `metadataCacheSizeLimit`: Maximum number of cached entries.

### FileInfo

Information about a scanned file.

```cpp
struct FileInfo {
    QString filePath;      // Full absolute path to the file
    qint64 fileSize;       // File size in bytes
    QString fileName;      // File name without path
    QString directory;     // Directory containing the file
    QDateTime lastModified; // Last modification timestamp
};
```

### ScanStatistics

Statistics collected during a scan operation.

```cpp
struct ScanStatistics {
    int totalFilesScanned = 0;          // Total files found and processed
    int totalDirectoriesScanned = 0;    // Total directories traversed
    qint64 totalBytesScanned = 0;       // Total bytes in scanned files
    int filesFilteredBySize = 0;        // Files excluded by size constraints
    int filesFilteredByPattern = 0;     // Files excluded by pattern matching
    int filesFilteredByHidden = 0;      // Files excluded (hidden files)
    int directoriesSkipped = 0;         // Directories skipped
    int errorsEncountered = 0;          // Total errors during scan
    qint64 scanDurationMs = 0;          // Scan duration in milliseconds
    double filesPerSecond = 0.0;        // Scan rate (files/second)
    qint64 peakMemoryUsage = 0;         // Peak memory usage (if available)
};
```

### ScanError

Enumeration of error types that can occur during scanning.

```cpp
enum class ScanError {
    None,                  // No error
    PermissionDenied,      // Cannot read file/directory
    FileSystemError,       // General I/O error
    NetworkTimeout,        // Network drive timeout
    DiskReadError,         // Bad sectors, disk errors
    PathTooLong,          // Path exceeds system limits
    UnknownError          // Catch-all for unexpected errors
};
```

### ScanErrorInfo

Detailed information about a scan error.

```cpp
struct ScanErrorInfo {
    ScanError errorType;       // Type of error
    QString filePath;          // Path where error occurred
    QString errorMessage;      // Human-readable error description
    QString systemErrorCode;   // OS-specific error code
    QDateTime timestamp;       // When the error occurred
};
```

## Methods

### Constructor

```cpp
explicit FileScanner(QObject* parent = nullptr);
```

Creates a new FileScanner instance.

**Parameters:**
- `parent`: Optional parent QObject for memory management

### startScan

```cpp
void startScan(const ScanOptions& options);
```

Starts an asynchronous scan operation with the specified options.

**Parameters:**
- `options`: ScanOptions structure containing scan configuration

**Behavior:**
- Emits `scanStarted()` signal when scan begins
- Emits `scanProgress()` signals periodically during scan
- Emits `fileFound()` for each file discovered (if not in streaming mode)
- Emits `scanCompleted()` when scan finishes successfully
- Emits `scanCancelled()` if scan is cancelled
- Emits `scanError()` for each error encountered
- Emits `scanStatistics()` with final statistics upon completion

**Thread Safety:** This method should be called from the main thread.

### cancelScan

```cpp
void cancelScan();
```

Requests cancellation of the current scan operation.

**Behavior:**
- Sets cancellation flag
- Scan will stop at the next safe point
- Emits `scanCancelled()` signal when cancellation is complete
- Already scanned files remain available via `getScannedFiles()`

### isScanning

```cpp
bool isScanning() const;
```

Checks if a scan operation is currently in progress.

**Returns:** `true` if scanning, `false` otherwise

### getScannedFiles

```cpp
QVector<FileInfo> getScannedFiles() const;
```

Retrieves all files found during the scan.

**Returns:** Vector of FileInfo structures for all scanned files

**Note:** Returns empty vector if `streamingMode` is enabled in ScanOptions.

### getTotalFilesFound

```cpp
int getTotalFilesFound() const;
```

Gets the total number of files found during the scan.

**Returns:** Count of files found

### getTotalBytesScanned

```cpp
qint64 getTotalBytesScanned() const;
```

Gets the total size of all scanned files.

**Returns:** Total bytes scanned

### getScanErrors

```cpp
QList<ScanErrorInfo> getScanErrors() const;
```

Retrieves detailed information about all errors encountered during the scan.

**Returns:** List of ScanErrorInfo structures

### getTotalErrorsEncountered

```cpp
int getTotalErrorsEncountered() const;
```

Gets the total number of errors encountered during the scan.

**Returns:** Error count

### getScanStatistics

```cpp
ScanStatistics getScanStatistics() const;
```

Retrieves comprehensive statistics about the scan operation.

**Returns:** ScanStatistics structure with detailed metrics

### clearMetadataCache

```cpp
void clearMetadataCache();
```

Clears the metadata cache (if caching is enabled).

**Use Case:** Call this when you know files have changed and cached metadata may be stale.

## Signals

### scanStarted

```cpp
void scanStarted();
```

Emitted when a scan operation begins.

### scanProgress

```cpp
void scanProgress(int filesProcessed, int totalFiles, const QString& currentPath);
```

Emitted periodically during scanning to report progress.

**Parameters:**
- `filesProcessed`: Number of files processed so far
- `totalFiles`: Total files to process (-1 if unknown)
- `currentPath`: Path currently being scanned

**Frequency:** Emitted based on `progressBatchSize` in ScanOptions (default: every 100 files)

### scanCompleted

```cpp
void scanCompleted();
```

Emitted when the scan completes successfully.

### scanCancelled

```cpp
void scanCancelled();
```

Emitted when the scan is cancelled by user request.

### errorOccurred

```cpp
void errorOccurred(const QString& error);
```

Emitted when a general error occurs.

**Parameters:**
- `error`: Error message string

### fileFound

```cpp
void fileFound(const FileInfo& fileInfo);
```

Emitted for each file discovered during scanning.

**Parameters:**
- `fileInfo`: Information about the discovered file

**Note:** Not emitted in streaming mode to reduce signal overhead.

### scanError

```cpp
void scanError(ScanError errorType, const QString& path, const QString& description);
```

Emitted when a specific error occurs during scanning.

**Parameters:**
- `errorType`: Type of error (from ScanError enum)
- `path`: File or directory path where error occurred
- `description`: Human-readable error description

### scanErrorSummary

```cpp
void scanErrorSummary(int totalErrors, const QList<ScanErrorInfo>& errors);
```

Emitted at the end of scanning with a summary of all errors.

**Parameters:**
- `totalErrors`: Total number of errors encountered
- `errors`: List of detailed error information

### scanStatistics

```cpp
void scanStatistics(const ScanStatistics& statistics);
```

Emitted when scan completes with comprehensive statistics.

**Parameters:**
- `statistics`: ScanStatistics structure with detailed metrics

## Pattern Matching

The FileScanner supports two types of patterns:

### Glob Patterns

Standard wildcard patterns commonly used in file systems.

**Syntax:**
- `*` - Matches any sequence of characters
- `?` - Matches any single character
- `[abc]` - Matches any character in the set
- `[!abc]` - Matches any character not in the set

**Examples:**
```cpp
options.includePatterns = {
    "*.jpg",           // All JPEG files
    "*.png",           // All PNG files
    "IMG_????.jpg",    // IMG_0001.jpg, IMG_9999.jpg, etc.
    "photo[0-9].png"   // photo0.png through photo9.png
};

options.excludePatterns = {
    "*.tmp",           // Exclude temporary files
    ".DS_Store",       // Exclude macOS metadata
    "Thumbs.db",       // Exclude Windows thumbnails
    "*~"               // Exclude backup files
};
```

### Regex Patterns

For more complex matching, you can use regular expressions.

**Examples:**
```cpp
options.includePatterns = {
    ".*\\.(jpg|jpeg|png|gif)$",  // Image files (regex)
    "^IMG_\\d{4}\\.jpg$"          // IMG_0000.jpg to IMG_9999.jpg
};
```

### Pattern Matching Behavior

1. **Include Patterns**: If specified, only files matching at least one include pattern are included
2. **Exclude Patterns**: Files matching any exclude pattern are excluded
3. **Priority**: Exclude patterns take precedence over include patterns
4. **Case Sensitivity**: Controlled by `caseSensitivePatterns` option (default: case-insensitive)

### Pattern Performance

- Patterns are compiled once and cached for performance
- Pattern matching adds minimal overhead (< 5% of scan time)
- Use simpler glob patterns when possible for best performance

## Error Handling

The FileScanner implements robust error handling to ensure scans can continue even when individual files or directories are inaccessible.

### Error Types

See [ScanError](#scanerror) enum for all error types.

### Error Behavior

1. **Non-Critical Errors**: Scan continues after logging the error
   - Permission denied on a file/directory
   - Individual file read errors
   - Path too long errors

2. **Transient Errors**: Automatic retry (up to 2 times)
   - Network timeouts
   - Temporary I/O errors

3. **Critical Errors**: Scan stops immediately
   - Invalid target path
   - Out of memory

### Error Signals

Connect to error signals to handle errors in your application:

```cpp
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    switch (errorType) {
        case FileScanner::ScanError::PermissionDenied:
            qWarning() << "Permission denied:" << path;
            break;
        case FileScanner::ScanError::NetworkTimeout:
            qWarning() << "Network timeout:" << path;
            break;
        default:
            qWarning() << "Error:" << desc << "at" << path;
    }
});

connect(&scanner, &FileScanner::scanErrorSummary,
        [](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    if (totalErrors > 0) {
        qWarning() << "Scan completed with" << totalErrors << "errors";
        // Log or display errors to user
    }
});
```

### Accessing Error Information

After scan completion, retrieve error details:

```cpp
auto errors = scanner.getScanErrors();
for (const auto& error : errors) {
    qDebug() << "Error at" << error.filePath
             << ":" << error.errorMessage
             << "(code:" << error.systemErrorCode << ")";
}
```

## Performance Optimization

The FileScanner is optimized for scanning large directory structures efficiently.

### Memory Optimization

**Problem**: Storing 100,000+ files in memory can consume significant RAM.

**Solutions**:

1. **Streaming Mode**: Don't store files in memory
```cpp
options.streamingMode = true;
connect(&scanner, &FileScanner::fileFound, [](const FileScanner::FileInfo& file) {
    // Process file immediately
    processFile(file);
});
```

2. **Estimated File Count**: Pre-allocate memory
```cpp
options.estimatedFileCount = 50000;  // If you know approximate count
```

### Progress Update Optimization

**Problem**: Emitting progress for every file can cause UI lag.

**Solution**: Batch progress updates
```cpp
options.progressBatchSize = 500;  // Update every 500 files instead of 100
```

### Metadata Caching

**Use Case**: Repeated scans of the same directories.

**Enable Caching**:
```cpp
options.enableMetadataCache = true;
options.metadataCacheSizeLimit = 20000;  // Cache up to 20,000 files
```

**Cache Behavior**:
- Caches file size and modification time
- Invalidates entries if file modification time changes
- Automatically enforces size limit (LRU eviction)

**Clear Cache**:
```cpp
scanner.clearMetadataCache();  // When you know files have changed
```

### Performance Targets

The FileScanner is designed to meet these performance targets:

- **Scan Rate**: ≥ 1,000 files/minute on SSD
- **Memory Usage**: ≤ 100MB for 100,000 files
- **Progress Latency**: ≤ 100ms per update
- **Pattern Overhead**: ≤ 5% of total scan time

### Performance Monitoring

Use scan statistics to monitor performance:

```cpp
connect(&scanner, &FileScanner::scanStatistics,
        [](const FileScanner::ScanStatistics& stats) {
    qDebug() << "Scan rate:" << stats.filesPerSecond << "files/sec";
    qDebug() << "Duration:" << stats.scanDurationMs << "ms";
    qDebug() << "Memory:" << stats.peakMemoryUsage << "bytes";
});
```

## Best Practices

1. **Always connect to error signals** to handle scan errors gracefully
2. **Use pattern matching** instead of post-scan filtering for better performance
3. **Enable streaming mode** for very large scans (100,000+ files)
4. **Provide estimated file count** when known for better memory allocation
5. **Adjust progress batch size** based on your UI responsiveness needs
6. **Clear metadata cache** when you know files have changed significantly
7. **Test with realistic data** to ensure performance meets your requirements

## See Also

- [FileScanner Usage Examples](FILESCANNER_EXAMPLES.md)
- [FileScanner Error Handling Guide](FILESCANNER_ERROR_HANDLING.md)
- [FileScanner Performance Tuning](FILESCANNER_PERFORMANCE.md)
- [Integration with HashCalculator](FILESCANNER_INTEGRATION.md)
- [Migration Guide](FILESCANNER_MIGRATION.md)
