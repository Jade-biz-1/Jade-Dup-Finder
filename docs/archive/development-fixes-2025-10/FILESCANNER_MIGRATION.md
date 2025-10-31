# FileScanner Migration Guide

This guide helps you migrate existing code to use the enhanced FileScanner features added in the completion phase.

## Table of Contents

- [Overview](#overview)
- [What's New](#whats-new)
- [Breaking Changes](#breaking-changes)
- [Migration Steps](#migration-steps)
- [Feature-by-Feature Migration](#feature-by-feature-migration)
- [Common Migration Scenarios](#common-migration-scenarios)
- [Troubleshooting](#troubleshooting)

## Overview

The FileScanner completion phase added several new features:

1. **Pattern Matching**: Glob and regex pattern support
2. **Enhanced Error Handling**: Specific error types and detailed reporting
3. **Performance Optimizations**: Streaming mode, metadata caching, progress batching
4. **Statistics**: Comprehensive scan statistics

This guide helps you adopt these features in existing code.

## What's New

### New Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| Pattern Matching | Include/exclude files by pattern | More flexible filtering |
| Error Types | Specific error classification | Better error handling |
| Streaming Mode | Process files without storing all | Lower memory usage |
| Metadata Caching | Cache file metadata | Faster repeated scans |
| Progress Batching | Configurable progress frequency | Better UI responsiveness |
| Scan Statistics | Detailed performance metrics | Performance monitoring |

### New ScanOptions Fields

```cpp
// Old ScanOptions (basic)
struct ScanOptions {
    QStringList targetPaths;
    qint64 minimumFileSize;
    qint64 maximumFileSize;
    bool includeHiddenFiles;
    bool followSymlinks;
    bool scanSystemDirectories;
};

// New ScanOptions (enhanced)
struct ScanOptions {
    // Existing fields (unchanged)
    QStringList targetPaths;
    qint64 minimumFileSize = 1024;
    qint64 maximumFileSize = -1;
    bool includeHiddenFiles = false;
    bool followSymlinks = false;
    bool scanSystemDirectories = false;
    
    // NEW: Pattern matching
    QStringList includePatterns;
    QStringList excludePatterns;
    bool caseSensitivePatterns = false;
    
    // NEW: Performance options
    bool streamingMode = false;
    int estimatedFileCount = 0;
    int progressBatchSize = 100;
    bool enableMetadataCache = false;
    int metadataCacheSizeLimit = 10000;
};
```

### New Signals

```cpp
// NEW: Enhanced error signals
void scanError(ScanError errorType, const QString& path, const QString& description);
void scanErrorSummary(int totalErrors, const QList<ScanErrorInfo>& errors);

// NEW: Statistics signal
void scanStatistics(const ScanStatistics& statistics);
```

### New Methods

```cpp
// NEW: Error access
QList<ScanErrorInfo> getScanErrors() const;
int getTotalErrorsEncountered() const;

// NEW: Statistics access
ScanStatistics getScanStatistics() const;

// NEW: Cache management
void clearMetadataCache();
```

## Breaking Changes

### None!

The FileScanner completion phase maintains **full backward compatibility**. All existing code will continue to work without modifications.

### Default Value Changes

Some defaults have changed for better performance:

| Field | Old Default | New Default | Impact |
|-------|-------------|-------------|--------|
| `minimumFileSize` | 0 | 1024 (1KB) | Small files now skipped by default |
| `maximumFileSize` | 0 (no limit) | -1 (no limit) | Semantic change, same behavior |

**Migration**: If you relied on scanning files < 1KB, explicitly set:
```cpp
options.minimumFileSize = 0;  // Include all files
```

## Migration Steps

### Step 1: Update Your Code (Optional)

Since there are no breaking changes, you can migrate incrementally:

1. **Keep existing code working** (no changes needed)
2. **Add new features gradually** (pattern matching, error handling, etc.)
3. **Optimize performance** (streaming mode, caching, etc.)

### Step 2: Add Pattern Matching (Optional)

If you were filtering files after scanning, move filtering to scan options:

**Before**:
```cpp
// Old approach: Filter after scanning
FileScanner::ScanOptions options;
options.targetPaths = {"/path"};
scanner.startScan(options);

// Later, filter results
auto files = scanner.getScannedFiles();
QVector<FileScanner::FileInfo> imageFiles;
for (const auto& file : files) {
    if (file.fileName.endsWith(".jpg") || file.fileName.endsWith(".png")) {
        imageFiles.append(file);
    }
}
```

**After**:
```cpp
// New approach: Filter during scanning
FileScanner::ScanOptions options;
options.targetPaths = {"/path"};
options.includePatterns = {"*.jpg", "*.png"};  // Filter during scan
scanner.startScan(options);

// All results are already filtered
auto imageFiles = scanner.getScannedFiles();
```

**Benefits**:
- Faster (no post-processing)
- Lower memory usage (fewer files stored)
- Cleaner code

### Step 3: Enhance Error Handling (Recommended)

Add error handling to make your application more robust:

**Before**:
```cpp
// Old approach: Basic error handling
connect(&scanner, &FileScanner::errorOccurred,
        [](const QString& error) {
    qWarning() << "Error:" << error;
});
```

**After**:
```cpp
// New approach: Detailed error handling
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
            qWarning() << "Error:" << desc;
    }
});

connect(&scanner, &FileScanner::scanErrorSummary,
        [](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    if (totalErrors > 0) {
        qWarning() << "Scan completed with" << totalErrors << "errors";
    }
});
```

**Benefits**:
- Better error classification
- More informative error messages
- Error summary for user feedback

### Step 4: Optimize Performance (As Needed)

Add performance optimizations based on your use case:

**For large scans (> 100k files)**:
```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/path"};
options.streamingMode = true;  // Don't store all files

connect(&scanner, &FileScanner::fileFound,
        [](const FileScanner::FileInfo& file) {
    // Process file immediately
    processFile(file);
});
```

**For repeated scans**:
```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/path"};
options.enableMetadataCache = true;  // Cache metadata
```

**For fast scans with many files**:
```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/path"};
options.progressBatchSize = 500;  // Reduce progress update frequency
```

### Step 5: Add Statistics Monitoring (Optional)

Monitor scan performance:

```cpp
connect(&scanner, &FileScanner::scanStatistics,
        [](const FileScanner::ScanStatistics& stats) {
    qDebug() << "Scan rate:" << stats.filesPerSecond << "files/sec";
    qDebug() << "Duration:" << (stats.scanDurationMs / 1000.0) << "seconds";
    qDebug() << "Errors:" << stats.errorsEncountered;
});
```

## Feature-by-Feature Migration

### Pattern Matching

**Use Case**: You want to scan only specific file types.

**Before**:
```cpp
// Manual filtering
auto allFiles = scanner.getScannedFiles();
QVector<FileScanner::FileInfo> filtered;
for (const auto& file : allFiles) {
    QString ext = QFileInfo(file.fileName).suffix().toLower();
    if (ext == "jpg" || ext == "png" || ext == "gif") {
        filtered.append(file);
    }
}
```

**After**:
```cpp
// Pattern-based filtering
FileScanner::ScanOptions options;
options.includePatterns = {"*.jpg", "*.png", "*.gif"};
scanner.startScan(options);
auto filtered = scanner.getScannedFiles();  // Already filtered
```

### Error Handling

**Use Case**: You want to handle permission errors gracefully.

**Before**:
```cpp
// Generic error handling
connect(&scanner, &FileScanner::errorOccurred,
        [](const QString& error) {
    qWarning() << error;
});
```

**After**:
```cpp
// Specific error handling
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    if (errorType == FileScanner::ScanError::PermissionDenied) {
        // Handle permission errors specifically
        qDebug() << "Skipped (no permission):" << path;
    } else {
        qWarning() << "Error:" << desc;
    }
});
```

### Streaming Mode

**Use Case**: You're scanning millions of files and running out of memory.

**Before**:
```cpp
// Store all files in memory
scanner.startScan(options);
// ... wait for completion ...
auto files = scanner.getScannedFiles();  // Millions of files in memory!
for (const auto& file : files) {
    processFile(file);
}
```

**After**:
```cpp
// Process files as they're found
options.streamingMode = true;
connect(&scanner, &FileScanner::fileFound,
        [](const FileScanner::FileInfo& file) {
    processFile(file);  // Process immediately, not stored
});
scanner.startScan(options);
```

### Metadata Caching

**Use Case**: You scan the same directories repeatedly.

**Before**:
```cpp
// No caching, every scan reads metadata from disk
for (int i = 0; i < 10; ++i) {
    scanner.startScan(options);
    // ... wait ...
    // Each scan reads all file metadata from disk
}
```

**After**:
```cpp
// Enable caching for faster repeated scans
options.enableMetadataCache = true;
for (int i = 0; i < 10; ++i) {
    scanner.startScan(options);
    // ... wait ...
    // Subsequent scans use cached metadata (much faster)
}
```

### Progress Batching

**Use Case**: Your UI lags during fast scans.

**Before**:
```cpp
// Default: Progress every 100 files
connect(&scanner, &FileScanner::scanProgress,
        [](int processed, int total, const QString& path) {
    updateProgressBar(processed, total);  // Called frequently, UI lags
});
```

**After**:
```cpp
// Reduce progress frequency
options.progressBatchSize = 500;  // Progress every 500 files
connect(&scanner, &FileScanner::scanProgress,
        [](int processed, int total, const QString& path) {
    updateProgressBar(processed, total);  // Called less often, smoother UI
});
```

### Statistics

**Use Case**: You want to monitor scan performance.

**Before**:
```cpp
// Manual timing
QElapsedTimer timer;
timer.start();
scanner.startScan(options);
// ... wait ...
qint64 duration = timer.elapsed();
qDebug() << "Scan took" << duration << "ms";
```

**After**:
```cpp
// Automatic statistics
connect(&scanner, &FileScanner::scanStatistics,
        [](const FileScanner::ScanStatistics& stats) {
    qDebug() << "Scan took" << stats.scanDurationMs << "ms";
    qDebug() << "Scan rate:" << stats.filesPerSecond << "files/sec";
    qDebug() << "Files scanned:" << stats.totalFilesScanned;
    qDebug() << "Errors:" << stats.errorsEncountered;
});
scanner.startScan(options);
```

## Common Migration Scenarios

### Scenario 1: Simple File Scanner

**Before**:
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Documents"};
scanner.startScan(options);
```

**After** (no changes needed):
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Documents"};
scanner.startScan(options);  // Works exactly the same
```

### Scenario 2: Duplicate File Finder

**Before**:
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.minimumFileSize = 1024 * 1024;  // 1MB

connect(&scanner, &FileScanner::scanCompleted, [&]() {
    auto files = scanner.getScannedFiles();
    // Find duplicates...
});

scanner.startScan(options);
```

**After** (with enhancements):
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.minimumFileSize = 1024 * 1024;  // 1MB
options.excludePatterns = {".git/*", "node_modules/*"};  // NEW: Exclude common dirs
options.enableMetadataCache = true;  // NEW: Cache for repeated scans

connect(&scanner, &FileScanner::scanCompleted, [&]() {
    auto files = scanner.getScannedFiles();
    auto stats = scanner.getScanStatistics();  // NEW: Get statistics
    qDebug() << "Scanned" << stats.totalFilesScanned << "files in"
             << (stats.scanDurationMs / 1000.0) << "seconds";
    // Find duplicates...
});

connect(&scanner, &FileScanner::scanError,  // NEW: Handle errors
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    qWarning() << "Error:" << desc << "at" << path;
});

scanner.startScan(options);
```

### Scenario 3: Media File Organizer

**Before**:
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Pictures"};

connect(&scanner, &FileScanner::scanCompleted, [&]() {
    auto files = scanner.getScannedFiles();
    
    // Filter for media files
    QVector<FileScanner::FileInfo> mediaFiles;
    for (const auto& file : files) {
        QString ext = QFileInfo(file.fileName).suffix().toLower();
        if (ext == "jpg" || ext == "png" || ext == "mp4") {
            mediaFiles.append(file);
        }
    }
    
    // Organize media files...
});

scanner.startScan(options);
```

**After** (with pattern matching):
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Pictures"};
options.includePatterns = {"*.jpg", "*.png", "*.mp4"};  // NEW: Filter during scan

connect(&scanner, &FileScanner::scanCompleted, [&]() {
    auto mediaFiles = scanner.getScannedFiles();  // Already filtered!
    // Organize media files...
});

scanner.startScan(options);
```

### Scenario 4: Large-Scale Scanner

**Before**:
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/"};  // Scan entire system

connect(&scanner, &FileScanner::scanCompleted, [&]() {
    auto files = scanner.getScannedFiles();  // Millions of files, high memory!
    // Process files...
});

scanner.startScan(options);
```

**After** (with streaming mode):
```cpp
FileScanner scanner;
FileScanner::ScanOptions options;
options.targetPaths = {"/"};
options.streamingMode = true;  // NEW: Don't store all files
options.progressBatchSize = 1000;  // NEW: Reduce progress frequency

connect(&scanner, &FileScanner::fileFound,  // NEW: Process as found
        [](const FileScanner::FileInfo& file) {
    processFile(file);  // Process immediately
});

connect(&scanner, &FileScanner::scanStatistics,  // NEW: Get statistics
        [](const FileScanner::ScanStatistics& stats) {
    qDebug() << "Scanned" << stats.totalFilesScanned << "files";
});

scanner.startScan(options);
```

## Troubleshooting

### Issue: Minimum File Size Changed

**Problem**: Small files are no longer being scanned.

**Cause**: Default `minimumFileSize` changed from 0 to 1024 (1KB).

**Solution**:
```cpp
options.minimumFileSize = 0;  // Include all files, even empty ones
```

### Issue: Too Many Progress Updates

**Problem**: UI is lagging during fast scans.

**Cause**: Default progress batch size (100) may be too frequent for very fast scans.

**Solution**:
```cpp
options.progressBatchSize = 500;  // Or 1000 for very fast scans
```

### Issue: High Memory Usage

**Problem**: Running out of memory when scanning millions of files.

**Cause**: All files are stored in memory by default.

**Solution**:
```cpp
options.streamingMode = true;  // Don't store all files
connect(&scanner, &FileScanner::fileFound,
        [](const FileScanner::FileInfo& file) {
    processFile(file);  // Process immediately
});
```

### Issue: Slow Repeated Scans

**Problem**: Scanning the same directories repeatedly is slow.

**Cause**: Metadata is read from disk every time.

**Solution**:
```cpp
options.enableMetadataCache = true;  // Cache metadata
```

### Issue: Missing Error Information

**Problem**: Not getting detailed error information.

**Cause**: Only connected to old `errorOccurred` signal.

**Solution**:
```cpp
// Connect to new error signals
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    // Handle specific error types
});

connect(&scanner, &FileScanner::scanErrorSummary,
        [](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    // Get error summary
});
```

## See Also

- [FileScanner API Documentation](API_FILESCANNER.md)
- [Usage Examples](FILESCANNER_EXAMPLES.md)
- [Error Handling Guide](FILESCANNER_ERROR_HANDLING.md)
- [Performance Tuning Guide](FILESCANNER_PERFORMANCE.md)
- [Integration Examples](FILESCANNER_INTEGRATION.md)
