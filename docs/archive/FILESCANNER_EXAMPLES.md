# FileScanner Usage Examples

This document provides practical examples of using the FileScanner component for various use cases.

## Table of Contents

- [Basic Scanning](#basic-scanning)
- [Pattern Matching Examples](#pattern-matching-examples)
- [Size-Based Filtering](#size-based-filtering)
- [Progress Monitoring](#progress-monitoring)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Advanced Use Cases](#advanced-use-cases)

## Basic Scanning

### Simple Directory Scan

Scan a directory with default options:

```cpp
#include "file_scanner.h"

FileScanner scanner;

// Configure basic scan
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Documents"};

// Connect to completion signal
connect(&scanner, &FileScanner::scanCompleted, [&scanner]() {
    auto files = scanner.getScannedFiles();
    qDebug() << "Found" << files.size() << "files";
    
    for (const auto& file : files) {
        qDebug() << file.filePath << "-" << file.fileSize << "bytes";
    }
});

// Start scan
scanner.startScan(options);
```

### Scanning Multiple Directories

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {
    "/home/user/Documents",
    "/home/user/Pictures",
    "/home/user/Downloads"
};

scanner.startScan(options);
```

### Including Hidden Files

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.includeHiddenFiles = true;  // Include .dotfiles

scanner.startScan(options);
```

## Pattern Matching Examples

### Scanning for Image Files

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Pictures"};
options.includePatterns = {"*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"};

connect(&scanner, &FileScanner::scanCompleted, [&scanner]() {
    qDebug() << "Found" << scanner.getTotalFilesFound() << "image files";
});

scanner.startScan(options);
```

### Excluding Temporary and System Files

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/project"};
options.excludePatterns = {
    "*.tmp",           // Temporary files
    "*.bak",           // Backup files
    "*~",              // Editor backups
    ".DS_Store",       // macOS metadata
    "Thumbs.db",       // Windows thumbnails
    "desktop.ini",     // Windows folder settings
    ".git/*",          // Git repository
    "node_modules/*",  // Node.js dependencies
    "__pycache__/*"    // Python cache
};

scanner.startScan(options);
```

### Case-Sensitive Pattern Matching

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/code"};
options.includePatterns = {"*.cpp", "*.h"};  // Only lowercase extensions
options.caseSensitivePatterns = true;        // Exclude .CPP, .H, etc.

scanner.startScan(options);
```

### Using Regex Patterns

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/photos"};

// Match IMG_0001.jpg through IMG_9999.jpg
options.includePatterns = {"^IMG_\\d{4}\\.jpg$"};

// Or match multiple image formats with regex
options.includePatterns = {".*\\.(jpg|jpeg|png|gif|bmp)$"};

scanner.startScan(options);
```

### Complex Pattern Combinations

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/media"};

// Include only media files
options.includePatterns = {
    "*.mp4", "*.avi", "*.mkv",  // Video
    "*.mp3", "*.flac", "*.wav"  // Audio
};

// But exclude certain patterns
options.excludePatterns = {
    "*_backup.*",      // Backup files
    "*.part",          // Partial downloads
    "sample_*"         // Sample files
};

scanner.startScan(options);
```

## Size-Based Filtering

### Finding Large Files

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.minimumFileSize = 100 * 1024 * 1024;  // 100MB minimum

connect(&scanner, &FileScanner::scanCompleted, [&scanner]() {
    auto files = scanner.getScannedFiles();
    qDebug() << "Found" << files.size() << "files larger than 100MB";
    
    // Sort by size
    std::sort(files.begin(), files.end(), 
              [](const auto& a, const auto& b) { return a.fileSize > b.fileSize; });
    
    // Show top 10 largest
    for (int i = 0; i < qMin(10, files.size()); ++i) {
        qDebug() << files[i].filePath << "-" 
                 << (files[i].fileSize / 1024.0 / 1024.0) << "MB";
    }
});

scanner.startScan(options);
```

### Finding Files in Size Range

```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Documents"};
options.minimumFileSize = 1 * 1024 * 1024;      // 1MB minimum
options.maximumFileSize = 10 * 1024 * 1024;     // 10MB maximum

scanner.startScan(options);
```

### Duplicate Detection Use Case

```cpp
// Scan for potential duplicates (files large enough to matter)
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.minimumFileSize = 1024 * 1024;  // 1MB minimum (skip small files)
options.includeHiddenFiles = false;      // Skip hidden files
options.excludePatterns = {
    ".git/*",
    "node_modules/*",
    ".cache/*"
};

scanner.startScan(options);
```

## Progress Monitoring

### Basic Progress Display

```cpp
FileScanner scanner;

connect(&scanner, &FileScanner::scanProgress,
        [](int processed, int total, const QString& currentPath) {
    if (total > 0) {
        int percent = (processed * 100) / total;
        qDebug() << "Progress:" << percent << "% (" << processed << "/" << total << ")";
    } else {
        qDebug() << "Processed:" << processed << "files";
    }
    qDebug() << "Current:" << currentPath;
});

FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
scanner.startScan(options);
```

### Progress Bar Integration (Qt Widgets)

```cpp
#include <QProgressDialog>

FileScanner scanner;
QProgressDialog progress("Scanning files...", "Cancel", 0, 0, parentWidget);
progress.setWindowModality(Qt::WindowModal);

connect(&scanner, &FileScanner::scanStarted, [&progress]() {
    progress.show();
});

connect(&scanner, &FileScanner::scanProgress,
        [&progress](int processed, int total, const QString& currentPath) {
    if (total > 0) {
        progress.setMaximum(total);
        progress.setValue(processed);
    } else {
        progress.setMaximum(0);  // Indeterminate progress
    }
    progress.setLabelText(QString("Scanning: %1").arg(currentPath));
});

connect(&scanner, &FileScanner::scanCompleted, [&progress]() {
    progress.close();
});

connect(&progress, &QProgressDialog::canceled, [&scanner]() {
    scanner.cancelScan();
});

FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
scanner.startScan(options);
```

### Adjusting Progress Update Frequency

```cpp
// For fast scans with many small files, reduce update frequency
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.progressBatchSize = 500;  // Update every 500 files instead of 100

scanner.startScan(options);
```

## Error Handling

### Basic Error Handling

```cpp
FileScanner scanner;

connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    qWarning() << "Scan error at" << path << ":" << desc;
});

connect(&scanner, &FileScanner::scanErrorSummary,
        [](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    if (totalErrors > 0) {
        qWarning() << "Scan completed with" << totalErrors << "errors";
    }
});

FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
scanner.startScan(options);
```

### Detailed Error Logging

```cpp
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    QString errorTypeName;
    switch (errorType) {
        case FileScanner::ScanError::PermissionDenied:
            errorTypeName = "Permission Denied";
            break;
        case FileScanner::ScanError::NetworkTimeout:
            errorTypeName = "Network Timeout";
            break;
        case FileScanner::ScanError::DiskReadError:
            errorTypeName = "Disk Read Error";
            break;
        case FileScanner::ScanError::PathTooLong:
            errorTypeName = "Path Too Long";
            break;
        default:
            errorTypeName = "Unknown Error";
    }
    
    qWarning() << "[" << errorTypeName << "]" << path << "-" << desc;
});
```

### Error Summary Report

```cpp
connect(&scanner, &FileScanner::scanErrorSummary,
        [](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    if (totalErrors == 0) {
        qDebug() << "Scan completed successfully with no errors";
        return;
    }
    
    qWarning() << "=== Scan Error Summary ===";
    qWarning() << "Total errors:" << totalErrors;
    
    // Group errors by type
    QMap<FileScanner::ScanError, int> errorCounts;
    for (const auto& error : errors) {
        errorCounts[error.errorType]++;
    }
    
    qWarning() << "Errors by type:";
    for (auto it = errorCounts.begin(); it != errorCounts.end(); ++it) {
        qWarning() << "  -" << static_cast<int>(it.key()) << ":" << it.value();
    }
    
    // Show first 5 errors
    qWarning() << "First errors:";
    for (int i = 0; i < qMin(5, errors.size()); ++i) {
        qWarning() << "  " << errors[i].filePath << "-" << errors[i].errorMessage;
    }
});
```

## Performance Optimization

### Streaming Mode for Large Scans

```cpp
// For very large scans (100,000+ files), use streaming mode
FileScanner scanner;
QVector<FileScanner::FileInfo> largeFiles;

connect(&scanner, &FileScanner::fileFound,
        [&largeFiles](const FileScanner::FileInfo& file) {
    // Process files as they're found instead of storing all
    if (file.fileSize > 10 * 1024 * 1024) {  // Only keep files > 10MB
        largeFiles.append(file);
    }
});

FileScanner::ScanOptions options;
options.targetPaths = {"/home/user"};
options.streamingMode = true;  // Don't store all files in memory
options.minimumFileSize = 1024 * 1024;  // 1MB minimum

scanner.startScan(options);
```

### Pre-allocating Memory

```cpp
// If you know approximate file count, pre-allocate for better performance
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/photos"};
options.estimatedFileCount = 50000;  // Hint: expect ~50,000 files

scanner.startScan(options);
```

### Metadata Caching for Repeated Scans

```cpp
FileScanner scanner;

// Enable caching for repeated scans
FileScanner::ScanOptions options;
options.targetPaths = {"/home/user/Documents"};
options.enableMetadataCache = true;
options.metadataCacheSizeLimit = 20000;

// First scan - builds cache
scanner.startScan(options);

// Wait for completion...

// Second scan - uses cache (much faster)
scanner.startScan(options);

// Clear cache when files have changed significantly
scanner.clearMetadataCache();
```

### Performance Monitoring

```cpp
connect(&scanner, &FileScanner::scanStatistics,
        [](const FileScanner::ScanStatistics& stats) {
    qDebug() << "=== Scan Performance ===";
    qDebug() << "Files scanned:" << stats.totalFilesScanned;
    qDebug() << "Directories scanned:" << stats.totalDirectoriesScanned;
    qDebug() << "Total bytes:" << (stats.totalBytesScanned / 1024.0 / 1024.0) << "MB";
    qDebug() << "Duration:" << (stats.scanDurationMs / 1000.0) << "seconds";
    qDebug() << "Scan rate:" << stats.filesPerSecond << "files/second";
    qDebug() << "Errors:" << stats.errorsEncountered;
    
    qDebug() << "Filtering:";
    qDebug() << "  - By size:" << stats.filesFilteredBySize;
    qDebug() << "  - By pattern:" << stats.filesFilteredByPattern;
    qDebug() << "  - By hidden:" << stats.filesFilteredByHidden;
});
```

## Advanced Use Cases

### Duplicate File Finder

```cpp
class DuplicateFinder : public QObject {
    Q_OBJECT
public:
    void findDuplicates(const QString& path) {
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        options.minimumFileSize = 1024 * 1024;  // Skip files < 1MB
        options.includeHiddenFiles = false;
        options.excludePatterns = {".git/*", "node_modules/*"};
        
        connect(&m_scanner, &FileScanner::scanCompleted, this, [this]() {
            auto files = m_scanner.getScannedFiles();
            
            // Group by size (potential duplicates)
            QMultiMap<qint64, FileScanner::FileInfo> bySize;
            for (const auto& file : files) {
                bySize.insert(file.fileSize, file);
            }
            
            // Find size groups with multiple files
            auto keys = bySize.uniqueKeys();
            for (qint64 size : keys) {
                auto sameSize = bySize.values(size);
                if (sameSize.size() > 1) {
                    qDebug() << "Potential duplicates (" << size << "bytes):";
                    for (const auto& file : sameSize) {
                        qDebug() << "  -" << file.filePath;
                    }
                }
            }
        });
        
        m_scanner.startScan(options);
    }

private:
    FileScanner m_scanner;
};
```

### Disk Space Analyzer

```cpp
class DiskSpaceAnalyzer : public QObject {
    Q_OBJECT
public:
    void analyzePath(const QString& path) {
        connect(&m_scanner, &FileScanner::scanCompleted, this, [this]() {
            auto stats = m_scanner.getScanStatistics();
            auto files = m_scanner.getScannedFiles();
            
            // Calculate space by extension
            QMap<QString, qint64> spaceByExtension;
            for (const auto& file : files) {
                QString ext = QFileInfo(file.fileName).suffix().toLower();
                if (ext.isEmpty()) ext = "(no extension)";
                spaceByExtension[ext] += file.fileSize;
            }
            
            // Sort by space used
            QList<QPair<QString, qint64>> sorted;
            for (auto it = spaceByExtension.begin(); it != spaceByExtension.end(); ++it) {
                sorted.append({it.key(), it.value()});
            }
            std::sort(sorted.begin(), sorted.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            qDebug() << "=== Disk Space Analysis ===";
            qDebug() << "Total space:" << (stats.totalBytesScanned / 1024.0 / 1024.0 / 1024.0) << "GB";
            qDebug() << "\nTop 10 file types by space:";
            for (int i = 0; i < qMin(10, sorted.size()); ++i) {
                double sizeMB = sorted[i].second / 1024.0 / 1024.0;
                qDebug() << QString("  %1: %2 MB")
                            .arg(sorted[i].first, -15)
                            .arg(sizeMB, 0, 'f', 2);
            }
        });
        
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        options.includeHiddenFiles = true;
        m_scanner.startScan(options);
    }

private:
    FileScanner m_scanner;
};
```

### Media File Organizer

```cpp
class MediaOrganizer : public QObject {
    Q_OBJECT
public:
    void scanMediaFiles(const QString& path) {
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        options.includePatterns = {
            // Images
            "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff",
            // Videos
            "*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv",
            // Audio
            "*.mp3", "*.flac", "*.wav", "*.aac", "*.ogg"
        };
        
        connect(&m_scanner, &FileScanner::scanCompleted, this, [this]() {
            auto files = m_scanner.getScannedFiles();
            
            // Organize by type and date
            for (const auto& file : files) {
                QString ext = QFileInfo(file.fileName).suffix().toLower();
                QString type = getMediaType(ext);
                QString dateFolder = file.lastModified.toString("yyyy/MM");
                
                qDebug() << "Move" << file.filePath
                         << "to" << type << "/" << dateFolder;
            }
        });
        
        m_scanner.startScan(options);
    }

private:
    QString getMediaType(const QString& ext) {
        if (QStringList{"jpg", "jpeg", "png", "gif", "bmp", "tiff"}.contains(ext))
            return "Images";
        if (QStringList{"mp4", "avi", "mkv", "mov", "wmv"}.contains(ext))
            return "Videos";
        if (QStringList{"mp3", "flac", "wav", "aac", "ogg"}.contains(ext))
            return "Audio";
        return "Other";
    }
    
    FileScanner m_scanner;
};
```

### Incremental Backup Scanner

```cpp
class BackupScanner : public QObject {
    Q_OBJECT
public:
    void scanForBackup(const QString& sourcePath, const QDateTime& lastBackup) {
        connect(&m_scanner, &FileScanner::fileFound,
                this, [lastBackup](const FileScanner::FileInfo& file) {
            // Only process files modified since last backup
            if (file.lastModified > lastBackup) {
                qDebug() << "Need to backup:" << file.filePath;
                // Add to backup queue...
            }
        });
        
        FileScanner::ScanOptions options;
        options.targetPaths = {sourcePath};
        options.streamingMode = true;  // Process files as found
        options.includeHiddenFiles = false;
        options.excludePatterns = {
            "*.tmp", "*.bak", ".cache/*", "Thumbs.db", ".DS_Store"
        };
        
        m_scanner.startScan(options);
    }

private:
    FileScanner m_scanner;
};
```

## See Also

- [FileScanner API Documentation](API_FILESCANNER.md)
- [Error Handling Guide](FILESCANNER_ERROR_HANDLING.md)
- [Performance Tuning Guide](FILESCANNER_PERFORMANCE.md)
- [Integration Examples](FILESCANNER_INTEGRATION.md)
