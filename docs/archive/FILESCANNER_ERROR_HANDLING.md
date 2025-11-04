# FileScanner Error Handling Guide

This guide explains how the FileScanner handles errors and how to implement robust error handling in your application.

## Table of Contents

- [Overview](#overview)
- [Error Types](#error-types)
- [Error Detection](#error-detection)
- [Error Recovery](#error-recovery)
- [Error Reporting](#error-reporting)
- [Best Practices](#best-practices)
- [Common Scenarios](#common-scenarios)

## Overview

The FileScanner implements a robust error handling system designed to:

1. **Continue scanning** even when individual files or directories are inaccessible
2. **Classify errors** into specific types for appropriate handling
3. **Retry transient errors** automatically (e.g., network timeouts)
4. **Report errors** through signals for application-level handling
5. **Accumulate error information** for post-scan analysis

### Error Handling Philosophy

- **Non-blocking**: Errors don't stop the entire scan
- **Informative**: Detailed error information is provided
- **Recoverable**: Transient errors are retried automatically
- **Transparent**: All errors are reported to the application

## Error Types

The FileScanner defines specific error types through the `ScanError` enum:

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

### Error Type Details

#### PermissionDenied

**Cause**: Insufficient permissions to read a file or directory.

**Common Scenarios**:
- System directories (e.g., `/root`, `/System`)
- Files owned by other users
- Files with restrictive permissions (e.g., `chmod 000`)

**Behavior**:
- File/directory is skipped
- Error is logged and reported
- Scan continues with next item

**Example**:
```
Error: Permission denied
Path: /root/.ssh/id_rsa
System Code: EACCES (13)
```

#### FileSystemError

**Cause**: General file system I/O error.

**Common Scenarios**:
- Corrupted file system
- Hardware issues
- File system not mounted properly
- File deleted during scan

**Behavior**:
- File/directory is skipped
- Error is logged and reported
- Scan continues with next item

**Example**:
```
Error: Input/output error
Path: /mnt/damaged_disk/file.dat
System Code: EIO (5)
```

#### NetworkTimeout

**Cause**: Network drive or remote file system timeout.

**Common Scenarios**:
- Slow network connection
- Network drive disconnected
- SMB/NFS share unavailable
- VPN connection dropped

**Behavior**:
- **Automatic retry** (up to 2 times)
- If retry fails, directory is skipped
- Error is logged and reported
- Scan continues with next item

**Example**:
```
Error: Network timeout
Path: /mnt/network_share/documents
System Code: ETIMEDOUT (110)
Retries: 2
```

#### DiskReadError

**Cause**: Physical disk read error.

**Common Scenarios**:
- Bad sectors on hard drive
- Failing disk
- Corrupted data
- RAID degradation

**Behavior**:
- File is skipped
- Error is logged and reported
- Scan continues with next item

**Example**:
```
Error: Disk read error
Path: /home/user/corrupted_file.dat
System Code: EIO (5)
```

#### PathTooLong

**Cause**: File path exceeds system limits.

**Common Scenarios**:
- Deeply nested directories
- Very long file names
- Platform-specific path length limits (e.g., 260 chars on Windows)

**Behavior**:
- File/directory is skipped
- Error is logged and reported
- Scan continues with next item

**Example**:
```
Error: Path too long
Path: /very/deeply/nested/directory/structure/...
System Code: ENAMETOOLONG (36)
```

#### UnknownError

**Cause**: Unexpected or unclassified error.

**Common Scenarios**:
- Rare file system errors
- Platform-specific issues
- Bugs in file system drivers

**Behavior**:
- File/directory is skipped
- Error is logged with full details
- Scan continues with next item

## Error Detection

The FileScanner detects errors at multiple points during scanning:

### 1. Directory Access

```cpp
QDir directory(path);
if (!directory.exists()) {
    recordError(ScanError::FileSystemError, path, "Directory does not exist");
    return;
}

if (!directory.isReadable()) {
    recordError(ScanError::PermissionDenied, path, "Cannot read directory");
    return;
}
```

### 2. File Information Retrieval

```cpp
QFileInfo fileInfo(filePath);
if (!fileInfo.exists()) {
    recordError(ScanError::FileSystemError, filePath, "File does not exist");
    return;
}

if (!fileInfo.isReadable()) {
    recordError(ScanError::PermissionDenied, filePath, "Cannot read file");
    return;
}
```

### 3. Directory Iteration

```cpp
QDirIterator iterator(path, QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot);
while (iterator.hasNext()) {
    try {
        QString filePath = iterator.next();
        QFileInfo fileInfo = iterator.fileInfo();
        
        // Check for errors
        if (fileInfo.filePath().isEmpty()) {
            recordError(ScanError::FileSystemError, path, "Failed to read entry");
            continue;
        }
        
        // Process file...
    } catch (const std::exception& e) {
        recordError(ScanError::UnknownError, path, e.what());
    }
}
```

### 4. Path Length Validation

```cpp
if (filePath.length() > MAX_PATH_LENGTH) {
    recordError(ScanError::PathTooLong, filePath, 
                QString("Path length %1 exceeds limit %2")
                .arg(filePath.length()).arg(MAX_PATH_LENGTH));
    return;
}
```

## Error Recovery

The FileScanner implements automatic recovery for certain error types:

### Transient Error Retry

Network timeouts and temporary I/O errors are retried automatically:

```cpp
bool FileScanner::retryOperation(const QString& directoryPath, int maxRetries) {
    for (int attempt = 0; attempt < maxRetries; ++attempt) {
        QThread::msleep(100 * (attempt + 1));  // Exponential backoff
        
        QDir directory(directoryPath);
        if (directory.exists() && directory.isReadable()) {
            return true;  // Success
        }
    }
    return false;  // All retries failed
}
```

**Retry Strategy**:
- Maximum 2 retries
- Exponential backoff (100ms, 200ms)
- Only for transient errors (NetworkTimeout)

### Continue-on-Error

For non-critical errors, scanning continues:

```cpp
void FileScanner::scanDirectory(const QString& directoryPath) {
    QDirIterator iterator(directoryPath, ...);
    
    while (iterator.hasNext()) {
        if (m_cancelRequested) break;
        
        try {
            QString filePath = iterator.next();
            processFile(filePath);
        } catch (...) {
            // Log error but continue with next file
            recordError(...);
            continue;  // Don't stop scanning
        }
    }
}
```

### Directory-Level Recovery

If a directory fails, sibling directories are still scanned:

```cpp
void FileScanner::processScanQueue() {
    while (!m_scanQueue.isEmpty() && !m_cancelRequested) {
        QString path = m_scanQueue.takeFirst();
        
        try {
            scanDirectory(path);
        } catch (...) {
            recordError(...);
            // Continue with next directory in queue
        }
    }
}
```

## Error Reporting

The FileScanner provides multiple ways to access error information:

### 1. Real-Time Error Signal

Emitted immediately when an error occurs:

```cpp
signals:
    void scanError(ScanError errorType, const QString& path, const QString& description);
```

**Usage**:
```cpp
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    qWarning() << "Error:" << static_cast<int>(errorType) << path << desc;
});
```

### 2. Error Summary Signal

Emitted at scan completion with all errors:

```cpp
signals:
    void scanErrorSummary(int totalErrors, const QList<ScanErrorInfo>& errors);
```

**Usage**:
```cpp
connect(&scanner, &FileScanner::scanErrorSummary,
        [](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    if (totalErrors > 0) {
        qWarning() << "Scan completed with" << totalErrors << "errors";
        for (const auto& error : errors) {
            qWarning() << "  -" << error.filePath << ":" << error.errorMessage;
        }
    }
});
```

### 3. Error List Access

Retrieve errors after scan completion:

```cpp
auto errors = scanner.getScanErrors();
for (const auto& error : errors) {
    qDebug() << "Error Type:" << static_cast<int>(error.errorType);
    qDebug() << "Path:" << error.filePath;
    qDebug() << "Message:" << error.errorMessage;
    qDebug() << "System Code:" << error.systemErrorCode;
    qDebug() << "Timestamp:" << error.timestamp.toString();
}
```

### 4. Error Count

Quick error count check:

```cpp
int errorCount = scanner.getTotalErrorsEncountered();
if (errorCount > 0) {
    qWarning() << "Scan had" << errorCount << "errors";
}
```

### 5. Statistics Integration

Errors are included in scan statistics:

```cpp
auto stats = scanner.getScanStatistics();
qDebug() << "Errors encountered:" << stats.errorsEncountered;
```

## Best Practices

### 1. Always Connect to Error Signals

```cpp
// Good: Handle errors
connect(&scanner, &FileScanner::scanError,
        this, &MyClass::handleScanError);

// Bad: Ignore errors
// (No error signal connection)
```

### 2. Log Errors for Debugging

```cpp
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    // Log to file for later analysis
    QFile logFile("scan_errors.log");
    if (logFile.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&logFile);
        out << QDateTime::currentDateTime().toString(Qt::ISODate) << " | "
            << static_cast<int>(errorType) << " | "
            << path << " | "
            << desc << "\n";
    }
});
```

### 3. Inform Users of Errors

```cpp
connect(&scanner, &FileScanner::scanErrorSummary,
        [this](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
    if (totalErrors > 0) {
        QString message = QString("Scan completed with %1 error(s).\n"
                                 "Some files or directories were inaccessible.")
                         .arg(totalErrors);
        QMessageBox::warning(this, "Scan Errors", message);
    }
});
```

### 4. Handle Specific Error Types

```cpp
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    switch (errorType) {
        case FileScanner::ScanError::PermissionDenied:
            // Suggest running with elevated privileges
            qWarning() << "Permission denied. Try running as administrator.";
            break;
            
        case FileScanner::ScanError::NetworkTimeout:
            // Suggest checking network connection
            qWarning() << "Network timeout. Check your network connection.";
            break;
            
        case FileScanner::ScanError::DiskReadError:
            // Suggest checking disk health
            qWarning() << "Disk read error. Check disk health with SMART tools.";
            break;
            
        default:
            qWarning() << "Error:" << desc;
    }
});
```

### 5. Provide Error Summary to Users

```cpp
void showErrorSummary(const QList<FileScanner::ScanErrorInfo>& errors) {
    if (errors.isEmpty()) return;
    
    // Group errors by type
    QMap<FileScanner::ScanError, QStringList> errorsByType;
    for (const auto& error : errors) {
        errorsByType[error.errorType].append(error.filePath);
    }
    
    QString summary = "Scan Errors:\n\n";
    
    for (auto it = errorsByType.begin(); it != errorsByType.end(); ++it) {
        QString typeName = getErrorTypeName(it.key());
        summary += QString("%1 (%2 files):\n").arg(typeName).arg(it.value().size());
        
        // Show first 5 paths
        for (int i = 0; i < qMin(5, it.value().size()); ++i) {
            summary += QString("  - %1\n").arg(it.value()[i]);
        }
        
        if (it.value().size() > 5) {
            summary += QString("  ... and %1 more\n").arg(it.value().size() - 5);
        }
        summary += "\n";
    }
    
    QMessageBox::information(nullptr, "Scan Error Summary", summary);
}
```

## Common Scenarios

### Scenario 1: Scanning System Directories

**Problem**: Many permission denied errors when scanning system directories.

**Solution**:
```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/"};
options.scanSystemDirectories = false;  // Skip system directories

// Handle expected permission errors gracefully
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString&) {
    if (errorType == FileScanner::ScanError::PermissionDenied) {
        // Expected for system directories, just log
        qDebug() << "Skipped (no permission):" << path;
    }
});
```

### Scenario 2: Scanning Network Drives

**Problem**: Network timeouts causing slow scans.

**Solution**:
```cpp
FileScanner::ScanOptions options;
options.targetPaths = {"/mnt/network_share"};

// Network errors are retried automatically
connect(&scanner, &FileScanner::scanError,
        [](FileScanner::ScanError errorType, const QString& path, const QString& desc) {
    if (errorType == FileScanner::ScanError::NetworkTimeout) {
        qWarning() << "Network timeout (after retries):" << path;
        // Consider notifying user to check network
    }
});
```

### Scenario 3: Scanning Failing Disk

**Problem**: Disk read errors on failing hardware.

**Solution**:
```cpp
int diskErrorCount = 0;

connect(&scanner, &FileScanner::scanError,
        [&diskErrorCount](FileScanner::ScanError errorType, const QString& path, const QString&) {
    if (errorType == FileScanner::ScanError::DiskReadError) {
        diskErrorCount++;
        qWarning() << "Disk read error:" << path;
    }
});

connect(&scanner, &FileScanner::scanCompleted, [&diskErrorCount]() {
    if (diskErrorCount > 10) {
        QMessageBox::warning(nullptr, "Disk Health Warning",
            QString("Encountered %1 disk read errors.\n"
                   "Your disk may be failing. Back up your data immediately!")
            .arg(diskErrorCount));
    }
});
```

### Scenario 4: Handling Path Length Limits

**Problem**: Path too long errors on deeply nested directories.

**Solution**:
```cpp
QStringList problematicPaths;

connect(&scanner, &FileScanner::scanError,
        [&problematicPaths](FileScanner::ScanError errorType, 
                           const QString& path, const QString&) {
    if (errorType == FileScanner::ScanError::PathTooLong) {
        problematicPaths.append(path);
    }
});

connect(&scanner, &FileScanner::scanCompleted, [&problematicPaths]() {
    if (!problematicPaths.isEmpty()) {
        qWarning() << "Paths too long (consider shortening):";
        for (const auto& path : problematicPaths) {
            qWarning() << "  -" << path;
        }
    }
});
```

### Scenario 5: Comprehensive Error Handling

**Complete example with all error handling**:

```cpp
class RobustScanner : public QObject {
    Q_OBJECT
public:
    void scan(const QString& path) {
        // Connect all error signals
        connect(&m_scanner, &FileScanner::scanError,
                this, &RobustScanner::handleError);
        connect(&m_scanner, &FileScanner::scanErrorSummary,
                this, &RobustScanner::handleErrorSummary);
        
        // Start scan
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        m_scanner.startScan(options);
    }

private slots:
    void handleError(FileScanner::ScanError errorType, 
                    const QString& path, const QString& desc) {
        // Log all errors
        m_errorLog.append(QString("[%1] %2: %3")
                         .arg(getErrorTypeName(errorType))
                         .arg(path)
                         .arg(desc));
        
        // Count by type
        m_errorCounts[errorType]++;
        
        // Handle critical errors
        if (errorType == FileScanner::ScanError::DiskReadError) {
            m_diskErrors++;
            if (m_diskErrors > 100) {
                qCritical() << "Too many disk errors, stopping scan";
                m_scanner.cancelScan();
            }
        }
    }
    
    void handleErrorSummary(int totalErrors, 
                           const QList<FileScanner::ScanErrorInfo>& errors) {
        if (totalErrors == 0) {
            qDebug() << "Scan completed successfully";
            return;
        }
        
        qWarning() << "=== Scan Error Summary ===";
        qWarning() << "Total errors:" << totalErrors;
        
        for (auto it = m_errorCounts.begin(); it != m_errorCounts.end(); ++it) {
            qWarning() << getErrorTypeName(it.key()) << ":" << it.value();
        }
        
        // Save error log to file
        saveErrorLog();
        
        // Notify user
        notifyUser(totalErrors);
    }
    
    QString getErrorTypeName(FileScanner::ScanError errorType) {
        switch (errorType) {
            case FileScanner::ScanError::PermissionDenied: return "Permission Denied";
            case FileScanner::ScanError::NetworkTimeout: return "Network Timeout";
            case FileScanner::ScanError::DiskReadError: return "Disk Read Error";
            case FileScanner::ScanError::PathTooLong: return "Path Too Long";
            case FileScanner::ScanError::FileSystemError: return "File System Error";
            default: return "Unknown Error";
        }
    }
    
    void saveErrorLog() {
        QFile file("scan_errors.log");
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            for (const auto& line : m_errorLog) {
                out << line << "\n";
            }
        }
    }
    
    void notifyUser(int totalErrors) {
        QString message = QString("Scan completed with %1 error(s).\n"
                                 "See scan_errors.log for details.")
                         .arg(totalErrors);
        QMessageBox::information(nullptr, "Scan Complete", message);
    }

private:
    FileScanner m_scanner;
    QStringList m_errorLog;
    QMap<FileScanner::ScanError, int> m_errorCounts;
    int m_diskErrors = 0;
};
```

## See Also

- [FileScanner API Documentation](API_FILESCANNER.md)
- [Usage Examples](FILESCANNER_EXAMPLES.md)
- [Performance Tuning Guide](FILESCANNER_PERFORMANCE.md)
