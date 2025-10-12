# FileScanner Integration Guide

This guide explains how to integrate the FileScanner component with other components in the duplicate file finder application.

## Table of Contents

- [Overview](#overview)
- [Integration with HashCalculator](#integration-with-hashcalculator)
- [Integration with DuplicateDetector](#integration-with-duplicatedetector)
- [End-to-End Workflow](#end-to-end-workflow)
- [Signal/Slot Connections](#signalslot-connections)
- [Error Propagation](#error-propagation)
- [Cancellation Handling](#cancellation-handling)
- [Progress Coordination](#progress-coordination)

## Overview

The FileScanner is the first component in the duplicate detection pipeline:

```
FileScanner → HashCalculator → DuplicateDetector → Results
```

Each component:
1. **FileScanner**: Finds files matching criteria
2. **HashCalculator**: Computes hashes for found files
3. **DuplicateDetector**: Identifies duplicates based on hashes
4. **Results**: Displays duplicates to user

## Integration with HashCalculator

The HashCalculator receives file information from the FileScanner and computes cryptographic hashes.

### Data Flow

```
FileScanner::FileInfo → HashCalculator::addFile() → Hash computation
```

### Basic Integration

```cpp
#include "file_scanner.h"
#include "hash_calculator.h"

class ScanAndHashWorkflow : public QObject {
    Q_OBJECT
public:
    void start(const QString& path) {
        // Configure scanner
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        options.minimumFileSize = 1024 * 1024;  // 1MB minimum
        
        // Connect scanner to hash calculator
        connect(&m_scanner, &FileScanner::scanCompleted,
                this, &ScanAndHashWorkflow::onScanCompleted);
        
        // Start scanning
        m_scanner.startScan(options);
    }

private slots:
    void onScanCompleted() {
        auto files = m_scanner.getScannedFiles();
        qDebug() << "Scan complete, found" << files.size() << "files";
        
        // Pass files to hash calculator
        for (const auto& file : files) {
            m_hashCalculator.addFile(file.filePath, file.fileSize);
        }
        
        // Start hashing
        m_hashCalculator.startCalculation();
    }
    
    void onHashingCompleted() {
        qDebug() << "Hashing complete";
        // Continue to duplicate detection...
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalculator;
};
```

### Streaming Integration

For large file sets, use streaming mode to avoid storing all files:

```cpp
class StreamingScanAndHash : public QObject {
    Q_OBJECT
public:
    void start(const QString& path) {
        // Enable streaming mode
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        options.streamingMode = true;  // Don't store all files
        
        // Process files as they're found
        connect(&m_scanner, &FileScanner::fileFound,
                this, &StreamingScanAndHash::onFileFound);
        
        connect(&m_scanner, &FileScanner::scanCompleted,
                this, &StreamingScanAndHash::onScanCompleted);
        
        m_scanner.startScan(options);
    }

private slots:
    void onFileFound(const FileScanner::FileInfo& file) {
        // Add file to hash calculator immediately
        m_hashCalculator.addFile(file.filePath, file.fileSize);
    }
    
    void onScanCompleted() {
        qDebug() << "Scan complete, starting hash calculation";
        m_hashCalculator.startCalculation();
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalculator;
};
```

### FileInfo Conversion

If HashCalculator expects a different format:

```cpp
void convertAndAddFiles() {
    auto scannerFiles = m_scanner.getScannedFiles();
    
    for (const auto& scanFile : scannerFiles) {
        // Convert FileScanner::FileInfo to HashCalculator format
        HashCalculator::FileInfo hashFile;
        hashFile.path = scanFile.filePath;
        hashFile.size = scanFile.fileSize;
        hashFile.modified = scanFile.lastModified;
        
        m_hashCalculator.addFile(hashFile);
    }
}
```

## Integration with DuplicateDetector

The DuplicateDetector receives file information and hashes to identify duplicates.

### Data Flow

```
FileScanner::FileInfo + Hash → DuplicateDetector::addFile() → Duplicate groups
```

### Basic Integration

```cpp
#include "file_scanner.h"
#include "hash_calculator.h"
#include "duplicate_detector.h"

class FullDuplicateWorkflow : public QObject {
    Q_OBJECT
public:
    void start(const QString& path) {
        // Step 1: Scan files
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        options.minimumFileSize = 1024 * 1024;
        
        connect(&m_scanner, &FileScanner::scanCompleted,
                this, &FullDuplicateWorkflow::onScanCompleted);
        
        m_scanner.startScan(options);
    }

private slots:
    void onScanCompleted() {
        auto files = m_scanner.getScannedFiles();
        qDebug() << "Scan complete:" << files.size() << "files";
        
        // Step 2: Calculate hashes
        for (const auto& file : files) {
            m_hashCalculator.addFile(file.filePath, file.fileSize);
        }
        
        connect(&m_hashCalculator, &HashCalculator::calculationCompleted,
                this, &FullDuplicateWorkflow::onHashingCompleted);
        
        m_hashCalculator.startCalculation();
    }
    
    void onHashingCompleted() {
        auto hashes = m_hashCalculator.getHashes();
        auto files = m_scanner.getScannedFiles();
        
        qDebug() << "Hashing complete:" << hashes.size() << "hashes";
        
        // Step 3: Detect duplicates
        for (int i = 0; i < files.size(); ++i) {
            const auto& file = files[i];
            const QString& hash = hashes.value(file.filePath);
            
            m_duplicateDetector.addFile(file.filePath, file.fileSize, hash);
        }
        
        connect(&m_duplicateDetector, &DuplicateDetector::detectionCompleted,
                this, &FullDuplicateWorkflow::onDetectionCompleted);
        
        m_duplicateDetector.startDetection();
    }
    
    void onDetectionCompleted() {
        auto duplicates = m_duplicateDetector.getDuplicateGroups();
        qDebug() << "Detection complete:" << duplicates.size() << "duplicate groups";
        
        // Display results...
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalculator;
    DuplicateDetector m_duplicateDetector;
};
```

### FileInfo Structure Compatibility

Ensure FileInfo structures are compatible:

```cpp
// FileScanner::FileInfo
struct FileInfo {
    QString filePath;
    qint64 fileSize;
    QString fileName;
    QString directory;
    QDateTime lastModified;
};

// DuplicateDetector may expect:
struct FileInfo {
    QString path;
    qint64 size;
    QString hash;
    QDateTime modified;
};

// Conversion helper
DuplicateDetector::FileInfo convertFileInfo(
    const FileScanner::FileInfo& scanFile,
    const QString& hash)
{
    DuplicateDetector::FileInfo detectorFile;
    detectorFile.path = scanFile.filePath;
    detectorFile.size = scanFile.fileSize;
    detectorFile.hash = hash;
    detectorFile.modified = scanFile.lastModified;
    return detectorFile;
}
```

## End-to-End Workflow

Complete example of the full duplicate detection workflow:

```cpp
class DuplicateFinderWorkflow : public QObject {
    Q_OBJECT
public:
    void findDuplicates(const QStringList& paths) {
        // Configure scan
        FileScanner::ScanOptions options;
        options.targetPaths = paths;
        options.minimumFileSize = 1024 * 1024;  // 1MB minimum
        options.includeHiddenFiles = false;
        options.excludePatterns = {".git/*", "node_modules/*"};
        
        // Connect workflow
        setupConnections();
        
        // Start workflow
        emit workflowStarted();
        m_scanner.startScan(options);
    }
    
    void cancel() {
        m_scanner.cancelScan();
        m_hashCalculator.cancelCalculation();
        m_duplicateDetector.cancelDetection();
    }

signals:
    void workflowStarted();
    void scanningProgress(int percent);
    void hashingProgress(int percent);
    void detectionProgress(int percent);
    void workflowCompleted(int duplicateGroups, qint64 wastedSpace);
    void workflowError(const QString& error);

private:
    void setupConnections() {
        // Scanner connections
        connect(&m_scanner, &FileScanner::scanProgress,
                this, &DuplicateFinderWorkflow::onScanProgress);
        connect(&m_scanner, &FileScanner::scanCompleted,
                this, &DuplicateFinderWorkflow::onScanCompleted);
        connect(&m_scanner, &FileScanner::scanError,
                this, &DuplicateFinderWorkflow::onScanError);
        
        // Hash calculator connections
        connect(&m_hashCalculator, &HashCalculator::progress,
                this, &DuplicateFinderWorkflow::onHashProgress);
        connect(&m_hashCalculator, &HashCalculator::calculationCompleted,
                this, &DuplicateFinderWorkflow::onHashCompleted);
        connect(&m_hashCalculator, &HashCalculator::error,
                this, &DuplicateFinderWorkflow::onHashError);
        
        // Duplicate detector connections
        connect(&m_duplicateDetector, &DuplicateDetector::progress,
                this, &DuplicateFinderWorkflow::onDetectionProgress);
        connect(&m_duplicateDetector, &DuplicateDetector::detectionCompleted,
                this, &DuplicateFinderWorkflow::onDetectionCompleted);
        connect(&m_duplicateDetector, &DuplicateDetector::error,
                this, &DuplicateFinderWorkflow::onDetectionError);
    }
    
private slots:
    void onScanProgress(int processed, int total, const QString& path) {
        int percent = (total > 0) ? (processed * 100 / total) : 0;
        emit scanningProgress(percent);
    }
    
    void onScanCompleted() {
        auto files = m_scanner.getScannedFiles();
        qDebug() << "Scan complete:" << files.size() << "files";
        
        if (files.isEmpty()) {
            emit workflowCompleted(0, 0);
            return;
        }
        
        // Start hashing
        for (const auto& file : files) {
            m_hashCalculator.addFile(file.filePath, file.fileSize);
        }
        m_hashCalculator.startCalculation();
    }
    
    void onScanError(FileScanner::ScanError errorType, 
                     const QString& path, const QString& desc) {
        qWarning() << "Scan error:" << path << desc;
        // Continue scanning, don't stop workflow
    }
    
    void onHashProgress(int processed, int total) {
        int percent = (total > 0) ? (processed * 100 / total) : 0;
        emit hashingProgress(percent);
    }
    
    void onHashCompleted() {
        auto hashes = m_hashCalculator.getHashes();
        auto files = m_scanner.getScannedFiles();
        
        qDebug() << "Hashing complete:" << hashes.size() << "hashes";
        
        // Start duplicate detection
        for (const auto& file : files) {
            QString hash = hashes.value(file.filePath);
            if (!hash.isEmpty()) {
                m_duplicateDetector.addFile(file.filePath, file.fileSize, hash);
            }
        }
        m_duplicateDetector.startDetection();
    }
    
    void onHashError(const QString& error) {
        qWarning() << "Hash error:" << error;
        emit workflowError("Hashing failed: " + error);
    }
    
    void onDetectionProgress(int processed, int total) {
        int percent = (total > 0) ? (processed * 100 / total) : 0;
        emit detectionProgress(percent);
    }
    
    void onDetectionCompleted() {
        auto duplicates = m_duplicateDetector.getDuplicateGroups();
        qint64 wastedSpace = calculateWastedSpace(duplicates);
        
        qDebug() << "Detection complete:" << duplicates.size() << "groups";
        qDebug() << "Wasted space:" << (wastedSpace / 1024.0 / 1024.0) << "MB";
        
        emit workflowCompleted(duplicates.size(), wastedSpace);
    }
    
    void onDetectionError(const QString& error) {
        qWarning() << "Detection error:" << error;
        emit workflowError("Detection failed: " + error);
    }
    
    qint64 calculateWastedSpace(const QList<DuplicateGroup>& groups) {
        qint64 wasted = 0;
        for (const auto& group : groups) {
            if (group.files.size() > 1) {
                // Wasted space = (count - 1) * file size
                wasted += (group.files.size() - 1) * group.fileSize;
            }
        }
        return wasted;
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalculator;
    DuplicateDetector m_duplicateDetector;
};
```

## Signal/Slot Connections

### Connection Patterns

#### Sequential Processing

Process each stage sequentially:

```cpp
// Scan → Hash → Detect
connect(&scanner, &FileScanner::scanCompleted,
        this, &MyClass::startHashing);
connect(&hashCalc, &HashCalculator::calculationCompleted,
        this, &MyClass::startDetection);
```

#### Parallel Processing

Start next stage while previous continues (if possible):

```cpp
// Start hashing as files are found
connect(&scanner, &FileScanner::fileFound,
        [this](const FileScanner::FileInfo& file) {
    m_hashCalculator.addFile(file.filePath, file.fileSize);
});

// Start hash calculation when enough files accumulated
connect(&scanner, &FileScanner::scanProgress,
        [this](int processed, int total, const QString&) {
    if (processed >= 100 && !m_hashingStarted) {
        m_hashCalculator.startCalculation();
        m_hashingStarted = true;
    }
});
```

### Connection Management

Use QObject parent-child relationships for automatic cleanup:

```cpp
class WorkflowManager : public QObject {
public:
    WorkflowManager(QObject* parent = nullptr) : QObject(parent) {
        // Components are children, will be deleted automatically
        m_scanner = new FileScanner(this);
        m_hashCalc = new HashCalculator(this);
        m_detector = new DuplicateDetector(this);
        
        setupConnections();
    }
    
private:
    void setupConnections() {
        connect(m_scanner, &FileScanner::scanCompleted,
                this, &WorkflowManager::onScanCompleted);
        // More connections...
    }
    
    FileScanner* m_scanner;
    HashCalculator* m_hashCalc;
    DuplicateDetector* m_detector;
};
```

## Error Propagation

Handle errors at each stage and propagate to user:

```cpp
class ErrorHandlingWorkflow : public QObject {
    Q_OBJECT
public:
    void start(const QString& path) {
        // Connect error handlers
        connect(&m_scanner, &FileScanner::scanError,
                this, &ErrorHandlingWorkflow::handleScanError);
        connect(&m_hashCalc, &HashCalculator::error,
                this, &ErrorHandlingWorkflow::handleHashError);
        connect(&m_detector, &DuplicateDetector::error,
                this, &ErrorHandlingWorkflow::handleDetectionError);
        
        // Start workflow
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        m_scanner.startScan(options);
    }

signals:
    void criticalError(const QString& stage, const QString& error);
    void nonCriticalError(const QString& stage, const QString& error);

private slots:
    void handleScanError(FileScanner::ScanError errorType,
                        const QString& path, const QString& desc) {
        // Most scan errors are non-critical
        QString errorMsg = QString("Scan error at %1: %2").arg(path, desc);
        emit nonCriticalError("Scanning", errorMsg);
        
        // Continue workflow
    }
    
    void handleHashError(const QString& error) {
        // Hash errors might be critical
        if (error.contains("out of memory")) {
            emit criticalError("Hashing", error);
            cancel();
        } else {
            emit nonCriticalError("Hashing", error);
        }
    }
    
    void handleDetectionError(const QString& error) {
        // Detection errors are usually critical
        emit criticalError("Detection", error);
        cancel();
    }
    
    void cancel() {
        m_scanner.cancelScan();
        m_hashCalc.cancelCalculation();
        m_detector.cancelDetection();
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalc;
    DuplicateDetector m_detector;
};
```

## Cancellation Handling

Implement proper cancellation across all components:

```cpp
class CancellableWorkflow : public QObject {
    Q_OBJECT
public:
    void start(const QString& path) {
        m_cancelled = false;
        
        // Connect cancellation signals
        connect(&m_scanner, &FileScanner::scanCancelled,
                this, &CancellableWorkflow::onScanCancelled);
        connect(&m_hashCalc, &HashCalculator::calculationCancelled,
                this, &CancellableWorkflow::onHashCancelled);
        
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        m_scanner.startScan(options);
    }
    
    void cancel() {
        if (m_cancelled) return;
        m_cancelled = true;
        
        qDebug() << "Cancelling workflow...";
        
        // Cancel all components
        m_scanner.cancelScan();
        m_hashCalc.cancelCalculation();
        m_detector.cancelDetection();
    }

signals:
    void workflowCancelled();

private slots:
    void onScanCancelled() {
        qDebug() << "Scan cancelled";
        if (m_cancelled) {
            emit workflowCancelled();
        }
    }
    
    void onHashCancelled() {
        qDebug() << "Hashing cancelled";
        if (m_cancelled) {
            emit workflowCancelled();
        }
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalc;
    DuplicateDetector m_detector;
    bool m_cancelled = false;
};
```

## Progress Coordination

Coordinate progress across multiple stages:

```cpp
class ProgressCoordinator : public QObject {
    Q_OBJECT
public:
    void start(const QString& path) {
        // Define stage weights (total = 100%)
        m_scanWeight = 20;      // Scanning: 20%
        m_hashWeight = 70;      // Hashing: 70% (slowest)
        m_detectWeight = 10;    // Detection: 10%
        
        // Connect progress signals
        connect(&m_scanner, &FileScanner::scanProgress,
                this, &ProgressCoordinator::onScanProgress);
        connect(&m_hashCalc, &HashCalculator::progress,
                this, &ProgressCoordinator::onHashProgress);
        connect(&m_detector, &DuplicateDetector::progress,
                this, &ProgressCoordinator::onDetectionProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        m_scanner.startScan(options);
    }

signals:
    void overallProgress(int percent, const QString& stage);

private slots:
    void onScanProgress(int processed, int total, const QString&) {
        if (total > 0) {
            int stagePercent = (processed * 100) / total;
            int overallPercent = (stagePercent * m_scanWeight) / 100;
            emit overallProgress(overallPercent, "Scanning");
        }
    }
    
    void onHashProgress(int processed, int total) {
        if (total > 0) {
            int stagePercent = (processed * 100) / total;
            int overallPercent = m_scanWeight + (stagePercent * m_hashWeight) / 100;
            emit overallProgress(overallPercent, "Hashing");
        }
    }
    
    void onDetectionProgress(int processed, int total) {
        if (total > 0) {
            int stagePercent = (processed * 100) / total;
            int overallPercent = m_scanWeight + m_hashWeight + 
                                (stagePercent * m_detectWeight) / 100;
            emit overallProgress(overallPercent, "Detecting");
        }
    }

private:
    FileScanner m_scanner;
    HashCalculator m_hashCalc;
    DuplicateDetector m_detector;
    
    int m_scanWeight;
    int m_hashWeight;
    int m_detectWeight;
};
```

## See Also

- [FileScanner API Documentation](API_FILESCANNER.md)
- [Usage Examples](FILESCANNER_EXAMPLES.md)
- [Error Handling Guide](FILESCANNER_ERROR_HANDLING.md)
- [Performance Tuning Guide](FILESCANNER_PERFORMANCE.md)
- [Migration Guide](FILESCANNER_MIGRATION.md)
