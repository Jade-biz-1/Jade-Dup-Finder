#pragma once

#include <QString>
#include <QStringList>
#include <QObject>
#include <QFileInfo>
#include <QDir>

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
    struct ScanOptions {
        QStringList targetPaths;           // Directories/files to scan
        qint64 minimumFileSize = 1024;     // Skip files smaller than this (1KB default)
        qint64 maximumFileSize = -1;       // No limit by default
        QStringList includePatterns;      // File patterns to include (*.jpg, etc.)
        QStringList excludePatterns;      // File patterns to exclude
        bool includeHiddenFiles = false;  // Include hidden/dot files
        bool followSymlinks = false;      // Follow symbolic links
        bool scanSystemDirectories = false; // Include system directories
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
    QList<FileInfo> getScannedFiles() const;
    int getTotalFilesFound() const;
    qint64 getTotalBytesScanned() const;

signals:
    void scanStarted();
    void scanProgress(int filesProcessed, int totalFiles, const QString& currentPath);
    void scanCompleted();
    void scanCancelled();
    void errorOccurred(const QString& error);
    void fileFound(const FileInfo& fileInfo);

private slots:
    void processScanQueue();

private:
    // TODO: Implement these methods
    bool shouldIncludeFile(const QFileInfo& fileInfo) const;
    bool shouldScanDirectory(const QDir& directory) const;
    void scanDirectory(const QString& directoryPath);
    
    // Member variables
    ScanOptions m_currentOptions;
    QList<FileInfo> m_scannedFiles;
    QStringList m_scanQueue;
    bool m_isScanning = false;
    bool m_cancelRequested = false;
    int m_filesProcessed = 0;
    qint64 m_totalBytesScanned = 0;
};