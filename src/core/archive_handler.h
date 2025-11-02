#ifndef ARCHIVE_HANDLER_H
#define ARCHIVE_HANDLER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QIODevice>
#include <QTemporaryDir>
#include <memory>

/**
 * @brief File information for files within archives
 */
struct ArchiveFileInfo {
    QString fileName;           // Name of file within archive
    QString fullPath;          // Full path within archive (including subdirectories)
    qint64 fileSize;           // Uncompressed file size
    qint64 compressedSize;     // Compressed file size
    QString archivePath;       // Path to the archive containing this file
    QByteArray content;        // File content (loaded on demand)
    bool isDirectory;          // Whether this is a directory entry
    
    ArchiveFileInfo() : fileSize(0), compressedSize(0), isDirectory(false) {}
};

/**
 * @brief Archive scanning configuration
 */
struct ArchiveScanConfig {
    bool scanZipFiles = true;           // Scan ZIP archives
    bool scanTarFiles = true;           // Scan TAR archives
    bool scanRarFiles = false;          // Scan RAR archives (requires external tool)
    bool scanNestedArchives = true;     // Scan archives within archives
    int maxNestingDepth = 3;            // Maximum nesting depth
    qint64 maxArchiveSize = 100 * 1024 * 1024; // Max archive size to scan (100MB)
    qint64 maxFileSize = 10 * 1024 * 1024;     // Max individual file size to extract (10MB)
    bool extractToMemory = true;        // Extract small files to memory vs temp files
    qint64 memoryThreshold = 1024 * 1024;      // Files larger than this use temp files (1MB)
};

/**
 * @brief Handler for scanning files within archives
 * 
 * This class provides functionality to scan inside ZIP, TAR, and other archive formats
 * to find duplicate files. It can handle nested archives and provides efficient
 * content extraction for duplicate detection algorithms.
 */
class ArchiveHandler : public QObject
{
    Q_OBJECT

public:
    explicit ArchiveHandler(QObject* parent = nullptr);
    ~ArchiveHandler();

    /**
     * @brief Check if a file is a supported archive format
     * @param filePath Path to the file to check
     * @return True if the file is a supported archive format
     */
    static bool isArchiveFile(const QString& filePath);
    
    /**
     * @brief Get list of supported archive extensions
     * @return List of supported extensions (e.g., "zip", "tar", "tar.gz")
     */
    static QStringList supportedExtensions();
    
    /**
     * @brief Set archive scanning configuration
     * @param config Configuration settings for archive scanning
     */
    void setConfiguration(const ArchiveScanConfig& config);
    
    /**
     * @brief Get current archive scanning configuration
     * @return Current configuration settings
     */
    ArchiveScanConfig configuration() const;
    
    /**
     * @brief Scan an archive file and return list of contained files
     * @param archivePath Path to the archive file
     * @return List of files found within the archive
     */
    QList<ArchiveFileInfo> scanArchive(const QString& archivePath);
    
    /**
     * @brief Extract file content from archive
     * @param archivePath Path to the archive file
     * @param internalPath Path to file within the archive
     * @return File content as byte array (empty if extraction failed)
     */
    QByteArray extractFileContent(const QString& archivePath, const QString& internalPath);
    
    /**
     * @brief Extract file to temporary location
     * @param archivePath Path to the archive file
     * @param internalPath Path to file within the archive
     * @return Path to extracted temporary file (empty if extraction failed)
     */
    QString extractToTempFile(const QString& archivePath, const QString& internalPath);
    
    /**
     * @brief Get archive format type
     * @param filePath Path to archive file
     * @return Archive format name (e.g., "ZIP", "TAR", "GZIP")
     */
    static QString getArchiveFormat(const QString& filePath);
    
    /**
     * @brief Check if archive scanning is enabled for given file type
     * @param filePath Path to archive file
     * @return True if scanning is enabled for this archive type
     */
    bool isScanningEnabled(const QString& filePath) const;

signals:
    /**
     * @brief Emitted when archive scanning starts
     * @param archivePath Path to archive being scanned
     * @param totalFiles Estimated number of files in archive
     */
    void scanStarted(const QString& archivePath, int totalFiles);
    
    /**
     * @brief Emitted when scanning progress updates
     * @param archivePath Path to archive being scanned
     * @param current Current file index
     * @param total Total files in archive
     * @param currentFile Name of current file being processed
     */
    void scanProgress(const QString& archivePath, int current, int total, const QString& currentFile);
    
    /**
     * @brief Emitted when archive scanning completes
     * @param archivePath Path to archive that was scanned
     * @param filesFound Number of files found in archive
     */
    void scanCompleted(const QString& archivePath, int filesFound);
    
    /**
     * @brief Emitted when an error occurs during archive operations
     * @param archivePath Path to archive that caused error
     * @param error Error message
     */
    void errorOccurred(const QString& archivePath, const QString& error);

private slots:
    void onExtractionProgress(int percentage);

private:
    // Archive format detection
    bool isZipFile(const QString& filePath) const;
    bool isTarFile(const QString& filePath) const;
    bool isRarFile(const QString& filePath) const;
    
    // Format-specific scanning
    QList<ArchiveFileInfo> scanZipArchive(const QString& archivePath);
    QList<ArchiveFileInfo> scanTarArchive(const QString& archivePath);
    QList<ArchiveFileInfo> scanRarArchive(const QString& archivePath);
    
    // Format-specific extraction
    QByteArray extractFromZip(const QString& archivePath, const QString& internalPath);
    QByteArray extractFromTar(const QString& archivePath, const QString& internalPath);
    QByteArray extractFromRar(const QString& archivePath, const QString& internalPath);
    
    // Utility methods
    QString createTempFile(const QByteArray& content, const QString& originalName);
    void cleanupTempFiles();
    bool shouldExtractToMemory(qint64 fileSize) const;
    QString sanitizeFileName(const QString& fileName) const;
    
    // Member variables
    ArchiveScanConfig m_config;
    std::unique_ptr<QTemporaryDir> m_tempDir;
    QStringList m_tempFiles;
    int m_currentNestingDepth;
    
    // Statistics
    int m_totalFilesScanned;
    int m_totalArchivesProcessed;
    qint64 m_totalBytesExtracted;
};

#endif // ARCHIVE_HANDLER_H