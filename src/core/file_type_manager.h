#ifndef FILE_TYPE_MANAGER_H
#define FILE_TYPE_MANAGER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantMap>
#include <memory>

#include "archive_handler.h"
#include "document_handler.h"
#include "media_handler.h"

/**
 * @brief File type processing configuration
 */
struct FileTypeConfig {
    // Archive processing
    bool enableArchiveScanning = true;
    ArchiveScanConfig archiveConfig;
    
    // Document processing
    bool enableDocumentProcessing = true;
    DocumentScanConfig documentConfig;
    
    // Media processing
    bool enableMediaProcessing = true;
    MediaScanConfig mediaConfig;
    
    // General settings
    bool autoDetectFileTypes = true;
    QStringList excludedExtensions;
    qint64 maxFileSize = 1024 * 1024 * 1024; // 1GB default limit
};

/**
 * @brief Enhanced file information with type-specific data
 */
struct EnhancedFileInfo {
    QString filePath;
    QString fileName;
    QString fileType;           // REGULAR, ARCHIVE, DOCUMENT, MEDIA
    qint64 fileSize;
    QDateTime lastModified;
    
    // Type-specific information
    QList<ArchiveFileInfo> archiveContents;  // For archives
    DocumentInfo documentInfo;               // For documents
    MediaInfo mediaInfo;                     // For media files
    
    // Processing status
    bool processed = false;
    bool processingFailed = false;
    QString errorMessage;
    
    EnhancedFileInfo() : fileSize(0) {}
};

/**
 * @brief Central manager for file type-specific processing
 * 
 * This class coordinates the processing of different file types using
 * specialized handlers for archives, documents, and media files.
 */
class FileTypeManager : public QObject
{
    Q_OBJECT

public:
    explicit FileTypeManager(QObject* parent = nullptr);
    ~FileTypeManager();

    /**
     * @brief Set file type processing configuration
     * @param config Configuration settings for all file type handlers
     */
    void setConfiguration(const FileTypeConfig& config);
    
    /**
     * @brief Get current file type processing configuration
     * @return Current configuration settings
     */
    FileTypeConfig configuration() const;
    
    /**
     * @brief Process a file and extract type-specific information
     * @param filePath Path to the file to process
     * @return Enhanced file information with type-specific data
     */
    EnhancedFileInfo processFile(const QString& filePath);
    
    /**
     * @brief Process multiple files
     * @param filePaths List of file paths to process
     * @return List of enhanced file information
     */
    QList<EnhancedFileInfo> processFiles(const QStringList& filePaths);
    
    /**
     * @brief Determine file type category
     * @param filePath Path to the file
     * @return File type (REGULAR, ARCHIVE, DOCUMENT, MEDIA)
     */
    static QString determineFileType(const QString& filePath);
    
    /**
     * @brief Check if a file should be processed
     * @param filePath Path to the file
     * @return True if the file should be processed
     */
    bool shouldProcessFile(const QString& filePath) const;
    
    /**
     * @brief Get archive handler instance
     * @return Pointer to archive handler
     */
    ArchiveHandler* archiveHandler() const;
    
    /**
     * @brief Get document handler instance
     * @return Pointer to document handler
     */
    DocumentHandler* documentHandler() const;
    
    /**
     * @brief Get media handler instance
     * @return Pointer to media handler
     */
    MediaHandler* mediaHandler() const;
    
    /**
     * @brief Get processing statistics
     * @return Statistics as variant map
     */
    QVariantMap getStatistics() const;
    
    /**
     * @brief Reset processing statistics
     */
    void resetStatistics();

signals:
    /**
     * @brief Emitted when file processing starts
     * @param filePath Path to file being processed
     * @param fileType Type of file being processed
     */
    void processingStarted(const QString& filePath, const QString& fileType);
    
    /**
     * @brief Emitted when processing progress updates
     * @param filePath Path to file being processed
     * @param percentage Progress percentage (0-100)
     */
    void processingProgress(const QString& filePath, int percentage);
    
    /**
     * @brief Emitted when file processing completes
     * @param filePath Path to file that was processed
     * @param success True if processing was successful
     */
    void processingCompleted(const QString& filePath, bool success);
    
    /**
     * @brief Emitted when batch processing completes
     * @param totalFiles Total files processed
     * @param successfulFiles Number of successfully processed files
     * @param failedFiles Number of failed files
     */
    void batchProcessingCompleted(int totalFiles, int successfulFiles, int failedFiles);
    
    /**
     * @brief Emitted when an error occurs during processing
     * @param filePath Path to file that caused error
     * @param error Error message
     */
    void errorOccurred(const QString& filePath, const QString& error);

private slots:
    void onArchiveProcessingStarted(const QString& archivePath, int totalFiles);
    void onArchiveProcessingProgress(const QString& archivePath, int current, int total, const QString& currentFile);
    void onArchiveProcessingCompleted(const QString& archivePath, int filesFound);
    void onArchiveErrorOccurred(const QString& archivePath, const QString& error);
    
    void onDocumentProcessingStarted(const QString& filePath);
    void onDocumentProcessingProgress(const QString& filePath, int percentage);
    void onDocumentProcessingCompleted(const QString& filePath, bool success);
    void onDocumentErrorOccurred(const QString& filePath, const QString& error);
    
    void onMediaProcessingStarted(const QString& filePath);
    void onMediaProcessingProgress(const QString& filePath, int percentage);
    void onMediaProcessingCompleted(const QString& filePath, bool success);
    void onMediaErrorOccurred(const QString& filePath, const QString& error);

private:
    // Helper methods
    EnhancedFileInfo processRegularFile(const QString& filePath);
    EnhancedFileInfo processArchiveFile(const QString& filePath);
    EnhancedFileInfo processDocumentFile(const QString& filePath);
    EnhancedFileInfo processMediaFile(const QString& filePath);
    
    bool isExcludedExtension(const QString& filePath) const;
    void updateStatistics(const QString& fileType, bool success);
    
    // Member variables
    FileTypeConfig m_config;
    
    // Handler instances
    std::unique_ptr<ArchiveHandler> m_archiveHandler;
    std::unique_ptr<DocumentHandler> m_documentHandler;
    std::unique_ptr<MediaHandler> m_mediaHandler;
    
    // Processing state
    QString m_currentFile;
    QString m_currentFileType;
    
    // Statistics
    struct Statistics {
        int totalFilesProcessed = 0;
        int regularFilesProcessed = 0;
        int archiveFilesProcessed = 0;
        int documentFilesProcessed = 0;
        int mediaFilesProcessed = 0;
        int processingErrors = 0;
        qint64 totalBytesProcessed = 0;
        QDateTime lastProcessingTime;
    } m_statistics;
};

#endif // FILE_TYPE_MANAGER_H