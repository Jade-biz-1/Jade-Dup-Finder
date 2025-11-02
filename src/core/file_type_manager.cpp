#include "file_type_manager.h"
#include <QDebug>
#include <QFileInfo>
#include <QDateTime>

FileTypeManager::FileTypeManager(QObject* parent)
    : QObject(parent)
{
    qDebug() << "FileTypeManager initialized";
    
    // Create handler instances
    m_archiveHandler = std::make_unique<ArchiveHandler>(this);
    m_documentHandler = std::make_unique<DocumentHandler>(this);
    m_mediaHandler = std::make_unique<MediaHandler>(this);
    
    // Connect archive handler signals
    connect(m_archiveHandler.get(), &ArchiveHandler::scanStarted,
            this, &FileTypeManager::onArchiveProcessingStarted);
    connect(m_archiveHandler.get(), &ArchiveHandler::scanProgress,
            this, &FileTypeManager::onArchiveProcessingProgress);
    connect(m_archiveHandler.get(), &ArchiveHandler::scanCompleted,
            this, &FileTypeManager::onArchiveProcessingCompleted);
    connect(m_archiveHandler.get(), &ArchiveHandler::errorOccurred,
            this, &FileTypeManager::onArchiveErrorOccurred);
    
    // Connect document handler signals
    connect(m_documentHandler.get(), &DocumentHandler::processingStarted,
            this, &FileTypeManager::onDocumentProcessingStarted);
    connect(m_documentHandler.get(), &DocumentHandler::processingProgress,
            this, &FileTypeManager::onDocumentProcessingProgress);
    connect(m_documentHandler.get(), &DocumentHandler::processingCompleted,
            this, &FileTypeManager::onDocumentProcessingCompleted);
    connect(m_documentHandler.get(), &DocumentHandler::errorOccurred,
            this, &FileTypeManager::onDocumentErrorOccurred);
    
    // Connect media handler signals
    connect(m_mediaHandler.get(), &MediaHandler::processingStarted,
            this, &FileTypeManager::onMediaProcessingStarted);
    connect(m_mediaHandler.get(), &MediaHandler::processingProgress,
            this, &FileTypeManager::onMediaProcessingProgress);
    connect(m_mediaHandler.get(), &MediaHandler::processingCompleted,
            this, &FileTypeManager::onMediaProcessingCompleted);
    connect(m_mediaHandler.get(), &MediaHandler::errorOccurred,
            this, &FileTypeManager::onMediaErrorOccurred);
    
    // Set default configuration
    m_config = FileTypeConfig();
}

FileTypeManager::~FileTypeManager()
{
    qDebug() << "FileTypeManager destroyed. Stats - Total files processed:" << m_statistics.totalFilesProcessed
             << "Archives:" << m_statistics.archiveFilesProcessed
             << "Documents:" << m_statistics.documentFilesProcessed
             << "Media:" << m_statistics.mediaFilesProcessed
             << "Errors:" << m_statistics.processingErrors;
}

void FileTypeManager::setConfiguration(const FileTypeConfig& config)
{
    m_config = config;
    
    // Update handler configurations
    if (m_archiveHandler) {
        m_archiveHandler->setConfiguration(config.archiveConfig);
    }
    if (m_documentHandler) {
        m_documentHandler->setConfiguration(config.documentConfig);
    }
    if (m_mediaHandler) {
        m_mediaHandler->setConfiguration(config.mediaConfig);
    }
    
    qDebug() << "FileTypeManager configuration updated:"
             << "Archives:" << config.enableArchiveScanning
             << "Documents:" << config.enableDocumentProcessing
             << "Media:" << config.enableMediaProcessing
             << "Auto-detect:" << config.autoDetectFileTypes;
}

FileTypeConfig FileTypeManager::configuration() const
{
    return m_config;
}

EnhancedFileInfo FileTypeManager::processFile(const QString& filePath)
{
    EnhancedFileInfo info;
    
    if (!QFileInfo::exists(filePath)) {
        info.filePath = filePath;
        info.processingFailed = true;
        info.errorMessage = "File does not exist";
        emit errorOccurred(filePath, info.errorMessage);
        return info;
    }
    
    if (!shouldProcessFile(filePath)) {
        info.filePath = filePath;
        info.fileName = QFileInfo(filePath).fileName();
        info.fileType = "REGULAR";
        info.processed = false;
        return info;
    }
    
    QFileInfo fileInfo(filePath);
    info.filePath = filePath;
    info.fileName = fileInfo.fileName();
    info.fileSize = fileInfo.size();
    info.lastModified = fileInfo.lastModified();
    
    // Determine file type
    if (m_config.autoDetectFileTypes) {
        info.fileType = determineFileType(filePath);
    } else {
        info.fileType = "REGULAR";
    }
    
    m_currentFile = filePath;
    m_currentFileType = info.fileType;
    
    emit processingStarted(filePath, info.fileType);
    
    try {
        // Process based on file type
        if (info.fileType == "ARCHIVE" && m_config.enableArchiveScanning) {
            info = processArchiveFile(filePath);
        } else if (info.fileType == "DOCUMENT" && m_config.enableDocumentProcessing) {
            info = processDocumentFile(filePath);
        } else if (info.fileType == "MEDIA" && m_config.enableMediaProcessing) {
            info = processMediaFile(filePath);
        } else {
            info = processRegularFile(filePath);
        }
        
        info.processed = true;
        updateStatistics(info.fileType, true);
        
        emit processingCompleted(filePath, true);
        
    } catch (const std::exception& e) {
        info.processingFailed = true;
        info.errorMessage = QString("Exception during processing: %1").arg(e.what());
        updateStatistics(info.fileType, false);
        
        emit errorOccurred(filePath, info.errorMessage);
        emit processingCompleted(filePath, false);
    }
    
    return info;
}

QList<EnhancedFileInfo> FileTypeManager::processFiles(const QStringList& filePaths)
{
    QList<EnhancedFileInfo> results;
    results.reserve(filePaths.size());
    
    int successfulFiles = 0;
    int failedFiles = 0;
    
    for (const QString& filePath : filePaths) {
        EnhancedFileInfo info = processFile(filePath);
        results.append(info);
        
        if (info.processingFailed) {
            failedFiles++;
        } else {
            successfulFiles++;
        }
    }
    
    emit batchProcessingCompleted(filePaths.size(), successfulFiles, failedFiles);
    
    return results;
}

QString FileTypeManager::determineFileType(const QString& filePath)
{
    if (ArchiveHandler::isArchiveFile(filePath)) {
        return "ARCHIVE";
    } else if (DocumentHandler::isDocumentFile(filePath)) {
        return "DOCUMENT";
    } else if (MediaHandler::isMediaFile(filePath)) {
        return "MEDIA";
    } else {
        return "REGULAR";
    }
}

bool FileTypeManager::shouldProcessFile(const QString& filePath) const
{
    QFileInfo fileInfo(filePath);
    
    // Check file size limit
    if (fileInfo.size() > m_config.maxFileSize) {
        return false;
    }
    
    // Check excluded extensions
    if (isExcludedExtension(filePath)) {
        return false;
    }
    
    // Check if file exists and is readable
    if (!fileInfo.exists() || !fileInfo.isReadable()) {
        return false;
    }
    
    return true;
}

ArchiveHandler* FileTypeManager::archiveHandler() const
{
    return m_archiveHandler.get();
}

DocumentHandler* FileTypeManager::documentHandler() const
{
    return m_documentHandler.get();
}

MediaHandler* FileTypeManager::mediaHandler() const
{
    return m_mediaHandler.get();
}

QVariantMap FileTypeManager::getStatistics() const
{
    QVariantMap stats;
    
    stats["totalFilesProcessed"] = m_statistics.totalFilesProcessed;
    stats["regularFilesProcessed"] = m_statistics.regularFilesProcessed;
    stats["archiveFilesProcessed"] = m_statistics.archiveFilesProcessed;
    stats["documentFilesProcessed"] = m_statistics.documentFilesProcessed;
    stats["mediaFilesProcessed"] = m_statistics.mediaFilesProcessed;
    stats["processingErrors"] = m_statistics.processingErrors;
    stats["totalBytesProcessed"] = static_cast<qint64>(m_statistics.totalBytesProcessed);
    stats["lastProcessingTime"] = m_statistics.lastProcessingTime;
    
    return stats;
}

void FileTypeManager::resetStatistics()
{
    m_statistics = Statistics();
    qDebug() << "FileTypeManager statistics reset";
}

// Signal handlers

void FileTypeManager::onArchiveProcessingStarted(const QString& archivePath, int totalFiles)
{
    Q_UNUSED(totalFiles)
    if (archivePath == m_currentFile) {
        emit processingProgress(archivePath, 10);
    }
}

void FileTypeManager::onArchiveProcessingProgress(const QString& archivePath, int current, int total, const QString& currentFile)
{
    Q_UNUSED(currentFile)
    if (archivePath == m_currentFile && total > 0) {
        int percentage = 10 + (static_cast<int>((static_cast<double>(current) / total) * 80));
        emit processingProgress(archivePath, percentage);
    }
}

void FileTypeManager::onArchiveProcessingCompleted(const QString& archivePath, int filesFound)
{
    Q_UNUSED(filesFound)
    if (archivePath == m_currentFile) {
        emit processingProgress(archivePath, 90);
    }
}

void FileTypeManager::onArchiveErrorOccurred(const QString& archivePath, const QString& error)
{
    if (archivePath == m_currentFile) {
        emit errorOccurred(archivePath, error);
    }
}

void FileTypeManager::onDocumentProcessingStarted(const QString& filePath)
{
    if (filePath == m_currentFile) {
        emit processingProgress(filePath, 10);
    }
}

void FileTypeManager::onDocumentProcessingProgress(const QString& filePath, int percentage)
{
    if (filePath == m_currentFile) {
        // Scale to 10-90% range
        int scaledPercentage = 10 + (percentage * 80 / 100);
        emit processingProgress(filePath, scaledPercentage);
    }
}

void FileTypeManager::onDocumentProcessingCompleted(const QString& filePath, bool success)
{
    Q_UNUSED(success)
    if (filePath == m_currentFile) {
        emit processingProgress(filePath, 90);
    }
}

void FileTypeManager::onDocumentErrorOccurred(const QString& filePath, const QString& error)
{
    if (filePath == m_currentFile) {
        emit errorOccurred(filePath, error);
    }
}

void FileTypeManager::onMediaProcessingStarted(const QString& filePath)
{
    if (filePath == m_currentFile) {
        emit processingProgress(filePath, 10);
    }
}

void FileTypeManager::onMediaProcessingProgress(const QString& filePath, int percentage)
{
    if (filePath == m_currentFile) {
        // Scale to 10-90% range
        int scaledPercentage = 10 + (percentage * 80 / 100);
        emit processingProgress(filePath, scaledPercentage);
    }
}

void FileTypeManager::onMediaProcessingCompleted(const QString& filePath, bool success)
{
    Q_UNUSED(success)
    if (filePath == m_currentFile) {
        emit processingProgress(filePath, 90);
    }
}

void FileTypeManager::onMediaErrorOccurred(const QString& filePath, const QString& error)
{
    if (filePath == m_currentFile) {
        emit errorOccurred(filePath, error);
    }
}

// Private implementation methods

EnhancedFileInfo FileTypeManager::processRegularFile(const QString& filePath)
{
    EnhancedFileInfo info;
    QFileInfo fileInfo(filePath);
    
    info.filePath = filePath;
    info.fileName = fileInfo.fileName();
    info.fileType = "REGULAR";
    info.fileSize = fileInfo.size();
    info.lastModified = fileInfo.lastModified();
    info.processed = true;
    
    return info;
}

EnhancedFileInfo FileTypeManager::processArchiveFile(const QString& filePath)
{
    EnhancedFileInfo info = processRegularFile(filePath);
    info.fileType = "ARCHIVE";
    
    if (m_archiveHandler) {
        try {
            info.archiveContents = m_archiveHandler->scanArchive(filePath);
            qDebug() << "Archive processed:" << filePath << "Files found:" << info.archiveContents.size();
        } catch (const std::exception& e) {
            info.processingFailed = true;
            info.errorMessage = QString("Archive processing failed: %1").arg(e.what());
        }
    }
    
    return info;
}

EnhancedFileInfo FileTypeManager::processDocumentFile(const QString& filePath)
{
    EnhancedFileInfo info = processRegularFile(filePath);
    info.fileType = "DOCUMENT";
    
    if (m_documentHandler) {
        try {
            info.documentInfo = m_documentHandler->extractDocumentInfo(filePath);
            qDebug() << "Document processed:" << filePath 
                     << "Text length:" << info.documentInfo.textContent.length()
                     << "Words:" << info.documentInfo.wordCount;
        } catch (const std::exception& e) {
            info.processingFailed = true;
            info.errorMessage = QString("Document processing failed: %1").arg(e.what());
        }
    }
    
    return info;
}

EnhancedFileInfo FileTypeManager::processMediaFile(const QString& filePath)
{
    EnhancedFileInfo info = processRegularFile(filePath);
    info.fileType = "MEDIA";
    
    if (m_mediaHandler) {
        try {
            info.mediaInfo = m_mediaHandler->extractMediaInfo(filePath);
            qDebug() << "Media processed:" << filePath 
                     << "Type:" << info.mediaInfo.mediaType
                     << "Format:" << info.mediaInfo.format
                     << "Dimensions:" << info.mediaInfo.dimensions;
        } catch (const std::exception& e) {
            info.processingFailed = true;
            info.errorMessage = QString("Media processing failed: %1").arg(e.what());
        }
    }
    
    return info;
}

bool FileTypeManager::isExcludedExtension(const QString& filePath) const
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    return m_config.excludedExtensions.contains(extension);
}

void FileTypeManager::updateStatistics(const QString& fileType, bool success)
{
    m_statistics.totalFilesProcessed++;
    m_statistics.lastProcessingTime = QDateTime::currentDateTime();
    
    if (success) {
        if (fileType == "REGULAR") {
            m_statistics.regularFilesProcessed++;
        } else if (fileType == "ARCHIVE") {
            m_statistics.archiveFilesProcessed++;
        } else if (fileType == "DOCUMENT") {
            m_statistics.documentFilesProcessed++;
        } else if (fileType == "MEDIA") {
            m_statistics.mediaFilesProcessed++;
        }
        
        QFileInfo fileInfo(m_currentFile);
        m_statistics.totalBytesProcessed += fileInfo.size();
    } else {
        m_statistics.processingErrors++;
    }
}