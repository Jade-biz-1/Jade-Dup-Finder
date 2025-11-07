#include "archive_handler.h"
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QMimeDatabase>
#include <QMimeType>
#include <QStandardPaths>
#include <QCoreApplication>
#include <QProcess>
#include <QTemporaryFile>
#include <QRegularExpression>

// Archive support using external tools and Qt's process handling

ArchiveHandler::ArchiveHandler(QObject* parent)
    : QObject(parent)
    , m_currentNestingDepth(0)
    , m_totalFilesScanned(0)
    , m_totalArchivesProcessed(0)
    , m_totalBytesExtracted(0)
{
    qDebug() << "ArchiveHandler initialized";
    
    // Create temporary directory for file extraction
    m_tempDir = std::make_unique<QTemporaryDir>();
    if (!m_tempDir->isValid()) {
        qWarning() << "Failed to create temporary directory for archive extraction";
    } else {
        qDebug() << "Archive temp directory:" << m_tempDir->path();
    }
}

ArchiveHandler::~ArchiveHandler()
{
    cleanupTempFiles();
    qDebug() << "ArchiveHandler destroyed. Stats - Files scanned:" << m_totalFilesScanned 
             << "Archives processed:" << m_totalArchivesProcessed 
             << "Bytes extracted:" << m_totalBytesExtracted;
}

bool ArchiveHandler::isArchiveFile(const QString& filePath)
{
    if (filePath.isEmpty() || !QFileInfo::exists(filePath)) {
        return false;
    }
    
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    // Check by extension first (fastest)
    QStringList archiveExtensions = {
        "zip", "jar", "war", "ear",  // ZIP-based formats
        "tar", "tar.gz", "tgz", "tar.bz2", "tbz2", "tar.xz", "txz",  // TAR formats
        "rar",  // RAR format
        "7z",   // 7-Zip format
        "gz", "bz2", "xz"  // Compressed single files
    };
    
    if (archiveExtensions.contains(extension)) {
        return true;
    }
    
    // Check for compound extensions (e.g., tar.gz)
    QString fullExtension = fileInfo.completeSuffix().toLower();
    QStringList compoundExtensions = {"tar.gz", "tar.bz2", "tar.xz"};
    if (compoundExtensions.contains(fullExtension)) {
        return true;
    }
    
    // Fallback to MIME type detection
    QMimeDatabase mimeDb;
    QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
    QString mimeName = mimeType.name();
    
    QStringList archiveMimeTypes = {
        "application/zip",
        "application/x-tar",
        "application/gzip",
        "application/x-bzip2",
        "application/x-xz",
        "application/x-rar-compressed",
        "application/x-7z-compressed"
    };
    
    return archiveMimeTypes.contains(mimeName);
}

QStringList ArchiveHandler::supportedExtensions()
{
    return {
        "zip", "jar", "war", "ear",
        "tar", "tar.gz", "tgz", "tar.bz2", "tbz2", "tar.xz", "txz",
        "rar", "7z", "gz", "bz2", "xz"
    };
}

void ArchiveHandler::setConfiguration(const ArchiveScanConfig& config)
{
    m_config = config;
    qDebug() << "Archive configuration updated:"
             << "ZIP:" << config.scanZipFiles
             << "TAR:" << config.scanTarFiles
             << "RAR:" << config.scanRarFiles
             << "Nested:" << config.scanNestedArchives
             << "Max depth:" << config.maxNestingDepth;
}

ArchiveScanConfig ArchiveHandler::configuration() const
{
    return m_config;
}

QList<ArchiveFileInfo> ArchiveHandler::scanArchive(const QString& archivePath)
{
    QList<ArchiveFileInfo> files;
    
    if (!QFileInfo::exists(archivePath)) {
        emit errorOccurred(archivePath, "Archive file does not exist");
        return files;
    }
    
    if (!isScanningEnabled(archivePath)) {
        qDebug() << "Scanning disabled for archive type:" << archivePath;
        return files;
    }
    
    QFileInfo archiveInfo(archivePath);
    if (archiveInfo.size() > m_config.maxArchiveSize) {
        emit errorOccurred(archivePath, QString("Archive too large (%1 bytes, max: %2)")
                          .arg(archiveInfo.size()).arg(m_config.maxArchiveSize));
        return files;
    }
    
    emit scanStarted(archivePath, -1); // Unknown file count initially
    
    try {
        QString format = getArchiveFormat(archivePath);
        qDebug() << "Scanning archive:" << archivePath << "Format:" << format;
        
        if (format == "ZIP") {
            files = scanZipArchive(archivePath);
        } else if (format == "TAR") {
            files = scanTarArchive(archivePath);
        } else if (format == "RAR") {
            files = scanRarArchive(archivePath);
        } else {
            emit errorOccurred(archivePath, "Unsupported archive format: " + format);
            return files;
        }
        
        m_totalArchivesProcessed++;
        m_totalFilesScanned += files.size();

        emit scanCompleted(archivePath, static_cast<int>(files.size()));
        qDebug() << "Archive scan completed:" << archivePath << "Files found:" << files.size();
        
    } catch (const std::exception& e) {
        emit errorOccurred(archivePath, QString("Exception during scan: %1").arg(e.what()));
    }
    
    return files;
}

QByteArray ArchiveHandler::extractFileContent(const QString& archivePath, const QString& internalPath)
{
    if (!QFileInfo::exists(archivePath)) {
        qWarning() << "Archive does not exist:" << archivePath;
        return QByteArray();
    }
    
    try {
        QString format = getArchiveFormat(archivePath);
        
        if (format == "ZIP") {
            return extractFromZip(archivePath, internalPath);
        } else if (format == "TAR") {
            return extractFromTar(archivePath, internalPath);
        } else if (format == "RAR") {
            return extractFromRar(archivePath, internalPath);
        } else {
            qWarning() << "Unsupported archive format for extraction:" << format;
            return QByteArray();
        }
        
    } catch (const std::exception& e) {
        qWarning() << "Exception during extraction:" << e.what();
        return QByteArray();
    }
}

QString ArchiveHandler::extractToTempFile(const QString& archivePath, const QString& internalPath)
{
    QByteArray content = extractFileContent(archivePath, internalPath);
    if (content.isEmpty()) {
        return QString();
    }
    
    QFileInfo pathInfo(internalPath);
    QString tempFilePath = createTempFile(content, pathInfo.fileName());
    
    if (!tempFilePath.isEmpty()) {
        m_totalBytesExtracted += content.size();
    }
    
    return tempFilePath;
}

QString ArchiveHandler::getArchiveFormat(const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    QString fullExtension = fileInfo.completeSuffix().toLower();
    
    // ZIP-based formats
    if (extension == "zip" || extension == "jar" || extension == "war" || extension == "ear") {
        return "ZIP";
    }
    
    // TAR-based formats
    if (extension == "tar" || fullExtension.contains("tar.")) {
        return "TAR";
    }
    
    // RAR format
    if (extension == "rar") {
        return "RAR";
    }
    
    // 7-Zip format
    if (extension == "7z") {
        return "7Z";
    }
    
    // Compressed single files
    if (extension == "gz" || extension == "bz2" || extension == "xz") {
        return "COMPRESSED";
    }
    
    return "UNKNOWN";
}

bool ArchiveHandler::isScanningEnabled(const QString& filePath) const
{
    QString format = getArchiveFormat(filePath);
    
    if (format == "ZIP") {
        return m_config.scanZipFiles;
    } else if (format == "TAR") {
        return m_config.scanTarFiles;
    } else if (format == "RAR") {
        return m_config.scanRarFiles;
    }
    
    return false;
}

// Private implementation methods

bool ArchiveHandler::isZipFile(const QString& filePath) const
{
    return getArchiveFormat(filePath) == "ZIP";
}

bool ArchiveHandler::isTarFile(const QString& filePath) const
{
    return getArchiveFormat(filePath) == "TAR";
}

bool ArchiveHandler::isRarFile(const QString& filePath) const
{
    return getArchiveFormat(filePath) == "RAR";
}

QList<ArchiveFileInfo> ArchiveHandler::scanZipArchive(const QString& archivePath)
{
    QList<ArchiveFileInfo> files;
    
    // Use external unzip command to list archive contents
    QProcess unzipProcess;
    QStringList arguments;
    arguments << "-l" << archivePath; // List contents
    
    unzipProcess.start("unzip", arguments);
    if (!unzipProcess.waitForStarted(5000)) {
        emit errorOccurred(archivePath, "Failed to start unzip command. Please ensure unzip is installed.");
        return files;
    }
    
    if (!unzipProcess.waitForFinished(30000)) {
        emit errorOccurred(archivePath, "Unzip command timed out");
        unzipProcess.kill();
        return files;
    }
    
    if (unzipProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(unzipProcess.readAllStandardError());
        emit errorOccurred(archivePath, "Unzip failed: " + error);
        return files;
    }
    
    // Parse unzip output
    QString output = QString::fromUtf8(unzipProcess.readAllStandardOutput());
    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    
    bool inFileList = false;
    int fileCount = 0;
    
    for (const QString& line : lines) {
        // Skip header lines until we reach the file list
        if (line.contains("Length") && line.contains("Name")) {
            inFileList = true;
            continue;
        }
        
        if (!inFileList) {
            continue;
        }
        
        // Stop at footer line
        if (line.contains("----") || line.contains("files")) {
            break;
        }
        
        // Parse file information from unzip -l output
        // Format: "  Length      Date    Time    Name"
        QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() < 4) {
            continue;
        }
        
        bool ok;
        qint64 fileSize = parts[0].toLongLong(&ok);
        if (!ok) {
            continue;
        }
        
        // Extract filename (everything after the time field)
        int nameIndex = static_cast<int>(line.lastIndexOf(parts[3])) + static_cast<int>(parts[3].length());
        if (nameIndex >= line.length()) {
            continue;
        }
        
        QString fileName = line.mid(nameIndex).trimmed();
        if (fileName.isEmpty() || fileName.endsWith('/')) {
            continue; // Skip directories
        }
        
        emit scanProgress(archivePath, ++fileCount, -1, fileName);
        
        // Check file size limits
        if (fileSize > m_config.maxFileSize) {
            qDebug() << "Skipping large file in archive:" << fileName << "Size:" << fileSize;
            continue;
        }
        
        ArchiveFileInfo archiveFile;
        archiveFile.fileName = QFileInfo(fileName).fileName();
        archiveFile.fullPath = fileName;
        archiveFile.fileSize = fileSize;
        archiveFile.compressedSize = 0; // Not available from unzip -l
        archiveFile.archivePath = archivePath;
        archiveFile.isDirectory = false;
        
        files.append(archiveFile);
        
        // Handle nested archives if enabled
        if (m_config.scanNestedArchives && 
            m_currentNestingDepth < m_config.maxNestingDepth &&
            isArchiveFile(archiveFile.fileName)) {
            
            qDebug() << "Found nested archive:" << archiveFile.fullPath;
            
            // Extract nested archive to temp file and scan it
            QString tempArchivePath = extractToTempFile(archivePath, fileName);
            if (!tempArchivePath.isEmpty()) {
                m_currentNestingDepth++;
                QList<ArchiveFileInfo> nestedFiles = scanArchive(tempArchivePath);
                m_currentNestingDepth--;
                
                // Update nested file paths to include parent archive info
                for (ArchiveFileInfo& nestedFile : nestedFiles) {
                    nestedFile.fullPath = archiveFile.fullPath + "/" + nestedFile.fullPath;
                    nestedFile.archivePath = archivePath; // Keep original archive path
                }
                
                files.append(nestedFiles);
            }
        }
    }
    
    return files;
}

QList<ArchiveFileInfo> ArchiveHandler::scanTarArchive(const QString& archivePath)
{
    QList<ArchiveFileInfo> files;
    
    // Use external tar command to list archive contents
    QProcess tarProcess;
    QStringList arguments;
    
    // Determine compression type and use appropriate tar options
    QString fileName = QFileInfo(archivePath).fileName().toLower();
    if (fileName.contains(".tar.gz") || fileName.contains(".tgz")) {
        arguments << "-tzf" << archivePath; // List gzipped tar
    } else if (fileName.contains(".tar.bz2") || fileName.contains(".tbz2")) {
        arguments << "-tjf" << archivePath; // List bzip2 tar
    } else if (fileName.contains(".tar.xz") || fileName.contains(".txz")) {
        arguments << "-tJf" << archivePath; // List xz tar
    } else {
        arguments << "-tf" << archivePath; // List uncompressed tar
    }
    
    tarProcess.start("tar", arguments);
    if (!tarProcess.waitForStarted(5000)) {
        emit errorOccurred(archivePath, "Failed to start tar command. Please ensure tar is installed.");
        return files;
    }
    
    if (!tarProcess.waitForFinished(30000)) {
        emit errorOccurred(archivePath, "Tar command timed out");
        tarProcess.kill();
        return files;
    }
    
    if (tarProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(tarProcess.readAllStandardError());
        emit errorOccurred(archivePath, "Tar listing failed: " + error);
        return files;
    }
    
    // Parse tar output
    QString output = QString::fromUtf8(tarProcess.readAllStandardOutput());
    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    
    int fileCount = 0;
    
    for (const QString& line : lines) {
        QString extractedFileName = line.trimmed();
        if (extractedFileName.isEmpty() || extractedFileName.endsWith('/')) {
            continue; // Skip directories
        }
        
        emit scanProgress(archivePath, ++fileCount, static_cast<int>(lines.size()), extractedFileName);
        
        // Get detailed file info using tar -tvf
        QProcess detailProcess;
        QStringList detailArgs;
        
        if (archivePath.contains(".tar.gz") || archivePath.contains(".tgz")) {
            detailArgs << "-tzvf" << archivePath << extractedFileName;
        } else if (archivePath.contains(".tar.bz2") || archivePath.contains(".tbz2")) {
            detailArgs << "-tjvf" << archivePath << extractedFileName;
        } else if (archivePath.contains(".tar.xz") || archivePath.contains(".txz")) {
            detailArgs << "-tJvf" << archivePath << extractedFileName;
        } else {
            detailArgs << "-tvf" << archivePath << extractedFileName;
        }
        
        detailProcess.start("tar", detailArgs);
        if (detailProcess.waitForFinished(5000)) {
            QString detailOutput = QString::fromUtf8(detailProcess.readAllStandardOutput());
            QStringList detailParts = detailOutput.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            
            qint64 fileSize = 0;
            if (detailParts.size() >= 3) {
                fileSize = detailParts[2].toLongLong();
            }
            
            // Check file size limits
            if (fileSize > m_config.maxFileSize) {
                qDebug() << "Skipping large file in tar archive:" << fileName << "Size:" << fileSize;
                continue;
            }
            
            ArchiveFileInfo archiveFile;
            archiveFile.fileName = QFileInfo(extractedFileName).fileName();
            archiveFile.fullPath = extractedFileName;
            archiveFile.fileSize = fileSize;
            archiveFile.compressedSize = fileSize; // TAR doesn't compress individual files
            archiveFile.archivePath = archivePath;
            archiveFile.isDirectory = false;
            
            files.append(archiveFile);
            
            // Handle nested archives if enabled
            if (m_config.scanNestedArchives && 
                m_currentNestingDepth < m_config.maxNestingDepth &&
                isArchiveFile(archiveFile.fileName)) {
                
                qDebug() << "Found nested archive in TAR:" << archiveFile.fullPath;
                
                // Extract nested archive to temp file and scan it
                QString tempArchivePath = extractToTempFile(archivePath, extractedFileName);
                if (!tempArchivePath.isEmpty()) {
                    m_currentNestingDepth++;
                    QList<ArchiveFileInfo> nestedFiles = scanArchive(tempArchivePath);
                    m_currentNestingDepth--;
                    
                    // Update nested file paths to include parent archive info
                    for (ArchiveFileInfo& nestedFile : nestedFiles) {
                        nestedFile.fullPath = archiveFile.fullPath + "/" + nestedFile.fullPath;
                        nestedFile.archivePath = archivePath; // Keep original archive path
                    }
                    
                    files.append(nestedFiles);
                }
            }
        }
    }
    
    return files;
}

QList<ArchiveFileInfo> ArchiveHandler::scanRarArchive(const QString& archivePath)
{
    QList<ArchiveFileInfo> files;
    
    // RAR support would require external tool (unrar)
    // For now, emit error indicating RAR support is not yet implemented
    emit errorOccurred(archivePath, "RAR archive support not yet implemented");
    
    return files;
}

QByteArray ArchiveHandler::extractFromZip(const QString& archivePath, const QString& internalPath)
{
    // Use external unzip command to extract specific file
    QProcess unzipProcess;
    QStringList arguments;
    arguments << "-p" << archivePath << internalPath; // Extract to stdout
    
    unzipProcess.start("unzip", arguments);
    if (!unzipProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start unzip command for extraction";
        return QByteArray();
    }
    
    if (!unzipProcess.waitForFinished(30000)) {
        qWarning() << "Unzip extraction timed out";
        unzipProcess.kill();
        return QByteArray();
    }
    
    if (unzipProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(unzipProcess.readAllStandardError());
        qWarning() << "Unzip extraction failed:" << error;
        return QByteArray();
    }
    
    return unzipProcess.readAllStandardOutput();
}

QByteArray ArchiveHandler::extractFromTar(const QString& archivePath, const QString& internalPath)
{
    // Use external tar command to extract specific file to stdout
    QProcess tarProcess;
    QStringList arguments;
    
    // Determine compression type and use appropriate tar options
    QString fileName = QFileInfo(archivePath).fileName().toLower();
    if (fileName.contains(".tar.gz") || fileName.contains(".tgz")) {
        arguments << "-xzf" << archivePath << "-O" << internalPath; // Extract gzipped tar to stdout
    } else if (fileName.contains(".tar.bz2") || fileName.contains(".tbz2")) {
        arguments << "-xjf" << archivePath << "-O" << internalPath; // Extract bzip2 tar to stdout
    } else if (fileName.contains(".tar.xz") || fileName.contains(".txz")) {
        arguments << "-xJf" << archivePath << "-O" << internalPath; // Extract xz tar to stdout
    } else {
        arguments << "-xf" << archivePath << "-O" << internalPath; // Extract uncompressed tar to stdout
    }
    
    tarProcess.start("tar", arguments);
    if (!tarProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start tar command for extraction";
        return QByteArray();
    }
    
    if (!tarProcess.waitForFinished(30000)) {
        qWarning() << "Tar extraction timed out";
        tarProcess.kill();
        return QByteArray();
    }
    
    if (tarProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(tarProcess.readAllStandardError());
        qWarning() << "Tar extraction failed:" << error;
        return QByteArray();
    }
    
    return tarProcess.readAllStandardOutput();
}

QByteArray ArchiveHandler::extractFromRar(const QString& archivePath, const QString& internalPath)
{
    Q_UNUSED(archivePath)
    Q_UNUSED(internalPath)
    
    // RAR extraction not yet implemented
    qWarning() << "RAR extraction not yet implemented";
    return QByteArray();
}

QString ArchiveHandler::createTempFile(const QByteArray& content, const QString& originalName)
{
    if (!m_tempDir || !m_tempDir->isValid()) {
        qWarning() << "Temporary directory not available";
        return QString();
    }
    
    QString sanitizedName = sanitizeFileName(originalName);
    QString tempFilePath = m_tempDir->path() + "/" + sanitizedName;
    
    // Ensure unique filename
    int counter = 1;
    QString basePath = tempFilePath;
    while (QFileInfo::exists(tempFilePath)) {
        QFileInfo info(basePath);
        tempFilePath = info.path() + "/" + info.baseName() + 
                      QString("_%1").arg(counter) + "." + info.suffix();
        counter++;
    }
    
    QFile tempFile(tempFilePath);
    if (!tempFile.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to create temporary file:" << tempFilePath;
        return QString();
    }
    
    qint64 bytesWritten = tempFile.write(content);
    tempFile.close();
    
    if (bytesWritten != content.size()) {
        qWarning() << "Failed to write complete content to temporary file";
        QFile::remove(tempFilePath);
        return QString();
    }
    
    m_tempFiles.append(tempFilePath);
    return tempFilePath;
}

void ArchiveHandler::cleanupTempFiles()
{
    for (const QString& tempFile : m_tempFiles) {
        if (QFileInfo::exists(tempFile)) {
            QFile::remove(tempFile);
        }
    }
    m_tempFiles.clear();
}

bool ArchiveHandler::shouldExtractToMemory(qint64 fileSize) const
{
    return m_config.extractToMemory && fileSize <= m_config.memoryThreshold;
}

QString ArchiveHandler::sanitizeFileName(const QString& fileName) const
{
    QString sanitized = fileName;
    
    // Replace invalid characters with underscores
    QRegularExpression invalidChars("[<>:\"/\\\\|?*]");
    sanitized.replace(invalidChars, "_");
    
    // Ensure filename is not empty
    if (sanitized.isEmpty()) {
        sanitized = "extracted_file";
    }
    
    return sanitized;
}

void ArchiveHandler::onExtractionProgress(int percentage)
{
    // This slot can be connected to extraction progress signals
    // Currently not used but available for future enhancements
    Q_UNUSED(percentage)
}