#include "media_handler.h"
#include <QDebug>
#include <QFileInfo>
#include <QMimeDatabase>
#include <QMimeType>
#include <QProcess>
#include <QRegularExpression>
#include <QImageReader>
#include <QBuffer>

MediaHandler::MediaHandler(QObject* parent)
    : QObject(parent)
    , m_totalMediaProcessed(0)
    , m_thumbnailsGenerated(0)
    , m_fingerprintsExtracted(0)
    , m_totalErrors(0)
{
    qDebug() << "MediaHandler initialized";
    
    // Set default configuration
    m_config = MediaScanConfig();
}

MediaHandler::~MediaHandler()
{
    qDebug() << "MediaHandler destroyed. Stats - Media processed:" << m_totalMediaProcessed 
             << "Thumbnails generated:" << m_thumbnailsGenerated
             << "Fingerprints extracted:" << m_fingerprintsExtracted
             << "Errors:" << m_totalErrors;
}

bool MediaHandler::isMediaFile(const QString& filePath)
{
    if (filePath.isEmpty() || !QFileInfo::exists(filePath)) {
        return false;
    }
    
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    // Check by extension first (fastest)
    QStringList mediaExtensions = {
        // Image formats
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "ico", "svg",
        "raw", "cr2", "nef", "arw", "dng", "orf", "rw2", "pef", "srw",
        
        // Video formats
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "3gp", "ogv",
        "mpg", "mpeg", "ts", "mts", "m2ts", "vob", "asf", "rm", "rmvb",
        
        // Audio formats
        "mp3", "wav", "flac", "aac", "ogg", "wma", "m4a", "opus", "ape", "ac3",
        "dts", "aiff", "au", "ra", "amr", "3ga"
    };
    
    if (mediaExtensions.contains(extension)) {
        return true;
    }
    
    // Fallback to MIME type detection
    QMimeDatabase mimeDb;
    QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
    QString mimeName = mimeType.name();
    
    return mimeName.startsWith("image/") || 
           mimeName.startsWith("video/") || 
           mimeName.startsWith("audio/");
}

QStringList MediaHandler::supportedExtensions()
{
    return {
        // Images
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "ico", "svg",
        "raw", "cr2", "nef", "arw", "dng", "orf", "rw2", "pef", "srw",
        
        // Videos
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "3gp", "ogv",
        "mpg", "mpeg", "ts", "mts", "m2ts", "vob", "asf", "rm", "rmvb",
        
        // Audio
        "mp3", "wav", "flac", "aac", "ogg", "wma", "m4a", "opus", "ape", "ac3",
        "dts", "aiff", "au", "ra", "amr", "3ga"
    };
}

void MediaHandler::setConfiguration(const MediaScanConfig& config)
{
    m_config = config;
    qDebug() << "Media configuration updated:"
             << "Image metadata:" << config.extractImageMetadata
             << "Video thumbnails:" << config.generateVideoThumbnails
             << "Audio fingerprints:" << config.extractAudioFingerprints
             << "Similarity threshold:" << config.similarityThreshold;
}

MediaScanConfig MediaHandler::configuration() const
{
    return m_config;
}

MediaInfo MediaHandler::extractMediaInfo(const QString& filePath)
{
    MediaInfo info;
    
    if (!QFileInfo::exists(filePath)) {
        emit errorOccurred(filePath, "Media file does not exist");
        return info;
    }
    
    if (!isProcessingEnabled(filePath)) {
        qDebug() << "Processing disabled for media type:" << filePath;
        return info;
    }
    
    QFileInfo fileInfo(filePath);
    QString mediaType = getMediaType(filePath);
    
    // Check file size limits
    if ((mediaType == "VIDEO" && fileInfo.size() > m_config.maxVideoSize) ||
        (mediaType == "AUDIO" && fileInfo.size() > m_config.maxAudioSize)) {
        emit errorOccurred(filePath, QString("Media file too large (%1 bytes)")
                          .arg(fileInfo.size()));
        return info;
    }
    
    emit processingStarted(filePath);
    
    try {
        // Basic file information
        info.filePath = filePath;
        info.fileName = fileInfo.fileName();
        info.mediaType = mediaType;
        info.format = getMediaFormat(filePath);
        info.fileSize = fileInfo.size();
        
        emit processingProgress(filePath, 25);
        
        // Process based on media type
        if (mediaType == "IMAGE") {
            info = processImageFile(filePath);
        } else if (mediaType == "VIDEO") {
            info = processVideoFile(filePath);
        } else if (mediaType == "AUDIO") {
            info = processAudioFile(filePath);
        }
        
        m_totalMediaProcessed++;
        
        emit processingProgress(filePath, 100);
        emit processingCompleted(filePath, true);
        
        qDebug() << "Media processed:" << filePath 
                 << "Type:" << info.mediaType
                 << "Format:" << info.format
                 << "Dimensions:" << info.dimensions
                 << "Duration:" << info.duration;
        
    } catch (const std::exception& e) {
        m_totalErrors++;
        emit errorOccurred(filePath, QString("Exception during processing: %1").arg(e.what()));
        emit processingCompleted(filePath, false);
    }
    
    return info;
}

QByteArray MediaHandler::generateVideoThumbnail(const QString& filePath, int timeOffset)
{
    if (!m_config.generateVideoThumbnails) {
        return QByteArray();
    }
    
    // Use ffmpeg to generate thumbnail
    QProcess ffmpegProcess;
    QStringList arguments;
    
    arguments << "-i" << filePath
              << "-ss" << QString::number(timeOffset)
              << "-vframes" << "1"
              << "-s" << QString("%1x%1").arg(m_config.thumbnailSize)
              << "-f" << "image2pipe"
              << "-vcodec" << "png"
              << "-";
    
    ffmpegProcess.start("ffmpeg", arguments);
    if (!ffmpegProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start ffmpeg for thumbnail generation";
        return QByteArray();
    }
    
    if (!ffmpegProcess.waitForFinished(30000)) {
        qWarning() << "Video thumbnail generation timed out";
        ffmpegProcess.kill();
        return QByteArray();
    }
    
    if (ffmpegProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(ffmpegProcess.readAllStandardError());
        qWarning() << "Video thumbnail generation failed:" << error;
        return QByteArray();
    }
    
    m_thumbnailsGenerated++;
    return ffmpegProcess.readAllStandardOutput();
}

QByteArray MediaHandler::extractAudioFingerprint(const QString& filePath)
{
    if (!m_config.extractAudioFingerprints) {
        return QByteArray();
    }
    
    // Basic audio fingerprinting using ffmpeg to extract audio features
    // This is a simplified implementation - real audio fingerprinting would use
    // specialized libraries like Chromaprint/AcoustID
    
    QProcess ffmpegProcess;
    QStringList arguments;
    
    arguments << "-i" << filePath
              << "-t" << "30" // First 30 seconds
              << "-ac" << "1" // Mono
              << "-ar" << "11025" // Low sample rate
              << "-f" << "wav"
              << "-";
    
    ffmpegProcess.start("ffmpeg", arguments);
    if (!ffmpegProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start ffmpeg for audio fingerprinting";
        return QByteArray();
    }
    
    if (!ffmpegProcess.waitForFinished(30000)) {
        qWarning() << "Audio fingerprint extraction timed out";
        ffmpegProcess.kill();
        return QByteArray();
    }
    
    if (ffmpegProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(ffmpegProcess.readAllStandardError());
        qWarning() << "Audio fingerprint extraction failed:" << error;
        return QByteArray();
    }
    
    QByteArray audioData = ffmpegProcess.readAllStandardOutput();
    
    // Simple fingerprint: hash of audio data (not a real audio fingerprint)
    // Real implementation would use spectral analysis
    m_fingerprintsExtracted++;
    return audioData.left(1024); // Take first 1KB as simple fingerprint
}

QVariantMap MediaHandler::extractImageMetadata(const QString& filePath)
{
    QVariantMap metadata;
    
    if (!m_config.extractImageMetadata) {
        return metadata;
    }
    
    // Use exiftool to extract EXIF metadata
    QProcess exifProcess;
    QStringList arguments;
    arguments << "-j" << filePath; // JSON output
    
    exifProcess.start("exiftool", arguments);
    if (!exifProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start exiftool for metadata extraction";
        return metadata;
    }
    
    if (!exifProcess.waitForFinished(10000)) {
        qWarning() << "EXIF metadata extraction timed out";
        exifProcess.kill();
        return metadata;
    }
    
    if (exifProcess.exitCode() != 0) {
        return metadata;
    }
    
    // Parse JSON output (simplified - would need proper JSON parsing)
    QString output = QString::fromUtf8(exifProcess.readAllStandardOutput());
    
    // Extract basic information using regex (simplified approach)
    QRegularExpression widthRegex("\"ImageWidth\"\\s*:\\s*(\\d+)");
    QRegularExpression heightRegex("\"ImageHeight\"\\s*:\\s*(\\d+)");
    QRegularExpression cameraRegex("\"Camera\"\\s*:\\s*\"([^\"]+)\"");
    QRegularExpression dateRegex("\"CreateDate\"\\s*:\\s*\"([^\"]+)\"");
    
    QRegularExpressionMatch widthMatch = widthRegex.match(output);
    QRegularExpressionMatch heightMatch = heightRegex.match(output);
    QRegularExpressionMatch cameraMatch = cameraRegex.match(output);
    QRegularExpressionMatch dateMatch = dateRegex.match(output);
    
    if (widthMatch.hasMatch()) {
        metadata["width"] = widthMatch.captured(1).toInt();
    }
    if (heightMatch.hasMatch()) {
        metadata["height"] = heightMatch.captured(1).toInt();
    }
    if (cameraMatch.hasMatch()) {
        metadata["camera"] = cameraMatch.captured(1);
    }
    if (dateMatch.hasMatch()) {
        metadata["createDate"] = dateMatch.captured(1);
    }
    
    return metadata;
}

double MediaHandler::compareMedia(const MediaInfo& media1, const MediaInfo& media2)
{
    if (media1.mediaType != media2.mediaType) {
        return 0.0; // Different media types
    }
    
    if (media1.mediaType == "IMAGE" && m_config.compareImageSimilarity) {
        return compareImages(media1.filePath, media2.filePath);
    } else if (media1.mediaType == "VIDEO" && m_config.compareVideoThumbnails) {
        return compareThumbnails(media1.thumbnail, media2.thumbnail);
    } else if (media1.mediaType == "AUDIO" && m_config.compareAudioFingerprints) {
        return compareAudioFingerprints(media1.audioFingerprint, media2.audioFingerprint);
    }
    
    // Fallback to basic comparison
    if (media1.fileSize == media2.fileSize && 
        media1.dimensions == media2.dimensions &&
        media1.duration == media2.duration) {
        return 0.9; // Likely similar
    }
    
    return 0.0;
}

double MediaHandler::compareImages(const QString& image1Path, const QString& image2Path)
{
    // This would integrate with the PerceptualHashAlgorithm
    // For now, return a placeholder implementation
    
    QFileInfo info1(image1Path);
    QFileInfo info2(image2Path);
    
    // Simple comparison based on file size and name similarity
    if (info1.size() == info2.size()) {
        return 0.8;
    }
    
    return 0.0;
}

double MediaHandler::compareThumbnails(const QByteArray& thumbnail1, const QByteArray& thumbnail2)
{
    if (thumbnail1.isEmpty() || thumbnail2.isEmpty()) {
        return 0.0;
    }
    
    // Simple byte comparison (not a real image similarity algorithm)
    if (thumbnail1 == thumbnail2) {
        return 1.0;
    }
    
    // Calculate similarity based on byte differences
    int differences = 0;
    int minSize = qMin(thumbnail1.size(), thumbnail2.size());
    
    for (int i = 0; i < minSize; ++i) {
        if (thumbnail1[i] != thumbnail2[i]) {
            differences++;
        }
    }
    
    double similarity = 1.0 - (static_cast<double>(differences) / minSize);
    return qMax(0.0, similarity);
}

double MediaHandler::compareAudioFingerprints(const QByteArray& fingerprint1, const QByteArray& fingerprint2)
{
    if (fingerprint1.isEmpty() || fingerprint2.isEmpty()) {
        return 0.0;
    }
    
    // Simple byte comparison (not a real audio fingerprint comparison)
    if (fingerprint1 == fingerprint2) {
        return 1.0;
    }
    
    // Calculate similarity based on byte differences
    int differences = 0;
    int minSize = qMin(fingerprint1.size(), fingerprint2.size());
    
    for (int i = 0; i < minSize; ++i) {
        if (fingerprint1[i] != fingerprint2[i]) {
            differences++;
        }
    }
    
    double similarity = 1.0 - (static_cast<double>(differences) / minSize);
    return qMax(0.0, similarity);
}

QString MediaHandler::getMediaFormat(const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    return fileInfo.suffix().toUpper();
}

QString MediaHandler::getMediaType(const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    // Image formats
    QStringList imageExtensions = {
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "ico", "svg",
        "raw", "cr2", "nef", "arw", "dng", "orf", "rw2", "pef", "srw"
    };
    
    // Video formats
    QStringList videoExtensions = {
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "3gp", "ogv",
        "mpg", "mpeg", "ts", "mts", "m2ts", "vob", "asf", "rm", "rmvb"
    };
    
    // Audio formats
    QStringList audioExtensions = {
        "mp3", "wav", "flac", "aac", "ogg", "wma", "m4a", "opus", "ape", "ac3",
        "dts", "aiff", "au", "ra", "amr", "3ga"
    };
    
    if (imageExtensions.contains(extension)) {
        return "IMAGE";
    } else if (videoExtensions.contains(extension)) {
        return "VIDEO";
    } else if (audioExtensions.contains(extension)) {
        return "AUDIO";
    }
    
    return "UNKNOWN";
}

bool MediaHandler::isProcessingEnabled(const QString& filePath) const
{
    QString mediaType = getMediaType(filePath);
    
    if (mediaType == "IMAGE") {
        return m_config.extractImageMetadata || m_config.compareImageSimilarity;
    } else if (mediaType == "VIDEO") {
        return m_config.generateVideoThumbnails || m_config.compareVideoThumbnails;
    } else if (mediaType == "AUDIO") {
        return m_config.extractAudioFingerprints || m_config.compareAudioFingerprints;
    }
    
    return false;
}

// Private implementation methods

MediaInfo MediaHandler::processImageFile(const QString& filePath)
{
    MediaInfo info;
    info.filePath = filePath;
    info.fileName = QFileInfo(filePath).fileName();
    info.mediaType = "IMAGE";
    info.format = getMediaFormat(filePath);
    info.fileSize = QFileInfo(filePath).size();
    
    emit processingProgress(filePath, 50);
    
    // Extract image dimensions
    QImageReader reader(filePath);
    if (reader.canRead()) {
        info.dimensions = reader.size();
    }
    
    emit processingProgress(filePath, 75);
    
    // Extract metadata if enabled
    if (m_config.extractImageMetadata) {
        info.metadata = extractImageMetadata(filePath);
        if (info.metadata.contains("width") && info.metadata.contains("height")) {
            info.dimensions = QSize(info.metadata["width"].toInt(), 
                                   info.metadata["height"].toInt());
        }
    }
    
    return info;
}

MediaInfo MediaHandler::processVideoFile(const QString& filePath)
{
    MediaInfo info;
    info.filePath = filePath;
    info.fileName = QFileInfo(filePath).fileName();
    info.mediaType = "VIDEO";
    info.format = getMediaFormat(filePath);
    info.fileSize = QFileInfo(filePath).size();
    
    emit processingProgress(filePath, 50);
    
    // Extract video metadata
    info.metadata = extractVideoMetadata(filePath);
    if (info.metadata.contains("duration")) {
        info.duration = info.metadata["duration"].toInt();
    }
    if (info.metadata.contains("width") && info.metadata.contains("height")) {
        info.dimensions = QSize(info.metadata["width"].toInt(), 
                               info.metadata["height"].toInt());
    }
    if (info.metadata.contains("bitrate")) {
        info.bitrate = info.metadata["bitrate"].toInt();
    }
    
    emit processingProgress(filePath, 75);
    
    // Generate thumbnail if enabled
    if (m_config.generateVideoThumbnails) {
        info.thumbnail = generateVideoThumbnail(filePath);
    }
    
    return info;
}

MediaInfo MediaHandler::processAudioFile(const QString& filePath)
{
    MediaInfo info;
    info.filePath = filePath;
    info.fileName = QFileInfo(filePath).fileName();
    info.mediaType = "AUDIO";
    info.format = getMediaFormat(filePath);
    info.fileSize = QFileInfo(filePath).size();
    
    emit processingProgress(filePath, 50);
    
    // Extract audio metadata
    info.metadata = extractAudioMetadata(filePath);
    if (info.metadata.contains("duration")) {
        info.duration = info.metadata["duration"].toInt();
    }
    if (info.metadata.contains("bitrate")) {
        info.bitrate = info.metadata["bitrate"].toInt();
    }
    
    emit processingProgress(filePath, 75);
    
    // Extract audio fingerprint if enabled
    if (m_config.extractAudioFingerprints) {
        info.audioFingerprint = extractAudioFingerprint(filePath);
    }
    
    return info;
}

QVariantMap MediaHandler::extractVideoMetadata(const QString& filePath)
{
    QVariantMap metadata;
    
    // Use ffprobe to extract video metadata
    QProcess ffprobeProcess;
    QStringList arguments;
    arguments << "-v" << "quiet"
              << "-print_format" << "json"
              << "-show_format"
              << "-show_streams"
              << filePath;
    
    ffprobeProcess.start("ffprobe", arguments);
    if (!ffprobeProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start ffprobe for video metadata";
        return metadata;
    }
    
    if (!ffprobeProcess.waitForFinished(10000)) {
        qWarning() << "Video metadata extraction timed out";
        ffprobeProcess.kill();
        return metadata;
    }
    
    if (ffprobeProcess.exitCode() != 0) {
        return metadata;
    }
    
    // Parse JSON output (simplified - would need proper JSON parsing)
    QString output = QString::fromUtf8(ffprobeProcess.readAllStandardOutput());
    
    // Extract basic information using regex (simplified approach)
    QRegularExpression durationRegex("\"duration\"\\s*:\\s*\"([^\"]+)\"");
    QRegularExpression widthRegex("\"width\"\\s*:\\s*(\\d+)");
    QRegularExpression heightRegex("\"height\"\\s*:\\s*(\\d+)");
    QRegularExpression bitrateRegex("\"bit_rate\"\\s*:\\s*\"(\\d+)\"");
    
    QRegularExpressionMatch durationMatch = durationRegex.match(output);
    QRegularExpressionMatch widthMatch = widthRegex.match(output);
    QRegularExpressionMatch heightMatch = heightRegex.match(output);
    QRegularExpressionMatch bitrateMatch = bitrateRegex.match(output);
    
    if (durationMatch.hasMatch()) {
        metadata["duration"] = static_cast<int>(durationMatch.captured(1).toDouble());
    }
    if (widthMatch.hasMatch()) {
        metadata["width"] = widthMatch.captured(1).toInt();
    }
    if (heightMatch.hasMatch()) {
        metadata["height"] = heightMatch.captured(1).toInt();
    }
    if (bitrateMatch.hasMatch()) {
        metadata["bitrate"] = bitrateMatch.captured(1).toInt();
    }
    
    return metadata;
}

QVariantMap MediaHandler::extractAudioMetadata(const QString& filePath)
{
    QVariantMap metadata;
    
    // Use ffprobe to extract audio metadata (similar to video)
    QProcess ffprobeProcess;
    QStringList arguments;
    arguments << "-v" << "quiet"
              << "-print_format" << "json"
              << "-show_format"
              << "-show_streams"
              << filePath;
    
    ffprobeProcess.start("ffprobe", arguments);
    if (!ffprobeProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start ffprobe for audio metadata";
        return metadata;
    }
    
    if (!ffprobeProcess.waitForFinished(10000)) {
        qWarning() << "Audio metadata extraction timed out";
        ffprobeProcess.kill();
        return metadata;
    }
    
    if (ffprobeProcess.exitCode() != 0) {
        return metadata;
    }
    
    // Parse JSON output (simplified)
    QString output = QString::fromUtf8(ffprobeProcess.readAllStandardOutput());
    
    QRegularExpression durationRegex("\"duration\"\\s*:\\s*\"([^\"]+)\"");
    QRegularExpression bitrateRegex("\"bit_rate\"\\s*:\\s*\"(\\d+)\"");
    
    QRegularExpressionMatch durationMatch = durationRegex.match(output);
    QRegularExpressionMatch bitrateMatch = bitrateRegex.match(output);
    
    if (durationMatch.hasMatch()) {
        metadata["duration"] = static_cast<int>(durationMatch.captured(1).toDouble());
    }
    if (bitrateMatch.hasMatch()) {
        metadata["bitrate"] = bitrateMatch.captured(1).toInt();
    }
    
    return metadata;
}

bool MediaHandler::isImageFile(const QString& filePath) const
{
    return getMediaType(filePath) == "IMAGE";
}

bool MediaHandler::isVideoFile(const QString& filePath) const
{
    return getMediaType(filePath) == "VIDEO";
}

bool MediaHandler::isAudioFile(const QString& filePath) const
{
    return getMediaType(filePath) == "AUDIO";
}

QByteArray MediaHandler::scaleImage(const QByteArray& imageData, const QSize& targetSize)
{
    // Simple image scaling implementation
    Q_UNUSED(imageData)
    Q_UNUSED(targetSize)
    
    // This would require proper image processing
    // For now, return original data
    return imageData;
}