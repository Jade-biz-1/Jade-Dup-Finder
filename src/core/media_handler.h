#ifndef MEDIA_HANDLER_H
#define MEDIA_HANDLER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QVariantMap>
#include <QSize>

/**
 * @brief Media file information for enhanced duplicate detection
 */
struct MediaInfo {
    QString filePath;           // Path to the media file
    QString fileName;           // Just the filename
    QString mediaType;          // Type of media (IMAGE, VIDEO, AUDIO)
    QString format;             // File format (JPG, MP4, MP3, etc.)
    QSize dimensions;           // Image/video dimensions
    int duration;               // Duration in seconds (for video/audio)
    int bitrate;                // Bitrate (for video/audio)
    QString codec;              // Codec information
    QByteArray thumbnail;       // Thumbnail data (for videos)
    QByteArray audioFingerprint; // Audio fingerprint (for audio files)
    QVariantMap metadata;       // EXIF and other metadata
    qint64 fileSize;           // File size in bytes
    
    MediaInfo() : duration(0), bitrate(0), fileSize(0) {}
};

/**
 * @brief Media scanning configuration
 */
struct MediaScanConfig {
    bool extractImageMetadata = true;       // Extract EXIF data from images
    bool generateVideoThumbnails = true;    // Generate thumbnails for videos
    bool extractAudioFingerprints = false; // Extract audio fingerprints (CPU intensive)
    bool compareImageSimilarity = true;     // Use perceptual hashing for images
    bool compareVideoThumbnails = true;     // Compare video thumbnails
    bool compareAudioFingerprints = false;  // Compare audio fingerprints
    int thumbnailSize = 128;                // Thumbnail size (pixels)
    int maxVideoSize = 500 * 1024 * 1024;  // Max video size to process (500MB)
    int maxAudioSize = 100 * 1024 * 1024;  // Max audio size to process (100MB)
    bool normalizeOrientation = true;       // Normalize image orientation using EXIF
    double similarityThreshold = 0.85;     // Similarity threshold for media comparison
};

/**
 * @brief Handler for media file processing and comparison
 * 
 * This class provides functionality to extract metadata, generate thumbnails,
 * and perform similarity comparison for images, videos, and audio files.
 */
class MediaHandler : public QObject
{
    Q_OBJECT

public:
    explicit MediaHandler(QObject* parent = nullptr);
    ~MediaHandler();

    /**
     * @brief Check if a file is a supported media format
     * @param filePath Path to the file to check
     * @return True if the file is a supported media format
     */
    static bool isMediaFile(const QString& filePath);
    
    /**
     * @brief Get list of supported media extensions
     * @return List of supported extensions (e.g., "jpg", "mp4", "mp3")
     */
    static QStringList supportedExtensions();
    
    /**
     * @brief Set media scanning configuration
     * @param config Configuration settings for media scanning
     */
    void setConfiguration(const MediaScanConfig& config);
    
    /**
     * @brief Get current media scanning configuration
     * @return Current configuration settings
     */
    MediaScanConfig configuration() const;
    
    /**
     * @brief Extract media information and metadata
     * @param filePath Path to the media file
     * @return Media information with extracted metadata
     */
    MediaInfo extractMediaInfo(const QString& filePath);
    
    /**
     * @brief Generate thumbnail for video file
     * @param filePath Path to the video file
     * @param timeOffset Time offset in seconds for thumbnail
     * @return Thumbnail data as byte array
     */
    QByteArray generateVideoThumbnail(const QString& filePath, int timeOffset = 10);
    
    /**
     * @brief Extract audio fingerprint
     * @param filePath Path to the audio file
     * @return Audio fingerprint as byte array
     */
    QByteArray extractAudioFingerprint(const QString& filePath);
    
    /**
     * @brief Extract EXIF metadata from image
     * @param filePath Path to the image file
     * @return EXIF metadata as variant map
     */
    QVariantMap extractImageMetadata(const QString& filePath);
    
    /**
     * @brief Compare two media files for similarity
     * @param media1 First media info
     * @param media2 Second media info
     * @return Similarity score (0.0-1.0, 1.0 = identical)
     */
    double compareMedia(const MediaInfo& media1, const MediaInfo& media2);
    
    /**
     * @brief Compare two images using perceptual hashing
     * @param image1Path Path to first image
     * @param image2Path Path to second image
     * @return Similarity score (0.0-1.0)
     */
    double compareImages(const QString& image1Path, const QString& image2Path);
    
    /**
     * @brief Compare two video thumbnails
     * @param thumbnail1 First thumbnail data
     * @param thumbnail2 Second thumbnail data
     * @return Similarity score (0.0-1.0)
     */
    double compareThumbnails(const QByteArray& thumbnail1, const QByteArray& thumbnail2);
    
    /**
     * @brief Compare two audio fingerprints
     * @param fingerprint1 First audio fingerprint
     * @param fingerprint2 Second audio fingerprint
     * @return Similarity score (0.0-1.0)
     */
    double compareAudioFingerprints(const QByteArray& fingerprint1, const QByteArray& fingerprint2);
    
    /**
     * @brief Get media format type
     * @param filePath Path to media file
     * @return Media format name (e.g., "JPG", "MP4", "MP3")
     */
    static QString getMediaFormat(const QString& filePath);
    
    /**
     * @brief Get media type category
     * @param filePath Path to media file
     * @return Media type (IMAGE, VIDEO, AUDIO)
     */
    static QString getMediaType(const QString& filePath);
    
    /**
     * @brief Check if media processing is enabled for given file type
     * @param filePath Path to media file
     * @return True if processing is enabled for this media type
     */
    bool isProcessingEnabled(const QString& filePath) const;

signals:
    /**
     * @brief Emitted when media processing starts
     * @param filePath Path to media file being processed
     */
    void processingStarted(const QString& filePath);
    
    /**
     * @brief Emitted when processing progress updates
     * @param filePath Path to media file being processed
     * @param percentage Progress percentage (0-100)
     */
    void processingProgress(const QString& filePath, int percentage);
    
    /**
     * @brief Emitted when media processing completes
     * @param filePath Path to media file that was processed
     * @param success True if processing was successful
     */
    void processingCompleted(const QString& filePath, bool success);
    
    /**
     * @brief Emitted when an error occurs during media operations
     * @param filePath Path to media file that caused error
     * @param error Error message
     */
    void errorOccurred(const QString& filePath, const QString& error);

private:
    // Format-specific processing methods
    MediaInfo processImageFile(const QString& filePath);
    MediaInfo processVideoFile(const QString& filePath);
    MediaInfo processAudioFile(const QString& filePath);
    
    // Metadata extraction methods
    QVariantMap extractVideoMetadata(const QString& filePath);
    QVariantMap extractAudioMetadata(const QString& filePath);
    
    // Utility methods
    bool isImageFile(const QString& filePath) const;
    bool isVideoFile(const QString& filePath) const;
    bool isAudioFile(const QString& filePath) const;
    QByteArray scaleImage(const QByteArray& imageData, const QSize& targetSize);
    
    // Member variables
    MediaScanConfig m_config;
    
    // Statistics
    int m_totalMediaProcessed;
    int m_thumbnailsGenerated;
    int m_fingerprintsExtracted;
    int m_totalErrors;
};

#endif // MEDIA_HANDLER_H