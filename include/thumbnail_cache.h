#ifndef THUMBNAIL_CACHE_H
#define THUMBNAIL_CACHE_H

#include <QtCore/QObject>
#include <QtCore/QCache>
#include <QtCore/QString>
#include <QtCore/QSize>
#include <QtCore/QMutex>
#include <QtCore/QThreadPool>
#include <QtCore/QRunnable>
#include <QtGui/QPixmap>
#include <QtGui/QImage>

/**
 * @brief Thread-safe thumbnail cache with background generation
 * 
 * This class provides in-memory caching of image and video thumbnails
 * with background thread processing using QThreadPool. It supports:
 * - Image thumbnail generation using Qt
 * - Video thumbnail generation (first frame extraction)
 * - Configurable cache size
 * - Thread-safe operations
 */
class ThumbnailCache : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new ThumbnailCache
     * @param parent Parent QObject
     */
    explicit ThumbnailCache(QObject* parent = nullptr);
    
    /**
     * @brief Destructor - waits for all background tasks to complete
     */
    ~ThumbnailCache();

    /**
     * @brief Get a thumbnail for the specified file
     * 
     * If the thumbnail is in cache, returns it immediately.
     * Otherwise, returns a null pixmap and queues generation in background.
     * When ready, thumbnailReady signal will be emitted.
     * 
     * @param filePath Path to the file
     * @param size Desired thumbnail size
     * @return QPixmap Cached thumbnail or null pixmap if not ready
     */
    QPixmap getThumbnail(const QString& filePath, const QSize& size);

    /**
     * @brief Preload thumbnails for multiple files in background
     * 
     * Useful for preloading visible items in a list/tree view.
     * 
     * @param filePaths List of file paths to preload
     * @param size Desired thumbnail size
     */
    void preloadThumbnails(const QStringList& filePaths, const QSize& size);

    /**
     * @brief Clear all cached thumbnails
     */
    void clearCache();

    /**
     * @brief Set maximum number of thumbnails to cache
     * @param maxItems Maximum cache size (default: 100)
     */
    void setCacheSize(int maxItems);

    /**
     * @brief Get current cache size
     * @return int Number of items currently in cache
     */
    int cacheSize() const;

    /**
     * @brief Get maximum cache capacity
     * @return int Maximum number of items that can be cached
     */
    int maxCacheSize() const;

    /**
     * @brief Check if a thumbnail is in cache
     * @param filePath File path
     * @param size Thumbnail size
     * @return bool True if thumbnail is cached
     */
    bool isCached(const QString& filePath, const QSize& size) const;

signals:
    /**
     * @brief Emitted when a thumbnail has been generated
     * @param filePath Path to the file
     * @param thumbnail Generated thumbnail
     */
    void thumbnailReady(const QString& filePath, const QPixmap& thumbnail);

    /**
     * @brief Emitted when thumbnail generation fails
     * @param filePath Path to the file
     * @param error Error message
     */
    void thumbnailFailed(const QString& filePath, const QString& error);

private:
    /**
     * @brief Generate cache key from file path and size
     * @param filePath File path
     * @param size Thumbnail size
     * @return QString Cache key
     */
    QString getCacheKey(const QString& filePath, const QSize& size) const;

    /**
     * @brief Generate thumbnail for an image file
     * @param filePath Path to image file
     * @param size Desired thumbnail size
     * @return QPixmap Generated thumbnail or null on failure
     */
    QPixmap generateImageThumbnail(const QString& filePath, const QSize& size);

    /**
     * @brief Generate thumbnail for a video file (first frame)
     * @param filePath Path to video file
     * @param size Desired thumbnail size
     * @return QPixmap Generated thumbnail or null on failure
     */
    QPixmap generateVideoThumbnail(const QString& filePath, const QSize& size);

    /**
     * @brief Check if file is an image
     * @param filePath File path
     * @return bool True if file is an image
     */
    bool isImageFile(const QString& filePath) const;

    /**
     * @brief Check if file is a video
     * @param filePath File path
     * @return bool True if file is a video
     */
    bool isVideoFile(const QString& filePath) const;

    /**
     * @brief Get default icon for unsupported file types
     * @param filePath File path
     * @param size Icon size
     * @return QPixmap Default icon
     */
    QPixmap getDefaultIcon(const QString& filePath, const QSize& size);

    // Thread-safe cache
    QCache<QString, QPixmap>* m_cache;
    mutable QMutex m_cacheMutex;

    // Thread pool for background generation
    QThreadPool* m_threadPool;

    // Default settings
    static const int DEFAULT_CACHE_SIZE = 100;
    static const int DEFAULT_THUMBNAIL_SIZE = 64;

    friend class ThumbnailGenerationTask;
};

/**
 * @brief Background task for thumbnail generation
 * 
 * This runnable task generates thumbnails in a background thread
 * and notifies the cache when complete.
 */
class ThumbnailGenerationTask : public QObject, public QRunnable
{
    Q_OBJECT

public:
    /**
     * @brief Construct a thumbnail generation task
     * @param cache Pointer to the thumbnail cache
     * @param filePath Path to the file
     * @param size Desired thumbnail size
     */
    ThumbnailGenerationTask(ThumbnailCache* cache, 
                           const QString& filePath, 
                           const QSize& size);

    /**
     * @brief Run the thumbnail generation task
     */
    void run() override;

signals:
    /**
     * @brief Emitted when thumbnail generation is complete
     * @param filePath File path
     * @param thumbnail Generated thumbnail
     */
    void thumbnailGenerated(const QString& filePath, const QPixmap& thumbnail);

    /**
     * @brief Emitted when thumbnail generation fails
     * @param filePath File path
     * @param error Error message
     */
    void generationFailed(const QString& filePath, const QString& error);

private:
    ThumbnailCache* m_cache;
    QString m_filePath;
    QSize m_size;
};

#endif // THUMBNAIL_CACHE_H
