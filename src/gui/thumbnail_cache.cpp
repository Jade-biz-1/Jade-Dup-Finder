#include "thumbnail_cache.h"
#include "logger.h"
#include <QtCore/QFileInfo>
#include <QtCore/QMutexLocker>

#include <QtGui/QImageReader>
#include <QtGui/QPainter>
#include <QtWidgets/QApplication>
#include <QtWidgets/QStyle>

// ThumbnailCache Implementation

ThumbnailCache::ThumbnailCache(QObject* parent)
    : QObject(parent)
    , m_cache(new QCache<QString, QPixmap>(DEFAULT_CACHE_SIZE))
    , m_threadPool(new QThreadPool(this))
{
    // Configure thread pool
    // Use half of available threads for thumbnail generation to avoid blocking
    int maxThreads = QThread::idealThreadCount();
    if (maxThreads > 2) {
        m_threadPool->setMaxThreadCount(maxThreads / 2);
    } else {
        m_threadPool->setMaxThreadCount(1);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("ThumbnailCache initialized with max threads: %1").arg(m_threadPool->maxThreadCount()));
}

ThumbnailCache::~ThumbnailCache()
{
    // Wait for all background tasks to complete
    m_threadPool->waitForDone();
    
    // Clean up cache
    QMutexLocker locker(&m_cacheMutex);
    m_cache->clear();
    delete m_cache;
}

QString ThumbnailCache::getCacheKey(const QString& filePath, const QSize& size) const
{
    return QString("%1_%2x%3").arg(filePath).arg(size.width()).arg(size.height());
}

QPixmap ThumbnailCache::getThumbnail(const QString& filePath, const QSize& size)
{
    QString key = getCacheKey(filePath, size);
    
    // Check cache first
    {
        QMutexLocker locker(&m_cacheMutex);
        QPixmap* cached = m_cache->object(key);
        if (cached) {
            return *cached;
        }
    }
    
    // Not in cache - queue background generation
    ThumbnailGenerationTask* task = new ThumbnailGenerationTask(this, filePath, size);
    
    // Connect signals
    connect(task, &ThumbnailGenerationTask::thumbnailGenerated,
            this, [this, key](const QString& path, const QPixmap& thumbnail) {
                // Store in cache
                {
                    QMutexLocker locker(&m_cacheMutex);
                    m_cache->insert(key, new QPixmap(thumbnail));
                }
                // Emit signal
                emit thumbnailReady(path, thumbnail);
            });
    
    connect(task, &ThumbnailGenerationTask::generationFailed,
            this, &ThumbnailCache::thumbnailFailed);
    
    // Queue the task
    m_threadPool->start(task);
    
    // Return null pixmap - caller will receive thumbnailReady signal when done
    return QPixmap();
}

void ThumbnailCache::preloadThumbnails(const QStringList& filePaths, const QSize& size)
{
    for (const QString& filePath : filePaths) {
        QString key = getCacheKey(filePath, size);
        
        // Skip if already cached
        {
            QMutexLocker locker(&m_cacheMutex);
            if (m_cache->contains(key)) {
                continue;
            }
        }
        
        // Queue generation
        ThumbnailGenerationTask* task = new ThumbnailGenerationTask(this, filePath, size);
        
        connect(task, &ThumbnailGenerationTask::thumbnailGenerated,
                this, [this, key](const QString& path, const QPixmap& thumbnail) {
                    QMutexLocker locker(&m_cacheMutex);
                    m_cache->insert(key, new QPixmap(thumbnail));
                    emit thumbnailReady(path, thumbnail);
                });
        
        connect(task, &ThumbnailGenerationTask::generationFailed,
                this, &ThumbnailCache::thumbnailFailed);
        
        m_threadPool->start(task);
    }
}

void ThumbnailCache::clearCache()
{
    QMutexLocker locker(&m_cacheMutex);
    m_cache->clear();
    LOG_DEBUG(LogCategories::UI, "ThumbnailCache cleared");
}

void ThumbnailCache::setCacheSize(int maxItems)
{
    QMutexLocker locker(&m_cacheMutex);
    m_cache->setMaxCost(maxItems);
    LOG_DEBUG(LogCategories::UI, QString("ThumbnailCache max size set to: %1").arg(maxItems));
}

int ThumbnailCache::cacheSize() const
{
    QMutexLocker locker(&m_cacheMutex);
    return static_cast<int>(m_cache->count());
}

int ThumbnailCache::maxCacheSize() const
{
    QMutexLocker locker(&m_cacheMutex);
    return static_cast<int>(m_cache->maxCost());
}

bool ThumbnailCache::isCached(const QString& filePath, const QSize& size) const
{
    QString key = getCacheKey(filePath, size);
    QMutexLocker locker(&m_cacheMutex);
    return m_cache->contains(key);
}

bool ThumbnailCache::isImageFile(const QString& filePath) const
{
    static const QStringList imageExtensions = {
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif",
        "webp", "svg", "ico", "ppm", "pgm", "pbm", "xpm"
    };
    
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    return imageExtensions.contains(extension);
}

bool ThumbnailCache::isVideoFile(const QString& filePath) const
{
    static const QStringList videoExtensions = {
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm",
        "m4v", "mpg", "mpeg", "3gp", "ogv"
    };
    
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    return videoExtensions.contains(extension);
}

QPixmap ThumbnailCache::generateImageThumbnail(const QString& filePath, const QSize& size)
{
    QImageReader reader(filePath);
    
    if (!reader.canRead()) {
        LOG_WARNING(LogCategories::PREVIEW, QString("Cannot read image: %1").arg(filePath));
        return QPixmap();
    }
    
    // Get original size
    QSize originalSize = reader.size();
    if (!originalSize.isValid()) {
        LOG_WARNING(LogCategories::PREVIEW, QString("Invalid image size: %1").arg(filePath));
        return QPixmap();
    }
    
    // Calculate scaled size maintaining aspect ratio
    QSize scaledSize = originalSize.scaled(size, Qt::KeepAspectRatio);
    reader.setScaledSize(scaledSize);
    
    // Read and convert to pixmap
    QImage image = reader.read();
    if (image.isNull()) {
        LOG_WARNING(LogCategories::PREVIEW, QString("Failed to read image: %1 - %2").arg(filePath).arg(reader.errorString()));
        return QPixmap();
    }
    
    return QPixmap::fromImage(image);
}

QPixmap ThumbnailCache::generateVideoThumbnail(const QString& filePath, const QSize& size)
{
    // For now, return a default video icon
    // Full video thumbnail extraction would require FFmpeg or similar
    // This is a placeholder implementation
    
    Q_UNUSED(filePath);
    
    // Create a simple video icon placeholder
    QPixmap pixmap(size);
    pixmap.fill(Qt::lightGray);
    
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw a play button icon
    QColor iconColor(100, 100, 100);
    painter.setBrush(iconColor);
    painter.setPen(Qt::NoPen);
    
    // Draw triangle (play button)
    int centerX = size.width() / 2;
    int centerY = size.height() / 2;
    int triangleSize = qMin(size.width(), size.height()) / 3;
    
    QPolygon triangle;
    triangle << QPoint(centerX - triangleSize/2, centerY - triangleSize/2)
             << QPoint(centerX - triangleSize/2, centerY + triangleSize/2)
             << QPoint(centerX + triangleSize/2, centerY);
    
    painter.drawPolygon(triangle);
    
    return pixmap;
}

QPixmap ThumbnailCache::getDefaultIcon(const QString& filePath, const QSize& size)
{
    Q_UNUSED(filePath);
    
    // Get file icon from system
    QPixmap pixmap(size);
    pixmap.fill(Qt::transparent);
    
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw a simple document icon
    QColor iconColor(150, 150, 150);
    painter.setBrush(Qt::white);
    painter.setPen(QPen(iconColor, 2));
    
    int margin = size.width() / 8;
    QRect rect(margin, margin, size.width() - 2*margin, size.height() - 2*margin);
    painter.drawRect(rect);
    
    // Draw some lines to represent text
    painter.setPen(QPen(iconColor, 1));
    int lineY = rect.top() + rect.height() / 4;
    int lineSpacing = rect.height() / 6;
    for (int i = 0; i < 3; ++i) {
        painter.drawLine(rect.left() + margin, lineY, 
                        rect.right() - margin, lineY);
        lineY += lineSpacing;
    }
    
    return pixmap;
}

// ThumbnailGenerationTask Implementation

ThumbnailGenerationTask::ThumbnailGenerationTask(ThumbnailCache* cache,
                                                 const QString& filePath,
                                                 const QSize& size)
    : m_cache(cache)
    , m_filePath(filePath)
    , m_size(size)
{
    setAutoDelete(true);  // Task will be deleted after run() completes
}

void ThumbnailGenerationTask::run()
{
    // Check if file exists
    QFileInfo fileInfo(m_filePath);
    if (!fileInfo.exists() || !fileInfo.isFile()) {
        emit generationFailed(m_filePath, "File does not exist or is not a file");
        return;
    }
    
    QPixmap thumbnail;
    
    // Generate thumbnail based on file type
    if (m_cache->isImageFile(m_filePath)) {
        thumbnail = m_cache->generateImageThumbnail(m_filePath, m_size);
    } else if (m_cache->isVideoFile(m_filePath)) {
        thumbnail = m_cache->generateVideoThumbnail(m_filePath, m_size);
    } else {
        thumbnail = m_cache->getDefaultIcon(m_filePath, m_size);
    }
    
    // Check if generation was successful
    if (thumbnail.isNull()) {
        // Try to get default icon as fallback
        thumbnail = m_cache->getDefaultIcon(m_filePath, m_size);
        
        if (thumbnail.isNull()) {
            emit generationFailed(m_filePath, "Failed to generate thumbnail");
            return;
        }
    }
    
    // Emit success signal
    emit thumbnailGenerated(m_filePath, thumbnail);
}
