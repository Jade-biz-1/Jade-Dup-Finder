# ThumbnailCache Usage Guide

## Overview

The `ThumbnailCache` class provides thread-safe, in-memory caching of image and video thumbnails with background generation using Qt's thread pool. It's designed for use in GUI applications that need to display thumbnails efficiently.

## Features

- **In-memory caching**: Fast access to previously generated thumbnails
- **Background processing**: Thumbnail generation happens in background threads
- **Thread-safe**: Safe to use from multiple threads
- **Configurable cache size**: Control memory usage
- **Multiple file types**: Supports images, videos, and provides default icons for other files
- **Signal-based notifications**: Asynchronous notification when thumbnails are ready

## Basic Usage

### 1. Create a ThumbnailCache Instance

```cpp
#include "thumbnail_cache.h"

// Create cache (typically as a member variable)
ThumbnailCache* cache = new ThumbnailCache(this);
```

### 2. Request a Thumbnail

```cpp
QString filePath = "/path/to/image.jpg";
QSize thumbnailSize(64, 64);

// Request thumbnail (returns immediately)
QPixmap thumbnail = cache->getThumbnail(filePath, thumbnailSize);

if (thumbnail.isNull()) {
    // Thumbnail not in cache - will be generated in background
    // Use a placeholder icon for now
} else {
    // Thumbnail was in cache - use it immediately
    label->setPixmap(thumbnail);
}
```

### 3. Handle Thumbnail Ready Signal

```cpp
// Connect to thumbnailReady signal
connect(cache, &ThumbnailCache::thumbnailReady,
        this, [this](const QString& filePath, const QPixmap& thumbnail) {
    // Update UI with the generated thumbnail
    updateThumbnailDisplay(filePath, thumbnail);
});
```

## Advanced Usage

### Preloading Thumbnails

Preload thumbnails for visible items in a list or tree view:

```cpp
QStringList visibleFiles = getVisibleFiles();
cache->preloadThumbnails(visibleFiles, QSize(64, 64));
```

### Configuring Cache Size

```cpp
// Set maximum number of cached thumbnails
cache->setCacheSize(200);  // Default is 100

// Check current cache usage
int currentSize = cache->cacheSize();
int maxSize = cache->maxCacheSize();
```

### Checking Cache Status

```cpp
QString filePath = "/path/to/image.jpg";
QSize size(64, 64);

if (cache->isCached(filePath, size)) {
    // Thumbnail is in cache
    QPixmap thumbnail = cache->getThumbnail(filePath, size);
}
```

### Clearing Cache

```cpp
// Clear all cached thumbnails (e.g., when memory is low)
cache->clearCache();
```

### Error Handling

```cpp
// Connect to thumbnailFailed signal
connect(cache, &ThumbnailCache::thumbnailFailed,
        this, [](const QString& filePath, const QString& error) {
    qWarning() << "Failed to generate thumbnail for" << filePath << ":" << error;
    // Use a default error icon
});
```

## Integration Example: QTreeWidget with Thumbnails

```cpp
class FileListWidget : public QWidget
{
    Q_OBJECT

public:
    FileListWidget(QWidget* parent = nullptr)
        : QWidget(parent)
        , m_cache(new ThumbnailCache(this))
    {
        m_tree = new QTreeWidget(this);
        
        // Connect thumbnail ready signal
        connect(m_cache, &ThumbnailCache::thumbnailReady,
                this, &FileListWidget::onThumbnailReady);
        
        // Setup layout
        QVBoxLayout* layout = new QVBoxLayout(this);
        layout->addWidget(m_tree);
    }
    
    void addFile(const QString& filePath)
    {
        QTreeWidgetItem* item = new QTreeWidgetItem(m_tree);
        item->setText(0, QFileInfo(filePath).fileName());
        item->setData(0, Qt::UserRole, filePath);
        
        // Request thumbnail
        QPixmap thumbnail = m_cache->getThumbnail(filePath, QSize(48, 48));
        if (!thumbnail.isNull()) {
            item->setIcon(0, QIcon(thumbnail));
        } else {
            // Use placeholder icon
            item->setIcon(0, style()->standardIcon(QStyle::SP_FileIcon));
        }
    }
    
private slots:
    void onThumbnailReady(const QString& filePath, const QPixmap& thumbnail)
    {
        // Find item and update icon
        for (int i = 0; i < m_tree->topLevelItemCount(); ++i) {
            QTreeWidgetItem* item = m_tree->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toString() == filePath) {
                item->setIcon(0, QIcon(thumbnail));
                break;
            }
        }
    }
    
private:
    QTreeWidget* m_tree;
    ThumbnailCache* m_cache;
};
```

## Supported File Types

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)
- SVG (.svg)
- ICO (.ico)
- And other formats supported by Qt

### Videos
- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- And others

**Note**: Video thumbnails currently show a placeholder play button icon. Full video frame extraction would require FFmpeg integration.

### Other Files
Files that are not images or videos will receive a default document icon.

## Performance Considerations

### Thread Pool Configuration

The cache automatically configures the thread pool to use half of available CPU cores to avoid blocking the main application:

```cpp
// Automatic configuration in constructor
int maxThreads = QThread::idealThreadCount();
if (maxThreads > 2) {
    m_threadPool->setMaxThreadCount(maxThreads / 2);
}
```

### Memory Usage

Each cached thumbnail consumes memory. Monitor cache size and adjust as needed:

```cpp
// For applications with many files
cache->setCacheSize(500);

// For memory-constrained environments
cache->setCacheSize(50);
```

### Preloading Strategy

Preload only visible items to optimize performance:

```cpp
void onScrolled()
{
    QStringList visibleFiles = getVisibleFiles();
    m_cache->preloadThumbnails(visibleFiles, QSize(64, 64));
}
```

## Best Practices

1. **Create one cache instance per application**: Share the cache across widgets to maximize cache hits

2. **Use consistent thumbnail sizes**: Different sizes create separate cache entries

3. **Preload visible items**: Improve perceived performance by preloading thumbnails for visible items

4. **Handle null pixmaps**: Always check if `getThumbnail()` returns null and use placeholders

5. **Connect signals early**: Connect to `thumbnailReady` before requesting thumbnails

6. **Clear cache when appropriate**: Clear the cache when switching directories or when memory is low

7. **Monitor cache size**: Adjust cache size based on your application's needs

## Thread Safety

The `ThumbnailCache` is fully thread-safe:

- All public methods can be called from any thread
- Internal cache access is protected by mutexes
- Signals are emitted in a thread-safe manner

## Example: Lazy Loading in a List View

```cpp
void MyListView::scrollContentsBy(int dx, int dy)
{
    QListView::scrollContentsBy(dx, dy);
    
    // Get visible items
    QModelIndexList visible = visibleIndexes();
    QStringList filePaths;
    
    for (const QModelIndex& index : visible) {
        QString path = index.data(FilePathRole).toString();
        filePaths << path;
    }
    
    // Preload thumbnails for visible items
    m_cache->preloadThumbnails(filePaths, QSize(64, 64));
}
```

## Troubleshooting

### Thumbnails not appearing

1. Check that you've connected to the `thumbnailReady` signal
2. Verify the file path is correct and the file exists
3. Check the `thumbnailFailed` signal for error messages

### High memory usage

1. Reduce cache size: `cache->setCacheSize(50)`
2. Clear cache periodically: `cache->clearCache()`
3. Use smaller thumbnail sizes

### Slow thumbnail generation

1. Reduce the number of concurrent requests
2. Use smaller thumbnail sizes
3. Preload thumbnails in batches

## See Also

- Qt Documentation: [QPixmap](https://doc.qt.io/qt-6/qpixmap.html)
- Qt Documentation: [QThreadPool](https://doc.qt.io/qt-6/qthreadpool.html)
- Qt Documentation: [QCache](https://doc.qt.io/qt-6/qcache.html)
