# Thumbnail Delegate Usage Guide

## Overview

The `ThumbnailDelegate` class provides thumbnail display functionality for the results tree in the CloneClean application. It integrates with `ThumbnailCache` to efficiently load and display image thumbnails alongside file information.

## Features

- **Automatic Thumbnail Loading**: Thumbnails are loaded asynchronously in the background
- **Lazy Loading**: Only visible thumbnails are loaded, improving performance
- **Configurable Size**: Thumbnail size can be adjusted (default: 48px)
- **Enable/Disable**: Thumbnails can be toggled on/off
- **Placeholder Display**: Shows a placeholder while thumbnails are loading
- **Smart Detection**: Only displays thumbnails for file items, not group headers

## Architecture

```
┌─────────────────┐
│ ResultsWindow   │
│                 │
│  ┌───────────┐  │
│  │ QTreeWidget│  │
│  │           │  │
│  │ Column 0  │◄─┼─── ThumbnailDelegate
│  └───────────┘  │
└────────┬────────┘
         │
         │ uses
         ▼
┌─────────────────┐
│ ThumbnailCache  │
│                 │
│ - In-memory     │
│ - Background    │
│ - Thread-safe   │
└─────────────────┘
```

## Usage

### Basic Setup

```cpp
// In ResultsWindow constructor
m_thumbnailCache = new ThumbnailCache(this);
m_thumbnailDelegate = new ThumbnailDelegate(m_thumbnailCache, this);

// Set delegate for first column
m_resultsTree->setItemDelegateForColumn(0, m_thumbnailDelegate);

// Connect thumbnail ready signal
connect(m_thumbnailCache, &ThumbnailCache::thumbnailReady,
        this, [this](const QString& filePath, const QPixmap& thumbnail) {
            Q_UNUSED(filePath);
            Q_UNUSED(thumbnail);
            // Force repaint to show new thumbnail
            m_resultsTree->viewport()->update();
        });
```

### Storing File Paths

For thumbnails to work, file items must store their file path in `Qt::UserRole`:

```cpp
void ResultsWindow::updateFileItem(QTreeWidgetItem* fileItem, const DuplicateFile& file)
{
    fileItem->setText(0, file.fileName);
    fileItem->setText(1, formatFileSize(file.fileSize));
    fileItem->setText(2, file.lastModified.toString("yyyy-MM-dd hh:mm"));
    fileItem->setText(3, file.directory);
    
    // IMPORTANT: Store file path for thumbnail delegate
    fileItem->setData(0, Qt::UserRole, QVariant::fromValue(file.filePath));
}
```

### Enabling/Disabling Thumbnails

```cpp
// Enable thumbnails
void ResultsWindow::enableThumbnails(bool enable)
{
    if (m_thumbnailDelegate) {
        m_thumbnailDelegate->setThumbnailsEnabled(enable);
        m_resultsTree->viewport()->update();
        
        if (enable) {
            preloadVisibleThumbnails();
        }
    }
}
```

### Configuring Thumbnail Size

```cpp
// Set thumbnail size (valid range: 1-256 pixels)
void ResultsWindow::setThumbnailSize(int size)
{
    if (m_thumbnailDelegate) {
        m_thumbnailDelegate->setThumbnailSize(size);
        m_resultsTree->viewport()->update();
        
        // Reload thumbnails with new size
        if (m_thumbnailDelegate->thumbnailsEnabled()) {
            m_thumbnailCache->clearCache();
            preloadVisibleThumbnails();
        }
    }
}
```

### Preloading Visible Thumbnails

```cpp
void ResultsWindow::preloadVisibleThumbnails()
{
    if (!m_thumbnailDelegate || !m_thumbnailDelegate->thumbnailsEnabled()) {
        return;
    }
    
    QStringList visibleFilePaths;
    
    // Iterate through visible items
    for (int i = 0; i < m_resultsTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem* groupItem = m_resultsTree->topLevelItem(i);
        
        if (groupItem->isExpanded()) {
            for (int j = 0; j < groupItem->childCount(); ++j) {
                QTreeWidgetItem* fileItem = groupItem->child(j);
                
                if (!m_resultsTree->visualItemRect(fileItem).isEmpty()) {
                    QVariant pathData = fileItem->data(0, Qt::UserRole);
                    if (pathData.isValid()) {
                        visibleFilePaths.append(pathData.toString());
                    }
                }
            }
        }
    }
    
    // Preload thumbnails
    if (!visibleFilePaths.isEmpty()) {
        QSize thumbSize(m_thumbnailDelegate->thumbnailSize(), 
                       m_thumbnailDelegate->thumbnailSize());
        m_thumbnailCache->preloadThumbnails(visibleFilePaths, thumbSize);
    }
}
```

### Lazy Loading on Scroll

```cpp
// In setupConnections()
connect(m_resultsTree->verticalScrollBar(), &QScrollBar::valueChanged,
        this, &ResultsWindow::preloadVisibleThumbnails);
```

## Configuration Options

### Thumbnail Size

- **Default**: 48 pixels
- **Valid Range**: 1-256 pixels
- **Recommended**: 32, 48, 64, 96, 128 pixels

### Cache Size

The thumbnail cache size can be configured:

```cpp
m_thumbnailCache->setCacheSize(200); // Store up to 200 thumbnails
```

## Performance Considerations

### Memory Usage

- Each thumbnail consumes approximately: `width × height × 4 bytes`
- Default (48×48): ~9 KB per thumbnail
- Cache of 100 thumbnails: ~900 KB

### CPU Usage

- Thumbnail generation uses background threads
- Thread pool size: `idealThreadCount() / 2`
- Prevents UI blocking during generation

### Optimization Tips

1. **Preload Strategically**: Only preload visible items
2. **Adjust Cache Size**: Balance memory vs. reload frequency
3. **Disable When Not Needed**: Turn off thumbnails for large result sets
4. **Use Appropriate Size**: Smaller thumbnails = faster generation

## Supported File Types

### Images (with thumbnails)
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)
- SVG (.svg)
- ICO (.ico)

### Videos (placeholder icon)
- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)

### Other Files (default icon)
- All other file types show a generic document icon

## Troubleshooting

### Thumbnails Not Showing

1. **Check file path storage**:
   ```cpp
   QVariant pathData = fileItem->data(0, Qt::UserRole);
   qDebug() << "File path:" << pathData.toString();
   ```

2. **Verify thumbnails are enabled**:
   ```cpp
   qDebug() << "Thumbnails enabled:" << m_thumbnailDelegate->thumbnailsEnabled();
   ```

3. **Check file exists**:
   ```cpp
   QFileInfo fileInfo(filePath);
   qDebug() << "File exists:" << fileInfo.exists();
   ```

### Performance Issues

1. **Reduce cache size**: Lower memory usage
2. **Increase thumbnail size threshold**: Only show for larger files
3. **Disable for large result sets**: > 1000 files

### Visual Issues

1. **Thumbnails too small**: Increase thumbnail size
2. **Thumbnails too large**: Decrease thumbnail size
3. **Alignment issues**: Check `THUMBNAIL_MARGIN` and `TEXT_MARGIN` constants

## API Reference

### ThumbnailDelegate

```cpp
class ThumbnailDelegate : public QStyledItemDelegate
{
public:
    explicit ThumbnailDelegate(ThumbnailCache* cache, QObject* parent = nullptr);
    
    void setThumbnailSize(int size);
    int thumbnailSize() const;
    
    void setThumbnailsEnabled(bool enabled);
    bool thumbnailsEnabled() const;
    
    // QStyledItemDelegate overrides
    void paint(QPainter* painter, 
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override;
    
    QSize sizeHint(const QStyleOptionViewItem& option,
                   const QModelIndex& index) const override;
};
```

### Constants

```cpp
static const int DEFAULT_THUMBNAIL_SIZE = 48;
static const int THUMBNAIL_MARGIN = 4;
static const int TEXT_MARGIN = 8;
```

## Examples

### Example 1: Basic Integration

```cpp
// Create cache and delegate
m_thumbnailCache = new ThumbnailCache(this);
m_thumbnailDelegate = new ThumbnailDelegate(m_thumbnailCache, this);
m_resultsTree->setItemDelegateForColumn(0, m_thumbnailDelegate);

// Enable thumbnails
m_thumbnailDelegate->setThumbnailsEnabled(true);
m_thumbnailDelegate->setThumbnailSize(64);
```

### Example 2: User Preference

```cpp
// Load from settings
QSettings settings;
bool thumbnailsEnabled = settings.value("thumbnails/enabled", true).toBool();
int thumbnailSize = settings.value("thumbnails/size", 48).toInt();

m_thumbnailDelegate->setThumbnailsEnabled(thumbnailsEnabled);
m_thumbnailDelegate->setThumbnailSize(thumbnailSize);
```

### Example 3: Dynamic Toggle

```cpp
// Add toggle button
QPushButton* toggleButton = new QPushButton("Toggle Thumbnails");
connect(toggleButton, &QPushButton::clicked, this, [this]() {
    bool enabled = m_thumbnailDelegate->thumbnailsEnabled();
    m_thumbnailDelegate->setThumbnailsEnabled(!enabled);
    m_resultsTree->viewport()->update();
});
```

## Testing

Unit tests are available in `tests/unit/test_thumbnail_delegate.cpp`:

```bash
# Build and run tests
cmake --build build --target test_thumbnail_delegate
./build/tests/test_thumbnail_delegate
```

## Future Enhancements

Potential improvements for future versions:

1. **Disk Cache**: Persist thumbnails to disk for faster loading
2. **Video Thumbnails**: Extract actual video frames using FFmpeg
3. **EXIF Rotation**: Respect EXIF orientation for images
4. **Hover Preview**: Show larger preview on hover
5. **Custom Icons**: Allow custom icons for specific file types
6. **Thumbnail Quality**: Configurable quality vs. size trade-off

## See Also

- [Thumbnail Cache Usage](THUMBNAIL_CACHE_USAGE.md)
- [Results Window API](API_RESULTSWINDOW.md)
- [UI Design Specification](UI_DESIGN_SPECIFICATION.md)
