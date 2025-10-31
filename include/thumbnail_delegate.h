#ifndef THUMBNAIL_DELEGATE_H
#define THUMBNAIL_DELEGATE_H

#include <QtWidgets/QStyledItemDelegate>
#include <QtCore/QSize>
#include <QtGui/QPixmap>

// Forward declaration
class ThumbnailCache;

/**
 * @brief Custom delegate for displaying thumbnails in QTreeWidget
 * 
 * This delegate renders file thumbnails alongside file information
 * in the results tree view. It integrates with ThumbnailCache for
 * efficient thumbnail loading and caching.
 */
class ThumbnailDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    /**
     * @brief Construct a ThumbnailDelegate
     * @param cache Pointer to thumbnail cache
     * @param parent Parent object
     */
    explicit ThumbnailDelegate(ThumbnailCache* cache, QObject* parent = nullptr);

    /**
     * @brief Paint the item with thumbnail
     * @param painter Painter to use
     * @param option Style options
     * @param index Model index
     */
    void paint(QPainter* painter, 
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override;

    /**
     * @brief Get size hint for item
     * @param option Style options
     * @param index Model index
     * @return QSize Size hint
     */
    QSize sizeHint(const QStyleOptionViewItem& option,
                   const QModelIndex& index) const override;

    /**
     * @brief Set thumbnail size
     * @param size Thumbnail size in pixels
     */
    void setThumbnailSize(int size);

    /**
     * @brief Get current thumbnail size
     * @return int Thumbnail size in pixels
     */
    int thumbnailSize() const { return m_thumbnailSize; }

    /**
     * @brief Enable or disable thumbnail display
     * @param enabled True to enable thumbnails
     */
    void setThumbnailsEnabled(bool enabled);

    /**
     * @brief Check if thumbnails are enabled
     * @return bool True if thumbnails are enabled
     */
    bool thumbnailsEnabled() const { return m_thumbnailsEnabled; }

    /**
     * @brief Handle editor events (mouse clicks on checkboxes)
     * @param event The event to handle
     * @param model The model
     * @param option Style options
     * @param index Model index
     * @return bool True if event was handled
     */
    bool editorEvent(QEvent* event,
                     QAbstractItemModel* model,
                     const QStyleOptionViewItem& option,
                     const QModelIndex& index) override;

private:
    /**
     * @brief Get file path from tree widget item
     * @param index Model index
     * @return QString File path or empty string
     */
    QString getFilePath(const QModelIndex& index) const;

    /**
     * @brief Check if item is a file item (not a group header)
     * @param index Model index
     * @return bool True if item is a file
     */
    bool isFileItem(const QModelIndex& index) const;

    /**
     * @brief Draw thumbnail in the specified rect
     * @param painter Painter to use
     * @param rect Rectangle to draw in
     * @param thumbnail Thumbnail pixmap
     */
    void drawThumbnail(QPainter* painter, const QRect& rect, const QPixmap& thumbnail) const;

    /**
     * @brief Draw placeholder when thumbnail is not available
     * @param painter Painter to use
     * @param rect Rectangle to draw in
     */
    void drawPlaceholder(QPainter* painter, const QRect& rect) const;

    ThumbnailCache* m_cache;
    int m_thumbnailSize;
    bool m_thumbnailsEnabled;

    static const int DEFAULT_THUMBNAIL_SIZE = 48;
    static const int THUMBNAIL_MARGIN = 4;
    static const int TEXT_MARGIN = 8;
};

#endif // THUMBNAIL_DELEGATE_H
