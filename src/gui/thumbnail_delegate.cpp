#include "thumbnail_delegate.h"
#include "thumbnail_cache.h"
#include "theme_manager.h"
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QApplication>
#include <QtWidgets/QStyle>
#include <QtWidgets/QAbstractItemView>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>
#include <QtCore/QFileInfo>


ThumbnailDelegate::ThumbnailDelegate(ThumbnailCache* cache, QObject* parent)
    : QStyledItemDelegate(parent)
    , m_cache(cache)
    , m_thumbnailSize(DEFAULT_THUMBNAIL_SIZE)
    , m_thumbnailsEnabled(true)
{
    Q_ASSERT(cache != nullptr);
    
    // Connect to theme changes to update rendering
    connect(ThemeManager::instance(), &ThemeManager::themeChanged,
            this, [this]() {
                // Force repaint of all items when theme changes
                if (auto* view = qobject_cast<QAbstractItemView*>(this->parent())) {
                    view->update();
                }
            });
}

void ThumbnailDelegate::paint(QPainter* painter,
                              const QStyleOptionViewItem& option,
                              const QModelIndex& index) const
{
    if (!index.isValid()) {
        QStyledItemDelegate::paint(painter, option, index);
        return;
    }

    // Only show thumbnails in the first column and for file items
    if (index.column() != 0 || !isFileItem(index)) {
        QStyledItemDelegate::paint(painter, option, index);
        return;
    }

    // If thumbnails are disabled, use default painting
    if (!m_thumbnailsEnabled) {
        QStyledItemDelegate::paint(painter, option, index);
        return;
    }

    painter->save();

    // Draw selection background
    if (option.state & QStyle::State_Selected) {
        painter->fillRect(option.rect, option.palette.highlight());
    } else if (index.row() % 2 == 0) {
        painter->fillRect(option.rect, option.palette.alternateBase());
    }

    // Get file path
    QString filePath = getFilePath(index);
    if (filePath.isEmpty()) {
        QStyledItemDelegate::paint(painter, option, index);
        painter->restore();
        return;
    }

    // Calculate thumbnail rect
    QRect thumbnailRect = option.rect;
    thumbnailRect.setLeft(thumbnailRect.left() + THUMBNAIL_MARGIN);
    thumbnailRect.setTop(thumbnailRect.top() + THUMBNAIL_MARGIN);
    thumbnailRect.setWidth(m_thumbnailSize);
    thumbnailRect.setHeight(m_thumbnailSize);

    // Try to get thumbnail from cache
    QSize thumbSize(m_thumbnailSize, m_thumbnailSize);
    QPixmap thumbnail = m_cache->getThumbnail(filePath, thumbSize);

    if (!thumbnail.isNull()) {
        drawThumbnail(painter, thumbnailRect, thumbnail);
    } else {
        drawPlaceholder(painter, thumbnailRect);
    }

    // Draw text next to thumbnail
    QRect textRect = option.rect;
    textRect.setLeft(thumbnailRect.right() + TEXT_MARGIN);
    textRect.setRight(option.rect.right() - TEXT_MARGIN);

    // Get display text
    QString displayText = index.data(Qt::DisplayRole).toString();

    // Set text color based on selection
    if (option.state & QStyle::State_Selected) {
        painter->setPen(option.palette.highlightedText().color());
    } else {
        painter->setPen(option.palette.text().color());
    }

    // Draw text with elision
    QFontMetrics fm(option.font);
    QString elidedText = fm.elidedText(displayText, Qt::ElideRight, textRect.width());
    painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter, elidedText);

    painter->restore();
}

QSize ThumbnailDelegate::sizeHint(const QStyleOptionViewItem& option,
                                  const QModelIndex& index) const
{
    QSize size = QStyledItemDelegate::sizeHint(option, index);

    // Only adjust size for file items in first column with thumbnails enabled
    if (index.column() == 0 && isFileItem(index) && m_thumbnailsEnabled) {
        // Ensure minimum height for thumbnail
        int minHeight = m_thumbnailSize + 2 * THUMBNAIL_MARGIN;
        if (size.height() < minHeight) {
            size.setHeight(minHeight);
        }
    }

    return size;
}

void ThumbnailDelegate::setThumbnailSize(int size)
{
    if (size > 0 && size <= 256) {
        m_thumbnailSize = size;
    }
}

void ThumbnailDelegate::setThumbnailsEnabled(bool enabled)
{
    m_thumbnailsEnabled = enabled;
}

QString ThumbnailDelegate::getFilePath(const QModelIndex& index) const
{
    // Try to get file path from item data
    // The file path should be stored in UserRole
    QVariant pathData = index.data(Qt::UserRole);
    if (pathData.isValid() && pathData.canConvert<QString>()) {
        return pathData.toString();
    }

    // Fallback: try to get from UserRole + 1
    pathData = index.data(Qt::UserRole + 1);
    if (pathData.isValid() && pathData.canConvert<QString>()) {
        return pathData.toString();
    }

    return QString();
}

bool ThumbnailDelegate::isFileItem(const QModelIndex& index) const
{
    // File items have a parent (they're children of group items)
    // Group items have no parent (they're top-level)
    return index.parent().isValid();
}

void ThumbnailDelegate::drawThumbnail(QPainter* painter, const QRect& rect, const QPixmap& thumbnail) const
{
    painter->save();

    // Draw border with theme-aware colors from ThemeManager
    ThemeData currentTheme = ThemeManager::instance()->getCurrentThemeData();
    QColor borderColor = currentTheme.colors.border;
    QColor backgroundColor = currentTheme.colors.background;
    painter->setPen(QPen(borderColor, 1));
    painter->setBrush(backgroundColor);
    painter->drawRect(rect);

    // Scale thumbnail to fit while maintaining aspect ratio
    QPixmap scaled = thumbnail.scaled(rect.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Center the thumbnail in the rect
    int x = rect.x() + (rect.width() - scaled.width()) / 2;
    int y = rect.y() + (rect.height() - scaled.height()) / 2;

    painter->drawPixmap(x, y, scaled);

    painter->restore();
}

void ThumbnailDelegate::drawPlaceholder(QPainter* painter, const QRect& rect) const
{
    painter->save();

    // Draw border with theme-aware colors from ThemeManager
    ThemeData currentTheme = ThemeManager::instance()->getCurrentThemeData();
    QColor borderColor = currentTheme.colors.border;
    QColor placeholderBg = currentTheme.colors.hover;
    painter->setPen(QPen(borderColor, 1));
    painter->setBrush(placeholderBg);
    painter->drawRect(rect);

    // Draw loading indicator with theme-aware color from ThemeManager
    QColor iconColor = currentTheme.colors.disabled;
    painter->setPen(QPen(iconColor, 2));
    
    int centerX = rect.center().x();
    int centerY = rect.center().y();
    int iconSize = rect.width() / 3;

    // Draw a simple document icon
    QRect iconRect(centerX - iconSize/2, centerY - iconSize/2, iconSize, iconSize);
    painter->drawRect(iconRect);

    // Draw lines to represent text
    int lineY = iconRect.top() + iconSize / 4;
    int lineSpacing = iconSize / 5;
    for (int i = 0; i < 3; ++i) {
        painter->drawLine(iconRect.left() + 2, lineY,
                         iconRect.right() - 2, lineY);
        lineY += lineSpacing;
    }

    painter->restore();
}
