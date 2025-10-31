#include "thumbnail_delegate.h"
#include "thumbnail_cache.h"
#include "theme_manager.h"
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QApplication>
#include <QtWidgets/QStyle>
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QStyleOptionButton>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>
#include <QtGui/QMouseEvent>
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

    painter->save();

    // Draw background and selection state first for ALL columns
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);
    
    // Enhanced selection background - make it more prominent
    if (opt.state & QStyle::State_Selected) {
        // Use a strong highlight color for selected rows
        ThemeData currentTheme = ThemeManager::instance()->getCurrentThemeData();
        QColor highlightColor = currentTheme.colors.accent;
        highlightColor.setAlpha(200); // Make it semi-transparent but visible
        painter->fillRect(opt.rect, highlightColor);
    } else if (opt.state & QStyle::State_MouseOver) {
        painter->fillRect(opt.rect, opt.palette.alternateBase());
    }

    // Handle first column with checkbox and thumbnail
    if (index.column() == 0) {
        // For group items, use Qt default painting but with our background
        if (!isFileItem(index)) {
            // Draw the text for group items
            QString text = index.data(Qt::DisplayRole).toString();
            if (!text.isEmpty()) {
                QColor textColor = opt.palette.text().color();
                if (opt.state & QStyle::State_Selected) {
                    textColor = Qt::white; // Force white text on selected rows
                }
                painter->setPen(textColor);
                painter->drawText(opt.rect.adjusted(4, 0, -4, 0), Qt::AlignLeft | Qt::AlignVCenter, text);
            }
            painter->restore();
            return;
        }

        // Draw checkbox manually for file items
        QRect checkboxRect = opt.rect;
        checkboxRect.setWidth(20);
        checkboxRect.setHeight(20);
        checkboxRect.moveTop(opt.rect.top() + (opt.rect.height() - 20) / 2);
        checkboxRect.moveLeft(opt.rect.left() + 4);

        QStyleOptionButton checkboxOption;
        checkboxOption.rect = checkboxRect;
        checkboxOption.state = QStyle::State_Enabled;
        
        // Get checkbox state from the model
        QVariant checkState = index.data(Qt::CheckStateRole);
        if (checkState.isValid()) {
            Qt::CheckState state = static_cast<Qt::CheckState>(checkState.toInt());
            if (state == Qt::Checked) {
                checkboxOption.state |= QStyle::State_On;
            } else if (state == Qt::PartiallyChecked) {
                checkboxOption.state |= QStyle::State_NoChange;
            } else {
                checkboxOption.state |= QStyle::State_Off;
            }
        }

        QApplication::style()->drawControl(QStyle::CE_CheckBox, &checkboxOption, painter);

        // Draw thumbnail if enabled
        if (m_thumbnailsEnabled) {
            QString filePath = getFilePath(index);
            if (!filePath.isEmpty()) {
                QSize thumbSize(m_thumbnailSize, m_thumbnailSize);
                QPixmap thumbnail = m_cache->getThumbnail(filePath, thumbSize);
                
                // Position thumbnail after checkbox
                int thumbnailX = checkboxRect.right() + THUMBNAIL_MARGIN;
                int thumbnailY = opt.rect.top() + (opt.rect.height() - m_thumbnailSize) / 2;
                QRect thumbnailRect(thumbnailX, thumbnailY, m_thumbnailSize, m_thumbnailSize);
                
                if (!thumbnail.isNull()) {
                    drawThumbnail(painter, thumbnailRect, thumbnail);
                } else {
                    drawPlaceholder(painter, thumbnailRect);
                }
                
                // Draw text after thumbnail
                QString text = index.data(Qt::DisplayRole).toString();
                if (!text.isEmpty()) {
                    int textX = thumbnailRect.right() + TEXT_MARGIN;
                    QRect textRect(textX, opt.rect.top(), 
                                  opt.rect.right() - textX, opt.rect.height());
                    
                    // Force white text on selected rows for better visibility
                    QColor textColor = opt.palette.text().color();
                    if (opt.state & QStyle::State_Selected) {
                        textColor = Qt::white;
                    }
                    
                    painter->setPen(textColor);
                    painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter, text);
                }
            }
        } else {
            // No thumbnails - just draw text after checkbox
            QString text = index.data(Qt::DisplayRole).toString();
            if (!text.isEmpty()) {
                int textX = checkboxRect.right() + TEXT_MARGIN;
                QRect textRect(textX, opt.rect.top(), 
                              opt.rect.right() - textX, opt.rect.height());
                
                QColor textColor = opt.palette.text().color();
                if (opt.state & QStyle::State_Selected) {
                    textColor = Qt::white;
                }
                
                painter->setPen(textColor);
                painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter, text);
            }
        }
    } else {
        // Handle other columns (Size, Modified, Path) with consistent selection appearance
        QString text = index.data(Qt::DisplayRole).toString();
        if (!text.isEmpty()) {
            QColor textColor = opt.palette.text().color();
            if (opt.state & QStyle::State_Selected) {
                textColor = Qt::white; // Force white text on selected rows
            }
            
            painter->setPen(textColor);
            
            // Align text based on column
            Qt::Alignment alignment = Qt::AlignLeft | Qt::AlignVCenter;
            if (index.column() == 1) { // Size column - right align
                alignment = Qt::AlignRight | Qt::AlignVCenter;
            }
            
            painter->drawText(opt.rect.adjusted(4, 0, -4, 0), alignment, text);
        }
    }

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

bool ThumbnailDelegate::editorEvent(QEvent* event,
                                    QAbstractItemModel* model,
                                    const QStyleOptionViewItem& option,
                                    const QModelIndex& index)
{
    if (!index.isValid() || index.column() != 0) {
        return QStyledItemDelegate::editorEvent(event, model, option, index);
    }

    // Handle mouse events for checkbox clicking
    if (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        
        // Calculate checkbox rect (same as in paint method)
        QRect checkboxRect = option.rect;
        checkboxRect.setWidth(20);
        checkboxRect.setHeight(20);
        checkboxRect.moveTop(option.rect.top() + (option.rect.height() - 20) / 2);
        checkboxRect.moveLeft(option.rect.left() + 4);
        
        // Check if click is within checkbox area
        if (checkboxRect.contains(mouseEvent->pos())) {
            if (event->type() == QEvent::MouseButtonRelease && mouseEvent->button() == Qt::LeftButton) {
                // Toggle checkbox state
                QVariant currentState = index.data(Qt::CheckStateRole);
                Qt::CheckState newState = Qt::Unchecked;
                
                if (currentState.isValid()) {
                    Qt::CheckState state = static_cast<Qt::CheckState>(currentState.toInt());
                    newState = (state == Qt::Checked) ? Qt::Unchecked : Qt::Checked;
                } else {
                    newState = Qt::Checked;
                }
                
                // Set the new state
                model->setData(index, static_cast<int>(newState), Qt::CheckStateRole);
                return true;
            }
            return true; // Consume the event
        }
    }

    // For other events, use default handling
    return QStyledItemDelegate::editorEvent(event, model, option, index);
}
