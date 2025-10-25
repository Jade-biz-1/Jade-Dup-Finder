#include "ui_enhancements.h"
#include "theme_manager.h"
#include "logger.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QLayout>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QGraphicsOpacityEffect>
#include <QtCore/QEvent>
#include <QtCore/QDateTime>
#include <QtCore/QLocale>
#include <QtCore/QDir>
#include <QtGui/QFontMetrics>
#include <QtGui/QPalette>
#include <QtGui/QKeyEvent>

// Section 1.5.3 - Visual Feedback

void UIEnhancements::addButtonHoverEffect(QPushButton* button, const QColor& hoverColor)
{
    if (!button) return;
    
    // Get current palette
    QPalette palette = button->palette();
    QColor baseColor = palette.color(QPalette::Button);
    QColor textColor = palette.color(QPalette::ButtonText);
    
    // Determine hover color based on theme if not provided
    QColor hover = hoverColor;
    if (!hover.isValid()) {
        // Use theme-aware hover color
        ThemeManager* themeManager = ThemeManager::instance();
        if (themeManager->currentTheme() == ThemeManager::Dark) {
            // Lighter color for dark theme
            hover = baseColor.lighter(120);
        } else {
            // Darker color for light theme
            hover = baseColor.darker(110);
        }
    }
    
    // Create stylesheet with hover effect
    QString styleSheet = QString(
        "QPushButton {"
        "    background-color: %1;"
        "    color: %2;"
        "    border: 1px solid %3;"
        "    border-radius: 4px;"
        "    padding: 5px 15px;"
        "    min-height: 24px;"
        "}"
        "QPushButton:hover {"
        "    background-color: %4;"
        "    border: 1px solid %5;"
        "}"
        "QPushButton:pressed {"
        "    background-color: %6;"
        "}"
        "QPushButton:disabled {"
        "    background-color: %7;"
        "    color: %8;"
        "}"
    ).arg(baseColor.name())
     .arg(textColor.name())
     .arg(baseColor.darker(120).name())
     .arg(hover.name())
     .arg(hover.darker(110).name())
     .arg(hover.darker(120).name())
     .arg(baseColor.lighter(130).name())
     .arg(textColor.lighter(170).name());
    
    button->setStyleSheet(styleSheet);
}

void UIEnhancements::addHoverEffectsToButtons(QWidget* parent)
{
    if (!parent) return;
    
    QList<QPushButton*> buttons = findAllButtons(parent);
    for (QPushButton* button : buttons) {
        addButtonHoverEffect(button);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Added hover effects to %1 buttons").arg(buttons.size()));
}

void UIEnhancements::applyDisabledStateStyle(QWidget* widget)
{
    if (!widget) return;
    
    // Apply opacity effect for disabled state
    if (!widget->isEnabled()) {
        QGraphicsOpacityEffect* effect = new QGraphicsOpacityEffect(widget);
        effect->setOpacity(0.5);
        widget->setGraphicsEffect(effect);
    } else {
        widget->setGraphicsEffect(nullptr);
    }
}

void UIEnhancements::setEnabledWithFeedback(QWidget* widget, bool enabled)
{
    if (!widget) return;
    
    widget->setEnabled(enabled);
    applyDisabledStateStyle(widget);
}

void UIEnhancements::addDragDropFeedback(QWidget* widget, const QString& acceptText)
{
    if (!widget) return;
    
    widget->setAcceptDrops(true);
    
    // Install event filter for drag/drop visual feedback
    class DragDropEventFilter : public QObject {
    public:
        DragDropEventFilter(QWidget* parent, const QString& text) 
            : QObject(parent), m_acceptText(text) {}
        
    protected:
        bool eventFilter(QObject* obj, QEvent* event) override {
            if (event->type() == QEvent::DragEnter) {
                // Add visual feedback (border highlight)
                if (QWidget* widget = qobject_cast<QWidget*>(obj)) {
                    widget->setStyleSheet(widget->styleSheet() + 
                        "; border: 2px dashed #0078d7;");
                }
            } else if (event->type() == QEvent::DragLeave || 
                       event->type() == QEvent::Drop) {
                // Remove visual feedback
                if (QWidget* widget = qobject_cast<QWidget*>(obj)) {
                    QString style = widget->styleSheet();
                    style.remove("; border: 2px dashed #0078d7;");
                    widget->setStyleSheet(style);
                }
            }
            return QObject::eventFilter(obj, event);
        }
        
    private:
        QString m_acceptText;
    };
    
    widget->installEventFilter(new DragDropEventFilter(widget, acceptText));
}

void UIEnhancements::showLoadingIndicator(QWidget* widget, const QString& text)
{
    if (!widget) return;
    
    // Store original cursor
    widget->setProperty("originalCursor", widget->cursor());
    widget->setCursor(Qt::WaitCursor);
    
    if (!text.isEmpty()) {
        widget->setToolTip(text);
    }
}

void UIEnhancements::hideLoadingIndicator(QWidget* widget)
{
    if (!widget) return;
    
    // Restore original cursor
    QVariant cursorVar = widget->property("originalCursor");
    if (cursorVar.isValid()) {
        widget->setCursor(cursorVar.value<QCursor>());
    } else {
        widget->setCursor(Qt::ArrowCursor);
    }
    
    widget->setToolTip(QString());
}

// Section 1.5.4 - Polish User Interactions

void UIEnhancements::setupLogicalTabOrder(QWidget* parent)
{
    if (!parent) return;
    
    QList<QWidget*> focusableWidgets = findAllFocusableWidgets(parent);
    
    // Sort by geometry (top to bottom, left to right)
    std::sort(focusableWidgets.begin(), focusableWidgets.end(), 
              [](QWidget* a, QWidget* b) {
                  if (qAbs(a->y() - b->y()) < 20) {
                      // Same row, sort by x
                      return a->x() < b->x();
                  }
                  return a->y() < b->y();
              });
    
    // Set tab order
    for (int i = 0; i < focusableWidgets.size() - 1; ++i) {
        QWidget::setTabOrder(focusableWidgets[i], focusableWidgets[i + 1]);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Set tab order for %1 widgets").arg(focusableWidgets.size()));
}

void UIEnhancements::addFocusIndicators(QWidget* widget)
{
    if (!widget) return;
    
    // Enable focus policy
    widget->setFocusPolicy(Qt::StrongFocus);
    
    // Set focus style
    QString focusStyle = "QWidget:focus { border: 2px solid #0078d7; }";
    QString currentStyle = widget->styleSheet();
    if (!currentStyle.contains(":focus")) {
        widget->setStyleSheet(currentStyle + " " + focusStyle);
    }
}

void UIEnhancements::addComprehensiveTooltips(QWidget* parent, const QMap<QString, QString>& defaultTooltips)
{
    if (!parent) return;
    
    int tooltipsAdded = 0;
    
    // Find all interactive widgets
    QList<QWidget*> widgets = parent->findChildren<QWidget*>();
    for (QWidget* widget : widgets) {
        // Skip if already has tooltip
        if (!widget->toolTip().isEmpty()) {
            continue;
        }
        
        QString tooltip;
        
        // Check for custom tooltip in map
        if (!widget->objectName().isEmpty() && defaultTooltips.contains(widget->objectName())) {
            tooltip = defaultTooltips[widget->objectName()];
        } else {
            // Generate default tooltip
            tooltip = getDefaultTooltipForWidget(widget);
        }
        
        if (!tooltip.isEmpty()) {
            widget->setToolTip(tooltip);
            tooltipsAdded++;
        }
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Added %1 tooltips").arg(tooltipsAdded));
}

void UIEnhancements::setupEscapeKeyHandler(QWidget* dialog)
{
    if (!dialog) return;
    
    // Install event filter for ESC key
    class EscapeKeyFilter : public QObject {
    public:
        EscapeKeyFilter(QWidget* parent) : QObject(parent) {}
        
    protected:
        bool eventFilter(QObject* obj, QEvent* event) override {
            if (event->type() == QEvent::KeyPress) {
                QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
                if (keyEvent->key() == Qt::Key_Escape) {
                    if (QWidget* widget = qobject_cast<QWidget*>(parent())) {
                        widget->close();
                        return true;
                    }
                }
            }
            return QObject::eventFilter(obj, event);
        }
    };
    
    dialog->installEventFilter(new EscapeKeyFilter(dialog));
}

void UIEnhancements::setupEnterKeyHandler(QWidget* dialog)
{
    if (!dialog) return;
    
    // Find default button in dialog button box
    QDialogButtonBox* buttonBox = dialog->findChild<QDialogButtonBox*>();
    if (buttonBox) {
        QPushButton* defaultButton = qobject_cast<QPushButton*>(buttonBox->button(QDialogButtonBox::Ok));
        if (!defaultButton) {
            defaultButton = qobject_cast<QPushButton*>(buttonBox->button(QDialogButtonBox::Save));
        }
        if (!defaultButton) {
            defaultButton = qobject_cast<QPushButton*>(buttonBox->button(QDialogButtonBox::Yes));
        }
        
        if (defaultButton) {
            defaultButton->setDefault(true);
        }
    }
}

void UIEnhancements::addHoverTooltip(QWidget* widget, const QString& text)
{
    if (!widget || text.isEmpty()) return;
    
    widget->setToolTip(text);
    widget->setToolTipDuration(3000);  // 3 seconds
}

// Section 1.5.5 - Text Display Improvements

void UIEnhancements::elideTextInLabel(QLabel* label, const QString& text, int maxWidth)
{
    if (!label) return;
    
    int width = maxWidth > 0 ? maxWidth : label->width();
    QFontMetrics metrics(label->font());
    QString elidedText = metrics.elidedText(text, Qt::ElideMiddle, width);
    label->setText(elidedText);
    label->setToolTip(text);  // Show full text in tooltip
}

QString UIEnhancements::formatPathWithEllipsis(const QString& path, int maxLength)
{
    if (path.length() <= maxLength) {
        return path;
    }
    
    // Try to preserve beginning and end of path
    QDir dir(path);
    QString fileName = dir.dirName();
    QString parentPath = dir.absolutePath();
    
    if (fileName.length() + 3 < maxLength) {
        int remainingLength = maxLength - fileName.length() - 3;  // 3 for "..."
        QString prefix = parentPath.left(remainingLength);
        return prefix + "..." + fileName;
    } else {
        // Just truncate with ellipsis in middle
        int half = maxLength / 2 - 2;
        return path.left(half) + "..." + path.right(half);
    }
}

QString UIEnhancements::formatFileSize(qint64 bytes)
{
    QLocale locale;
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;
    
    if (bytes >= TB) {
        return locale.toString(bytes / static_cast<double>(TB), 'f', 2) + " TB";
    } else if (bytes >= GB) {
        return locale.toString(bytes / static_cast<double>(GB), 'f', 2) + " GB";
    } else if (bytes >= MB) {
        return locale.toString(bytes / static_cast<double>(MB), 'f', 2) + " MB";
    } else if (bytes >= KB) {
        return locale.toString(bytes / static_cast<double>(KB), 'f', 1) + " KB";
    } else {
        return locale.toString(bytes) + " bytes";
    }
}

QString UIEnhancements::formatNumber(qint64 number)
{
    QLocale locale;
    return locale.toString(number);
}

QString UIEnhancements::formatDateTime(const QDateTime& dateTime, const QString& format)
{
    QLocale locale;
    if (!format.isEmpty()) {
        return locale.toString(dateTime, format);
    } else {
        // Use locale's default format
        return locale.toString(dateTime, QLocale::ShortFormat);
    }
}

QList<QWidget*> UIEnhancements::findNonTranslatableText(QWidget* widget)
{
    QList<QWidget*> nonTranslatable;
    
    if (!widget) return nonTranslatable;
    
    // This is a simplified check - in practice, detecting tr() usage requires
    // source code analysis, not runtime checks
    // Here we just return empty list as placeholder
    
    return nonTranslatable;
}

void UIEnhancements::applyConsistentSpacing(QWidget* dialog)
{
    if (!dialog) return;
    
    // Standard spacing values
    const int STANDARD_MARGIN = 12;
    const int STANDARD_SPACING = 8;
    
    // Apply to all layouts
    QList<QLayout*> layouts = dialog->findChildren<QLayout*>();
    for (QLayout* layout : layouts) {
        layout->setContentsMargins(STANDARD_MARGIN, STANDARD_MARGIN, STANDARD_MARGIN, STANDARD_MARGIN);
        layout->setSpacing(STANDARD_SPACING);
    }
    
    // Also set dialog's main layout
    if (dialog->layout()) {
        dialog->layout()->setContentsMargins(STANDARD_MARGIN, STANDARD_MARGIN, STANDARD_MARGIN, STANDARD_MARGIN);
        dialog->layout()->setSpacing(STANDARD_SPACING);
    }
}

// Private Helper Methods

QList<QPushButton*> UIEnhancements::findAllButtons(QWidget* parent)
{
    return parent->findChildren<QPushButton*>();
}

QList<QWidget*> UIEnhancements::findAllFocusableWidgets(QWidget* parent)
{
    QList<QWidget*> focusable;
    QList<QWidget*> allWidgets = parent->findChildren<QWidget*>();
    
    for (QWidget* widget : allWidgets) {
        if (widget->focusPolicy() != Qt::NoFocus && widget->isVisible()) {
            focusable.append(widget);
        }
    }
    
    return focusable;
}

QString UIEnhancements::getDefaultTooltipForWidget(QWidget* widget)
{
    if (!widget) return QString();
    
    // Generate default tooltip based on widget type and text
    if (QPushButton* button = qobject_cast<QPushButton*>(widget)) {
        QString text = button->text().remove('&');  // Remove mnemonic
        if (!text.isEmpty()) {
            return QObject::tr("Click to %1").arg(text.toLower());
        }
    } else if (QCheckBox* checkbox = qobject_cast<QCheckBox*>(widget)) {
        QString text = checkbox->text();
        if (!text.isEmpty()) {
            return QObject::tr("Check to enable: %1").arg(text);
        }
    } else if (QLineEdit* lineEdit = qobject_cast<QLineEdit*>(widget)) {
        QString placeholder = lineEdit->placeholderText();
        if (!placeholder.isEmpty()) {
            return placeholder;
        }
    }
    
    return QString();
}

// Section 2.1.6 - Theme Styling Helpers
// Consolidate duplicate styling code by providing centralized helper methods

void UIEnhancements::applyButtonStyle(QPushButton* button)
{
    if (!button) return;
    
    QString buttonStyle = ThemeManager::instance()->getComponentStyle(ThemeManager::ComponentType::Button);
    button->setStyleSheet(buttonStyle);
    button->setMinimumSize(ThemeManager::instance()->getMinimumControlSize(ThemeManager::ControlType::Button));
}

void UIEnhancements::applyCheckBoxStyle(QCheckBox* checkbox)
{
    if (!checkbox) return;
    
    QString checkboxStyle = ThemeManager::instance()->getComponentStyle(ThemeManager::ComponentType::CheckBox);
    checkbox->setStyleSheet(checkboxStyle);
    checkbox->setMinimumSize(ThemeManager::instance()->getMinimumControlSize(ThemeManager::ControlType::CheckBox));
}

void UIEnhancements::applyLabelStyle(QLabel* label)
{
    if (!label) return;
    
    QString labelStyle = ThemeManager::instance()->getComponentStyle(ThemeManager::ComponentType::Label);
    label->setStyleSheet(labelStyle);
}

void UIEnhancements::applyTreeWidgetStyle(QTreeWidget* treeWidget)
{
    if (!treeWidget) return;
    
    QString treeStyle = ThemeManager::instance()->getComponentStyle(ThemeManager::ComponentType::TreeView);
    treeWidget->setStyleSheet(treeStyle);
}

void UIEnhancements::applyProgressBarStyle(QProgressBar* progressBar, const QString& type)
{
    if (!progressBar) return;
    
    ThemeManager::ProgressType progressType = ThemeManager::ProgressType::Normal;
    if (type == "Performance") {
        progressType = ThemeManager::ProgressType::Performance;
    } else if (type == "Queue") {
        progressType = ThemeManager::ProgressType::Queue;
    }
    
    QString progressStyle = ThemeManager::instance()->getProgressBarStyle(progressType);
    progressBar->setStyleSheet(progressStyle);
}

void UIEnhancements::applyStatusIndicatorStyle(QLabel* label, const QString& statusType)
{
    if (!label) return;
    
    ThemeManager::StatusType status = ThemeManager::StatusType::Neutral;
    if (statusType == "Error") {
        status = ThemeManager::StatusType::Error;
    } else if (statusType == "Warning") {
        status = ThemeManager::StatusType::Warning;
    } else if (statusType == "Success") {
        status = ThemeManager::StatusType::Success;
    } else if (statusType == "Info") {
        status = ThemeManager::StatusType::Info;
    }
    
    QString statusStyle = ThemeManager::instance()->getStatusIndicatorStyle(status);
    label->setStyleSheet(statusStyle);
}

void UIEnhancements::applyButtonStyles(const QList<QPushButton*>& buttons)
{
    for (QPushButton* button : buttons) {
        applyButtonStyle(button);
    }
}

void UIEnhancements::applyCheckBoxStyles(const QList<QCheckBox*>& checkboxes)
{
    for (QCheckBox* checkbox : checkboxes) {
        applyCheckBoxStyle(checkbox);
    }
}
