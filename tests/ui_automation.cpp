#include "ui_automation.h"
#include <QApplication>
#include <QScreen>
#include <QWindow>
#include <QSignalSpy>
#include <QThread>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QDebug>
#include <QMetaObject>
#include <QMetaProperty>
#include <QAccessible>
#include <QAccessibleInterface>
#include <QStyle>
#include <QStyleOption>
#include <QRegularExpression>

UIAutomation::UIAutomation(QObject* parent)
    : QObject(parent)
    , m_defaultTimeoutMs(5000)
    , m_defaultDelayMs(100)
    , m_automaticScreenshots(false)
    , m_detailedLogging(false)
    , m_retryAttempts(3)
{
    // Set default screenshot directory
    m_screenshotDirectory = QDir::currentPath() + "/ui_test_screenshots";
    QDir().mkpath(m_screenshotDirectory);
}

UIAutomation::~UIAutomation() = default;

QWidget* UIAutomation::findWidget(const WidgetSelector& selector, QWidget* parent) {
    logAction("findWidget", QString("Searching for widget: %1").arg(selector.value), true);
    
    QWidget* searchRoot = parent ? parent : QApplication::activeWindow();
    if (!searchRoot) {
        // Try all top-level widgets
        QWidgetList topLevelWidgets = QApplication::topLevelWidgets();
        for (QWidget* widget : topLevelWidgets) {
            if (widget->isVisible()) {
                QWidget* found = findWidgetRecursive(widget, selector);
                if (found) {
                    emit widgetFound(selector.value, found);
                    return found;
                }
            }
        }
    } else {
        QWidget* found = findWidgetRecursive(searchRoot, selector);
        if (found) {
            emit widgetFound(selector.value, found);
            return found;
        }
    }
    
    emit widgetNotFound(selector.value);
    return nullptr;
}

QList<QWidget*> UIAutomation::findAllWidgets(const WidgetSelector& selector, QWidget* parent) {
    QList<QWidget*> results;
    
    QWidget* searchRoot = parent ? parent : QApplication::activeWindow();
    if (!searchRoot) {
        QWidgetList topLevelWidgets = QApplication::topLevelWidgets();
        for (QWidget* widget : topLevelWidgets) {
            if (widget->isVisible()) {
                QList<QWidget*> children = widget->findChildren<QWidget*>();
                for (QWidget* child : children) {
                    if (matchesSelector(child, selector)) {
                        results.append(child);
                    }
                }
            }
        }
    } else {
        QList<QWidget*> children = searchRoot->findChildren<QWidget*>();
        for (QWidget* child : children) {
            if (matchesSelector(child, selector)) {
                results.append(child);
            }
        }
    }
    
    return results;
}

bool UIAutomation::waitForWidget(const WidgetSelector& selector, int timeoutMs) {
    logAction("waitForWidget", QString("Waiting for widget: %1").arg(selector.value), true);
    
    QElapsedTimer timer;
    timer.start();
    
    while (timer.elapsed() < timeoutMs) {
        QWidget* widget = findWidget(selector);
        if (widget && (!selector.mustBeVisible || widget->isVisible()) && 
            (!selector.mustBeEnabled || widget->isEnabled())) {
            return true;
        }
        
        QApplication::processEvents();
        QThread::msleep(50);
    }
    
    logAction("waitForWidget", QString("Timeout waiting for widget: %1").arg(selector.value), false);
    return false;
}

bool UIAutomation::isWidgetAccessible(const WidgetSelector& selector) {
    QWidget* widget = findWidget(selector);
    return widget && isWidgetInteractable(widget);
}

bool UIAutomation::clickWidget(const WidgetSelector& selector, const MouseOptions& options) {
    emit actionStarted("click", selector.value);
    
    auto action = [this, &selector, &options]() -> bool {
        QWidget* widget = findWidget(selector);
        if (!widget) {
            logAction("clickWidget", QString("Widget not found: %1").arg(selector.value), false);
            return false;
        }
        
        if (!isWidgetInteractable(widget)) {
            logAction("clickWidget", QString("Widget not interactable: %1").arg(selector.value), false);
            return false;
        }
        
        // Calculate click point
        QPoint clickPoint = getWidgetClickPoint(widget, options.offset);
        
        // Apply delay if specified
        if (options.delayMs > 0) {
            QThread::msleep(options.delayMs);
        }
        
        // Perform click
        if (options.doubleClick) {
            QTest::mouseDClick(widget, options.button, options.modifiers, clickPoint);
        } else {
            QTest::mouseClick(widget, options.button, options.modifiers, clickPoint);
        }
        
        QApplication::processEvents();
        
        logAction("clickWidget", QString("Clicked widget: %1").arg(selector.value), true);
        return true;
    };
    
    bool success = retryAction(action, m_retryAttempts);
    
    if (m_automaticScreenshots) {
        takeAutomaticScreenshot("click_" + selector.value);
    }
    
    emit actionCompleted("click", selector.value, success);
    return success;
}

bool UIAutomation::doubleClickWidget(const WidgetSelector& selector, const MouseOptions& options) {
    MouseOptions doubleClickOptions = options;
    doubleClickOptions.doubleClick = true;
    return clickWidget(selector, doubleClickOptions);
}

bool UIAutomation::rightClickWidget(const WidgetSelector& selector, const MouseOptions& options) {
    MouseOptions rightClickOptions = options;
    rightClickOptions.button = Qt::RightButton;
    return clickWidget(selector, rightClickOptions);
}

bool UIAutomation::hoverWidget(const WidgetSelector& selector, int durationMs) {
    emit actionStarted("hover", selector.value);
    
    QWidget* widget = findWidget(selector);
    if (!widget) {
        emit actionCompleted("hover", selector.value, false);
        return false;
    }
    
    QPoint hoverPoint = getWidgetCenter(widget);
    QTest::mouseMove(widget, hoverPoint);
    
    if (durationMs > 0) {
        QThread::msleep(durationMs);
    }
    
    QApplication::processEvents();
    
    emit actionCompleted("hover", selector.value, true);
    return true;
}

bool UIAutomation::typeText(const WidgetSelector& selector, const QString& text, const KeyboardOptions& options) {
    emit actionStarted("typeText", selector.value);
    
    auto action = [this, &selector, &text, &options]() -> bool {
        QWidget* widget = findWidget(selector);
        if (!widget) {
            return false;
        }
        
        if (!isWidgetInteractable(widget)) {
            return false;
        }
        
        // Set focus to widget
        widget->setFocus();
        QApplication::processEvents();
        
        // Clear existing text if requested
        if (options.clearFirst) {
            clearText(selector);
        }
        
        // Select all text if requested
        if (options.selectAll) {
            selectAllText(selector);
        }
        
        // Apply delay if specified
        if (options.delayMs > 0) {
            QThread::msleep(options.delayMs);
        }
        
        // Type text with specified speed
        if (options.typingSpeedMs > 0) {
            for (const QChar& ch : text) {
                QTest::keyClick(widget, ch.toLatin1(), options.modifiers);
                QThread::msleep(options.typingSpeedMs);
            }
        } else {
            QTest::keyClicks(widget, text, options.modifiers);
        }
        
        QApplication::processEvents();
        
        logAction("typeText", QString("Typed text in widget: %1").arg(selector.value), true);
        return true;
    };
    
    bool success = retryAction(action, m_retryAttempts);
    emit actionCompleted("typeText", selector.value, success);
    return success;
}

bool UIAutomation::pressKey(const WidgetSelector& selector, Qt::Key key, const KeyboardOptions& options) {
    emit actionStarted("pressKey", selector.value);
    
    QWidget* widget = findWidget(selector);
    if (!widget) {
        emit actionCompleted("pressKey", selector.value, false);
        return false;
    }
    
    widget->setFocus();
    QApplication::processEvents();
    
    if (options.delayMs > 0) {
        QThread::msleep(options.delayMs);
    }
    
    QTest::keyClick(widget, key, options.modifiers);
    QApplication::processEvents();
    
    emit actionCompleted("pressKey", selector.value, true);
    return true;
}

bool UIAutomation::pressKeySequence(const WidgetSelector& selector, const QKeySequence& sequence, const KeyboardOptions& options) {
    emit actionStarted("pressKeySequence", selector.value);
    
    QWidget* widget = findWidget(selector);
    if (!widget) {
        emit actionCompleted("pressKeySequence", selector.value, false);
        return false;
    }
    
    widget->setFocus();
    QApplication::processEvents();
    
    if (options.delayMs > 0) {
        QThread::msleep(options.delayMs);
    }
    
    QTest::keySequence(widget, sequence);
    QApplication::processEvents();
    
    emit actionCompleted("pressKeySequence", selector.value, true);
    return true;
}

bool UIAutomation::setWidgetValue(const WidgetSelector& selector, const QVariant& value) {
    QWidget* widget = findWidget(selector);
    if (!widget) {
        return false;
    }
    
    // Try different widget types
    if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
        return setLineEditValue(lineEdit, value);
    } else if (auto* textEdit = qobject_cast<QTextEdit*>(widget)) {
        return setTextEditValue(textEdit, value);
    } else if (auto* comboBox = qobject_cast<QComboBox*>(widget)) {
        return setComboBoxValue(comboBox, value);
    } else if (auto* spinBox = qobject_cast<QSpinBox*>(widget)) {
        return setSpinBoxValue(spinBox, value);
    } else if (auto* slider = qobject_cast<QSlider*>(widget)) {
        return setSliderValue(slider, value);
    } else if (auto* checkBox = qobject_cast<QCheckBox*>(widget)) {
        return setCheckBoxValue(checkBox, value);
    } else if (auto* radioButton = qobject_cast<QRadioButton*>(widget)) {
        return setRadioButtonValue(radioButton, value);
    }
    
    // Try setting as property
    return widget->setProperty("value", value);
}

QVariant UIAutomation::getWidgetValue(const WidgetSelector& selector) {
    QWidget* widget = findWidget(selector);
    if (!widget) {
        return QVariant();
    }
    
    // Try different widget types
    if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
        return getLineEditValue(lineEdit);
    } else if (auto* textEdit = qobject_cast<QTextEdit*>(widget)) {
        return getTextEditValue(textEdit);
    } else if (auto* comboBox = qobject_cast<QComboBox*>(widget)) {
        return getComboBoxValue(comboBox);
    } else if (auto* spinBox = qobject_cast<QSpinBox*>(widget)) {
        return getSpinBoxValue(spinBox);
    } else if (auto* slider = qobject_cast<QSlider*>(widget)) {
        return getSliderValue(slider);
    } else if (auto* checkBox = qobject_cast<QCheckBox*>(widget)) {
        return getCheckBoxValue(checkBox);
    } else if (auto* radioButton = qobject_cast<QRadioButton*>(widget)) {
        return getRadioButtonValue(radioButton);
    }
    
    // Try getting as property
    return widget->property("value");
}

bool UIAutomation::fillForm(const QList<FormField>& fields) {
    emit actionStarted("fillForm", QString("Filling %1 fields").arg(fields.size()));
    
    bool allSuccess = true;
    
    for (const FormField& field : fields) {
        bool success = false;
        
        if (field.customSetter) {
            QWidget* widget = findWidget(field.selector);
            if (widget) {
                success = field.customSetter(widget, field.value);
            }
        } else {
            success = setWidgetValue(field.selector, field.value);
        }
        
        if (!success && field.required) {
            logAction("fillForm", QString("Failed to set required field: %1").arg(field.selector.value), false);
            allSuccess = false;
            break;
        }
        
        // Small delay between fields
        QThread::msleep(50);
    }
    
    emit actionCompleted("fillForm", QString("%1 fields").arg(fields.size()), allSuccess);
    return allSuccess;
}

bool UIAutomation::waitForCondition(const WaitSpec& waitSpec) {
    logAction("waitForCondition", QString("Waiting for condition: %1").arg(static_cast<int>(waitSpec.condition)), true);
    
    QElapsedTimer timer;
    timer.start();
    
    while (timer.elapsed() < waitSpec.timeoutMs) {
        if (evaluateWaitCondition(waitSpec)) {
            emit waitConditionMet(QString::number(static_cast<int>(waitSpec.condition)));
            return true;
        }
        
        QApplication::processEvents();
        QThread::msleep(waitSpec.pollIntervalMs);
    }
    
    emit waitConditionTimeout(QString::number(static_cast<int>(waitSpec.condition)));
    return false;
}

bool UIAutomation::verifyWidgetExists(const WidgetSelector& selector) {
    return findWidget(selector) != nullptr;
}

bool UIAutomation::verifyWidgetVisible(const WidgetSelector& selector) {
    QWidget* widget = findWidget(selector);
    return widget && widget->isVisible();
}

bool UIAutomation::verifyWidgetEnabled(const WidgetSelector& selector) {
    QWidget* widget = findWidget(selector);
    return widget && widget->isEnabled();
}

bool UIAutomation::verifyWidgetText(const WidgetSelector& selector, const QString& expectedText) {
    QWidget* widget = findWidget(selector);
    if (!widget) {
        return false;
    }
    
    QString actualText;
    
    // Try different ways to get text
    if (auto* label = qobject_cast<QLabel*>(widget)) {
        actualText = label->text();
    } else if (auto* button = qobject_cast<QPushButton*>(widget)) {
        actualText = button->text();
    } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
        actualText = lineEdit->text();
    } else if (auto* textEdit = qobject_cast<QTextEdit*>(widget)) {
        actualText = textEdit->toPlainText();
    } else {
        // Try as property
        actualText = widget->property("text").toString();
    }
    
    return actualText == expectedText;
}

QPixmap UIAutomation::captureWidget(const WidgetSelector& selector) {
    QWidget* widget = findWidget(selector);
    if (!widget) {
        return QPixmap();
    }
    
    return widget->grab();
}

QPixmap UIAutomation::captureScreen() {
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return QPixmap();
    }
    
    return screen->grabWindow(0);
}

bool UIAutomation::saveScreenshot(const WidgetSelector& selector, const QString& filePath) {
    QPixmap screenshot = captureWidget(selector);
    if (screenshot.isNull()) {
        return false;
    }
    
    QFileInfo fileInfo(filePath);
    QDir().mkpath(fileInfo.absolutePath());
    
    bool success = screenshot.save(filePath);
    if (success) {
        emit screenshotCaptured(filePath);
    }
    
    return success;
}

// Static utility functions
UIAutomation::WidgetSelector UIAutomation::byObjectName(const QString& objectName) {
    WidgetSelector selector;
    selector.strategy = SelectorStrategy::ObjectName;
    selector.value = objectName;
    return selector;
}

UIAutomation::WidgetSelector UIAutomation::byText(const QString& text) {
    WidgetSelector selector;
    selector.strategy = SelectorStrategy::Text;
    selector.value = text;
    return selector;
}

UIAutomation::WidgetSelector UIAutomation::byClassName(const QString& className) {
    WidgetSelector selector;
    selector.strategy = SelectorStrategy::ClassName;
    selector.value = className;
    return selector;
}

UIAutomation::WidgetSelector UIAutomation::byProperty(const QString& propertyName, const QVariant& value) {
    WidgetSelector selector;
    selector.strategy = SelectorStrategy::Property;
    selector.value = propertyName;
    selector.properties[propertyName] = value;
    return selector;
}

// Configuration methods
void UIAutomation::setDefaultTimeout(int timeoutMs) {
    m_defaultTimeoutMs = timeoutMs;
}

void UIAutomation::setDefaultDelay(int delayMs) {
    m_defaultDelayMs = delayMs;
}

void UIAutomation::setScreenshotDirectory(const QString& directory) {
    m_screenshotDirectory = directory;
    QDir().mkpath(directory);
}

void UIAutomation::enableAutomaticScreenshots(bool enable) {
    m_automaticScreenshots = enable;
}

void UIAutomation::enableDetailedLogging(bool enable) {
    m_detailedLogging = enable;
}

// Private helper methods
QWidget* UIAutomation::findWidgetRecursive(QWidget* parent, const WidgetSelector& selector) {
    if (!parent) {
        return nullptr;
    }
    
    // Check if parent matches
    if (matchesSelector(parent, selector)) {
        return parent;
    }
    
    // Check children
    QList<QWidget*> children = parent->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (QWidget* child : children) {
        if (matchesSelector(child, selector)) {
            return child;
        }
        
        // Recurse into child
        QWidget* found = findWidgetRecursive(child, selector);
        if (found) {
            return found;
        }
    }
    
    return nullptr;
}

bool UIAutomation::matchesSelector(QWidget* widget, const WidgetSelector& selector) {
    if (!widget) {
        return false;
    }
    
    // Check visibility and enabled state if required
    if (selector.mustBeVisible && !widget->isVisible()) {
        return false;
    }
    
    if (selector.mustBeEnabled && !widget->isEnabled()) {
        return false;
    }
    
    // Check selector strategy
    switch (selector.strategy) {
        case SelectorStrategy::ObjectName:
            return widget->objectName() == selector.value;
            
        case SelectorStrategy::Text: {
            QString text;
            if (auto* label = qobject_cast<QLabel*>(widget)) {
                text = label->text();
            } else if (auto* button = qobject_cast<QPushButton*>(widget)) {
                text = button->text();
            } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
                text = lineEdit->text();
            } else {
                text = widget->property("text").toString();
            }
            return text == selector.value;
        }
        
        case SelectorStrategy::ClassName:
            return widget->metaObject()->className() == selector.value;
            
        case SelectorStrategy::Property:
            for (auto it = selector.properties.begin(); it != selector.properties.end(); ++it) {
                if (widget->property(it.key().toUtf8().constData()) != it.value()) {
                    return false;
                }
            }
            return true;
            
        case SelectorStrategy::Custom:
            return selector.customPredicate && selector.customPredicate(widget);
            
        default:
            return false;
    }
}

QPoint UIAutomation::getWidgetCenter(QWidget* widget) {
    if (!widget) {
        return QPoint();
    }
    
    QRect rect = widget->rect();
    return rect.center();
}

QPoint UIAutomation::getWidgetClickPoint(QWidget* widget, const QPoint& offset) {
    QPoint center = getWidgetCenter(widget);
    return center + offset;
}

bool UIAutomation::isWidgetInteractable(QWidget* widget) {
    return widget && widget->isVisible() && widget->isEnabled();
}

void UIAutomation::logAction(const QString& action, const QString& details, bool success) {
    if (m_detailedLogging) {
        QString logMessage = QString("[UI] %1: %2 - %3")
                            .arg(action)
                            .arg(details)
                            .arg(success ? "SUCCESS" : "FAILED");
        qDebug() << logMessage;
    }
    
    m_actionHistory.append(QString("%1: %2").arg(action).arg(details));
    if (m_actionHistory.size() > 100) {
        m_actionHistory.removeFirst();
    }
}

void UIAutomation::takeAutomaticScreenshot(const QString& action) {
    if (!m_automaticScreenshots) {
        return;
    }
    
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz");
    QString fileName = QString("%1_%2.png").arg(action).arg(timestamp);
    QString filePath = QDir(m_screenshotDirectory).absoluteFilePath(fileName);
    
    QPixmap screenshot = captureScreen();
    if (!screenshot.isNull()) {
        screenshot.save(filePath);
        emit screenshotCaptured(filePath);
    }
}

bool UIAutomation::retryAction(std::function<bool()> action, int maxAttempts) {
    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        if (action()) {
            return true;
        }
        
        if (attempt < maxAttempts - 1) {
            QThread::msleep(m_defaultDelayMs);
        }
    }
    
    return false;
}

// Widget-specific value setters
bool UIAutomation::setLineEditValue(QLineEdit* lineEdit, const QVariant& value) {
    if (!lineEdit) return false;
    
    lineEdit->clear();
    lineEdit->setText(value.toString());
    return true;
}

bool UIAutomation::setTextEditValue(QTextEdit* textEdit, const QVariant& value) {
    if (!textEdit) return false;
    
    textEdit->clear();
    textEdit->setPlainText(value.toString());
    return true;
}

bool UIAutomation::setComboBoxValue(QComboBox* comboBox, const QVariant& value) {
    if (!comboBox) return false;
    
    // Try to find by text first
    int index = comboBox->findText(value.toString());
    if (index >= 0) {
        comboBox->setCurrentIndex(index);
        return true;
    }
    
    // Try to set by index
    bool ok;
    int intValue = value.toInt(&ok);
    if (ok && intValue >= 0 && intValue < comboBox->count()) {
        comboBox->setCurrentIndex(intValue);
        return true;
    }
    
    return false;
}

bool UIAutomation::setSpinBoxValue(QSpinBox* spinBox, const QVariant& value) {
    if (!spinBox) return false;
    
    bool ok;
    int intValue = value.toInt(&ok);
    if (ok) {
        spinBox->setValue(intValue);
        return true;
    }
    
    return false;
}

bool UIAutomation::setCheckBoxValue(QCheckBox* checkBox, const QVariant& value) {
    if (!checkBox) return false;
    
    checkBox->setChecked(value.toBool());
    return true;
}

// Widget-specific value getters
QVariant UIAutomation::getLineEditValue(QLineEdit* lineEdit) {
    return lineEdit ? lineEdit->text() : QVariant();
}

QVariant UIAutomation::getTextEditValue(QTextEdit* textEdit) {
    return textEdit ? textEdit->toPlainText() : QVariant();
}

QVariant UIAutomation::getComboBoxValue(QComboBox* comboBox) {
    return comboBox ? comboBox->currentText() : QVariant();
}

QVariant UIAutomation::getSpinBoxValue(QSpinBox* spinBox) {
    return spinBox ? spinBox->value() : QVariant();
}

QVariant UIAutomation::getCheckBoxValue(QCheckBox* checkBox) {
    return checkBox ? checkBox->isChecked() : QVariant();
}

// Wait condition evaluation
bool UIAutomation::evaluateWaitCondition(const WaitSpec& waitSpec) {
    switch (waitSpec.condition) {
        case WaitCondition::WidgetVisible: {
            QWidget* widget = findWidget(waitSpec.selector);
            return widget && widget->isVisible();
        }
        
        case WaitCondition::WidgetHidden: {
            QWidget* widget = findWidget(waitSpec.selector);
            return !widget || !widget->isVisible();
        }
        
        case WaitCondition::WidgetEnabled: {
            QWidget* widget = findWidget(waitSpec.selector);
            return widget && widget->isEnabled();
        }
        
        case WaitCondition::TextChanged: {
            QWidget* widget = findWidget(waitSpec.selector);
            if (!widget) return false;
            
            QString currentText;
            if (auto* label = qobject_cast<QLabel*>(widget)) {
                currentText = label->text();
            } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
                currentText = lineEdit->text();
            } else {
                currentText = widget->property("text").toString();
            }
            
            return currentText == waitSpec.expectedValue;
        }
        
        case WaitCondition::CustomCondition:
            return waitSpec.customCondition && waitSpec.customCondition();
            
        default:
            return false;
    }
}