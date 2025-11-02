#include "component_registry.h"
#include "core/logger.h"
#include <QMutexLocker>
#include <QApplication>
#include <QDialog>

ComponentRegistry::ComponentRegistry(QObject* parent)
    : QObject(parent)
    , m_realTimeUpdateTimer(new QTimer(this))
    , m_monitoringTimer(new QTimer(this))
    , m_realTimeUpdatesEnabled(false)
    , m_monitoringEnabled(false)
    , m_updateInterval(100)
    , m_lastCleanup(QDateTime::currentDateTime())
{
    // Setup real-time update timer
    m_realTimeUpdateTimer->setSingleShot(false);
    m_realTimeUpdateTimer->setInterval(m_updateInterval);
    connect(m_realTimeUpdateTimer, &QTimer::timeout, this, &ComponentRegistry::onRealTimeUpdate);
    
    // Setup monitoring timer (runs every 5 seconds)
    m_monitoringTimer->setSingleShot(false);
    m_monitoringTimer->setInterval(5000);
    connect(m_monitoringTimer, &QTimer::timeout, this, &ComponentRegistry::onMonitoringUpdate);
    
    LOG_INFO(LogCategories::UI, "ComponentRegistry initialized with enhanced capabilities");
}

ComponentRegistry::~ComponentRegistry()
{
    stopMonitoring();
    enableRealTimeUpdates(false);
    LOG_INFO(LogCategories::UI, "ComponentRegistry destroyed");
}

void ComponentRegistry::registerComponent(QWidget* component, ThemeManager::ComponentType type)
{
    if (!component) return;
    
    QMutexLocker locker(&m_mutex);
    
    // Check if already registered
    if (m_components.contains(component)) {
        LOG_DEBUG(LogCategories::UI, QString("Component %1 already registered, updating type")
                 .arg(component->metaObject()->className()));
        m_components[component].type = type;
        m_components[component].lastUpdated = QDateTime::currentDateTime();
        return;
    }
    
    // Create component info
    ComponentInfo info;
    info.widget = QPointer<QWidget>(component);
    info.type = type;
    info.registered = QDateTime::currentDateTime();
    info.lastUpdated = info.registered;
    info.isValid = true;
    info.updateFailed = false;
    info.objectName = component->objectName();
    info.className = component->metaObject()->className();
    info.failureCount = 0;
    
    // Add to registry
    m_components[component] = info;
    
    // Connect to destruction signal
    connect(component, &QObject::destroyed, this, &ComponentRegistry::onComponentDestroyed);
    
    LOG_DEBUG(LogCategories::UI, QString("Registered component %1 (%2) with type %3")
             .arg(info.className)
             .arg(info.objectName.isEmpty() ? "unnamed" : info.objectName)
             .arg(static_cast<int>(type)));
    
    emit componentRegistered(component, type);
}

void ComponentRegistry::registerDialog(QDialog* dialog)
{
    if (!dialog) return;
    
    QMutexLocker locker(&m_mutex);
    
    // Check if already registered
    if (m_dialogs.contains(dialog)) {
        LOG_DEBUG(LogCategories::UI, QString("Dialog %1 already registered")
                 .arg(dialog->metaObject()->className()));
        m_dialogs[dialog].lastUpdated = QDateTime::currentDateTime();
        return;
    }
    
    // Create dialog info
    DialogInfo info;
    info.dialog = QPointer<QDialog>(dialog);
    info.registered = QDateTime::currentDateTime();
    info.lastUpdated = info.registered;
    info.isValid = true;
    info.updateFailed = false;
    info.objectName = dialog->objectName();
    info.className = dialog->metaObject()->className();
    info.failureCount = 0;
    
    // Add to registry
    m_dialogs[dialog] = info;
    
    // Connect to destruction signal
    connect(dialog, &QObject::destroyed, this, &ComponentRegistry::onDialogDestroyed);
    
    LOG_DEBUG(LogCategories::UI, QString("Registered dialog %1 (%2)")
             .arg(info.className)
             .arg(info.objectName.isEmpty() ? "unnamed" : info.objectName));
    
    emit dialogRegistered(dialog);
}

void ComponentRegistry::unregisterComponent(QWidget* component)
{
    if (!component) return;
    
    QMutexLocker locker(&m_mutex);
    
    if (m_components.contains(component)) {
        QString className = m_components[component].className;
        QString objectName = m_components[component].objectName;
        
        m_components.remove(component);
        
        // Remove from failed components list if present
        QString componentId = QString("%1::%2").arg(className).arg(objectName);
        m_failedComponents.removeAll(componentId);
        
        LOG_DEBUG(LogCategories::UI, QString("Unregistered component %1 (%2)")
                 .arg(className)
                 .arg(objectName.isEmpty() ? "unnamed" : objectName));
        
        emit componentUnregistered(component);
    }
}

void ComponentRegistry::unregisterDialog(QDialog* dialog)
{
    if (!dialog) return;
    
    QMutexLocker locker(&m_mutex);
    
    if (m_dialogs.contains(dialog)) {
        QString className = m_dialogs[dialog].className;
        QString objectName = m_dialogs[dialog].objectName;
        
        m_dialogs.remove(dialog);
        
        // Remove from failed components list if present
        QString dialogId = QString("%1::%2").arg(className).arg(objectName);
        m_failedComponents.removeAll(dialogId);
        
        LOG_DEBUG(LogCategories::UI, QString("Unregistered dialog %1 (%2)")
                 .arg(className)
                 .arg(objectName.isEmpty() ? "unnamed" : objectName));
        
        emit dialogUnregistered(dialog);
    }
}

QList<QWidget*> ComponentRegistry::getComponentsByType(ThemeManager::ComponentType type) const
{
    QMutexLocker locker(&m_mutex);
    
    QList<QWidget*> result;
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (it.value().type == type && !it.value().widget.isNull()) {
            result.append(it.value().widget.data());
        }
    }
    
    return result;
}

QList<QWidget*> ComponentRegistry::getAllComponents() const
{
    QMutexLocker locker(&m_mutex);
    
    QList<QWidget*> result;
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (!it.value().widget.isNull()) {
            result.append(it.value().widget.data());
        }
    }
    
    return result;
}

QList<QDialog*> ComponentRegistry::getAllDialogs() const
{
    QMutexLocker locker(&m_mutex);
    
    QList<QDialog*> result;
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (!it.value().dialog.isNull()) {
            result.append(it.value().dialog.data());
        }
    }
    
    return result;
}

void ComponentRegistry::applyThemeToAll(const ThemeData& theme)
{
    QMutexLocker locker(&m_mutex);
    
    int appliedCount = 0;
    int failedCount = 0;
    
    // Apply theme to components
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (!it.value().widget.isNull()) {
            try {
                applyThemeToComponent(it.value().widget.data(), theme);
                updateComponentInfo(it.key(), true);
                appliedCount++;
            } catch (const std::exception& e) {
                updateComponentInfo(it.key(), false, QString::fromStdString(e.what()));
                failedCount++;
                LOG_WARNING(LogCategories::UI, QString("Failed to apply theme to component %1: %2")
                           .arg(it.value().className).arg(e.what()));
            }
        }
    }
    
    // Apply theme to dialogs
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (!it.value().dialog.isNull()) {
            try {
                applyThemeToDialog(it.value().dialog.data(), theme);
                updateDialogInfo(it.key(), true);
                appliedCount++;
            } catch (const std::exception& e) {
                updateDialogInfo(it.key(), false, QString::fromStdString(e.what()));
                failedCount++;
                LOG_WARNING(LogCategories::UI, QString("Failed to apply theme to dialog %1: %2")
                           .arg(it.value().className).arg(e.what()));
            }
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Applied theme to %1 components/dialogs, %2 failed")
             .arg(appliedCount).arg(failedCount));
    
    emit themeAppliedToAll();
}

void ComponentRegistry::validateAllComponents()
{
    QMutexLocker locker(&m_mutex);
    
    cleanupInvalidComponents();
    
    int validCount = 0;
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (!it.value().widget.isNull()) {
            validCount++;
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Validated components: %1 valid, %2 total")
             .arg(validCount).arg(m_components.size()));
}

int ComponentRegistry::getComponentCount() const
{
    QMutexLocker locker(&m_mutex);
    
    int validCount = 0;
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (!it.value().widget.isNull()) {
            validCount++;
        }
    }
    
    return validCount;
}

int ComponentRegistry::getComponentCountByType(ThemeManager::ComponentType type) const
{
    QMutexLocker locker(&m_mutex);
    
    int count = 0;
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (it.value().type == type && !it.value().widget.isNull()) {
            count++;
        }
    }
    
    return count;
}

void ComponentRegistry::onComponentDestroyed(QObject* obj)
{
    QMutexLocker locker(&m_mutex);
    
    // Find and remove the destroyed component
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (it.key() == obj || it.value().widget.isNull()) {
            QString className = it.value().className;
            QString objectName = it.value().objectName;
            
            // Remove from failed components list
            QString componentId = QString("%1::%2").arg(className).arg(objectName);
            m_failedComponents.removeAll(componentId);
            
            m_components.erase(it);
            
            LOG_DEBUG(LogCategories::UI, QString("Auto-removed destroyed component %1 (%2)")
                     .arg(className)
                     .arg(objectName.isEmpty() ? "unnamed" : objectName));
            break;
        }
    }
}

void ComponentRegistry::onDialogDestroyed(QObject* obj)
{
    QMutexLocker locker(&m_mutex);
    
    // Find and remove the destroyed dialog
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (it.key() == obj || it.value().dialog.isNull()) {
            QString className = it.value().className;
            QString objectName = it.value().objectName;
            
            // Remove from failed components list
            QString dialogId = QString("%1::%2").arg(className).arg(objectName);
            m_failedComponents.removeAll(dialogId);
            
            m_dialogs.erase(it);
            
            LOG_DEBUG(LogCategories::UI, QString("Auto-removed destroyed dialog %1 (%2)")
                     .arg(className)
                     .arg(objectName.isEmpty() ? "unnamed" : objectName));
            break;
        }
    }
}

void ComponentRegistry::cleanupInvalidComponents()
{
    // Remove null pointers and invalid components
    for (auto it = m_components.begin(); it != m_components.end();) {
        if (it.value().widget.isNull()) {
            it = m_components.erase(it);
        } else {
            ++it;
        }
    }
}

void ComponentRegistry::applyThemeToComponent(QWidget* component, const ThemeData& theme)
{
    if (!component) return;
    
    // Find the component type
    ThemeManager::ComponentType type = ThemeManager::ComponentType::Widget;
    
    QMutexLocker locker(&m_mutex);
    if (m_components.contains(component)) {
        type = m_components[component].type;
    }
    locker.unlock();
    
    // Generate and apply style sheet
    QString styleSheet = generateComponentStyleSheet(type, theme);
    component->setStyleSheet(styleSheet);
    component->update();
}

QString ComponentRegistry::generateComponentStyleSheet(ThemeManager::ComponentType type, const ThemeData& theme) const
{
    QString styleSheet;
    
    switch (type) {
        case ThemeManager::ComponentType::Button:
            styleSheet = QString(R"(
                QPushButton {
                    background-color: %1;
                    color: %2;
                    border: %3px solid %4;
                    border-radius: %5px;
                    padding: %6px;
                    min-height: 24px;
                    font-family: %7;
                    font-size: %8pt;
                }
                QPushButton:hover {
                    background-color: %9;
                }
                QPushButton:pressed {
                    background-color: %10;
                }
                QPushButton:disabled {
                    background-color: %11;
                    color: %12;
                }
            )").arg(theme.colors.accent.name())
               .arg(theme.colors.background.name())
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.spacing.borderRadius)
               .arg(theme.spacing.padding)
               .arg(theme.typography.fontFamily)
               .arg(theme.typography.baseFontSize)
               .arg(theme.colors.hover.name())
               .arg(theme.colors.accent.darker(120).name())
               .arg(theme.colors.disabled.name())
               .arg(theme.colors.disabled.darker(150).name());
            break;
            
        case ThemeManager::ComponentType::LineEdit:
            styleSheet = QString(R"(
                QLineEdit {
                    background-color: %1;
                    color: %2;
                    border: %3px solid %4;
                    border-radius: %5px;
                    padding: %6px;
                    min-height: 20px;
                    font-family: %7;
                    font-size: %8pt;
                }
                QLineEdit:focus {
                    border-color: %9;
                    border-width: 2px;
                }
            )").arg(theme.colors.background.name())
               .arg(theme.colors.foreground.name())
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.spacing.borderRadius)
               .arg(theme.spacing.padding)
               .arg(theme.typography.fontFamily)
               .arg(theme.typography.baseFontSize)
               .arg(theme.colors.accent.name());
            break;
            
        case ThemeManager::ComponentType::CheckBox:
            styleSheet = QString(R"(
                QCheckBox {
                    color: %1;
                    spacing: 8px;
                    font-family: %2;
                    font-size: %3pt;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: %4px solid %5;
                    border-radius: 3px;
                    background-color: %6;
                }
                QCheckBox::indicator:checked {
                    background-color: %7;
                    border-color: %7;
                }
                QCheckBox::indicator:hover {
                    border-color: %8;
                }
                QCheckBox::indicator:disabled {
                    background-color: %9;
                    border-color: %10;
                }
            )").arg(theme.colors.foreground.name())
               .arg(theme.typography.fontFamily)
               .arg(theme.typography.baseFontSize)
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.colors.background.name())
               .arg(theme.colors.accent.name())
               .arg(theme.colors.hover.name())
               .arg(theme.colors.disabled.name())
               .arg(theme.colors.disabled.darker(120).name());
            break;
            
        case ThemeManager::ComponentType::ComboBox:
            styleSheet = QString(R"(
                QComboBox {
                    background-color: %1;
                    color: %2;
                    border: %3px solid %4;
                    border-radius: %5px;
                    padding: %6px;
                    min-height: 24px;
                    font-family: %7;
                    font-size: %8pt;
                }
                QComboBox:hover {
                    border-color: %9;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox QAbstractItemView {
                    background-color: %1;
                    color: %2;
                    selection-background-color: %10;
                }
            )").arg(theme.colors.background.name())
               .arg(theme.colors.foreground.name())
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.spacing.borderRadius)
               .arg(theme.spacing.padding)
               .arg(theme.typography.fontFamily)
               .arg(theme.typography.baseFontSize)
               .arg(theme.colors.accent.name())
               .arg(theme.colors.hover.name());
            break;
            
        case ThemeManager::ComponentType::ProgressBar:
            styleSheet = QString(R"(
                QProgressBar {
                    background-color: %1;
                    color: %2;
                    border: %3px solid %4;
                    border-radius: %5px;
                    text-align: center;
                    font-family: %6;
                    font-size: %7pt;
                    font-weight: bold;
                    min-height: 20px;
                }
                QProgressBar::chunk {
                    background-color: %8;
                    border-radius: %9px;
                }
            )").arg(theme.colors.background.name())
               .arg(theme.colors.foreground.name())
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.spacing.borderRadius)
               .arg(theme.typography.fontFamily)
               .arg(theme.typography.baseFontSize)
               .arg(theme.colors.accent.name())
               .arg(theme.spacing.borderRadius - 1);
            break;
            
        default:
            styleSheet = QString(R"(
                QWidget {
                    background-color: %1;
                    color: %2;
                    font-family: %3;
                    font-size: %4pt;
                }
            )").arg(theme.colors.background.name())
               .arg(theme.colors.foreground.name())
               .arg(theme.typography.fontFamily)
               .arg(theme.typography.baseFontSize);
            break;
    }
    
    return styleSheet;
}
void ComponentRegistry::applyThemeToDialog(QDialog* dialog, const ThemeData& theme)
{
    if (!dialog) return;
    
    // Generate and apply style sheet for dialog
    QString styleSheet = generateDialogStyleSheet(theme);
    dialog->setStyleSheet(styleSheet);
    
    // Apply theme to all child widgets recursively
    QList<QWidget*> children = dialog->findChildren<QWidget*>();
    for (QWidget* child : children) {
        if (child) {
            // Determine component type for child widget
            ThemeManager::ComponentType childType = determineComponentType(child);
            QString childStyleSheet = generateComponentStyleSheet(childType, theme);
            child->setStyleSheet(childStyleSheet);
        }
    }
    
    dialog->update();
}

void ComponentRegistry::enableRealTimeUpdates(bool enabled)
{
    m_realTimeUpdatesEnabled = enabled;
    
    if (enabled) {
        m_realTimeUpdateTimer->start();
        LOG_INFO(LogCategories::UI, QString("Real-time theme updates enabled (interval: %1ms)")
                 .arg(m_updateInterval));
    } else {
        m_realTimeUpdateTimer->stop();
        LOG_INFO(LogCategories::UI, "Real-time theme updates disabled");
    }
}

void ComponentRegistry::setUpdateInterval(int milliseconds)
{
    m_updateInterval = qMax(50, milliseconds); // Minimum 50ms to prevent excessive updates
    m_realTimeUpdateTimer->setInterval(m_updateInterval);
    
    LOG_DEBUG(LogCategories::UI, QString("Real-time update interval set to %1ms").arg(m_updateInterval));
}

void ComponentRegistry::forceUpdateAll()
{
    emit realTimeUpdateTriggered();
    LOG_DEBUG(LogCategories::UI, "Forced real-time update triggered");
}

void ComponentRegistry::startMonitoring()
{
    m_monitoringEnabled = true;
    m_monitoringTimer->start();
    LOG_INFO(LogCategories::UI, "Component monitoring started");
}

void ComponentRegistry::stopMonitoring()
{
    m_monitoringEnabled = false;
    m_monitoringTimer->stop();
    LOG_INFO(LogCategories::UI, "Component monitoring stopped");
}

bool ComponentRegistry::isMonitoring() const
{
    return m_monitoringEnabled;
}

void ComponentRegistry::attemptRecovery()
{
    QMutexLocker locker(&m_mutex);
    
    int recoveredCount = 0;
    
    // Attempt to recover failed components
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (it.value().updateFailed && !it.value().widget.isNull()) {
            try {
                // Reset failure state and try again
                it.value().updateFailed = false;
                it.value().failureCount = 0;
                it.value().lastError.clear();
                
                recoveredCount++;
                LOG_DEBUG(LogCategories::UI, QString("Recovered component %1").arg(it.value().className));
            } catch (...) {
                // Recovery failed, increment failure count
                it.value().failureCount++;
            }
        }
    }
    
    // Attempt to recover failed dialogs
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (it.value().updateFailed && !it.value().dialog.isNull()) {
            try {
                // Reset failure state and try again
                it.value().updateFailed = false;
                it.value().failureCount = 0;
                it.value().lastError.clear();
                
                recoveredCount++;
                LOG_DEBUG(LogCategories::UI, QString("Recovered dialog %1").arg(it.value().className));
            } catch (...) {
                // Recovery failed, increment failure count
                it.value().failureCount++;
            }
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Recovery attempt completed: %1 components/dialogs recovered")
             .arg(recoveredCount));
}

QStringList ComponentRegistry::getFailedComponents() const
{
    QMutexLocker locker(&m_mutex);
    return m_failedComponents;
}

void ComponentRegistry::retryFailedComponents(const ThemeData& theme)
{
    QMutexLocker locker(&m_mutex);
    
    int retryCount = 0;
    int successCount = 0;
    
    // Retry failed components
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (it.value().updateFailed && !it.value().widget.isNull()) {
            retryCount++;
            try {
                applyThemeToComponent(it.value().widget.data(), theme);
                updateComponentInfo(it.key(), true);
                successCount++;
            } catch (const std::exception& e) {
                updateComponentInfo(it.key(), false, QString::fromStdString(e.what()));
            }
        }
    }
    
    // Retry failed dialogs
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (it.value().updateFailed && !it.value().dialog.isNull()) {
            retryCount++;
            try {
                applyThemeToDialog(it.value().dialog.data(), theme);
                updateDialogInfo(it.key(), true);
                successCount++;
            } catch (const std::exception& e) {
                updateDialogInfo(it.key(), false, QString::fromStdString(e.what()));
            }
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Retried %1 failed components/dialogs, %2 succeeded")
             .arg(retryCount).arg(successCount));
}

int ComponentRegistry::getDialogCount() const
{
    QMutexLocker locker(&m_mutex);
    
    int validCount = 0;
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (!it.value().dialog.isNull()) {
            validCount++;
        }
    }
    
    return validCount;
}

void ComponentRegistry::onRealTimeUpdate()
{
    if (!m_realTimeUpdatesEnabled) return;
    
    emit realTimeUpdateTriggered();
}

void ComponentRegistry::onMonitoringUpdate()
{
    if (!m_monitoringEnabled) return;
    
    QMutexLocker locker(&m_mutex);
    
    // Clean up invalid components periodically
    QDateTime now = QDateTime::currentDateTime();
    if (m_lastCleanup.secsTo(now) > 300) { // Clean up every 5 minutes
        cleanupInvalidComponents();
        m_lastCleanup = now;
    }
    
    // Count valid components
    int validComponents = 0;
    int validDialogs = 0;
    
    for (auto it = m_components.begin(); it != m_components.end(); ++it) {
        if (!it.value().widget.isNull()) {
            validComponents++;
        }
    }
    
    for (auto it = m_dialogs.begin(); it != m_dialogs.end(); ++it) {
        if (!it.value().dialog.isNull()) {
            validDialogs++;
        }
    }
    
    emit componentValidationCompleted(validComponents + validDialogs, 
                                    m_components.size() + m_dialogs.size());
}

void ComponentRegistry::updateComponentInfo(QWidget* component, bool success, const QString& error)
{
    if (!component || !m_components.contains(component)) return;
    
    ComponentInfo& info = m_components[component];
    info.lastUpdated = QDateTime::currentDateTime();
    info.updateFailed = !success;
    info.lastError = error;
    
    if (!success) {
        info.failureCount++;
        QString componentId = QString("%1::%2").arg(info.className).arg(info.objectName);
        if (!m_failedComponents.contains(componentId)) {
            m_failedComponents.append(componentId);
        }
        emit themeApplicationFailed(component, error);
    } else {
        info.failureCount = 0;
        QString componentId = QString("%1::%2").arg(info.className).arg(info.objectName);
        m_failedComponents.removeAll(componentId);
    }
}

void ComponentRegistry::updateDialogInfo(QDialog* dialog, bool success, const QString& error)
{
    if (!dialog || !m_dialogs.contains(dialog)) return;
    
    DialogInfo& info = m_dialogs[dialog];
    info.lastUpdated = QDateTime::currentDateTime();
    info.updateFailed = !success;
    info.lastError = error;
    
    if (!success) {
        info.failureCount++;
        QString dialogId = QString("%1::%2").arg(info.className).arg(info.objectName);
        if (!m_failedComponents.contains(dialogId)) {
            m_failedComponents.append(dialogId);
        }
        emit themeApplicationFailed(dialog, error);
    } else {
        info.failureCount = 0;
        QString dialogId = QString("%1::%2").arg(info.className).arg(info.objectName);
        m_failedComponents.removeAll(dialogId);
    }
}

QString ComponentRegistry::generateDialogStyleSheet(const ThemeData& theme) const
{
    return QString(R"(
        QDialog {
            background-color: %1;
            color: %2;
            font-family: %3;
            font-size: %4pt;
        }
        QDialog QWidget {
            background-color: %1;
            color: %2;
        }
    )").arg(theme.colors.background.name())
       .arg(theme.colors.foreground.name())
       .arg(theme.typography.fontFamily)
       .arg(theme.typography.baseFontSize);
}

ThemeManager::ComponentType ComponentRegistry::determineComponentType(QWidget* widget) const
{
    if (!widget) return ThemeManager::ComponentType::Widget;
    
    QString className = widget->metaObject()->className();
    
    if (className == "QPushButton") return ThemeManager::ComponentType::Button;
    if (className == "QLineEdit") return ThemeManager::ComponentType::LineEdit;
    if (className == "QComboBox") return ThemeManager::ComponentType::ComboBox;
    if (className == "QCheckBox") return ThemeManager::ComponentType::CheckBox;
    if (className == "QRadioButton") return ThemeManager::ComponentType::RadioButton;
    if (className == "QLabel") return ThemeManager::ComponentType::Label;
    if (className == "QGroupBox") return ThemeManager::ComponentType::GroupBox;
    if (className == "QTabWidget") return ThemeManager::ComponentType::TabWidget;
    if (className == "QProgressBar") return ThemeManager::ComponentType::ProgressBar;
    if (className == "QSlider") return ThemeManager::ComponentType::Slider;
    if (className == "QTreeWidget" || className == "QTreeView") return ThemeManager::ComponentType::TreeView;
    if (className == "QTableWidget" || className == "QTableView") return ThemeManager::ComponentType::TableView;
    if (className == "QDialog") return ThemeManager::ComponentType::Dialog;
    
    return ThemeManager::ComponentType::Widget;
}