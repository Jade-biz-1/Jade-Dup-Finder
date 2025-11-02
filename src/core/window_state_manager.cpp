#include "window_state_manager.h"
#include "logger.h"
#include <QApplication>
#include <QScreen>
#include <QWindow>
#include <QMainWindow>
#include <QDialog>
#include <QCloseEvent>
#include <QMoveEvent>
#include <QResizeEvent>
#include <QShowEvent>
#include <QDebug>

// Static member initialization
WindowStateManager* WindowStateManager::s_instance = nullptr;
const QString WindowStateManager::DEFAULT_SETTINGS_GROUP = "WindowStates";

WindowStateManager* WindowStateManager::instance()
{
    if (!s_instance) {
        s_instance = new WindowStateManager();
    }
    return s_instance;
}

WindowStateManager::WindowStateManager(QObject* parent)
    : QObject(parent)
    , m_settings(new QSettings(this))
    , m_settingsGroup(DEFAULT_SETTINGS_GROUP)
    , m_autoSaveEnabled(true)
    , m_saveTimer(new QTimer(this))
{
    m_saveTimer->setSingleShot(true);
    m_saveTimer->setInterval(SAVE_DELAY_MS);
    connect(m_saveTimer, &QTimer::timeout, this, &WindowStateManager::saveDelayedStates);
    
    // Connect to application exit to save all states
    connect(qApp, &QApplication::aboutToQuit, this, &WindowStateManager::onApplicationExit);
    
    LOG_DEBUG(LogCategories::UI, "WindowStateManager initialized");
}

WindowStateManager::~WindowStateManager()
{
    // Save all states before destruction
    saveAllWindowStates();
    LOG_DEBUG(LogCategories::UI, "WindowStateManager destroyed");
}

void WindowStateManager::registerWindow(QWidget* window, const QString& identifier)
{
    if (!window) {
        LOG_WARNING(LogCategories::UI, "Attempted to register null window");
        return;
    }

    QString windowId = getWindowIdentifier(window, identifier);
    
    WindowInfo info;
    info.window = window;
    info.identifier = windowId;
    info.isDialog = qobject_cast<QDialog*>(window) != nullptr;
    info.lastGeometry = window->geometry();
    info.lastState = window->windowState();

    m_registeredWindows[window] = info;
    
    // Connect signals for automatic state tracking
    connectWindowSignals(window);
    
    // Try to restore state immediately if window is already visible
    if (window->isVisible()) {
        restoreWindowState(window);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Registered window: %1 (ID: %2)").arg(window->objectName()).arg(windowId));
}

void WindowStateManager::registerDialog(QDialog* dialog, const QString& identifier)
{
    registerWindow(dialog, identifier);
}

void WindowStateManager::unregisterWindow(QWidget* window)
{
    if (!window || !m_registeredWindows.contains(window)) {
        return;
    }

    // Save state before unregistering
    if (m_autoSaveEnabled) {
        saveWindowState(window);
    }

    disconnectWindowSignals(window);
    m_registeredWindows.remove(window);
    m_pendingSaves.remove(window);
    
    LOG_DEBUG(LogCategories::UI, QString("Unregistered window: %1").arg(window->objectName()));
}

void WindowStateManager::saveWindowState(QWidget* window)
{
    if (!window || !m_registeredWindows.contains(window)) {
        return;
    }

    const WindowInfo& info = m_registeredWindows[window];
    
    // Don't save if window is minimized or not visible
    if (window->isMinimized() || !window->isVisible()) {
        return;
    }

    saveWindowGeometry(info.identifier, window);
    
    // Update cached geometry
    m_registeredWindows[window].lastGeometry = window->geometry();
    m_registeredWindows[window].lastState = window->windowState();
    
    LOG_DEBUG(LogCategories::UI, QString("Saved state for window: %1").arg(info.identifier));
}

bool WindowStateManager::restoreWindowState(QWidget* window)
{
    if (!window || !m_registeredWindows.contains(window)) {
        return false;
    }

    const WindowInfo& info = m_registeredWindows[window];
    return restoreWindowGeometry(info.identifier, window);
}

void WindowStateManager::saveAllWindowStates()
{
    if (!m_autoSaveEnabled) {
        return;
    }

    int savedCount = 0;
    for (auto it = m_registeredWindows.begin(); it != m_registeredWindows.end(); ++it) {
        QWidget* window = it.key();
        if (window && window->isVisible() && !window->isMinimized()) {
            saveWindowState(window);
            savedCount++;
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Saved states for %1 windows").arg(savedCount));
}

void WindowStateManager::setAutoSaveEnabled(bool enabled)
{
    m_autoSaveEnabled = enabled;
    LOG_DEBUG(LogCategories::UI, QString("Auto-save %1").arg(enabled ? "enabled" : "disabled"));
}

bool WindowStateManager::isAutoSaveEnabled() const
{
    return m_autoSaveEnabled;
}

void WindowStateManager::setSettingsGroup(const QString& groupName)
{
    m_settingsGroup = groupName.isEmpty() ? DEFAULT_SETTINGS_GROUP : groupName;
    LOG_DEBUG(LogCategories::UI, QString("Settings group set to: %1").arg(m_settingsGroup));
}

QString WindowStateManager::settingsGroup() const
{
    return m_settingsGroup;
}

void WindowStateManager::clearAllSavedStates()
{
    m_settings->beginGroup(m_settingsGroup);
    m_settings->remove("");  // Remove all keys in the group
    m_settings->endGroup();
    
    LOG_INFO(LogCategories::UI, "Cleared all saved window states");
}

bool WindowStateManager::hasSavedState(const QString& identifier) const
{
    m_settings->beginGroup(m_settingsGroup);
    bool exists = m_settings->contains(identifier + "/geometry");
    m_settings->endGroup();
    return exists;
}

void WindowStateManager::onApplicationExit()
{
    LOG_INFO(LogCategories::UI, "Application exiting, saving all window states");
    saveAllWindowStates();
}

void WindowStateManager::onWindowDestroyed(QObject* window)
{
    QWidget* widget = static_cast<QWidget*>(window);
    if (m_registeredWindows.contains(widget)) {
        m_registeredWindows.remove(widget);
        m_pendingSaves.remove(widget);
    }
}

void WindowStateManager::saveDelayedStates()
{
    for (auto it = m_pendingSaves.begin(); it != m_pendingSaves.end(); ++it) {
        if (it.value()) {  // If save is pending
            saveWindowState(it.key());
        }
    }
    m_pendingSaves.clear();
}

QString WindowStateManager::getWindowIdentifier(QWidget* window, const QString& customId) const
{
    if (!customId.isEmpty()) {
        return customId;
    }

    QString className = window->metaObject()->className();
    QString objectName = window->objectName();
    
    if (!objectName.isEmpty()) {
        return QString("%1_%2").arg(className).arg(objectName);
    }
    
    return className;
}

void WindowStateManager::saveWindowGeometry(const QString& identifier, QWidget* window)
{
    m_settings->beginGroup(m_settingsGroup);
    
    // Save geometry
    m_settings->setValue(identifier + "/geometry", window->saveGeometry());
    
    // Save window state for main windows
    if (QMainWindow* mainWindow = qobject_cast<QMainWindow*>(window)) {
        m_settings->setValue(identifier + "/windowState", mainWindow->saveState());
    }
    
    // Save additional properties
    m_settings->setValue(identifier + "/maximized", window->isMaximized());
    m_settings->setValue(identifier + "/position", window->pos());
    m_settings->setValue(identifier + "/size", window->size());
    
    m_settings->endGroup();
}

bool WindowStateManager::restoreWindowGeometry(const QString& identifier, QWidget* window)
{
    m_settings->beginGroup(m_settingsGroup);
    
    bool restored = false;
    
    // Try to restore from saved geometry first (most reliable)
    QByteArray geometry = m_settings->value(identifier + "/geometry").toByteArray();
    if (!geometry.isEmpty()) {
        restored = window->restoreGeometry(geometry);
        
        // Restore window state for main windows
        if (QMainWindow* mainWindow = qobject_cast<QMainWindow*>(window)) {
            QByteArray windowState = m_settings->value(identifier + "/windowState").toByteArray();
            if (!windowState.isEmpty()) {
                mainWindow->restoreState(windowState);
            }
        }
    }
    
    // Fallback to manual position/size restoration
    if (!restored) {
        QPoint position = m_settings->value(identifier + "/position").toPoint();
        QSize size = m_settings->value(identifier + "/size").toSize();
        bool wasMaximized = m_settings->value(identifier + "/maximized", false).toBool();
        
        if (!position.isNull() && !size.isEmpty()) {
            QRect geometry(position, size);
            geometry = ensureValidGeometry(geometry);
            
            window->setGeometry(geometry);
            
            if (wasMaximized) {
                window->showMaximized();
            }
            
            restored = true;
        }
    }
    
    m_settings->endGroup();
    
    if (restored) {
        LOG_DEBUG(LogCategories::UI, QString("Restored state for window: %1").arg(identifier));
    }
    
    return restored;
}

bool WindowStateManager::isValidGeometry(const QRect& geometry) const
{
    // Check if geometry is within any available screen
    QList<QScreen*> screens = QApplication::screens();
    for (QScreen* screen : screens) {
        QRect screenGeometry = screen->availableGeometry();
        if (screenGeometry.intersects(geometry)) {
            return true;
        }
    }
    return false;
}

QRect WindowStateManager::ensureValidGeometry(const QRect& geometry) const
{
    if (isValidGeometry(geometry)) {
        return geometry;
    }
    
    // If geometry is not valid, move it to the primary screen
    QScreen* primaryScreen = QApplication::primaryScreen();
    if (!primaryScreen) {
        return geometry;  // Fallback to original if no primary screen
    }
    
    QRect screenGeometry = primaryScreen->availableGeometry();
    QRect validGeometry = geometry;
    
    // Ensure the window is not larger than the screen
    if (validGeometry.width() > screenGeometry.width()) {
        validGeometry.setWidth(static_cast<int>(screenGeometry.width() * 0.9));  // 90% of screen width
    }
    if (validGeometry.height() > screenGeometry.height()) {
        validGeometry.setHeight(static_cast<int>(screenGeometry.height() * 0.9));  // 90% of screen height
    }
    
    // Ensure the window is visible on screen
    if (validGeometry.right() < screenGeometry.left() || 
        validGeometry.left() > screenGeometry.right() ||
        validGeometry.bottom() < screenGeometry.top() || 
        validGeometry.top() > screenGeometry.bottom()) {
        
        // Move to center of primary screen
        validGeometry.moveCenter(screenGeometry.center());
    }
    
    // Ensure at least part of the title bar is visible
    if (validGeometry.top() < screenGeometry.top()) {
        validGeometry.moveTop(screenGeometry.top());
    }
    
    return validGeometry;
}

void WindowStateManager::connectWindowSignals(QWidget* window)
{
    // Connect to destroyed signal to clean up
    connect(window, &QObject::destroyed, this, &WindowStateManager::onWindowDestroyed);
    
    // Install event filter to track geometry changes
    window->installEventFilter(this);
}

void WindowStateManager::disconnectWindowSignals(QWidget* window)
{
    disconnect(window, &QObject::destroyed, this, &WindowStateManager::onWindowDestroyed);
    window->removeEventFilter(this);
}

bool WindowStateManager::eventFilter(QObject* watched, QEvent* event)
{
    QWidget* window = qobject_cast<QWidget*>(watched);
    if (!window || !m_registeredWindows.contains(window)) {
        return QObject::eventFilter(watched, event);
    }

    switch (event->type()) {
        case QEvent::Move:
        case QEvent::Resize:
            if (m_autoSaveEnabled && window->isVisible() && !window->isMinimized()) {
                // Schedule delayed save to avoid too frequent saves during dragging
                m_pendingSaves[window] = true;
                m_saveTimer->start();
            }
            break;
            
        case QEvent::Show:
            // Try to restore state when window is first shown
            if (!hasSavedState(m_registeredWindows[window].identifier)) {
                // No saved state, let Qt handle default positioning
            } else {
                restoreWindowState(window);
            }
            break;
            
        case QEvent::Close:
            if (m_autoSaveEnabled) {
                saveWindowState(window);
            }
            break;
            
        default:
            break;
    }

    return QObject::eventFilter(watched, event);
}