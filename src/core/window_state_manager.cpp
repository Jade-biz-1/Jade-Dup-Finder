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

    // CRITICAL FIX: Always try to restore state immediately upon registration
    // Don't wait for Show event - restore now so the window opens at the saved position
    if (hasSavedState(windowId)) {
        restoreWindowState(window);
        LOG_DEBUG(LogCategories::UI, QString("Restored saved state for registered window: %1").arg(windowId));
    } else {
        LOG_DEBUG(LogCategories::UI, QString("No saved state for registered window: %1").arg(windowId));
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

    // Don't save if window is minimized
    if (window->isMinimized()) {
        return;
    }

    // CRITICAL FIX: Allow saving even if window is not visible
    // During application exit, windows may already be hidden but we still
    // need to save their geometry. Use cached geometry if window is hidden.
    if (!window->isVisible()) {
        // Use cached geometry from last known good state
        if (!info.lastGeometry.isNull() && !info.lastGeometry.isEmpty()) {
            LOG_DEBUG(LogCategories::UI, QString("Saving cached state for hidden window: %1").arg(info.identifier));
            // Save the cached geometry instead of current (which might be invalid)
            m_settings->beginGroup(m_settingsGroup);
            m_settings->setValue(info.identifier + "/position", info.lastGeometry.topLeft());
            m_settings->setValue(info.identifier + "/size", info.lastGeometry.size());
            m_settings->endGroup();
            return;
        }
        return;  // No valid cached geometry
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
        if (window) {
            // CRITICAL FIX: Save all registered windows, regardless of visibility
            // During application exit, windows may already be hidden
            // saveWindowState() will use cached geometry for hidden windows
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

    // CRITICAL FIX: Use geometry() instead of pos() and size() separately
    // On some Linux window managers, pos() returns incorrect values
    // but geometry() is more reliable and consistent
    QRect geometry = window->geometry();
    QPoint position = geometry.topLeft();
    QSize size = geometry.size();

    LOG_INFO(LogCategories::UI, QString("Saving geometry for %1: pos=(%2,%3) size=(%4x%5)")
            .arg(identifier)
            .arg(position.x())
            .arg(position.y())
            .arg(size.width())
            .arg(size.height()));

    m_settings->setValue(identifier + "/maximized", window->isMaximized());
    m_settings->setValue(identifier + "/position", position);
    m_settings->setValue(identifier + "/size", size);

    m_settings->endGroup();
}

bool WindowStateManager::restoreWindowGeometry(const QString& identifier, QWidget* window)
{
    m_settings->beginGroup(m_settingsGroup);

    bool restored = false;

    // Try to restore from saved geometry first (most reliable)
    QByteArray geometry = m_settings->value(identifier + "/geometry").toByteArray();
    if (!geometry.isEmpty()) {
        LOG_INFO(LogCategories::UI, QString("Attempting to restore window %1 from saved geometry ByteArray").arg(identifier));

        restored = window->restoreGeometry(geometry);

        if (restored) {
            // Log the restored geometry
            QRect restoredGeometry = window->geometry();
            LOG_INFO(LogCategories::UI, QString("Window %1 restoreGeometry() returned true, geometry is now: pos=(%2,%3) size=(%4x%5)")
                    .arg(identifier)
                    .arg(restoredGeometry.x())
                    .arg(restoredGeometry.y())
                    .arg(restoredGeometry.width())
                    .arg(restoredGeometry.height()));
        } else {
            LOG_WARNING(LogCategories::UI, QString("Window %1 restoreGeometry() returned false").arg(identifier));
        }

        // CRITICAL FIX: Validate and fix geometry after restore
        // Qt's restoreGeometry may restore to invalid positions (e.g., disconnected monitors)
        QRect currentGeometry = window->geometry();
        if (!isValidGeometry(currentGeometry)) {
            LOG_WARNING(LogCategories::UI, QString("Window %1 restored to invalid position (%2, %3), correcting...")
                       .arg(identifier).arg(currentGeometry.x()).arg(currentGeometry.y()));
            QRect validGeometry = ensureValidGeometry(currentGeometry);
            window->setGeometry(validGeometry);
        }

        // Restore window state for main windows
        if (QMainWindow* mainWindow = qobject_cast<QMainWindow*>(window)) {
            QByteArray windowState = m_settings->value(identifier + "/windowState").toByteArray();
            if (!windowState.isEmpty()) {
                mainWindow->restoreState(windowState);
            }
        }
    } else {
        LOG_INFO(LogCategories::UI, QString("No saved geometry ByteArray found for window %1").arg(identifier));
    }

    // Fallback to manual position/size restoration
    if (!restored) {
        QPoint position = m_settings->value(identifier + "/position").toPoint();
        QSize size = m_settings->value(identifier + "/size").toSize();
        bool wasMaximized = m_settings->value(identifier + "/maximized", false).toBool();

        if (!position.isNull() && !size.isEmpty()) {
            QRect windowGeometry(position, size);
            windowGeometry = ensureValidGeometry(windowGeometry);

            window->setGeometry(windowGeometry);

            if (wasMaximized) {
                window->showMaximized();
            }

            restored = true;
        }
    }

    m_settings->endGroup();

    if (restored) {
        LOG_DEBUG(LogCategories::UI, QString("Restored state for window: %1 at position (%2, %3)")
                 .arg(identifier).arg(window->x()).arg(window->y()));
    }

    return restored;
}

bool WindowStateManager::isValidGeometry(const QRect& geometry) const
{
    // Check if geometry is within any available screen
    QList<QScreen*> screens = QApplication::screens();
    for (QScreen* screen : screens) {
        QRect screenGeometry = screen->availableGeometry();

        // CRITICAL FIX: Check if the window is substantially visible on screen
        // A window is considered valid only if at least 50% of it is on screen
        // This prevents windows from being positioned mostly off-screen
        QRect intersection = screenGeometry.intersected(geometry);
        if (!intersection.isEmpty()) {
            int intersectionArea = intersection.width() * intersection.height();
            int windowArea = geometry.width() * geometry.height();

            // Window must have at least 50% visible on screen
            if (windowArea > 0 && intersectionArea >= (windowArea / 2)) {
                return true;
            }
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
                // CRITICAL FIX: Update cached geometry immediately on move/resize
                // This ensures we have the latest geometry when the window is closed
                QRect newGeometry = window->geometry();
                m_registeredWindows[window].lastGeometry = newGeometry;
                m_registeredWindows[window].lastState = window->windowState();

                const QString& identifier = m_registeredWindows[window].identifier;
                LOG_INFO(LogCategories::UI, QString("Window %1 moved/resized to position (%2, %3) size (%4x%5)")
                        .arg(identifier)
                        .arg(newGeometry.x())
                        .arg(newGeometry.y())
                        .arg(newGeometry.width())
                        .arg(newGeometry.height()));

                // Schedule delayed save to avoid too frequent disk writes during dragging
                m_pendingSaves[window] = true;
                m_saveTimer->start();
            }
            break;
            
        case QEvent::Show: {
            // Try to restore state when window is first shown
            const QString& identifier = m_registeredWindows[window].identifier;
            if (hasSavedState(identifier)) {
                LOG_DEBUG(LogCategories::UI, QString("Restoring state for window on show: %1").arg(identifier));
                restoreWindowState(window);
            } else {
                LOG_DEBUG(LogCategories::UI, QString("No saved state found for window: %1").arg(identifier));
            }
            break;
        }
            
        case QEvent::Close:
            if (m_autoSaveEnabled) {
                // CRITICAL FIX: Update cached geometry before the window is hidden
                // The window is still visible during Close event, capture its geometry now
                // This is ESSENTIAL on Linux where Move events may not fire during window dragging
                if (window->isVisible() && !window->isMinimized()) {
                    // Try multiple methods to get the window position
                    // On some Linux WMs, geometry() and pos() return (0,0) but frameGeometry() works
                    QRect geometry = window->geometry();
                    QRect frameGeometry = window->frameGeometry();
                    QPoint pos = window->pos();

                    const QString& identifier = m_registeredWindows[window].identifier;
                    LOG_INFO(LogCategories::UI, QString("Window %1 closing: geometry=(%2,%3 %4x%5) frameGeometry=(%6,%7 %8x%9) pos=(%10,%11)")
                            .arg(identifier)
                            .arg(geometry.x()).arg(geometry.y()).arg(geometry.width()).arg(geometry.height())
                            .arg(frameGeometry.x()).arg(frameGeometry.y()).arg(frameGeometry.width()).arg(frameGeometry.height())
                            .arg(pos.x()).arg(pos.y()));

                    // Use frameGeometry if geometry returns (0,0)
                    QRect currentGeometry = (geometry.topLeft() == QPoint(0, 0) && frameGeometry.topLeft() != QPoint(0, 0))
                                           ? frameGeometry
                                           : geometry;

                    m_registeredWindows[window].lastGeometry = currentGeometry;
                    m_registeredWindows[window].lastState = window->windowState();
                }
                // Now save the window state (will use cached geometry if needed)
                saveWindowState(window);
            }
            break;
            
        default:
            break;
    }

    return QObject::eventFilter(watched, event);
}