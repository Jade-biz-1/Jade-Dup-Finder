#ifndef WINDOW_STATE_MANAGER_H
#define WINDOW_STATE_MANAGER_H

#include <QObject>
#include <QWidget>
#include <QDialog>
#include <QMainWindow>
#include <QSettings>
#include <QRect>
#include <QSize>
#include <QPoint>
#include <QHash>
#include <QTimer>

/**
 * @class WindowStateManager
 * @brief Manages saving and restoring window positions, sizes, and states
 * 
 * This class automatically saves window geometry and state when windows are closed
 * and restores them when windows are created. It handles:
 * - Window position and size
 * - Window state (maximized, minimized, etc.)
 * - Multi-monitor support
 * - Validation of restored positions
 */
class WindowStateManager : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Get the singleton instance
     */
    static WindowStateManager* instance();

    /**
     * @brief Register a window for state management
     * @param window The window to manage
     * @param identifier Unique identifier for the window (optional, uses class name if empty)
     */
    void registerWindow(QWidget* window, const QString& identifier = QString());

    /**
     * @brief Register a dialog for state management
     * @param dialog The dialog to manage
     * @param identifier Unique identifier for the dialog (optional, uses class name if empty)
     */
    void registerDialog(QDialog* dialog, const QString& identifier = QString());

    /**
     * @brief Unregister a window from state management
     * @param window The window to unregister
     */
    void unregisterWindow(QWidget* window);

    /**
     * @brief Save window state immediately
     * @param window The window to save
     */
    void saveWindowState(QWidget* window);

    /**
     * @brief Restore window state immediately
     * @param window The window to restore
     * @return true if state was restored, false if no saved state exists
     */
    bool restoreWindowState(QWidget* window);

    /**
     * @brief Save all registered window states
     */
    void saveAllWindowStates();

    /**
     * @brief Set whether to save window states automatically on close
     * @param enabled True to enable automatic saving
     */
    void setAutoSaveEnabled(bool enabled);

    /**
     * @brief Check if auto-save is enabled
     */
    bool isAutoSaveEnabled() const;

    /**
     * @brief Set the settings group name for window states
     * @param groupName The group name to use in QSettings
     */
    void setSettingsGroup(const QString& groupName);

    /**
     * @brief Get the current settings group name
     */
    QString settingsGroup() const;

    /**
     * @brief Clear all saved window states
     */
    void clearAllSavedStates();

    /**
     * @brief Check if a window has saved state
     * @param identifier The window identifier
     * @return true if saved state exists
     */
    bool hasSavedState(const QString& identifier) const;

public slots:
    /**
     * @brief Save states for all registered windows (called on app exit)
     */
    void onApplicationExit();

private slots:
    void onWindowDestroyed(QObject* window);
    void saveDelayedStates();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    explicit WindowStateManager(QObject* parent = nullptr);
    ~WindowStateManager();

    struct WindowInfo {
        QWidget* window;
        QString identifier;
        bool isDialog;
        QRect lastGeometry;
        Qt::WindowStates lastState;
    };

    // Helper methods
    QString getWindowIdentifier(QWidget* window, const QString& customId = QString()) const;
    void saveWindowGeometry(const QString& identifier, QWidget* window);
    bool restoreWindowGeometry(const QString& identifier, QWidget* window);
    bool isValidGeometry(const QRect& geometry) const;
    QRect ensureValidGeometry(const QRect& geometry) const;
    void connectWindowSignals(QWidget* window);
    void disconnectWindowSignals(QWidget* window);

    // Member variables
    static WindowStateManager* s_instance;
    QHash<QWidget*, WindowInfo> m_registeredWindows;
    QSettings* m_settings;
    QString m_settingsGroup;
    bool m_autoSaveEnabled;
    QTimer* m_saveTimer;  // For delayed saving to avoid too frequent saves
    QHash<QWidget*, bool> m_pendingSaves;  // Track windows that need saving

    // Constants
    static const QString DEFAULT_SETTINGS_GROUP;
    static const int SAVE_DELAY_MS = 500;  // Delay before saving to avoid frequent saves
};

#endif // WINDOW_STATE_MANAGER_H