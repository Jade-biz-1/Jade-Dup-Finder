# Window Position Restoration Investigation

**Date:** 2025-11-26
**Platform Tested:** Linux (Ubuntu with Wayland)
**Status:** Partially Fixed - Needs macOS Testing

## Problem Statement

Application windows and dialogs were not restoring to their saved positions when the application was reopened. Users would move windows to specific locations, close the app, and upon reopening, windows would appear at default positions instead of where they were last positioned.

## Root Causes Identified

### 1. Windows Not Being Saved During Application Exit
**Issue:** The `saveAllWindowStates()` function only saved windows that were `isVisible()`, but Qt had already hidden windows before calling the `aboutToQuit` signal, resulting in 0 windows being saved.

**Fix:** Modified `saveAllWindowStates()` to save all registered windows regardless of visibility status, relying on cached geometry for hidden windows.

**File:** `src/core/window_state_manager.cpp:155-174`

### 2. Qt Move Events Not Firing on Linux
**Issue:** On Linux (especially Wayland), Qt doesn't generate Move/Resize events during window manager-controlled dragging. Only one Move event was logged on initial window show, but no events when dragging the window.

**Fix:** Added geometry capture in the Close event handler to get the position when the window is closed, since we can't rely on Move events during dragging.

**File:** `src/core/window_state_manager.cpp:489-519`

### 3. Incorrect Geometry Retrieval Method
**Issue:** Using `window->pos()` and `window->size()` separately returned inconsistent values. `pos()` was returning incorrect values like `(1917, -30)` while `geometry()` showed `(1920, 0)`.

**Fix:** Changed to use `window->geometry()` as a single call and extract position/size from it for consistency.

**File:** `src/core/window_state_manager.cpp:256-287`

### 4. Wayland Limitation (UNRESOLVED on Linux)
**Issue:** On Wayland, all Qt geometry methods (`geometry()`, `pos()`, `frameGeometry()`) return `(0,0)` or incorrect values due to Wayland's security model preventing applications from knowing their absolute screen position.

**Status:** This is a fundamental Wayland limitation. **Needs testing on macOS and X11 to verify fixes work on other platforms.**

## Code Changes Made

### 1. `window_state_manager.cpp` - Save All Windows Regardless of Visibility

**Location:** Lines 155-174

```cpp
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
```

### 2. `window_state_manager.cpp` - Cache Geometry on Move/Resize Events

**Location:** Lines 441-462

```cpp
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
```

### 3. `window_state_manager.cpp` - Capture Geometry in Close Event

**Location:** Lines 489-519

```cpp
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
```

### 4. `window_state_manager.cpp` - Use geometry() Instead of pos()/size()

**Location:** Lines 256-287

```cpp
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
```

### 5. `window_state_manager.cpp` - Enhanced Restoration Logging

**Location:** Lines 289-334

```cpp
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
    // ... fallback code continues
}
```

### 6. `main_window.cpp` - Removed Duplicate Geometry Handling

**Location:** Lines 1187-1205

Removed manual `restoreGeometry()` and `saveGeometry()` calls from MainWindow's `loadSettings()` and `saveSettings()` methods to avoid conflicts with WindowStateManager.

```cpp
void MainWindow::loadSettings()
{
    // NOTE: Window geometry and state restoration is now handled by WindowStateManager
    // This is registered in the constructor via WindowStateManager::instance()->registerWindow()
    // No need to manually restore geometry here - it creates conflicts
}

void MainWindow::saveSettings()
{
    // NOTE: Window geometry and state saving is now handled by WindowStateManager
    // Automatic saving occurs on Move/Resize/Close events and application exit
    // No need to manually save geometry here - it creates conflicts and redundancy
}
```

## Testing Results on Linux (Wayland)

### Build System
- Successfully built using: `cd build/linux/x64/linux-ninja-cpu && ninja cloneclean-1.0.0`
- Launch using: `/home/deepak/Public/cloneclean/launch.sh`

### Test Results
1. ✅ Application now saves window states (logs show "Saved states for 1 windows")
2. ✅ Geometry is cached on Move/Resize events (when they fire)
3. ✅ Geometry is captured in Close event handler
4. ✅ Qt's `restoreGeometry()` is called and returns true
5. ❌ **Position values are incorrect (0,0) due to Wayland limitation**

### Wayland-Specific Issues
On Wayland, Qt's position APIs return (0,0) regardless of actual window position:
- `window->geometry()` → returns (0,0) even after moving window
- `window->pos()` → returns (0,0) or incorrect values
- `window->frameGeometry()` → returns frame offset, not actual position

This is a **security feature of Wayland**, not a bug in our code.

## macOS Testing Plan

When testing on macOS, verify the following:

### 1. Basic Position Persistence
- [ ] Launch the application
- [ ] Move the main window to a different position (e.g., bottom-right corner)
- [ ] Close the application
- [ ] Relaunch and verify window appears at the saved position

### 2. Multi-Monitor Support
- [ ] If you have multiple monitors, move window to second monitor
- [ ] Close and relaunch
- [ ] Verify window appears on the correct monitor at the correct position

### 3. Dialog Position Persistence
- [ ] Open any dialog (if applicable)
- [ ] Move the dialog to a different position
- [ ] Close the dialog
- [ ] Reopen the same dialog
- [ ] Verify dialog appears at the saved position

### 4. Check Logs
Look for these log messages to verify the fixes are working:

**On window movement:**
```
Window MainWindow moved/resized to position (X, Y) size (WxH)
```

**On window close:**
```
Window MainWindow closing: geometry=(X,Y WxH) frameGeometry=(...) pos=(X,Y)
Saving geometry for MainWindow: pos=(X,Y) size=(WxH)
```

**On application restart:**
```
Attempting to restore window MainWindow from saved geometry ByteArray
Window MainWindow restoreGeometry() returned true, geometry is now: pos=(X,Y) size=(WxH)
```

### 5. Verify Position Values
Unlike Wayland, macOS should report actual window positions. Check that:
- The logged position values match where you actually moved the window
- The position values are NOT (0,0) when you move the window elsewhere
- The restored position matches the saved position

### 6. Settings File Location
On macOS, the QSettings file should be at:
```
~/Library/Preferences/com.CloneCleanTeam.CloneClean.plist
```

You can inspect it with:
```bash
defaults read com.CloneCleanTeam.CloneClean
```

Look for `WindowStates/MainWindow/position` values.

## Expected Behavior on macOS

If the fixes work correctly on macOS, you should see:

1. **Non-zero position values** when moving windows (unlike Wayland's (0,0))
2. **Consistent values** across all three methods (geometry(), pos(), frameGeometry())
3. **Accurate restoration** - windows opening exactly where you closed them
4. **Multi-monitor support** - windows appearing on the correct monitor

## Known Limitations

1. **Wayland on Linux**: Position tracking does not work reliably due to Wayland's security model
2. **Disconnected monitors**: If a window was on a monitor that's now disconnected, the `ensureValidGeometry()` function should correct it to the primary screen
3. **Minimized windows**: Minimized state is tracked but position may not update while minimized

## Next Steps if macOS Testing Succeeds

1. Document that window position persistence is **not supported on Wayland**
2. Consider detecting Wayland at runtime and displaying a warning or disabling position persistence
3. Recommend X11 for users who need window position persistence on Linux
4. Extend the fixes to all dialogs in the application

## Next Steps if macOS Testing Fails

1. Add more detailed logging to identify platform-specific issues
2. Investigate macOS-specific window manager behaviors
3. Consider alternative approaches like using Qt's native window positioning APIs
4. Check if Qt version or macOS version affects window geometry APIs

## Files Modified

1. `/home/deepak/Public/cloneclean/src/core/window_state_manager.cpp`
   - Lines 66-81: Always restore state during registration
   - Lines 155-174: Save all windows regardless of visibility
   - Lines 256-287: Use geometry() instead of pos()/size()
   - Lines 289-334: Enhanced restoration with detailed logging
   - Lines 441-462: Cache geometry on Move/Resize events
   - Lines 489-519: Capture geometry in Close event

2. `/home/deepak/Public/cloneclean/src/gui/main_window.cpp`
   - Lines 1187-1195: Removed duplicate loadSettings() geometry handling
   - Lines 1197-1205: Removed duplicate saveSettings() geometry handling

## Related Configuration

**QSettings storage:**
- Linux: `~/.config/CloneClean Team/CloneClean.conf`
- macOS: `~/Library/Preferences/com.CloneCleanTeam.CloneClean.plist`

**Settings group:** `WindowStates`

**Keys saved per window:**
- `{identifier}/geometry` - Qt's saveGeometry() ByteArray
- `{identifier}/windowState` - Qt's saveState() ByteArray (for QMainWindow)
- `{identifier}/maximized` - Boolean
- `{identifier}/position` - QPoint
- `{identifier}/size` - QSize

## References

- Qt Documentation: [QWidget::saveGeometry()](https://doc.qt.io/qt-6/qwidget.html#saveGeometry)
- Qt Documentation: [QWidget::restoreGeometry()](https://doc.qt.io/qt-6/qwidget.html#restoreGeometry)
- Wayland Position Limitation: [Qt Bug QTBUG-68619](https://bugreports.qt.io/browse/QTBUG-68619)
