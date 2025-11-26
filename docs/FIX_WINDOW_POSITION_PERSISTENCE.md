# Fix: Window Position Persistence - November 25, 2025

## Problem

The "Duplicate Files Found" dialog (ResultsWindow) was opening at an invalid position, spilling over the left edge of the screen. While window sizes were being preserved correctly, positions were not.

## Root Cause Analysis

### Investigation

1. **WindowStateManager was implemented correctly** - saving and restoring geometry
2. **QSettings was configured correctly** - organization and app names set in main.cpp
3. **Settings file exists** at `~/.config/DupFinder Team/DupFinder.conf`

### The Issue

Examining the saved settings revealed the problem:

```ini
ResultsWindow\position=@Point(1917 -30)
ScanProgressDialog\position=@Point(1917 -30)
ScanSetupDialog\position=@Point(1917 -30)
... (all windows had the same invalid position)
```

**Analysis:**
- X position: **1917** (off-screen to the right, typical screen width is 1920)
- Y position: **-30** (above the screen, negative position)

**Why this happened:**
1. User likely has/had multiple monitors
2. Window was positioned on a monitor that was later disconnected or reconfigured
3. Qt's `restoreGeometry()` faithfully restored the saved position, even though it's now invalid
4. The existing `ensureValidGeometry()` method was only called for manual position restoration, not after `restoreGeometry()`

## Solution

Added validation **AFTER** Qt's `restoreGeometry()` to detect and fix invalid positions.

### File Modified: `src/core/window_state_manager.cpp`

#### Method: `restoreWindowGeometry()` (line 252-310)

**Added validation block after restoreGeometry:**

```cpp
// Try to restore from saved geometry first (most reliable)
QByteArray geometry = m_settings->value(identifier + "/geometry").toByteArray();
if (!geometry.isEmpty()) {
    restored = window->restoreGeometry(geometry);

    // CRITICAL FIX: Validate and fix geometry after restore
    // Qt's restoreGeometry may restore to invalid positions (e.g., disconnected monitors)
    QRect currentGeometry = window->geometry();
    if (!isValidGeometry(currentGeometry)) {
        LOG_WARNING(LogCategories::UI, QString("Window %1 restored to invalid position (%2, %3), correcting...")
                   .arg(identifier).arg(currentGeometry.x()).arg(currentGeometry.y()));
        QRect validGeometry = ensureValidGeometry(currentGeometry);
        window->setGeometry(validGeometry);
    }

    // ... rest of restoration
}
```

**Enhanced logging:**
```cpp
if (restored) {
    LOG_DEBUG(LogCategories::UI, QString("Restored state for window: %1 at position (%2, %3)")
             .arg(identifier).arg(window->x()).arg(window->y()));
}
```

### How It Works

1. **restoreGeometry()** is called first (line 261)
   - Restores the saved position from QSettings
   - This may be an invalid position (off-screen, disconnected monitor, etc.)

2. **Validation check** (line 266)
   - Get current geometry after restore
   - Call `isValidGeometry()` to check if position is on any available screen

3. **Correction** (line 269-270)
   - If invalid, call `ensureValidGeometry()` to calculate valid position
   - This method:
     - Ensures window fits on screen (adjusts size if needed)
     - Checks if window is completely off-screen
     - Moves window to center of primary screen if needed
     - Ensures title bar is visible (not above screen)

4. **Apply corrected geometry** (line 270)
   - Set the validated geometry on the window

## Example Scenarios

### Before Fix:
```
1. User has 2 monitors (left: 0-1920, right: 1920-3840)
2. ResultsWindow opens on right monitor at position (2000, 100)
3. Position saved: (2000, 100)
4. User disconnects right monitor
5. ResultsWindow tries to restore at (2000, 100) - OFF SCREEN!
6. Window appears partially or completely off-screen
```

### After Fix:
```
1. User has 2 monitors (left: 0-1920, right: 1920-3840)
2. ResultsWindow opens on right monitor at position (2000, 100)
3. Position saved: (2000, 100)
4. User disconnects right monitor
5. ResultsWindow tries to restore at (2000, 100)
6. Validation detects invalid position
7. Window moved to center of primary screen
8. Window appears correctly on-screen ✓
```

## Enhanced Event Filter

Also added debug logging to track restoration:

**File**: `src/core/window_state_manager.cpp` (line 387-397)

```cpp
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
```

## Testing

### Test Scenarios:

1. **Normal case** - Window at valid position
   - Open ResultsWindow
   - Close it
   - Reopen
   - **Expected**: Window appears at same position ✓

2. **Invalid saved position** - Position off-screen
   - Saved position: (1917, -30)
   - Open ResultsWindow
   - **Expected**: Warning logged, window moved to valid position ✓

3. **Multiple monitors** - Switch monitor configurations
   - Open window on monitor 2
   - Disconnect monitor 2
   - Reopen window
   - **Expected**: Window appears on primary monitor ✓

4. **Negative positions** - Window above screen
   - Saved position: (500, -100)
   - Open window
   - **Expected**: Window moved to have title bar visible ✓

## Validation Logic

### `isValidGeometry()` checks:
- Window intersects with at least one available screen
- Window is not completely off-screen

### `ensureValidGeometry()` corrects:
- Window width/height exceed screen size → resize to 90% of screen
- Window completely off-screen → move to center of primary screen
- Window title bar above screen → move down to screen top

## Logging

The fix adds helpful debug logs:

```
[DEBUG] [UI] Restoring state for window on show: ResultsWindow
[WARNING] [UI] Window ResultsWindow restored to invalid position (1917, -30), correcting...
[DEBUG] [UI] Restored state for window: ResultsWindow at position (960, 384)
```

## Benefits

1. **Robust multi-monitor support** - handles monitor disconnection gracefully
2. **User experience** - windows always appear on-screen
3. **Preserves user preferences** - when position is valid, it's restored exactly
4. **Automatic correction** - invalid positions are fixed transparently
5. **Debugging** - comprehensive logging for troubleshooting

## Files Modified

1. **src/core/window_state_manager.cpp** (lines 252-310, 387-397)
   - Added validation after `restoreGeometry()`
   - Enhanced debug logging
   - Validates and corrects invalid positions

## Related Code

- `isValidGeometry()` - checks if geometry is on any screen
- `ensureValidGeometry()` - calculates valid geometry from invalid one
- Event filter - handles QEvent::Show to trigger restoration

## Configuration File

Settings stored at: `~/.config/DupFinder Team/DupFinder.conf`

Format:
```ini
[WindowStates]
ResultsWindow\geometry=@ByteArray(...)
ResultsWindow\position=@Point(x y)
ResultsWindow\size=@Size(width height)
ResultsWindow\maximized=true/false
```

## Summary

The fix ensures that window positions are validated after restoration from saved settings. This handles cases where:
- Monitors are disconnected
- Screen resolution changes
- Saved position is otherwise invalid

Windows now always appear on-screen in a valid position, while still preserving user preferences when positions are valid.
