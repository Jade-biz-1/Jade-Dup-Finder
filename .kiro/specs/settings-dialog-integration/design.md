# Settings Dialog Integration - Design Document

## Overview

This design addresses the integration bug where the Settings button in the Results window doesn't open the Settings dialog. The fix involves connecting the existing Settings button to create and show the SettingsDialog instance.

## Architecture

### Current State
- ✅ SettingsDialog class exists and is fully implemented
- ✅ Settings button exists in Results window UI
- ❌ Settings button is connected to debug print instead of dialog

### Target State
- ✅ Settings button properly opens SettingsDialog
- ✅ Settings dialog displays with all tabs
- ✅ Settings changes are applied and persisted

## Components and Interfaces

### ResultsWindow Class
**File:** `src/gui/results_window.cpp`

**Current Implementation:**
```cpp
connect(m_settingsButton, &QPushButton::clicked, this, [this]() {
    qDebug() << "Settings clicked";
});
```

**Required Changes:**
1. Add SettingsDialog member variable
2. Replace lambda with proper slot method
3. Implement dialog creation and display logic

### SettingsDialog Class
**File:** `src/gui/settings_dialog.cpp`

**Current State:** Fully implemented with all tabs and functionality
**Required Changes:** None - dialog is already complete

## Data Models

No data model changes required. The SettingsDialog already handles:
- QSettings persistence
- Tab-based organization
- Form validation
- Apply/OK/Cancel logic

## Error Handling

### Dialog Creation Errors
- **Issue:** SettingsDialog constructor fails
- **Handling:** Log error, show message box to user
- **Recovery:** Continue without opening dialog

### Settings Load/Save Errors
- **Issue:** QSettings operations fail
- **Handling:** Already handled by SettingsDialog implementation
- **Recovery:** Use default values, notify user

## Testing Strategy

### Unit Tests
- Test Settings button click triggers dialog creation
- Test dialog shows with correct parent window
- Test dialog modal behavior

### Integration Tests
- Test Settings button → dialog → settings persistence flow
- Test multiple dialog open attempts (should reuse existing)
- Test dialog close → settings applied correctly

### Manual Testing
1. Click Settings button in Results window
2. Verify Settings dialog opens with all tabs
3. Modify settings and click Apply
4. Verify changes are persisted
5. Close and reopen dialog to verify persistence

## Implementation Plan

### Phase 1: Basic Integration
1. Add SettingsDialog include to results_window.h
2. Add SettingsDialog* member variable
3. Replace debug lambda with proper slot method
4. Implement dialog creation and show logic

### Phase 2: Enhancement
1. Add dialog reuse logic (don't create multiple instances)
2. Add proper parent-child relationship
3. Add error handling for dialog creation

### Phase 3: Testing
1. Test basic functionality
2. Test edge cases (multiple clicks, etc.)
3. Verify settings persistence works correctly