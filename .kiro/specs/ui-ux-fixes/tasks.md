# UI/UX Fixes Implementation Tasks

## Task 1: Fix File Thumbnail Visibility
**Priority**: High
**Files**: `src/gui/results_window.cpp`, `src/gui/thumbnail_delegate.cpp`

### Subtasks:
1. Verify thumbnail delegate is properly set on tree widget
2. Ensure thumbnail cache initialization
3. Add debug logging for thumbnail generation
4. Fix column configuration for thumbnail display
5. Test thumbnail visibility with sample images

## Task 2: Add Group Selection Checkboxes
**Priority**: Medium
**Files**: `src/gui/results_window.cpp`

### Subtasks:
1. Enable checkboxes for group items in `updateGroupItem()`
2. Implement group checkbox change handler
3. Add logic to toggle all child checkboxes when group is toggled
4. Update selection summary to include group selections
5. Test group selection functionality

## Task 3: Fix Light Theme Contrast Issues
**Priority**: High (Accessibility)
**Files**: `src/gui/results_window.cpp`

### Subtasks:
1. Update `applyTheme()` method for light theme handling
2. Define high-contrast selection colors for light theme
3. Ensure selected text remains readable
4. Test both active and inactive selection states
5. Verify contrast ratios meet accessibility standards

## Task 4: Fix Scan History Date Input Cutoff
**Priority**: Medium
**Files**: `src/gui/scan_history_dialog.cpp`

### Subtasks:
1. Set explicit minimum width/height for QDateEdit widgets
2. Adjust layout spacing and margins
3. Test with different system fonts and DPI settings
4. Ensure calendar popup works correctly
5. Verify date input functionality

## Task 5: Add Loading Indicator for Large Scans
**Priority**: Medium
**Files**: `src/gui/scan_history_dialog.cpp`

### Subtasks:
1. Add loading cursor before processing large scans
2. Implement QTimer-based deferred loading
3. Show progress status messages
4. Add threshold for showing loading indicator (>1000 files)
5. Test with large scan datasets

## Task 6: Fix Dialog Navigation
**Priority**: Low
**Files**: `src/gui/scan_history_dialog.cpp`, `src/gui/main_window.cpp`

### Subtasks:
1. Modify dialog invocation to use `hide()` instead of `accept()`
2. Implement proper dialog stacking
3. Add signal connections for dialog returns
4. Ensure Scan History reappears when Results closes
5. Test complete dialog workflow

## Implementation Order

### Phase 1 (Critical Fixes)
1. Task 1: Fix File Thumbnail Visibility
2. Task 3: Fix Light Theme Contrast Issues  
3. Task 4: Fix Scan History Date Input Cutoff

### Phase 2 (Enhancement Fixes)
1. Task 2: Add Group Selection Checkboxes
2. Task 5: Add Loading Indicator for Large Scans
3. Task 6: Fix Dialog Navigation

## Testing Checklist

### For Each Task:
- [ ] Test with light theme
- [ ] Test with dark theme
- [ ] Test with different screen resolutions
- [ ] Test with large datasets
- [ ] Verify no regressions in existing functionality
- [ ] Test keyboard navigation
- [ ] Test accessibility features

### Integration Testing:
- [ ] Complete workflow from scan to results
- [ ] Dialog navigation flow
- [ ] Theme switching while dialogs are open
- [ ] Performance with large result sets