# UI/UX Fixes Requirements

## Overview
Fix critical UI/UX issues identified during user testing of the Duplicate File Results dialog and Scan History dialog.

## Issues to Fix

### 1. File Thumbnail Visibility Issue
- **Problem**: File thumbnails are no longer visible in the results tree view
- **Impact**: Users cannot visually identify image files quickly
- **Root Cause**: Thumbnail delegate may not be properly configured or enabled

### 2. Group Selection Checkbox Missing
- **Problem**: No checkbox appears near group names for selecting entire groups
- **Impact**: Users cannot select all files in a group at once
- **Root Cause**: Group items don't have checkboxes enabled

### 3. Light Theme Contrast Issues
- **Problem**: In Light theme, text color and background colors lack sufficient contrast, especially for selected items
- **Impact**: Text becomes unreadable when items are selected
- **Root Cause**: Selection colors not properly configured for light theme

### 4. Scan History Date Input Cutoff
- **Problem**: Date input boxes in Scan History dialog are cut off from bottom and right
- **Impact**: Users cannot properly interact with date controls
- **Root Cause**: Insufficient minimum size settings for QDateEdit widgets

### 5. Missing Loading Indicator
- **Problem**: When clicking "View Results" for large scans, no loading indicator is shown before the "Force Quit" or "Wait" dialog appears
- **Impact**: Poor user experience during loading of large result sets
- **Root Cause**: No loading state management for large operations

### 6. Dialog Navigation Issue
- **Problem**: When Duplicate File Results is invoked from Scan History dialog, closing the results doesn't return to Scan History
- **Impact**: Users lose their place in the workflow
- **Root Cause**: Dialog management doesn't maintain parent-child relationship properly

## Success Criteria
1. Thumbnails are visible for image files in results tree
2. Group checkboxes allow selection of entire groups
3. Light theme has proper contrast for all text/background combinations
4. Date inputs in Scan History are fully visible and functional
5. Loading indicator appears for operations taking >1 second
6. Closing results dialog returns to Scan History when appropriate