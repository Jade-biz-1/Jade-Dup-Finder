# P1 Features - Manual Testing Guide

## Date: October 13, 2025
## Purpose: Verify P1 features work correctly

---

## Prerequisites

1. ✅ Application builds successfully
2. ✅ No compilation errors
3. ✅ Test data available (some duplicate files)

---

## Test Scenarios

### Scenario 1: Preset Button - Downloads ✅

**Objective:** Verify Downloads preset button works

**Steps:**
1. Launch CloneClean application
2. Click "Downloads" preset button
3. Verify scan dialog opens
4. Verify Downloads folder is pre-selected
5. Verify minimum file size is 0 MB
6. Verify "All" file types is selected
7. Click "Start Scan"
8. Wait for scan to complete

**Expected Results:**
- ✅ Dialog opens immediately
- ✅ Downloads folder path visible in tree
- ✅ Settings match Downloads preset
- ✅ Scan runs successfully

**Status:** ⬜ Not Tested

---

### Scenario 2: Preset Button - Photos ✅

**Objective:** Verify Photos preset button works

**Steps:**
1. Click "Photos" preset button
2. Verify scan dialog opens
3. Verify Pictures folder is pre-selected
4. Verify "Images" file type is selected
5. Verify other file types are unchecked
6. Click "Start Scan"

**Expected Results:**
- ✅ Dialog opens with Pictures folder
- ✅ Only Images checkbox is checked
- ✅ Scan runs successfully

**Status:** ⬜ Not Tested

---

### Scenario 3: Preset Button - Documents ✅

**Objective:** Verify Documents preset button works

**Steps:**
1. Click "Documents" preset button
2. Verify scan dialog opens
3. Verify Documents folder is pre-selected
4. Verify "Documents" file type is selected
5. Click "Start Scan"

**Expected Results:**
- ✅ Dialog opens with Documents folder
- ✅ Only Documents checkbox is checked
- ✅ Scan runs successfully

**Status:** ⬜ Not Tested

---

### Scenario 4: Preset Button - Quick Scan ✅

**Objective:** Verify Quick Scan preset works

**Steps:**
1. Click "Quick Scan" preset button
2. Verify scan dialog opens
3. Verify multiple folders selected (Home, Downloads, Documents)
4. Verify minimum file size is 1 MB
5. Click "Start Scan"

**Expected Results:**
- ✅ Dialog opens with 3 folders
- ✅ Min size is 1 MB
- ✅ Scan runs successfully

**Status:** ⬜ Not Tested

---

### Scenario 5: Preset Button - Full System ✅

**Objective:** Verify Full System preset works

**Steps:**
1. Click "Full System" preset button
2. Verify scan dialog opens
3. Verify Home folder is selected
4. Verify "Include hidden files" is checked
5. Verify minimum file size is 1 MB
6. Click "Start Scan"

**Expected Results:**
- ✅ Dialog opens with Home folder
- ✅ Hidden files option is checked
- ✅ Scan runs successfully

**Status:** ⬜ Not Tested

---

### Scenario 6: Preset Button - Custom ✅

**Objective:** Verify Custom preset works

**Steps:**
1. Click "Custom" preset button
2. Verify scan dialog opens
3. Verify default settings are loaded
4. Manually select a folder
5. Click "Start Scan"

**Expected Results:**
- ✅ Dialog opens with defaults
- ✅ Can customize settings
- ✅ Scan runs successfully

**Status:** ⬜ Not Tested

---

### Scenario 7: Automatic History Saving ✅

**Objective:** Verify scans automatically save to history

**Steps:**
1. Run any scan (e.g., Downloads)
2. Wait for detection to complete
3. Look at "Scan History" widget in main window
4. Verify new scan appears in history list

**Expected Results:**
- ✅ Scan appears in history immediately
- ✅ Shows correct date/time
- ✅ Shows scan type (e.g., "Downloads")
- ✅ Shows duplicate count
- ✅ Shows space saved

**Status:** ⬜ Not Tested

---

### Scenario 8: History Display Format ✅

**Objective:** Verify history items display correctly

**Steps:**
1. Look at history widget
2. Check date format
3. Check scan type
4. Check statistics

**Expected Results:**
- ✅ Recent scans show "Today, HH:MM AM/PM"
- ✅ Yesterday's scans show "Yesterday, HH:MM AM/PM"
- ✅ Older scans show "MMM D, HH:MM AM/PM"
- ✅ Scan type is descriptive
- ✅ Duplicate count is accurate
- ✅ Space saved is formatted (KB/MB/GB)

**Status:** ⬜ Not Tested

---

### Scenario 9: Load Scan from History ✅

**Objective:** Verify clicking history item loads results

**Steps:**
1. Ensure at least one scan in history
2. Double-click a history item
3. Wait for results window to open
4. Verify results are displayed

**Expected Results:**
- ✅ Results window opens
- ✅ Shows correct duplicate groups
- ✅ Shows correct file details
- ✅ Statistics match history item
- ✅ Can interact with results (select files, etc.)

**Status:** ⬜ Not Tested

---

### Scenario 10: Multiple Scans in History ✅

**Objective:** Verify multiple scans are tracked

**Steps:**
1. Run Downloads scan
2. Run Photos scan
3. Run Documents scan
4. Check history widget

**Expected Results:**
- ✅ All 3 scans appear in history
- ✅ Sorted by date (newest first)
- ✅ Each has correct type
- ✅ Each has correct statistics
- ✅ Can click any to view results

**Status:** ⬜ Not Tested

---

### Scenario 11: History Persistence ✅

**Objective:** Verify history persists across app restarts

**Steps:**
1. Run a scan
2. Verify it appears in history
3. Close the application
4. Reopen the application
5. Check history widget

**Expected Results:**
- ✅ History still shows previous scan
- ✅ Can click to view results
- ✅ Results load correctly
- ✅ All data intact

**Status:** ⬜ Not Tested

---

### Scenario 12: Empty Scan (No Duplicates) ✅

**Objective:** Verify scans with no duplicates are handled

**Steps:**
1. Run a scan on folder with no duplicates
2. Wait for detection to complete
3. Check history widget

**Expected Results:**
- ✅ Shows message "No duplicates found"
- ✅ Scan still saves to history
- ✅ History shows "0 groups"
- ✅ Can click history item
- ✅ Shows empty results window

**Status:** ⬜ Not Tested

---

### Scenario 13: Large Scan ✅

**Objective:** Verify large scans work correctly

**Steps:**
1. Run Full System scan
2. Wait for completion (may take several minutes)
3. Check results
4. Check history

**Expected Results:**
- ✅ Scan completes without errors
- ✅ Results display correctly
- ✅ History saves successfully
- ✅ Can load from history

**Status:** ⬜ Not Tested

---

### Scenario 14: History File Storage ✅

**Objective:** Verify JSON files are created correctly

**Steps:**
1. Run a scan
2. Navigate to `~/.local/share/CloneClean/history/`
3. Check for JSON files

**Expected Results:**
- ✅ Directory exists
- ✅ JSON files present (scan_<uuid>.json)
- ✅ Files are readable
- ✅ JSON is valid
- ✅ Contains all scan data

**Status:** ⬜ Not Tested

---

### Scenario 15: Modify Preset Before Scanning ✅

**Objective:** Verify preset settings can be modified

**Steps:**
1. Click "Downloads" preset
2. Add another folder to scan
3. Change minimum file size
4. Click "Start Scan"

**Expected Results:**
- ✅ Can modify preset settings
- ✅ Scan uses modified settings
- ✅ Results reflect modifications

**Status:** ⬜ Not Tested

---

## Error Scenarios

### Error 1: Missing Folder ✅

**Steps:**
1. Click preset for non-existent folder
2. Try to start scan

**Expected Results:**
- ✅ Handles gracefully
- ✅ Shows appropriate message
- ✅ Doesn't crash

**Status:** ⬜ Not Tested

---

### Error 2: Permission Denied ✅

**Steps:**
1. Try to scan folder without read permission
2. Check error handling

**Expected Results:**
- ✅ Shows error message
- ✅ Logs error
- ✅ Continues with accessible files

**Status:** ⬜ Not Tested

---

### Error 3: Disk Full ✅

**Steps:**
1. Run scan when disk is nearly full
2. Check if history saves

**Expected Results:**
- ✅ Handles gracefully
- ✅ Shows error if can't save
- ✅ Doesn't crash

**Status:** ⬜ Not Tested

---

### Error 4: Corrupted History File ✅

**Steps:**
1. Manually corrupt a JSON file in history
2. Restart application
3. Check history widget

**Expected Results:**
- ✅ Skips corrupted file
- ✅ Logs warning
- ✅ Shows other scans
- ✅ Doesn't crash

**Status:** ⬜ Not Tested

---

## Performance Tests

### Performance 1: Preset Loading Speed ✅

**Objective:** Verify presets load quickly

**Steps:**
1. Click each preset button
2. Measure time to dialog open

**Expected Results:**
- ✅ Dialog opens in < 100ms
- ✅ No noticeable delay
- ✅ Responsive UI

**Status:** ⬜ Not Tested

---

### Performance 2: History Save Speed ✅

**Objective:** Verify history saves quickly

**Steps:**
1. Run scan with moderate results (100-500 groups)
2. Measure time from detection complete to history update

**Expected Results:**
- ✅ Saves in < 500ms
- ✅ UI remains responsive
- ✅ No freezing

**Status:** ⬜ Not Tested

---

### Performance 3: History Load Speed ✅

**Objective:** Verify history loads quickly

**Steps:**
1. Have 10+ scans in history
2. Click a history item
3. Measure time to results display

**Expected Results:**
- ✅ Loads in < 200ms
- ✅ Results display smoothly
- ✅ No lag

**Status:** ⬜ Not Tested

---

## Test Summary Template

```
Date: _______________
Tester: _______________
Build: _______________

Scenarios Passed: _____ / 15
Error Scenarios Passed: _____ / 4
Performance Tests Passed: _____ / 3

Total: _____ / 22

Critical Issues Found: _____
Minor Issues Found: _____

Overall Status: ⬜ PASS  ⬜ FAIL  ⬜ PARTIAL

Notes:
_________________________________
_________________________________
_________________________________
```

---

## Quick 5-Minute Smoke Test

For a quick verification, test these essential scenarios:

1. ✅ Click "Downloads" preset → Dialog opens
2. ✅ Start scan → Completes successfully
3. ✅ Check history → Scan appears
4. ✅ Click history item → Results load
5. ✅ Close and reopen app → History persists

If all 5 pass, core functionality is working!

---

## Logging Verification

Check logs at: `~/.local/share/CloneClean/logs/`

Look for:
- ✅ "ScanHistoryManager created"
- ✅ "Saving scan to: ..."
- ✅ "Scan saved successfully"
- ✅ "Loading scan from: ..."
- ✅ "Scan loaded successfully"
- ✅ No ERROR messages (unless expected)

---

## Known Issues

Document any issues found during testing:

| Issue # | Severity | Description | Steps to Reproduce | Status |
|---------|----------|-------------|-------------------|--------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

---

**Prepared by:** Kiro AI Assistant  
**Date:** October 13, 2025  
**Version:** P1 Features v1.0  
**Status:** Ready for Testing
