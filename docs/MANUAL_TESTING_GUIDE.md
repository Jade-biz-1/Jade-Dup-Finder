# Manual Testing Guide - CloneClean

## Date: December 10, 2025
## Purpose: Step-by-step guide for testing the application manually

---

## What is Manual Testing?

Manual testing means **using the application like a real user** and checking that everything works as expected. You click buttons, enter data, and verify the results match what you expect.

---

## Prerequisites

### 1. Make Sure Application is Built
```bash
# Build the application
cmake --build build --target cloneclean -j$(nproc)

# Check for errors
echo $?  # Should output 0 (success)
```

### 2. Run the Application
```bash
# Run from workspace root
./build/cloneclean

# Or run in background to see logs
./build/cloneclean > /tmp/cloneclean_test.log 2>&1 &
```

### 3. Check Application Started
```bash
# Check if running
ps aux | grep cloneclean | grep -v grep

# Should show something like:
# deepak    12345  1.0  0.1 4838508 126992 pts/2  Sl   21:21   0:00 ./build/cloneclean
```

---

## Testing Checklist

## Phase 1: Application Launch (5 minutes)

### Test 1.1: Application Starts
**Steps:**
1. Run `./build/cloneclean`
2. Wait 2-3 seconds

**Expected:**
- ✅ Application window appears
- ✅ No error messages
- ✅ Window title shows "CloneClean - Duplicate File Finder"
- ✅ Main window has header buttons visible

**If it fails:**
- Check terminal for error messages
- Check log file: `~/.local/share/CloneClean Team/CloneClean/logs/cloneclean.log`

---

### Test 1.2: UI Elements Visible
**Steps:**
1. Look at the main window

**Expected:**
- ✅ Header buttons: "New Scan", "Settings", "Help", "View Results"
- ✅ Quick Actions section with 6 buttons
- ✅ Scan History section
- ✅ System Overview section
- ✅ Status bar at bottom

**Screenshot what you see** (optional but helpful)

---

## Phase 2: Help Button (2 minutes)

### Test 2.1: Help Button Works
**Steps:**
1. Click the "❓ Help" button in the header

**Expected:**
- ✅ Dialog box appears
- ✅ Title says "CloneClean Help"
- ✅ Content includes:
  - Quick Start section
  - Quick Actions section
  - Keyboard Shortcuts section
  - Safety Features section
  - Link to documentation

**Take a screenshot of the help dialog**

### Test 2.2: Help Dialog Closes
**Steps:**
1. Click "OK" or "Close" button

**Expected:**
- ✅ Dialog closes
- ✅ Main window still visible

---

## Phase 3: Quick Action Presets (10 minutes)

### Test 3.1: Quick Scan Button
**Steps:**
1. Click "🚀 Start Quick Scan" button

**Expected:**
- ✅ Scan configuration dialog opens
- ✅ Dialog title shows "Scan Setup"
- ✅ Three folders are selected:
  - Home folder
  - Downloads folder
  - Documents folder
- ✅ Minimum file size is set to 1 MB
- ✅ "Include hidden files" is unchecked

**Screenshot the dialog**

**Steps to close:**
1. Click "Cancel" button

---

### Test 3.2: Downloads Cleanup Button
**Steps:**
1. Click "📂 Downloads Cleanup" button

**Expected:**
- ✅ Scan configuration dialog opens
- ✅ Only Downloads folder is selected
- ✅ Minimum file size is 0 (all files)
- ✅ All file types are checked

**Steps to close:**
1. Click "Cancel"

---

### Test 3.3: Photo Cleanup Button
**Steps:**
1. Click "📸 Photo Cleanup" button

**Expected:**
- ✅ Scan configuration dialog opens
- ✅ Pictures folder is selected
- ✅ Only "Images" file type is checked
- ✅ Other file types (Documents, Videos, etc.) are unchecked

**Steps to close:**
1. Click "Cancel"

---

### Test 3.4: Documents Button
**Steps:**
1. Click "📄 Documents" button

**Expected:**
- ✅ Scan configuration dialog opens
- ✅ Documents folder is selected
- ✅ Only "Documents" file type is checked

**Steps to close:**
1. Click "Cancel"

---

### Test 3.5: Full System Scan Button
**Steps:**
1. Click "🖥️ Full System Scan" button

**Expected:**
- ✅ Scan configuration dialog opens
- ✅ Home folder is selected
- ✅ "Include hidden files" is checked
- ✅ Minimum file size is 1 MB

**Steps to close:**
1. Click "Cancel"

---

### Test 3.6: Custom Preset Button
**Steps:**
1. Click "⭐ Custom Preset" button

**Expected:**
- ✅ Scan configuration dialog opens
- ✅ Default settings are shown
- ✅ No folders pre-selected (or default folders)

**Steps to close:**
1. Click "Cancel"

---

## Phase 4: Complete Scan Workflow (15 minutes)

### Test 4.1: Configure a Test Scan
**Steps:**
1. Create a test folder with some duplicate files:
```bash
# Create test directory
mkdir -p ~/cloneclean_test
cd ~/cloneclean_test

# Create some duplicate files
echo "test content" > file1.txt
echo "test content" > file2.txt
echo "different content" > file3.txt
cp file1.txt file1_copy.txt

# Create a subfolder with more duplicates
mkdir subfolder
cp file1.txt subfolder/file1_duplicate.txt
```

2. In CloneClean, click "📁 New Scan"
3. Click "Add Folder" button
4. Navigate to `~/cloneclean_test`
5. Select the folder
6. Click "Open" or "Select Folder"

**Expected:**
- ✅ Folder appears in the scan locations list
- ✅ Folder has a checkbox next to it
- ✅ Checkbox is checked

---

### Test 4.2: Start the Scan
**Steps:**
1. In the scan dialog, click "Start Scan" button

**Expected:**
- ✅ Dialog closes
- ✅ Progress bar appears in main window
- ✅ Status shows "Scanning..."
- ✅ File count updates (should show 5 files found)

**Watch the progress** - it should complete in a few seconds

---

### Test 4.3: Duplicate Detection
**Expected (automatically after scan):**
- ✅ Status changes to "Detecting duplicates..."
- ✅ Progress continues
- ✅ Detection completes
- ✅ Success message appears: "Found X duplicate groups"

**Click "OK" on the success message**

---

### Test 4.4: Results Window Opens
**Expected:**
- ✅ Results window opens automatically
- ✅ Window title shows "Scan Results"
- ✅ Duplicate groups are displayed
- ✅ Should show at least 1 group (file1.txt, file2.txt, file1_copy.txt, file1_duplicate.txt)
- ✅ Statistics show:
  - Total files scanned
  - Duplicate groups found
  - Potential space savings

**Screenshot the results window**

---

### Test 4.5: Explore Results
**Steps:**
1. Click on a duplicate group to expand it
2. Look at the file details

**Expected:**
- ✅ Group expands to show all duplicate files
- ✅ Each file shows:
  - File path
  - File size
  - Last modified date
- ✅ One file is marked as "recommended to keep"

---

## Phase 5: File Operations (10 minutes)

### Test 5.1: Select Files
**Steps:**
1. In results window, check the checkbox next to a file (NOT the recommended one)
2. Check another file

**Expected:**
- ✅ Checkboxes become checked
- ✅ Selection count updates at bottom
- ✅ Total size of selected files shown

---

### Test 5.2: Preview File
**Steps:**
1. Select a file
2. Click "Preview" button (or right-click → Preview)

**Expected:**
- ✅ Preview dialog opens
- ✅ File information shown (name, size, path, date)
- ✅ File content shown (for text files)
- ✅ Image shown (for image files)

**Steps to close:**
1. Click "Close"

---

### Test 5.3: Export Results
**Steps:**
1. Click "Export" button
2. Choose "CSV" format
3. Choose a location (e.g., ~/cloneclean_export.csv)
4. Click "Save"

**Expected:**
- ✅ Success message appears
- ✅ File is created

**Verify:**
```bash
# Check file exists
ls -lh ~/cloneclean_export.csv

# View contents
cat ~/cloneclean_export.csv
```

---

### Test 5.4: Delete Files (CAREFUL!)
**⚠️ WARNING: This will actually delete files! Use test files only!**

**Steps:**
1. Select one or two duplicate files (NOT the recommended one)
2. Click "Delete Selected" button
3. Read the confirmation dialog
4. Click "Yes" to confirm

**Expected:**
- ✅ Confirmation dialog appears
- ✅ Shows number of files to delete
- ✅ Shows total size
- ✅ After confirming:
  - Progress shown
  - Success message appears
  - Files are removed from results
  - Statistics update

**Verify files are deleted:**
```bash
# Check if files are gone
ls -la ~/cloneclean_test/
```

**Verify backups were created:**
```bash
# Check backup directory
ls -la ~/.local/share/CloneClean\ Team/CloneClean/backups/
```

---

## Phase 6: Logging Verification (5 minutes)

### Test 6.1: Check Log File
**Steps:**
```bash
# View the log file
tail -50 ~/.local/share/CloneClean\ Team/CloneClean/logs/cloneclean.log
```

**Expected:**
- ✅ Log file exists
- ✅ Contains entries like:
  - `[INFO ] [SYSTEM] Logging system initialized`
  - `[INFO ] [UI] User clicked 'Help' button`
  - `[INFO ] [UI] User selected preset: quick`
  - `[INFO ] [SCAN] Scan completed with X files`
  - `[INFO ] [DUPLICATE] Detection completed`

---

### Test 6.2: Check Log Rotation
**Steps:**
```bash
# Check log directory
ls -lh ~/.local/share/CloneClean\ Team/CloneClean/logs/
```

**Expected:**
- ✅ `cloneclean.log` file exists
- ✅ File size is reasonable (< 10MB)
- ✅ May have rotated logs if you've run the app many times

---

## Phase 7: Cleanup (2 minutes)

### Test 7.1: Close Application
**Steps:**
1. Close results window (if open)
2. Close main window

**Expected:**
- ✅ Application closes cleanly
- ✅ No error messages
- ✅ No crash

---

### Test 7.2: Clean Up Test Files
```bash
# Remove test directory
rm -rf ~/cloneclean_test

# Remove export file
rm -f ~/cloneclean_export.csv

# Optional: Remove backups
rm -rf ~/.local/share/CloneClean\ Team/CloneClean/backups/*
```

---

## Common Issues & Solutions

### Issue 1: Application Won't Start
**Symptoms:** Nothing happens when you run `./build/cloneclean`

**Solutions:**
1. Check if it's already running: `ps aux | grep cloneclean`
2. Kill existing instance: `pkill -f cloneclean`
3. Check build succeeded: `cmake --build build --target cloneclean`
4. Check for errors: `./build/cloneclean` (run in foreground)

---

### Issue 2: Buttons Don't Work
**Symptoms:** Clicking buttons does nothing

**Solutions:**
1. Check logs: `tail -f ~/.local/share/CloneClean\ Team/CloneClean/logs/cloneclean.log`
2. Look for error messages in terminal
3. Try restarting the application

---

### Issue 3: No Duplicates Found
**Symptoms:** Scan completes but says "No duplicates found"

**Solutions:**
1. Make sure you created actual duplicate files (same content)
2. Check minimum file size setting (set to 0 to include all files)
3. Check file type filters (set to "All")
4. Look at logs to see what was scanned

---

### Issue 4: Results Window Doesn't Open
**Symptoms:** Detection completes but no results window

**Solutions:**
1. Check if window is hidden behind main window
2. Check logs for errors
3. Try clicking "View Results" button manually

---

## Test Results Template

Use this template to record your test results:

```
# CloneClean Manual Test Results
Date: ___________
Tester: ___________
Build: ___________

## Phase 1: Application Launch
- [ ] Test 1.1: Application Starts - PASS/FAIL
- [ ] Test 1.2: UI Elements Visible - PASS/FAIL
Notes: ___________

## Phase 2: Help Button
- [ ] Test 2.1: Help Button Works - PASS/FAIL
- [ ] Test 2.2: Help Dialog Closes - PASS/FAIL
Notes: ___________

## Phase 3: Quick Action Presets
- [ ] Test 3.1: Quick Scan - PASS/FAIL
- [ ] Test 3.2: Downloads - PASS/FAIL
- [ ] Test 3.3: Photos - PASS/FAIL
- [ ] Test 3.4: Documents - PASS/FAIL
- [ ] Test 3.5: Full System - PASS/FAIL
- [ ] Test 3.6: Custom - PASS/FAIL
Notes: ___________

## Phase 4: Complete Scan Workflow
- [ ] Test 4.1: Configure Scan - PASS/FAIL
- [ ] Test 4.2: Start Scan - PASS/FAIL
- [ ] Test 4.3: Duplicate Detection - PASS/FAIL
- [ ] Test 4.4: Results Window - PASS/FAIL
- [ ] Test 4.5: Explore Results - PASS/FAIL
Notes: ___________

## Phase 5: File Operations
- [ ] Test 5.1: Select Files - PASS/FAIL
- [ ] Test 5.2: Preview File - PASS/FAIL
- [ ] Test 5.3: Export Results - PASS/FAIL
- [ ] Test 5.4: Delete Files - PASS/FAIL
Notes: ___________

## Phase 6: Logging
- [ ] Test 6.1: Check Log File - PASS/FAIL
- [ ] Test 6.2: Check Log Rotation - PASS/FAIL
Notes: ___________

## Phase 7: Cleanup
- [ ] Test 7.1: Close Application - PASS/FAIL
- [ ] Test 7.2: Clean Up - PASS/FAIL
Notes: ___________

## Overall Result: PASS/FAIL

## Issues Found:
1. ___________
2. ___________
3. ___________

## Screenshots Attached:
- [ ] Main window
- [ ] Help dialog
- [ ] Scan configuration
- [ ] Results window
- [ ] Preview dialog
```

---

## Quick Test (5 minutes)

If you're short on time, do this quick smoke test:

1. **Start app:** `./build/cloneclean`
2. **Click Help:** Verify dialog shows
3. **Click Quick Scan:** Verify dialog opens with presets
4. **Close dialogs**
5. **Close app**

If all 5 steps work, the critical fixes are working!

---

## Tips for Effective Testing

1. **Take Screenshots:** Visual proof helps track issues
2. **Note Everything:** Write down anything unexpected
3. **Test Systematically:** Follow the checklist in order
4. **Don't Skip Steps:** Each test builds on previous ones
5. **Check Logs:** Logs show what's happening behind the scenes
6. **Use Test Data:** Don't test on real important files!
7. **Be Patient:** Some operations take time
8. **Report Issues:** Note exact steps to reproduce problems

---

## Next Steps After Testing

### If All Tests Pass:
1. Document success
2. Move to next development tasks
3. Consider automated testing

### If Tests Fail:
1. Document which tests failed
2. Note error messages
3. Check logs for details
4. Report to developer (me!)
5. I'll fix the issues

---

**Prepared by:** Kiro AI Assistant  
**Date:** December 10, 2025  
**Estimated Time:** 45-60 minutes for complete testing  
**Quick Test Time:** 5 minutes
