# Scan Progress Not Showing - Test Instructions

**Date:** October 14, 2025  
**Issue:** No progress UI, no results reported  
**Status:** 🧪 READY FOR TESTING

---

## 🎯 What We're Testing

1. **Progress Bar** - Should appear in status bar during scan
2. **Status Updates** - Should show "Scanning... X files found"
3. **File Count** - Should update in real-time
4. **Results** - Should open results window when complete

---

## 📋 Step-by-Step Test

### Step 1: Create Test Files

```bash
# Create test directory with duplicates
mkdir -p ~/Downloads/cloneclean_test
cd ~/Downloads/cloneclean_test

# Create 5 original files
for i in {1..5}; do
    echo "Test file content $i" > "file_$i.txt"
done

# Create duplicates of each
for i in {1..5}; do
    cp "file_$i.txt" "file_${i}_copy1.txt"
    cp "file_$i.txt" "file_${i}_copy2.txt"
done

# Verify - should show 15 files
ls -l
echo "Total files: $(ls | wc -l)"
```

**Expected:** 15 files (5 originals + 10 duplicates)

### Step 2: Run Application with Debug

```bash
# Navigate to project directory
cd /path/to/cloneclean

# Run with debug output
./build/cloneclean 2>&1 | tee scan_test.log
```

### Step 3: Start Scan

1. **Click** "📂 Downloads Cleanup" button
2. **Verify** scan dialog opens
3. **Check** Downloads folder is selected
4. **Click** "▶ Start Scan" button

### Step 4: Observe UI

**Watch for these UI changes:**

#### A. Status Bar (Bottom of Window)
- [ ] Status changes from "Ready" to "Scanning..."
- [ ] Progress bar appears (small bar on right side)
- [ ] File count updates: "Files: 1", "Files: 2", etc.

#### B. Progress Bar
- [ ] Becomes visible
- [ ] Shows activity (fills up or animates)
- [ ] Disappears when complete

#### C. Status Messages
- [ ] "Scanning..." appears
- [ ] "Scanning... X files found" updates
- [ ] "Scan complete! Found X files" appears
- [ ] "Detecting duplicates..." appears
- [ ] "Detection complete! Found X groups" appears

#### D. Results Window
- [ ] Opens automatically after detection
- [ ] Shows duplicate groups
- [ ] Lists all duplicate files

### Step 5: Check Terminal Output

**Look for these messages in order:**

```
1. MainWindow::onPresetSelected called with preset: downloads
2. MainWindow::handleScanConfiguration called
3. === Starting New Scan ===
4. FileScanner: Starting scan of 1 paths
5. === FileScanner: Scan Started ===
6. FileScanner: Processing path: /home/user/Downloads
7. Scan progress: X files processed
8. FileScanner: Scan completed - found 15 files
9. === onScanCompleted called ===
10. Files found: 15
11. m_duplicateDetector: EXISTS
12. detectorFiles.size(): 15
13. Starting duplicate detection with 15 files
14. === Duplicate Detection Started ===
15. === Duplicate Detection Completed ===
16. Groups found: 5
```

---

## 🔍 Diagnostic Questions

### Question 1: Does the scan dialog open?
- **YES** → Continue to Q2
- **NO** → Quick actions buttons not working (already fixed)

### Question 2: What happens when you click "Start Scan"?
- **Dialog closes, nothing else** → Scan not starting
- **Dialog closes, UI freezes** → Scan blocking UI thread
- **Dialog closes, no visible change** → Progress UI not updating
- **Dialog closes, progress shows** → Working correctly!

### Question 3: Do you see ANY terminal output?
- **YES** → Share the output
- **NO** → Application not running or output redirected

### Question 4: Does the status bar exist?
- **YES, at bottom of window** → Good
- **NO, can't see it** → UI layout issue

### Question 5: After waiting 10 seconds, what do you see?
- **Still says "Ready"** → Scan never started
- **Says "Scanning..."** → Scan started but stuck
- **Says "Scan complete"** → Scan finished, check for results
- **Results window opened** → Everything working!

---

## 🐛 Common Issues & Solutions

### Issue 1: Scan Never Starts

**Symptoms:**
- Status stays "Ready"
- No terminal output after "Starting New Scan"
- No progress bar

**Diagnosis:** FileScanner not starting

**Check:**
```bash
# In terminal output, look for:
grep "FileScanner: Starting scan" scan_test.log
```

**If missing:** FileScanner.startScan() not being called

**Solution:** Check handleScanConfiguration() is being called

### Issue 2: Scan Starts But No Progress

**Symptoms:**
- Terminal shows "Scan Started"
- Status stays "Scanning..."
- No file count updates
- Eventually says "Scan complete"

**Diagnosis:** Progress signals not connected or not emitted

**Check:**
```bash
# Look for progress updates:
grep "Scan progress" scan_test.log
```

**If missing:** FileScanner not emitting scanProgress signal

**Solution:** Check FileScanner::processScanQueue() emits progress

### Issue 3: Scan Completes But No Results

**Symptoms:**
- Status shows "Scan complete! Found X files"
- No "Detecting duplicates..." message
- No results window

**Diagnosis:** Duplicate detection not starting

**Check:**
```bash
# Look for detection start:
grep "Starting duplicate detection" scan_test.log
```

**If missing:** Check onScanCompleted() logic

**Solution:** Verify m_duplicateDetector exists and has files

### Issue 4: Detection Runs But No Results Window

**Symptoms:**
- Status shows "Detection complete! Found X groups"
- No results window opens

**Diagnosis:** Results window not being shown

**Check:**
```bash
# Look for results display:
grep "Groups found" scan_test.log
```

**If shows 0 groups:** No duplicates detected (check test files)
**If shows >0 groups:** Results window not opening

**Solution:** Check onDuplicateDetectionCompleted() calls showScanResults()

---

## 📊 Expected vs Actual

### Expected Behavior (15 test files, 5 groups)

**Timeline:**
1. **0s** - Click "Start Scan"
2. **0s** - Dialog closes
3. **0s** - Status: "Scanning..."
4. **0-1s** - Progress bar appears
5. **0-1s** - File count updates: 1, 2, 3... 15
6. **1s** - Status: "Scan complete! Found 15 files"
7. **1s** - Status: "Detecting duplicates..."
8. **1-2s** - Progress bar shows detection progress
9. **2s** - Status: "Detection complete! Found 5 groups"
10. **2s** - Results window opens
11. **2s** - Shows 5 groups with 3 files each

### What Are You Seeing?

**Please describe:**
1. What happens when you click "Start Scan"?
2. What does the status bar show?
3. Does the progress bar appear?
4. What terminal output do you see?
5. Does a results window open?

---

## 🔧 Quick Fixes to Try

### Fix 1: Clean Rebuild
```bash
rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
```

### Fix 2: Check File Permissions
```bash
# Verify Downloads folder is readable
ls -la ~/Downloads
```

### Fix 3: Reduce Minimum File Size
In scan dialog, set "Minimum file size" to **0 MB**

### Fix 4: Check for Errors
```bash
# Look for errors in log
grep -i error scan_test.log
grep -i warning scan_test.log
```

---

## 📝 Information Needed

To diagnose the issue, please provide:

1. **Terminal output** from running the application
2. **Screenshot** of the main window during scan
3. **Description** of what you see in the status bar
4. **Number of files** in your Downloads folder
5. **Any error messages** that appear

---

## ✅ Success Criteria

**Scan is working when:**
- ✅ Status bar updates during scan
- ✅ Progress bar appears and animates
- ✅ File count increases
- ✅ "Scan complete" message appears
- ✅ "Detecting duplicates" message appears
- ✅ Results window opens
- ✅ Duplicate groups are shown

---

**Test Ready:** ✅ Yes  
**Test Files:** Create in ~/Downloads/cloneclean_test  
**Next Action:** Run test and report findings

**Waiting for:** Test results and terminal output
