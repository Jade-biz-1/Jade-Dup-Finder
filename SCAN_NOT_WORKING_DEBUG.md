# Scan Not Working - Debug Guide

**Date:** October 14, 2025  
**Issue:** Scan doesn't show progress, doesn't find duplicates  
**User Story:** 3.1.2 - Downloads Cleanup fails  
**Status:** 🔍 DEBUGGING - Added comprehensive debug output

---

## 🐛 Problem Description

**User Report:**
1. Click "Downloads Cleanup" button
2. Scan dialog opens
3. Click "Start Scan"
4. **No progress shown**
5. **No results displayed**
6. Known duplicate files in Downloads folder not detected

**Expected Behavior:**
1. Progress bar shows scanning activity
2. File count updates in real-time
3. Scan completes
4. Duplicate detection runs
5. Results window opens showing duplicates

---

## 🔍 Debug Output Added

### 1. Scan Completion Tracking

**File:** `src/gui/main_window.cpp` - `onScanCompleted()`

```cpp
qDebug() << "=== onScanCompleted called ===";
qDebug() << "Files found:" << filesFound;
qDebug() << "Bytes scanned:" << bytesScanned;
qDebug() << "Errors:" << errorsEncountered;
```

**Purpose:** Verify scan actually completes and how many files were found

### 2. Duplicate Detection Check

```cpp
qDebug() << "Checking duplicate detector...";
qDebug() << "m_duplicateDetector:" << (m_duplicateDetector ? "EXISTS" : "NULL");
qDebug() << "detectorFiles.size():" << detectorFiles.size();
```

**Purpose:** Verify duplicate detector exists and files are passed to it

### 3. Detection Start Confirmation

```cpp
if (m_duplicateDetector && !detectorFiles.isEmpty()) {
    qDebug() << "Starting duplicate detection with" << detectorFiles.size() << "files";
    m_duplicateDetector->findDuplicates(detectorFiles);
}
```

**Purpose:** Confirm duplicate detection is actually initiated

---

## 🧪 Testing Instructions

### Step 1: Create Test Duplicates

```bash
# Create test directory with duplicates
mkdir -p ~/Downloads/test_duplicates
cd ~/Downloads/test_duplicates

# Create original file
echo "This is a test file for duplicate detection" > original.txt

# Create exact duplicates
cp original.txt duplicate1.txt
cp original.txt duplicate2.txt
cp original.txt duplicate3.txt

# Verify files exist
ls -lh
```

### Step 2: Run Application with Debug Output

```bash
# Run from terminal to see all debug output
cd /path/to/dupfinder
./build/dupfinder 2>&1 | tee scan_debug.log
```

### Step 3: Perform Scan

1. Click "📂 Downloads Cleanup" button
2. Verify scan dialog opens
3. Click "▶ Start Scan"
4. **Watch the terminal output carefully**

### Step 4: Analyze Output

**Look for these key messages:**

#### A. Scan Configuration
```
MainWindow::handleScanConfiguration called with preset: downloads
=== Starting New Scan ===
Scan Configuration:
  - Target paths (1): /home/user/Downloads
  - Minimum file size: 0 MB
```

#### B. Scan Start
```
FileScanner: Starting scan of 1 paths
=== FileScanner: Scan Started ===
```

#### C. Scan Progress
```
FileScanner: Processing path: /home/user/Downloads
Scan progress: X files processed
```

#### D. Scan Completion
```
FileScanner: Scan completed - found X files
=== onScanCompleted called ===
Files found: X
Bytes scanned: Y
```

#### E. Duplicate Detection
```
Checking duplicate detector...
m_duplicateDetector: EXISTS
detectorFiles.size(): X
Starting duplicate detection with X files
=== Duplicate Detection Started ===
```

#### F. Results Display
```
=== Duplicate Detection Completed ===
  - Groups found: X
```

---

## 📊 Diagnostic Scenarios

### Scenario 1: No Output at All
**Symptoms:**
- No debug messages appear
- Application seems frozen

**Diagnosis:** Application not starting or crashing  
**Solution:** Check for segfaults, missing libraries

### Scenario 2: Scan Never Starts
**Symptoms:**
```
MainWindow::handleScanConfiguration called
(nothing else)
```

**Diagnosis:** FileScanner not initialized or startScan() not called  
**Possible Causes:**
- m_fileScanner is NULL
- Signal not connected
- startScan() fails silently

**Check:**
```cpp
// In main.cpp, verify:
mainWindow.setFileScanner(&fileScanner);
```

### Scenario 3: Scan Starts But Finds No Files
**Symptoms:**
```
=== FileScanner: Scan Started ===
FileScanner: Scan completed - found 0 files
Files found: 0
```

**Diagnosis:** Files being filtered out or path incorrect  
**Possible Causes:**
- Minimum file size too high
- Path doesn't exist
- Permission issues
- Files filtered by pattern

**Solutions:**
- Check minimum file size setting (should be 0 for Downloads)
- Verify Downloads path exists
- Check file permissions
- Review exclude patterns

### Scenario 4: Files Found But No Duplicates
**Symptoms:**
```
Files found: 10
Starting duplicate detection with 10 files
=== Duplicate Detection Completed ===
  - Groups found: 0
```

**Diagnosis:** Duplicate detection working but no duplicates found  
**Possible Causes:**
- Files are actually unique
- Hash calculation failing
- Duplicate detection algorithm issue

**Solutions:**
- Verify test files are truly identical
- Check hash calculator is working
- Review duplicate detection logic

### Scenario 5: Duplicates Found But Not Displayed
**Symptoms:**
```
=== Duplicate Detection Completed ===
  - Groups found: 2
(Results window doesn't open)
```

**Diagnosis:** Results window not opening  
**Possible Causes:**
- ResultsWindow not created
- show() not called
- Window hidden behind main window

**Solutions:**
- Check onDuplicateDetectionCompleted()
- Verify ResultsWindow creation
- Check window management

---

## 🔧 Common Issues & Fixes

### Issue 1: FileScanner Not Initialized
**Symptom:** m_fileScanner is NULL

**Fix in main.cpp:**
```cpp
FileScanner fileScanner;
mainWindow.setFileScanner(&fileScanner);
```

### Issue 2: Minimum File Size Too High
**Symptom:** Files found: 0 (but files exist)

**Fix:** Check scan configuration
```cpp
// In loadPreset("downloads"):
m_minimumSize->setValue(0); // Should be 0, not 1
```

### Issue 3: Path Not Found
**Symptom:** Scan completes instantly with 0 files

**Fix:** Verify path
```cpp
QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
qDebug() << "Downloads path:" << downloadsPath;
qDebug() << "Path exists:" << QDir(downloadsPath).exists();
```

### Issue 4: Duplicate Detector Not Initialized
**Symptom:** m_duplicateDetector: NULL

**Fix in main.cpp:**
```cpp
DuplicateDetector duplicateDetector;
mainWindow.setDuplicateDetector(&duplicateDetector);
```

### Issue 5: Hash Calculator Not Set
**Symptom:** Duplicate detection starts but never completes

**Fix:** Verify HashCalculator connection
```cpp
// In main.cpp:
duplicateDetector.setHashCalculator(&hashCalculator);
```

---

## 📝 Expected Complete Output

**Successful scan should show:**

```
MainWindow::onPresetSelected called with preset: downloads
=== Starting New Scan ===
Scan Configuration:
  - Target paths (1): /home/user/Downloads
  - Minimum file size: 0 MB
FileScanner: Starting scan of 1 paths
=== FileScanner: Scan Started ===
FileScanner: Processing path: /home/user/Downloads
Scan progress: 4 files processed
FileScanner: Scan completed - found 4 files
=== onScanCompleted called ===
Files found: 4
Bytes scanned: 180
Checking duplicate detector...
m_duplicateDetector: EXISTS
detectorFiles.size(): 4
Starting duplicate detection with 4 files
=== Duplicate Detection Started ===
  - Total files to process: 4
Detecting duplicates...
=== Duplicate Detection Completed ===
  - Groups found: 1
Opening results window...
```

---

## 🎯 Next Steps

### 1. Run Test
```bash
./build/dupfinder 2>&1 | tee scan_debug.log
```

### 2. Perform Scan
- Click Downloads Cleanup
- Start scan
- Watch output

### 3. Report Findings
**Which messages appear?**
- [ ] Scan configuration
- [ ] Scan started
- [ ] Scan progress
- [ ] Scan completed
- [ ] Files found (how many?)
- [ ] Duplicate detector check
- [ ] Detection started
- [ ] Detection completed
- [ ] Groups found (how many?)

### 4. Share Log
```bash
# Save the log
cat scan_debug.log

# Or share specific sections
grep "===" scan_debug.log
grep "Files found" scan_debug.log
grep "Groups found" scan_debug.log
```

---

## 🔍 Additional Debug (If Needed)

If the above doesn't reveal the issue, add more debug:

### In FileScanner::scanDirectory()
```cpp
qDebug() << "Scanning directory:" << dirPath;
qDebug() << "Files in directory:" << dir.count();
```

### In DuplicateDetector::findDuplicates()
```cpp
qDebug() << "findDuplicates called with" << files.size() << "files";
```

### In HashCalculator
```cpp
qDebug() << "Calculating hash for:" << filePath;
```

---

## ✅ Success Criteria

**Scan is working when you see:**
1. ✅ Scan starts
2. ✅ Files found > 0
3. ✅ Duplicate detection starts
4. ✅ Groups found (if duplicates exist)
5. ✅ Results window opens

---

**Debug Build:** ✅ Ready  
**Test Files:** Create in ~/Downloads/test_duplicates  
**Next Action:** Run and report terminal output

**Status:** Waiting for test results to diagnose issue
