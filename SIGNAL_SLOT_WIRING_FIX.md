# Signal/Slot Wiring - Complete Fix

**Date:** October 14, 2025  
**Issue:** Basic scan functionality not working - signals not reaching slots  
**Status:** 🔧 COMPREHENSIVE FIX IN PROGRESS

---

## 🎯 Root Cause Identified

**CRITICAL ISSUE:** Signal/slot connections are being set up in `setupConnections()` which is called in the MainWindow constructor, but the component pointers (`m_fileScanner`, `m_duplicateDetector`, etc.) are set AFTER construction in `main.cpp`.

**Result:** All connections fail because the pointers are NULL when connections are attempted.

---

## 🔧 Fix Strategy

Move all component signal connections into their respective `set*()` methods so connections are made AFTER the pointers are set.

### Components to Fix:
1. ✅ FileScanner - FIXED (moved to setFileScanner())
2. ⏳ DuplicateDetector - NEEDS FIX
3. ⏳ HashCalculator - NEEDS FIX (if needed)
4. ⏳ SafetyManager - NEEDS FIX (if needed)
5. ⏳ FileManager - NEEDS FIX (if needed)

---

## 📋 Implementation Plan

### Step 1: Fix DuplicateDetector Connections ⏳

**Current Problem:**
```cpp
// In setupConnections() - m_duplicateDetector is NULL here!
if (m_duplicateDetector) {
    connect(m_duplicateDetector, &DuplicateDetector::detectionStarted, ...);
}
```

**Solution:**
```cpp
// Move to setDuplicateDetector()
void MainWindow::setDuplicateDetector(DuplicateDetector* detector)
{
    m_duplicateDetector = detector;
    
    if (m_duplicateDetector) {
        connect(m_duplicateDetector, &DuplicateDetector::detectionStarted,
                this, &MainWindow::onDuplicateDetectionStarted);
        
        connect(m_duplicateDetector, &DuplicateDetector::detectionProgress,
                this, &MainWindow::onDuplicateDetectionProgress);
        
        connect(m_duplicateDetector, &DuplicateDetector::detectionCompleted,
                this, &MainWindow::onDuplicateDetectionCompleted);
        
        connect(m_duplicateDetector, &DuplicateDetector::detectionError,
                this, &MainWindow::onDuplicateDetectionError);
    }
}
```

### Step 2: Clean Up setupConnections() ⏳

Remove all component connections from `setupConnections()` and leave only UI element connections (buttons, etc.) that are created in `initializeUI()`.

### Step 3: Verify Connection Order in main.cpp ✅

```cpp
// main.cpp - This is the correct order
MainWindow mainWindow;  // Constructor calls setupConnections() with NULL pointers

// NOW set the components - connections should be made HERE
mainWindow.setFileScanner(&fileScanner);        // ✅ Connections made here
mainWindow.setDuplicateDetector(&duplicateDetector);  // ⏳ Need to add connections
mainWindow.setHashCalculator(&hashCalculator);
mainWindow.setSafetyManager(&safetyManager);
mainWindow.setFileManager(&fileManager);
```

---

## 🔍 Current Status

### What's Working ✅
- FileScanner connections moved to setFileScanner()
- Scan finds files correctly
- Signal is emitted

### What's Broken ❌
- onScanCompleted() not being called
- No duplicate detection starts
- No results window opens
- No progress UI updates

### Why It's Broken
- DuplicateDetector connections still in setupConnections() where detector is NULL
- Other component connections may have same issue

---

## 📝 Files to Modify

1. **src/gui/main_window.cpp**
   - Move DuplicateDetector connections to setDuplicateDetector()
   - Clean up setupConnections()
   - Verify all set*() methods

2. **include/main_window.h**
   - No changes needed (methods already exist)

---

## ✅ Testing Checklist

After fix, verify:
- [ ] "Setting up FileScanner connections..." appears
- [ ] "FileScanner::scanCompleted connection result: true" appears
- [ ] "!!! MainWindow::onScanCompleted() CALLED !!!" appears
- [ ] "Starting duplicate detection" appears
- [ ] Results window opens
- [ ] Duplicate groups shown

---

**Status:** Ready to implement fix  
**Next:** Move DuplicateDetector connections to setDuplicateDetector()
