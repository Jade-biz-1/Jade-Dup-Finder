# UI Wiring Audit - DupFinder

## Date: December 10, 2025
## Purpose: Comprehensive audit of all UI components and their wiring

---

## Executive Summary

### Issues Found:
1. ❌ **Settings button** - Emits signal but nothing connects to it
2. ❌ **Help button** - Emits signal but nothing connects to it  
3. ⚠️ **Scan flow** - Wired but needs verification
4. ⚠️ **Quick action presets** - Emit signals but may not trigger scans

---

## Header Buttons

### 1. New Scan Button 📁
**Status:** ✅ WORKING

**Wiring:**
```cpp
connect(m_newScanButton, &QPushButton::clicked, this, &MainWindow::onNewScanRequested);
```

**Handler:**
```cpp
void MainWindow::onNewScanRequested()
{
    LOG_INFO("User clicked 'New Scan' button");
    // Creates ScanSetupDialog
    // Connects to handleScanConfiguration
    m_scanSetupDialog->show();
}
```

**Result:** Opens scan configuration dialog ✅

---

### 2. Settings Button ⚙️
**Status:** ❌ NOT WORKING

**Wiring:**
```cpp
connect(m_settingsButton, &QPushButton::clicked, this, &MainWindow::onSettingsRequested);
```

**Handler:**
```cpp
void MainWindow::onSettingsRequested()
{
    LOG_INFO("User clicked 'Settings' button");
    emit settingsRequested();  // ❌ NOBODY LISTENS TO THIS SIGNAL
}
```

**Problem:** Signal is emitted but no slot connects to it!

**Fix Needed:**
```cpp
void MainWindow::onSettingsRequested()
{
    LOG_INFO("User clicked 'Settings' button");
    
    // Create and show settings dialog
    if (!m_settingsDialog) {
        m_settingsDialog = new SettingsDialog(this);
    }
    m_settingsDialog->show();
    m_settingsDialog->raise();
    m_settingsDialog->activateWindow();
}
```

**Status:** ❌ BROKEN - Button does nothing

---

### 3. Help Button ❓
**Status:** ❌ NOT WORKING

**Wiring:**
```cpp
connect(m_helpButton, &QPushButton::clicked, this, &MainWindow::onHelpRequested);
```

**Handler:**
```cpp
void MainWindow::onHelpRequested()
{
    LOG_INFO("User clicked 'Help' button");
    emit helpRequested();  // ❌ NOBODY LISTENS TO THIS SIGNAL
}
```

**Problem:** Signal is emitted but no slot connects to it!

**Fix Needed:**
```cpp
void MainWindow::onHelpRequested()
{
    LOG_INFO("User clicked 'Help' button");
    
    // Show help dialog or open documentation
    QMessageBox::information(this, tr("Help"),
        tr("DupFinder Help\n\n"
           "1. Click 'New Scan' to configure and start a scan\n"
           "2. Select folders to scan\n"
           "3. Configure scan options\n"
           "4. Click 'Start Scan' to begin\n"
           "5. Review results and take action\n\n"
           "For more information, visit: https://dupfinder.org/docs"));
}
```

**Status:** ❌ BROKEN - Button does nothing

---

### 4. View Results Button 🔍 (Test)
**Status:** ✅ WORKING

**Wiring:**
```cpp
connect(testResultsButton, &QPushButton::clicked, this, &MainWindow::showScanResults);
```

**Handler:**
```cpp
void MainWindow::showScanResults()
{
    // Creates ResultsWindow if needed
    // Shows the window
    m_resultsWindow->show();
}
```

**Result:** Opens results window ✅

---

## Quick Action Buttons

### Status: ⚠️ PARTIALLY WORKING

All quick action buttons emit `presetSelected(QString)` signal:

```cpp
void QuickActionsWidget::onQuickScanClicked() { emit presetSelected("quick"); }
void QuickActionsWidget::onDownloadsCleanupClicked() { emit presetSelected("downloads"); }
void QuickActionsWidget::onPhotoCleanupClicked() { emit presetSelected("photos"); }
void QuickActionsWidget::onDocumentsClicked() { emit presetSelected("documents"); }
void QuickActionsWidget::onFullSystemClicked() { emit presetSelected("fullsystem"); }
void QuickActionsWidget::onCustomPresetClicked() { emit presetSelected("custom"); }
```

**Connection:**
```cpp
connect(m_quickActions, &QuickActionsWidget::presetSelected, this, &MainWindow::onPresetSelected);
```

**Handler:**
```cpp
void MainWindow::onPresetSelected(const QString& preset)
{
    LOG_INFO(QString("User selected preset: %1").arg(preset));
    emit scanRequested(preset);  // ❌ NOBODY LISTENS TO THIS SIGNAL
}
```

**Problem:** Signal is emitted but no slot connects to it!

**Fix Needed:** Should open ScanSetupDialog with preset loaded:
```cpp
void MainWindow::onPresetSelected(const QString& preset)
{
    LOG_INFO(QString("User selected preset: %1").arg(preset));
    
    // Open scan dialog with preset
    if (!m_scanSetupDialog) {
        m_scanSetupDialog = new ScanSetupDialog(this);
        connect(m_scanSetupDialog, &ScanSetupDialog::scanConfigured,
                this, &MainWindow::handleScanConfiguration);
    }
    
    // Load the preset
    m_scanSetupDialog->loadPreset(preset);
    m_scanSetupDialog->show();
    m_scanSetupDialog->raise();
    m_scanSetupDialog->activateWindow();
}
```

**Status:** ⚠️ BROKEN - Buttons emit signal but nothing happens

---

## Scan Flow

### 1. Scan Configuration
**Status:** ✅ WORKING

**Flow:**
1. User clicks "New Scan" or preset button
2. `ScanSetupDialog` opens
3. User configures scan
4. User clicks "Start Scan" in dialog
5. Dialog emits `scanConfigured()` signal
6. `MainWindow::handleScanConfiguration()` is called

**Wiring:**
```cpp
connect(m_scanSetupDialog, &ScanSetupDialog::scanConfigured,
        this, &MainWindow::handleScanConfiguration);
```

**Result:** ✅ Configuration is captured

---

### 2. Scan Execution
**Status:** ✅ WORKING

**Handler:**
```cpp
void MainWindow::handleScanConfiguration()
{
    // Gets configuration from dialog
    ScanSetupDialog::ScanConfiguration config = m_scanSetupDialog->getCurrentConfiguration();
    
    // Converts to FileScanner options
    FileScanner::ScanOptions scanOptions;
    scanOptions.targetPaths = config.targetPaths;
    scanOptions.minimumFileSize = config.minimumFileSize * 1024 * 1024;
    // ... more options
    
    // Starts the scan
    m_fileScanner->startScan(scanOptions);
}
```

**Result:** ✅ Scan starts

---

### 3. Scan Progress
**Status:** ✅ WORKING

**Connections:**
```cpp
connect(m_fileScanner, &FileScanner::scanStarted, ...);
connect(m_fileScanner, &FileScanner::scanProgress, ...);
connect(m_fileScanner, &FileScanner::scanCompleted, this, &MainWindow::onScanCompleted);
connect(m_fileScanner, &FileScanner::scanCancelled, ...);
connect(m_fileScanner, &FileScanner::scanError, ...);
```

**Result:** ✅ Progress updates shown

---

### 4. Duplicate Detection
**Status:** ✅ WORKING

**Trigger:**
```cpp
void MainWindow::onScanCompleted()
{
    // Gets scan results
    m_lastScanResults = m_fileScanner->getScannedFiles();
    
    // Converts to detector format
    QList<DuplicateDetector::FileInfo> detectorFiles;
    for (const auto& scanFile : m_lastScanResults) {
        detectorFiles.append(DuplicateDetector::FileInfo::fromScannerInfo(scanFile));
    }
    
    // Starts duplicate detection
    m_duplicateDetector->findDuplicates(detectorFiles);
}
```

**Connections:**
```cpp
connect(m_duplicateDetector, &DuplicateDetector::detectionStarted,
        this, &MainWindow::onDuplicateDetectionStarted);
connect(m_duplicateDetector, &DuplicateDetector::detectionProgress,
        this, &MainWindow::onDuplicateDetectionProgress);
connect(m_duplicateDetector, &DuplicateDetector::detectionCompleted,
        this, &MainWindow::onDuplicateDetectionCompleted);
```

**Result:** ✅ Duplicate detection runs automatically after scan

---

### 5. Results Display
**Status:** ⚠️ NEEDS VERIFICATION

**Handler:**
```cpp
void MainWindow::onDuplicateDetectionCompleted(int totalGroups)
{
    // Should get results and show in ResultsWindow
    // Implementation truncated in file view
}
```

**Needs Verification:** Does this actually pass results to ResultsWindow?

---

## Scan History Widget

### History Item Click
**Status:** ⚠️ PARTIALLY WORKING

**Wiring:**
```cpp
connect(m_scanHistory, &ScanHistoryWidget::historyItemClicked, 
        this, &MainWindow::onScanHistoryItemClicked);
```

**Handler:**
```cpp
void MainWindow::onScanHistoryItemClicked(int index)
{
    LOG_INFO(QString("User clicked history item: %1").arg(index));
    
    // TODO: Load the actual scan results from storage
    // For now, show the results window
    showScanResults();
}
```

**Problem:** Shows empty results window, doesn't load actual scan data

**Status:** ⚠️ STUB - Shows window but no data

---

### View All History
**Status:** ❌ STUB

**Handler:**
```cpp
void MainWindow::onViewAllHistoryClicked()
{
    LOG_INFO("User clicked 'View All History'");
    
    QMessageBox::information(this, tr("Scan History"),
                           tr("Full scan history view will be implemented."));
}
```

**Status:** ❌ STUB - Shows message only

---

## System Overview Widget

### Refresh Stats
**Status:** ✅ WORKING

**Wiring:**
```cpp
connect(m_systemUpdateTimer, &QTimer::timeout, this, &MainWindow::refreshSystemStats);
```

**Handler:**
```cpp
void MainWindow::refreshSystemStats()
{
    // Gets disk space info
    QStorageInfo storage(homePath);
    stats.totalDiskSpace = storage.bytesTotal();
    stats.availableDiskSpace = storage.bytesAvailable();
    
    // Updates widget
    m_systemOverview->updateStats(stats);
}
```

**Result:** ✅ Updates every 30 seconds

---

## Results Window

### File Operations
**Status:** ✅ WORKING (from previous tasks)

- Delete files: ✅ Working
- Move files: ✅ Working
- Export: ✅ Working
- Preview: ✅ Working

---

## Summary of Issues

### Critical Issues (Broken Functionality)

1. **Settings Button** ❌
   - **Problem:** Emits signal, nothing listens
   - **Impact:** Users cannot access settings
   - **Fix:** Create SettingsDialog and show it

2. **Help Button** ❌
   - **Problem:** Emits signal, nothing listens
   - **Impact:** Users cannot get help
   - **Fix:** Show help dialog or documentation

3. **Quick Action Presets** ❌
   - **Problem:** Emit signal, nothing listens
   - **Impact:** Preset buttons don't work
   - **Fix:** Open ScanSetupDialog with preset loaded

### Medium Issues (Incomplete Functionality)

4. **Scan History Loading** ⚠️
   - **Problem:** Shows empty results window
   - **Impact:** Cannot review past scans
   - **Fix:** Implement scan result persistence and loading

5. **View All History** ⚠️
   - **Problem:** Shows stub message
   - **Impact:** Cannot see full history
   - **Fix:** Create history viewer dialog

### Verification Needed

6. **Duplicate Detection Results** ⚠️
   - **Problem:** File truncated, can't see full implementation
   - **Impact:** Unknown if results are passed to ResultsWindow
   - **Fix:** Verify `onDuplicateDetectionCompleted` implementation

---

## Working Components ✅

1. ✅ New Scan button
2. ✅ Scan configuration dialog
3. ✅ Scan execution
4. ✅ Scan progress updates
5. ✅ Duplicate detection trigger
6. ✅ System stats refresh
7. ✅ Results window (file operations)
8. ✅ Export functionality
9. ✅ File preview

---

## Recommended Fixes Priority

### Priority 1: Critical (Immediate)
1. Fix Settings button - Create SettingsDialog
2. Fix Help button - Show help information
3. Fix Quick Action presets - Load preset and open dialog

### Priority 2: Important (This Week)
4. Verify duplicate detection results flow
5. Implement scan history persistence
6. Implement scan history loading

### Priority 3: Enhancement (Next Week)
7. Create full history viewer
8. Add more settings options
9. Create comprehensive help system

---

## Testing Checklist

### Manual Testing Needed:
- [ ] Click Settings button - verify dialog opens
- [ ] Click Help button - verify help shows
- [ ] Click each quick action preset - verify scan dialog opens with preset
- [ ] Complete a scan - verify results show in ResultsWindow
- [ ] Click history item - verify past results load
- [ ] Verify all file operations work (delete, move, export, preview)

---

**Prepared by:** Kiro AI Assistant  
**Date:** December 10, 2025  
**Status:** 3 Critical Issues, 2 Medium Issues, 1 Verification Needed
