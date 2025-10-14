# Design Document - P1 Features

## Overview

This design document outlines the implementation approach for the P1 (High Priority) features of the DupFinder application. These features enhance the user experience by implementing preset loading functionality, verifying the duplicate detection flow, and adding scan history persistence.

The P1 features build upon the successfully completed P0 critical fixes and integrate seamlessly with the existing architecture. The design focuses on:

1. **Preset Loading** - Automatic configuration of scan dialogs based on user-selected presets
2. **Detection Flow Verification** - Ensuring duplicate detection results properly flow to the results window
3. **Scan History Persistence** - Saving and loading scan results for future reference

## Architecture

### System Context

The P1 features integrate with the existing DupFinder architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                        MainWindow                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Quick Actions│  │ Scan History │  │ System Info  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ├────────────────────┼────────────────────┤
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ScanSetupDialog │  │ ScanHistoryMgr  │  │ ResultsWindow   │
│  - loadPreset() │  │  - saveScan()   │  │  - display()    │
│  - configure()  │  │  - loadScan()   │  │  - show()       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ DuplicateDetector│
                    │  - findDups()   │
                    │  - getGroups()  │
                    └─────────────────┘
```

### Component Interaction Flow


#### Preset Loading Flow

```
User clicks preset button
    → MainWindow::onPresetSelected(preset)
        → Create/reuse ScanSetupDialog
        → ScanSetupDialog::loadPreset(preset)
            → Configure dialog based on preset type
            → Set target paths
            → Set scan options
            → Update UI elements
        → Show dialog
    → User can modify settings
    → User clicks "Start Scan"
        → Emit scanConfigured signal
```

#### Detection Results Flow

```
Scan completes
    → FileScanner emits scanCompleted()
        → MainWindow::onScanCompleted()
            → Get scanned files
            → Convert to detector format
            → DuplicateDetector::findDuplicates()
                → Emit detectionCompleted(groupCount)
                    → MainWindow::onDuplicateDetectionCompleted()
                        → Get duplicate groups
                        → Create/reuse ResultsWindow
                        → ResultsWindow::displayDuplicateGroups()
                        → Show and activate window
                        → Save to history
```

#### Scan History Flow

```
Detection completes
    → MainWindow::onDuplicateDetectionCompleted()
        → Create ScanRecord
        → ScanHistoryManager::saveScan(record)
            → Generate unique scan ID
            → Serialize to JSON
            → Write to file
        → ScanHistoryWidget::refreshHistory()
            → Load all scans
            → Update display

User clicks history item
    → ScanHistoryWidget emits historyItemClicked(index)
        → MainWindow::onScanHistoryItemClicked(index)
            → ScanHistoryManager::loadScan(scanId)
                → Read JSON file
                → Deserialize to ScanRecord
            → ResultsWindow::displayDuplicateGroups()
            → Show window
```

## Components and Interfaces

### 1. ScanSetupDialog Enhancement



**Purpose:** Implement preset loading functionality to automatically configure scan settings.

**Existing Interface:**
```cpp
class ScanSetupDialog : public QDialog {
    // Already exists
    void setConfiguration(const ScanConfiguration& config);
    ScanConfiguration getCurrentConfiguration() const;
    
    // To be implemented
    void loadPreset(const QString& presetName);
};
```

**Implementation Details:**

The `loadPreset()` method will:
1. Accept a preset name string ("quick", "downloads", "photos", "documents", "fullsystem", "custom")
2. Determine appropriate configuration based on preset type
3. Use QStandardPaths to get system-specific folder locations
4. Configure all dialog elements (paths, options, filters)
5. Update UI to reflect the loaded preset

**Preset Configurations:**

| Preset | Target Paths | Min Size | File Types | Hidden Files |
|--------|-------------|----------|------------|--------------|
| quick | Home, Downloads, Documents | 1 MB | All | No |
| downloads | Downloads | 0 | All | No |
| photos | Pictures | 0 | Images | No |
| documents | Documents | 0 | Documents | No |
| fullsystem | Home | 1 MB | All | Yes |
| custom | Last used | Last used | Last used | Last used |

**Methods to Add:**
```cpp
private:
    void setTargetPaths(const QStringList& paths);
    void setMinimumFileSize(qint64 sizeMB);
    void setFileTypeFilter(const QString& filterName);
    void setIncludeHidden(bool include);
    void loadLastConfiguration();
```

### 2. MainWindow Integration

**Purpose:** Connect preset buttons to scan dialog and handle detection results.

**Methods to Modify:**
```cpp
void MainWindow::onPresetSelected(const QString& preset) {
    // Current: emits signal that nobody listens to
    // New: Create dialog, load preset, show dialog
}

void MainWindow::onDuplicateDetectionCompleted(int totalGroups) {
    // Current: May not properly display results
    // New: Get groups, show in results window, save to history
}
```

**New Private Methods:**
```cpp
private:
    void showResultsWindow(const QList<DuplicateDetector::DuplicateGroup>& groups);
    void saveScanToHistory(const QList<DuplicateDetector::DuplicateGroup>& groups);
    qint64 calculatePotentialSavings(const QList<DuplicateDetector::DuplicateGroup>& groups);
```

### 3. ScanHistoryManager (New Component)



**Purpose:** Centralized management of scan history persistence.

**Class Definition:**
```cpp
class ScanHistoryManager : public QObject {
    Q_OBJECT
    
public:
    struct ScanRecord {
        QString scanId;              // UUID
        QDateTime timestamp;         // When scan was performed
        QStringList targetPaths;     // Scanned locations
        int filesScanned;            // Total files scanned
        int duplicateGroups;         // Number of duplicate groups found
        qint64 potentialSavings;     // Total size of duplicates
        QList<DuplicateDetector::DuplicateGroup> groups;  // Full results
        
        bool isValid() const { return !scanId.isEmpty(); }
    };
    
    static ScanHistoryManager* instance();
    
    // Core operations
    void saveScan(const ScanRecord& record);
    ScanRecord loadScan(const QString& scanId);
    QList<ScanRecord> getAllScans();
    void deleteScan(const QString& scanId);
    void clearOldScans(int daysToKeep = 30);
    
signals:
    void scanSaved(const QString& scanId);
    void scanDeleted(const QString& scanId);
    void historyCleared();
    
private:
    ScanHistoryManager();
    ~ScanHistoryManager();
    
    QString getHistoryFilePath(const QString& scanId) const;
    QString getHistoryDirectory() const;
    void ensureHistoryDirectory();
    
    QJsonObject serializeScanRecord(const ScanRecord& record) const;
    ScanRecord deserializeScanRecord(const QJsonObject& json) const;
    
    static ScanHistoryManager* s_instance;
};
```

**Storage Format:**

Scans will be stored as individual JSON files in:
- Linux: `~/.local/share/DupFinder/history/`
- Each file named: `scan_<uuid>.json`

**JSON Structure:**
```json
{
    "scanId": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-13T14:30:00",
    "targetPaths": ["/home/user/Downloads", "/home/user/Documents"],
    "filesScanned": 1523,
    "duplicateGroups": 42,
    "potentialSavings": 2147483648,
    "groups": [
        {
            "groupId": "group_001",
            "files": [
                {
                    "filePath": "/path/to/file1.jpg",
                    "fileSize": 1048576,
                    "hash": "abc123...",
                    "lastModified": "2025-10-01T10:00:00"
                }
            ],
            "totalSize": 2097152,
            "recommendedToKeep": "/path/to/file1.jpg"
        }
    ]
}
```

### 4. ScanHistoryWidget Enhancement

**Purpose:** Display scan history and handle user interactions.

**Methods to Modify:**
```cpp
void ScanHistoryWidget::refreshHistory() {
    // Current: Does nothing or shows sample data
    // New: Load from ScanHistoryManager and display
}
```

**New Implementation:**
```cpp
void ScanHistoryWidget::refreshHistory() {
    clearHistory();
    
    QList<ScanHistoryManager::ScanRecord> scans = 
        ScanHistoryManager::instance()->getAllScans();
    
    // Convert to widget format and display
    for (const auto& record : scans) {
        ScanHistoryItem item;
        item.scanId = record.scanId;
        item.date = formatDateTime(record.timestamp);
        item.type = determineType(record.targetPaths);
        item.duplicateCount = record.duplicateGroups;
        item.spaceSaved = record.potentialSavings;
        
        addScanResult(item);
    }
}
```

## Data Models

### ScanRecord Structure



The `ScanRecord` structure contains all information needed to recreate a scan's results:

**Fields:**
- `scanId` (QString): Unique identifier generated using QUuid
- `timestamp` (QDateTime): When the scan was performed
- `targetPaths` (QStringList): Folders that were scanned
- `filesScanned` (int): Total number of files processed
- `duplicateGroups` (int): Number of duplicate groups found
- `potentialSavings` (qint64): Total bytes that could be freed
- `groups` (QList<DuplicateGroup>): Complete duplicate group data

**Validation:**
- `scanId` must not be empty
- `timestamp` must be valid
- `targetPaths` must contain at least one path
- `filesScanned` must be >= 0
- `duplicateGroups` must be >= 0
- `potentialSavings` must be >= 0

### Preset Configuration Mapping

Each preset maps to specific `ScanConfiguration` values:

**Quick Preset:**
```cpp
config.targetPaths = {
    QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
    QStandardPaths::writableLocation(QStandardPaths::DownloadLocation),
    QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
};
config.minimumFileSize = 1 * 1024 * 1024; // 1 MB
config.includeHidden = false;
config.fileTypeFilter = FileTypeFilter::All;
```

**Downloads Preset:**
```cpp
config.targetPaths = {
    QStandardPaths::writableLocation(QStandardPaths::DownloadLocation)
};
config.minimumFileSize = 0;
config.includeHidden = false;
config.fileTypeFilter = FileTypeFilter::All;
```

**Photos Preset:**
```cpp
config.targetPaths = {
    QStandardPaths::writableLocation(QStandardPaths::PicturesLocation)
};
config.minimumFileSize = 0;
config.includeHidden = false;
config.fileTypeFilter = FileTypeFilter::Images;
```

**Documents Preset:**
```cpp
config.targetPaths = {
    QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
};
config.minimumFileSize = 0;
config.includeHidden = false;
config.fileTypeFilter = FileTypeFilter::Documents;
```

**Full System Preset:**
```cpp
config.targetPaths = {
    QStandardPaths::writableLocation(QStandardPaths::HomeLocation)
};
config.minimumFileSize = 1 * 1024 * 1024; // 1 MB
config.includeHidden = true;
config.followSymlinks = false;
config.fileTypeFilter = FileTypeFilter::All;
```

## Error Handling

### Preset Loading Errors



**Scenario:** Unknown preset name
- **Handling:** Log warning, load default configuration
- **User Impact:** Dialog opens with default settings
- **Recovery:** User can manually configure

**Scenario:** Standard path not available
- **Handling:** Skip that path, continue with others
- **User Impact:** Some paths may not be pre-selected
- **Recovery:** User can manually add paths

**Scenario:** No valid paths found
- **Handling:** Log error, show empty dialog
- **User Impact:** User must manually select paths
- **Recovery:** User adds paths manually

### Detection Flow Errors

**Scenario:** No duplicate groups found
- **Handling:** Show informational message
- **User Impact:** User sees "No duplicates found" message
- **Recovery:** User can run different scan

**Scenario:** Detection fails with error
- **Handling:** Log error, show error dialog
- **User Impact:** User sees error message
- **Recovery:** User can retry scan

**Scenario:** Results window creation fails
- **Handling:** Log error, show error dialog
- **User Impact:** Results not displayed
- **Recovery:** User can view from history

### History Persistence Errors

**Scenario:** Cannot create history directory
- **Handling:** Log error, continue without saving
- **User Impact:** Scan results not saved
- **Recovery:** Manual export of results

**Scenario:** Cannot write scan file
- **Handling:** Log error, show warning
- **User Impact:** This scan not saved to history
- **Recovery:** User can export manually

**Scenario:** Cannot read scan file
- **Handling:** Log error, skip corrupted file
- **User Impact:** That scan not shown in history
- **Recovery:** Other scans still accessible

**Scenario:** JSON parsing fails
- **Handling:** Log error, return invalid record
- **User Impact:** Cannot load that scan
- **Recovery:** User can run new scan

**Scenario:** Disk full
- **Handling:** Catch exception, show error
- **User Impact:** Cannot save new scans
- **Recovery:** User clears old scans or frees space

## Testing Strategy

### Unit Tests

**ScanSetupDialog::loadPreset()**
- Test each preset type loads correct configuration
- Test unknown preset falls back to default
- Test preset with missing paths handles gracefully
- Test UI elements update correctly

**ScanHistoryManager**
- Test saveScan() creates file correctly
- Test loadScan() reads file correctly
- Test getAllScans() returns sorted list
- Test deleteScan() removes file
- Test clearOldScans() removes old files only
- Test JSON serialization/deserialization
- Test invalid JSON handling
- Test missing file handling

**MainWindow Integration**
- Test onPresetSelected() creates dialog
- Test onDuplicateDetectionCompleted() shows results
- Test onDuplicateDetectionCompleted() saves to history
- Test onScanHistoryItemClicked() loads scan

### Integration Tests

**Preset to Scan Flow**
1. Click preset button
2. Verify dialog opens with correct settings
3. Start scan
4. Verify scan uses preset configuration

**Scan to Results Flow**
1. Complete a scan
2. Verify detection starts automatically
3. Verify results window opens
4. Verify results are displayed correctly

**Scan to History Flow**
1. Complete a scan
2. Verify scan is saved to history
3. Verify history widget updates
4. Click history item
5. Verify results load correctly

**History Persistence**
1. Complete multiple scans
2. Close application
3. Reopen application
4. Verify history is loaded
5. Verify scans can be opened

### Manual Testing

**Preset Loading**
- [ ] Click each preset button
- [ ] Verify correct folders are selected
- [ ] Verify correct options are set
- [ ] Modify settings and start scan
- [ ] Verify scan uses modified settings

**Detection Results**
- [ ] Run a scan that finds duplicates
- [ ] Verify results window opens automatically
- [ ] Verify all groups are displayed
- [ ] Verify statistics are correct
- [ ] Run a scan with no duplicates
- [ ] Verify appropriate message is shown

**Scan History**
- [ ] Complete several scans
- [ ] Verify each appears in history widget
- [ ] Click a history item
- [ ] Verify results load correctly
- [ ] Close and reopen application
- [ ] Verify history persists
- [ ] Delete a history item
- [ ] Verify it's removed

## Performance Considerations



### Preset Loading
- **Impact:** Minimal - configuration is lightweight
- **Optimization:** Cache QStandardPaths results
- **Expected Time:** < 10ms

### Detection Results Display
- **Impact:** Depends on number of groups
- **Optimization:** Use Qt's model/view for large datasets
- **Expected Time:** < 100ms for typical scans (< 1000 groups)

### History Persistence
- **Impact:** Depends on number of duplicate groups
- **Optimization:** 
  - Write asynchronously to avoid blocking UI
  - Compress large scan results
  - Limit stored groups to reasonable number
- **Expected Time:** < 500ms for typical scans

### History Loading
- **Impact:** Depends on number of saved scans
- **Optimization:**
  - Load metadata only for list display
  - Load full data only when scan is opened
  - Cache recently accessed scans
- **Expected Time:** < 200ms for 100 scans

### Storage Requirements
- **Typical Scan:** 10-100 KB per scan
- **Large Scan:** Up to 1 MB per scan
- **100 Scans:** ~10 MB total
- **Mitigation:** Auto-delete scans older than 30 days

## Security Considerations

### File System Access
- **Risk:** Writing to user's home directory
- **Mitigation:** Use QStandardPaths for proper locations
- **Validation:** Ensure directory permissions before writing

### JSON Parsing
- **Risk:** Malformed JSON could crash application
- **Mitigation:** Wrap parsing in try-catch blocks
- **Validation:** Validate all required fields exist

### Path Injection
- **Risk:** Malicious paths in saved scans
- **Mitigation:** Validate paths before using
- **Validation:** Check paths exist and are accessible

### Data Privacy
- **Risk:** Scan history contains file paths
- **Mitigation:** Store in user-only accessible directory
- **Permissions:** Set appropriate file permissions (0600)

## Backward Compatibility

### Existing Functionality
- All existing features continue to work unchanged
- No breaking changes to existing APIs
- Existing scan configurations remain valid

### Future Compatibility
- JSON format allows for easy extension
- Version field in JSON for future migrations
- Graceful handling of unknown fields

## Dependencies

### Qt Modules
- QtCore: QStandardPaths, QDateTime, QUuid, QJsonDocument
- QtWidgets: Existing UI components
- QtGui: No new dependencies

### Internal Components
- FileScanner: Existing, no changes needed
- DuplicateDetector: Existing, no changes needed
- ResultsWindow: Existing, no changes needed
- Logger: Existing, used for error logging

### External Libraries
- None required

## Implementation Notes

### File Organization
```
include/
  scan_history_manager.h       # New file
  scan_dialog.h                 # Modified
  main_window.h                 # Modified

src/core/
  scan_history_manager.cpp      # New file

src/gui/
  scan_dialog.cpp               # Modified
  main_window.cpp               # Modified
  main_window_widgets.cpp       # Modified

CMakeLists.txt                  # Modified to add new files
```

### Build System Changes
Add to CMakeLists.txt:
```cmake
# Core sources
set(CORE_SOURCES
    # ... existing ...
    src/core/scan_history_manager.cpp
)

# Headers
set(HEADERS
    # ... existing ...
    include/scan_history_manager.h
)
```

### Logging Strategy
Use existing Logger with appropriate categories:
- `LogCategories::SCAN` for preset loading
- `LogCategories::DUPLICATE` for detection flow
- `LogCategories::SYSTEM` for history persistence

Example log messages:
```cpp
LOG_INFO(LogCategories::SCAN, "Loading preset: " + presetName);
LOG_DEBUG(LogCategories::DUPLICATE, "Detection completed: " + QString::number(groupCount) + " groups");
LOG_INFO(LogCategories::SYSTEM, "Saving scan to history: " + scanId);
LOG_ERROR(LogCategories::SYSTEM, "Failed to save scan: " + error);
```

## Migration Path

### Phase 1: Preset Loading (Task 4)
1. Implement `loadPreset()` method
2. Add helper methods for configuration
3. Update `onPresetSelected()` in MainWindow
4. Test each preset type
5. Verify UI updates correctly

### Phase 2: Detection Flow (Task 5)
1. Review existing `onDuplicateDetectionCompleted()`
2. Verify group retrieval
3. Verify results window creation
4. Verify display call
5. Test end-to-end flow

### Phase 3: History Persistence (Task 6)
1. Create ScanHistoryManager class
2. Implement save/load methods
3. Implement JSON serialization
4. Update MainWindow to save scans
5. Update ScanHistoryWidget to load scans
6. Test persistence across restarts

## Success Criteria

### Functional Requirements
- [ ] All 6 preset buttons open dialog with correct settings
- [ ] Users can modify preset settings before scanning
- [ ] Detection results automatically display in results window
- [ ] Scan results are saved to history automatically
- [ ] History persists across application restarts
- [ ] Users can click history items to view past results
- [ ] Old scans can be deleted manually or automatically

### Non-Functional Requirements
- [ ] Preset loading completes in < 10ms
- [ ] Results display in < 100ms
- [ ] History save completes in < 500ms
- [ ] History load completes in < 200ms
- [ ] No memory leaks in history management
- [ ] Graceful handling of all error conditions
- [ ] Clear error messages for users
- [ ] Comprehensive logging for debugging

### User Experience
- [ ] Preset buttons feel responsive
- [ ] Results appear immediately after detection
- [ ] History updates without user action
- [ ] History items are clearly labeled
- [ ] Loading past scans is intuitive
- [ ] No confusing error messages
- [ ] Application remains stable

## Future Enhancements

### Preset Management
- User-defined custom presets
- Preset import/export
- Preset sharing between users
- Preset templates

### History Features
- Search/filter history
- Export history to CSV
- History statistics and trends
- Comparison between scans
- Scheduled scans with history

### Performance Optimizations
- Incremental history loading
- Background history cleanup
- Compressed storage format
- Database instead of JSON files

### Advanced Features
- Cloud sync of history
- Multi-device history
- Collaborative scanning
- History analytics dashboard
