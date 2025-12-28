# Design Document

## Overview

This design addresses the critical integration gaps in the CloneClean application identified in the comprehensive code review. The application has well-implemented individual components but lacks proper integration between them, resulting in a broken core workflow. This design focuses on connecting FileScanner → DuplicateDetector → ResultsWindow → FileManager → SafetyManager to create a functional end-to-end duplicate detection and management system.

The design follows the existing architecture patterns and Qt signal/slot mechanisms already established in the codebase, ensuring consistency and maintainability.

## Architecture

### Current State

The application currently has isolated components:
- **FileScanner**: Fully functional, scans directories and reports files
- **DuplicateDetector**: Implemented but never invoked
- **ResultsWindow**: Displays sample data, actions are stubs
- **FileManager**: Partially implemented, missing restore/backup operations
- **SafetyManager**: Partially implemented, missing validation features

### Target State

Components will be connected in a pipeline:
```
User Action → MainWindow → FileScanner → DuplicateDetector → ResultsWindow
                                                                      ↓
                                                              FileManager ↔ SafetyManager
```

### Integration Points

1. **MainWindow ↔ FileScanner**: Already connected (working)
2. **MainWindow ↔ DuplicateDetector**: NEW - needs signal/slot connections
3. **MainWindow ↔ ResultsWindow**: Partially connected - needs data passing
4. **ResultsWindow ↔ FileManager**: NEW - needs integration for file operations
5. **FileManager ↔ SafetyManager**: Partially connected - needs completion

## Components and Interfaces

### 1. MainWindow Integration Layer

**Purpose**: Orchestrate the scan → detect → display workflow

**New Methods**:
```cpp
private slots:
    void onScanCompleted();
    void onDuplicateDetectionStarted(int totalFiles);
    void onDuplicateDetectionProgress(const DuplicateDetector::DetectionProgress& progress);
    void onDuplicateDetectionCompleted(int totalGroups);
    void onDuplicateDetectionError(const QString& error);
```

**New Member Variables**:
```cpp
private:
    QList<FileScanner::FileInfo> m_lastScanResults;  // Cache scan results for detection
```

**Modified Methods**:
- `setupConnections()`: Add DuplicateDetector signal connections
- `handleScanConfiguration()`: Existing, no changes needed

**Integration Logic**:
1. When FileScanner::scanCompleted is emitted, cache the results
2. Convert FileScanner::FileInfo to DuplicateDetector::FileInfo
3. Invoke DuplicateDetector::findDuplicates() with converted data
4. Show progress updates during detection
5. When detection completes, pass results to ResultsWindow

### 2. DuplicateDetector Completion

**Purpose**: Implement missing synchronous detection method

**Implementation for findDuplicatesSync()**:
```cpp
QList<DuplicateGroup> DuplicateDetector::findDuplicatesSync(const QList<FileInfo>& files)
{
    // Use same logic as async version but execute synchronously
    // 1. Group by size
    // 2. Calculate hashes for potential duplicates
    // 3. Group by hash
    // 4. Generate recommendations
    // 5. Return results
}
```

**Key Considerations**:
- Reuse existing private methods (groupFilesBySize, groupFilesByHash, etc.)
- No signal emissions during sync operation
- Return results directly instead of storing in member variables
- Handle errors by logging and returning partial results

### 3. ResultsWindow Data Binding

**Purpose**: Display real duplicate detection results and enable file operations

**New Methods**:
```cpp
public:
    void displayDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups);
    void setFileManager(FileManager* fileManager);
    
private:
    void convertDetectorGroupToDisplayGroup(const DuplicateDetector::DuplicateGroup& source, 
                                            DuplicateGroup& target);
    void updateStatisticsDisplay();
    void removeFilesFromDisplay(const QStringList& filePaths);
```

**New Member Variables**:
```cpp
private:
    FileManager* m_fileManager;  // Reference to file manager for operations
```

**Modified Methods**:
- `deleteSelectedFiles()`: Replace stub with FileManager integration
- `moveSelectedFiles()`: Replace stub with FileManager integration
- `exportResults()`: Implement CSV/JSON export
- `previewSelectedFile()`: Implement basic file preview

**Data Flow**:
1. Receive DuplicateDetector::DuplicateGroup list
2. Convert to ResultsWindow::DuplicateGroup format
3. Populate m_currentResults
4. Call populateResultsTree() to display
5. Update statistics labels

### 4. FileManager Operation Implementation

**Purpose**: Complete missing file operations (restore, backup)

**Restore Operation Implementation**:
```cpp
case OperationType::Restore:
    // 1. Get backup info from SafetyManager
    // 2. Verify backup exists and is valid
    // 3. Check if original location is available
    // 4. Copy backup to original location
    // 5. Verify restore success
    // 6. Update SafetyManager undo history
    fileSuccess = performRestore(sourceFile, operationId);
    break;
```

**Backup Creation Implementation**:
```cpp
case OperationType::CreateBackup:
    // 1. Validate source file exists
    // 2. Call SafetyManager::createBackup()
    // 3. Verify backup was created
    // 4. Log backup location
    fileSuccess = performBackupCreation(sourceFile, operationId);
    break;
```

**New Private Methods**:
```cpp
private:
    bool performRestore(const QString& backupPath, const QString& operationId);
    bool performBackupCreation(const QString& sourceFile, const QString& operationId);
```

**Integration with SafetyManager**:
- Before delete: Call `m_safetyManager->createBackup(filePath)`
- Before move: Call `m_safetyManager->createBackup(filePath)`
- Check `m_safetyManager->isProtectedFile(filePath)` before operations
- After successful operation: Update SafetyManager undo history

### 5. SafetyManager Backup Validation

**Purpose**: Implement missing backup integrity validation

**Backup Validation Implementation**:
```cpp
bool SafetyManager::validateBackupIntegrity()
{
    // 1. Iterate through all backups in undo history
    // 2. For each backup:
    //    a. Check file exists
    //    b. Verify file size matches original
    //    c. Calculate hash and compare with original
    // 3. Mark invalid backups
    // 4. Return true if all valid, false if any invalid
}
```

**Backup Storage Optimization**:
```cpp
void SafetyManager::optimizeBackupStorage()
{
    // 1. Find old backups (> 30 days)
    // 2. Find large backups (> 100MB)
    // 3. Prompt user for cleanup
    // 4. Remove approved backups
    // 5. Update undo history
}
```

## Data Models

### FileInfo Conversion

**From FileScanner to DuplicateDetector**:
```cpp
DuplicateDetector::FileInfo DuplicateDetector::FileInfo::fromScannerInfo(
    const FileScanner::FileInfo& scanInfo)
{
    DuplicateDetector::FileInfo info;
    info.filePath = scanInfo.absolutePath;
    info.fileSize = scanInfo.size;
    info.fileName = QFileInfo(scanInfo.absolutePath).fileName();
    info.directory = QFileInfo(scanInfo.absolutePath).absolutePath();
    info.lastModified = scanInfo.lastModified;
    // hash will be calculated by DuplicateDetector
    return info;
}
```

**From DuplicateDetector to ResultsWindow**:
```cpp
void ResultsWindow::convertDetectorGroupToDisplayGroup(
    const DuplicateDetector::DuplicateGroup& source,
    ResultsWindow::DuplicateGroup& target)
{
    target.groupId = source.groupId;
    target.totalSize = source.totalSize;
    target.wastedSpace = source.wastedSpace;
    target.fileCount = source.fileCount;
    target.primaryFile = source.recommendedAction;
    
    for (const auto& detectorFile : source.files) {
        DuplicateFile displayFile;
        displayFile.filePath = detectorFile.filePath;
        displayFile.fileName = detectorFile.fileName;
        displayFile.directory = detectorFile.directory;
        displayFile.fileSize = detectorFile.fileSize;
        displayFile.lastModified = detectorFile.lastModified;
        displayFile.hash = detectorFile.hash;
        displayFile.isSelected = false;
        displayFile.isMarkedForDeletion = false;
        target.files.append(displayFile);
    }
}
```

## Error Handling

### Scan to Detection Errors

**Scenario**: FileScanner completes but DuplicateDetector fails to start
- **Detection**: Check if m_duplicateDetector is null
- **Response**: Log error, show message to user, don't crash
- **Recovery**: Allow user to retry or view scan results without duplicates

**Scenario**: File list conversion fails
- **Detection**: Catch exceptions during FileInfo conversion
- **Response**: Log failed files, continue with valid files
- **Recovery**: Process partial results

### Detection to Display Errors

**Scenario**: DuplicateDetector finds no duplicates
- **Detection**: detectionCompleted signal with totalGroups = 0
- **Response**: Show "No duplicates found" message
- **Recovery**: Normal operation, no error

**Scenario**: ResultsWindow fails to display results
- **Detection**: Exception during populateResultsTree()
- **Response**: Log error, show error dialog
- **Recovery**: Allow user to retry or export raw results

### File Operation Errors

**Scenario**: Delete operation fails (permission denied)
- **Detection**: QFile::remove() returns false
- **Response**: Add to failedFiles list, emit operationError signal
- **Recovery**: Continue with remaining files, show summary of failures

**Scenario**: Backup creation fails before delete
- **Detection**: SafetyManager::createBackup() returns false
- **Response**: Abort delete for that file, log warning
- **Recovery**: Skip file, continue with others

**Scenario**: Restore operation fails (backup missing)
- **Detection**: Backup file doesn't exist
- **Response**: Emit operationFailed signal with error message
- **Recovery**: Show error to user, suggest checking backup location

### SafetyManager Errors

**Scenario**: Backup validation finds corrupt backup
- **Detection**: Hash mismatch or file size mismatch
- **Response**: Mark backup as invalid, log warning
- **Recovery**: Notify user, suggest re-creating backup

**Scenario**: Backup directory is full
- **Detection**: QFile::copy() fails with disk full error
- **Response**: Abort operation, emit error signal
- **Recovery**: Prompt user to free space or change backup location

## Testing Strategy

### Unit Tests

**DuplicateDetector::findDuplicatesSync()**:
- Test with empty file list → returns empty list
- Test with files of unique sizes → returns empty list
- Test with duplicate files → returns correct groups
- Test with mixed duplicates and unique files → returns only duplicate groups
- Test error handling → logs warnings, returns partial results

**FileManager Restore Operation**:
- Test restore with valid backup → file restored to original location
- Test restore with missing backup → returns false, emits error
- Test restore with occupied location → prompts for conflict resolution
- Test restore updates undo history → SafetyManager updated

**FileManager Backup Creation**:
- Test backup of existing file → backup created successfully
- Test backup of non-existent file → returns false, emits error
- Test backup with insufficient space → returns false, emits error

**SafetyManager Validation**:
- Test validation with all valid backups → returns true
- Test validation with corrupt backup → returns false, marks invalid
- Test validation with missing backup → returns false, marks invalid

### Integration Tests

**Complete Workflow Test**:
```cpp
void TestIntegration::testCompleteWorkflow()
{
    // 1. Create test files with duplicates
    // 2. Start scan via MainWindow
    // 3. Verify DuplicateDetector is triggered
    // 4. Verify ResultsWindow receives results
    // 5. Select files for deletion
    // 6. Verify FileManager deletes files
    // 7. Verify backups created
    // 8. Verify files removed from filesystem
    // 9. Test restore operation
    // 10. Verify files restored
}
```

**FileScanner to DuplicateDetector Integration**:
- Test scan completion triggers detection
- Test file info conversion is accurate
- Test progress updates are emitted
- Test cancellation during detection

**DuplicateDetector to ResultsWindow Integration**:
- Test results are displayed correctly
- Test statistics are accurate
- Test empty results handled gracefully
- Test large result sets (1000+ groups)

**ResultsWindow to FileManager Integration**:
- Test delete operation actually deletes files
- Test move operation actually moves files
- Test operations update display
- Test operations update statistics
- Test error handling for failed operations

**FileManager to SafetyManager Integration**:
- Test backups created before delete
- Test backups created before move
- Test protected files are skipped
- Test restore functionality works
- Test undo history is maintained

### End-to-End Tests

**User Workflow Test**:
1. User clicks "New Scan"
2. User configures scan settings
3. User starts scan
4. Scan completes, detection starts automatically
5. Results displayed in ResultsWindow
6. User selects files to delete
7. User confirms deletion
8. Files deleted, backups created
9. User realizes mistake, clicks undo
10. Files restored from backup

**Error Scenario Tests**:
- Permission denied during scan
- Disk full during backup
- Network timeout for network drives
- Corrupt files during hash calculation
- User cancels during operation

## Performance Considerations

### Memory Management

**Large File Lists**:
- FileScanner already handles streaming
- DuplicateDetector processes in batches
- ResultsWindow uses lazy loading for thumbnails
- No changes needed, existing design is efficient

**Duplicate Group Display**:
- Use QTreeWidget with lazy expansion
- Generate thumbnails on-demand
- Limit visible items with pagination if needed

### Threading

**Current Threading Model**:
- FileScanner: Uses QThread for scanning
- DuplicateDetector: Uses HashCalculator thread pool
- FileManager: Uses operation queue with timer
- No changes needed, existing model is appropriate

**Signal/Slot Connections**:
- All cross-thread signals use Qt::QueuedConnection (automatic)
- No manual thread synchronization needed
- Qt's event loop handles thread safety

### Progress Reporting

**Frequency**:
- FileScanner: Every 100 files or 1 second
- DuplicateDetector: Every 50 files or 500ms
- FileManager: Every file or 500ms
- Prevents UI freezing while providing feedback

## Security Considerations

### File Operation Safety

**Protected Files**:
- SafetyManager maintains list of protected paths
- FileManager checks before operations
- System files, application files, OS directories blocked

**Backup Verification**:
- Hash verification before delete
- Size verification before delete
- Backup integrity check before restore

**User Confirmation**:
- Confirm before bulk delete (> 10 files)
- Confirm before large delete (> 100MB)
- Show preview of files to be deleted

### Data Privacy

**File Hashes**:
- SHA-256 hashes stored temporarily
- Cleared after detection completes
- Not persisted to disk (unless user exports)

**Scan History**:
- File paths stored in history
- User can clear history
- History stored locally, not transmitted

## Migration and Compatibility

### Backward Compatibility

**Existing Code**:
- All changes are additions, not modifications
- Existing FileScanner functionality unchanged
- Existing UI components unchanged
- No breaking changes to public APIs

**Configuration**:
- No configuration changes needed
- Existing settings remain valid
- New features use sensible defaults

### Future Extensibility

**Plugin Architecture**:
- Design allows for custom duplicate detection algorithms
- FileManager operations can be extended
- ResultsWindow can support custom views

**Export Formats**:
- CSV export for spreadsheet analysis
- JSON export for programmatic processing
- HTML export for sharing reports

## Deployment Considerations

### Build System

**No Changes Required**:
- All new code in existing files
- No new dependencies
- CMake configuration unchanged

### Testing Before Release

**Critical Path Testing**:
1. Scan → Detect → Display workflow
2. Delete operation with backup
3. Restore operation
4. Error handling for common failures

**Platform Testing**:
- Linux (primary platform)
- Windows (if supported)
- macOS (if supported)

### Documentation Updates

**User Documentation**:
- Update user guide with new workflow
- Add troubleshooting section
- Document backup/restore features

**Developer Documentation**:
- Update architecture diagrams
- Document integration points
- Add code examples for extensions

## Summary

This design addresses all critical integration gaps identified in the code review by:

1. **Connecting the Pipeline**: FileScanner → DuplicateDetector → ResultsWindow → FileManager → SafetyManager
2. **Completing Missing Features**: Synchronous detection, restore operation, backup creation, validation
3. **Enabling Core Functionality**: Users can now scan, detect, view, and delete duplicates
4. **Maintaining Safety**: Backups created automatically, restore functionality available
5. **Following Existing Patterns**: Uses established Qt signal/slot architecture, consistent with codebase

The implementation will transform CloneClean from a demonstration application to a functional duplicate file manager while maintaining code quality, safety, and user experience.
