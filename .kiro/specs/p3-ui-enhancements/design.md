# Design Document: P3 UI Enhancements

## Overview

This design document outlines the implementation approach for five P3 enhancement tasks that will improve the user experience of the Duplicate File Finder application. All enhancements build upon existing working features and follow established Qt/C++ patterns used throughout the codebase.

The enhancements are designed to be:
- **Incremental**: Each can be implemented independently
- **Non-breaking**: No changes to existing functionality
- **Performance-conscious**: Minimal overhead on existing operations
- **User-friendly**: Following established UI patterns

## Architecture

### Current Architecture Overview

The application follows a layered architecture:

```
┌─────────────────────────────────────────┐
│         GUI Layer (Qt Widgets)          │
│  MainWindow, ResultsWindow, Dialogs    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       Business Logic Layer              │
│  FileScanner, DuplicateDetector        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Core Services Layer             │
│  FileManager, SafetyManager, Settings  │
└─────────────────────────────────────────┘
```

### Enhancement Integration Points

Each enhancement integrates at specific points:

1. **Scan Configuration** → ScanSetupDialog (GUI Layer)
2. **Scan Progress** → MainWindow status bar + new ProgressDialog (GUI Layer)
3. **Results Display** → ResultsWindow (GUI Layer)
4. **File Selection** → ResultsWindow selection logic (GUI Layer)
5. **File Operations** → FileManager + ResultsWindow (Business + GUI)

## Components and Interfaces

### 1. Enhanced Scan Configuration Dialog

#### New Components

**ExcludePatternWidget**
```cpp
class ExcludePatternWidget : public QWidget {
    Q_OBJECT
public:
    explicit ExcludePatternWidget(QWidget* parent = nullptr);
    
    QStringList getPatterns() const;
    void setPatterns(const QStringList& patterns);
    void addPattern(const QString& pattern);
    void removePattern(const QString& pattern);
    
signals:
    void patternsChanged(const QStringList& patterns);
    
private:
    QListWidget* m_patternList;
    QLineEdit* m_patternInput;
    QPushButton* m_addButton;
    QPushButton* m_removeButton;
};
```

**PresetManagerDialog**
```cpp
class PresetManagerDialog : public QDialog {
    Q_OBJECT
public:
    struct PresetInfo {
        QString name;
        QString description;
        ScanSetupDialog::ScanConfiguration config;
        bool isBuiltIn;
    };
    
    explicit PresetManagerDialog(QWidget* parent = nullptr);
    
    void loadPresets();
    void savePreset(const PresetInfo& preset);
    void deletePreset(const QString& name);
    QList<PresetInfo> getUserPresets() const;
    
signals:
    void presetSelected(const QString& name);
    void presetDeleted(const QString& name);
    
private:
    QListWidget* m_presetList;
    QTextEdit* m_presetDetails;
    QPushButton* m_editButton;
    QPushButton* m_deleteButton;
    QPushButton* m_loadButton;
};
```

**ScanScopePreviewWidget**
```cpp
class ScanScopePreviewWidget : public QWidget {
    Q_OBJECT
public:
    struct ScopeStats {
        int folderCount;
        int estimatedFileCount;
        qint64 estimatedSize;
        QStringList includedPaths;
        QStringList excludedPaths;
    };
    
    explicit ScanScopePreviewWidget(QWidget* parent = nullptr);
    
    void updatePreview(const ScanSetupDialog::ScanConfiguration& config);
    ScopeStats getCurrentStats() const;
    
signals:
    void previewUpdated(const ScopeStats& stats);
    
private:
    QLabel* m_folderCountLabel;
    QLabel* m_fileCountLabel;
    QLabel* m_sizeLabel;
    QTreeWidget* m_pathsTree;
    QTimer* m_updateTimer;
    
    void calculateStats(const ScanSetupDialog::ScanConfiguration& config);
};
```

#### Modified Components

**ScanSetupDialog** - Add new methods:
```cpp
// Validation
QString validateConfiguration() const;
bool isConfigurationValid() const;
void showValidationError(const QString& error);

// Preset management
void openPresetManager();
void saveCustomPreset(const QString& name, const QString& description);
void loadCustomPreset(const QString& name);

// Scope preview
void updateScopePreview();
void showScopePreview(bool show);
```

### 2. Enhanced Scan Progress Display

#### New Components

**ScanProgressDialog**
```cpp
class ScanProgressDialog : public QDialog {
    Q_OBJECT
public:
    struct ProgressInfo {
        int filesScanned;
        int totalFiles;
        qint64 bytesScanned;
        qint64 totalBytes;
        QString currentFolder;
        QString currentFile;
        double filesPerSecond;
        int secondsRemaining;
        bool isPaused;
    };
    
    explicit ScanProgressDialog(QWidget* parent = nullptr);
    
    void updateProgress(const ProgressInfo& info);
    void setPaused(bool paused);
    bool isPaused() const;
    
signals:
    void pauseRequested();
    void resumeRequested();
    void cancelRequested();
    
private:
    QProgressBar* m_overallProgress;
    QLabel* m_filesLabel;
    QLabel* m_sizeLabel;
    QLabel* m_currentFolderLabel;
    QLabel* m_currentFileLabel;
    QLabel* m_rateLabel;
    QLabel* m_etaLabel;
    QPushButton* m_pauseButton;
    QPushButton* m_cancelButton;
    QElapsedTimer m_scanTimer;
    
    void updateETA(const ProgressInfo& info);
    QString formatTime(int seconds) const;
};
```

#### Modified Components

**FileScanner** - Add progress tracking:
```cpp
struct ScanProgress {
    int filesScanned;
    qint64 bytesScanned;
    QString currentFolder;
    QString currentFile;
    QElapsedTimer timer;
};

signals:
    void detailedProgress(const ScanProgress& progress);
    void scanPaused();
    void scanResumed();

public slots:
    void pauseScan();
    void resumeScan();
```

### 3. Enhanced Results Display

#### New Components

**ThumbnailCache**
```cpp
class ThumbnailCache : public QObject {
    Q_OBJECT
public:
    explicit ThumbnailCache(QObject* parent = nullptr);
    
    QPixmap getThumbnail(const QString& filePath, const QSize& size);
    void preloadThumbnails(const QStringList& filePaths, const QSize& size);
    void clearCache();
    void setCacheSize(int maxItems);
    
signals:
    void thumbnailReady(const QString& filePath, const QPixmap& thumbnail);
    
private:
    QCache<QString, QPixmap> m_cache;
    QThreadPool* m_threadPool;
    
    QPixmap generateThumbnail(const QString& filePath, const QSize& size);
    QPixmap generateVideoThumbnail(const QString& filePath, const QSize& size);
};
```

**ThumbnailDelegate**
```cpp
class ThumbnailDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    explicit ThumbnailDelegate(ThumbnailCache* cache, QObject* parent = nullptr);
    
    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option,
                   const QModelIndex& index) const override;
    
private:
    ThumbnailCache* m_cache;
    static const int THUMBNAIL_SIZE = 48;
};
```

**AdvancedFilterDialog**
```cpp
class AdvancedFilterDialog : public QDialog {
    Q_OBJECT
public:
    struct FilterCriteria {
        QDateTime dateFrom;
        QDateTime dateTo;
        QStringList extensions;
        QString pathPattern;
        qint64 minSize;
        qint64 maxSize;
        bool useDate;
        bool useExtensions;
        bool usePath;
        bool useSize;
    };
    
    explicit AdvancedFilterDialog(QWidget* parent = nullptr);
    
    FilterCriteria getCriteria() const;
    void setCriteria(const FilterCriteria& criteria);
    void saveAsPreset(const QString& name);
    void loadPreset(const QString& name);
    
signals:
    void filterApplied(const FilterCriteria& criteria);
    
private:
    QDateTimeEdit* m_dateFrom;
    QDateTimeEdit* m_dateTo;
    QLineEdit* m_extensions;
    QLineEdit* m_pathPattern;
    QSpinBox* m_minSize;
    QSpinBox* m_maxSize;
    QCheckBox* m_useDateCheck;
    QCheckBox* m_useExtensionsCheck;
    QCheckBox* m_usePathCheck;
    QCheckBox* m_useSizeCheck;
};
```

**GroupingOptionsDialog**
```cpp
class GroupingOptionsDialog : public QDialog {
    Q_OBJECT
public:
    enum GroupingMode {
        ByHash,
        BySize,
        ByType,
        ByDate,
        ByLocation,
        Custom
    };
    
    explicit GroupingOptionsDialog(QWidget* parent = nullptr);
    
    GroupingMode getGroupingMode() const;
    void setGroupingMode(GroupingMode mode);
    
signals:
    void groupingChanged(GroupingMode mode);
    
private:
    QComboBox* m_groupingCombo;
    QWidget* m_customOptions;
};
```

#### Modified Components

**ResultsWindow** - Add new methods:
```cpp
// Thumbnail support
void enableThumbnails(bool enable);
void setThumbnailSize(int size);
void preloadVisibleThumbnails();

// Advanced filtering
void showAdvancedFilter();
void applyAdvancedFilter(const AdvancedFilterDialog::FilterCriteria& criteria);
void clearAdvancedFilter();

// Grouping
void setGroupingMode(GroupingOptionsDialog::GroupingMode mode);
void regroupResults();

// Export with thumbnails
void exportResultsWithThumbnails(const QString& filePath);
```

### 4. Enhanced File Selection

#### New Components

**SelectionHistoryManager**
```cpp
class SelectionHistoryManager : public QObject {
    Q_OBJECT
public:
    struct SelectionState {
        QStringList selectedFiles;
        QDateTime timestamp;
        QString description;
    };
    
    explicit SelectionHistoryManager(QObject* parent = nullptr);
    
    void pushState(const SelectionState& state);
    SelectionState undo();
    SelectionState redo();
    bool canUndo() const;
    bool canRedo() const;
    void clear();
    
signals:
    void historyChanged();
    void undoAvailable(bool available);
    void redoAvailable(bool available);
    
private:
    QStack<SelectionState> m_undoStack;
    QStack<SelectionState> m_redoStack;
    static const int MAX_HISTORY = 50;
};
```

**SmartSelectionDialog**
```cpp
class SmartSelectionDialog : public QDialog {
    Q_OBJECT
public:
    enum SelectionMode {
        OldestFiles,
        NewestFiles,
        LargestFiles,
        SmallestFiles,
        ByPath,
        ByCriteria
    };
    
    struct SelectionCriteria {
        SelectionMode mode;
        QString pathPattern;
        QDateTime dateFrom;
        QDateTime dateTo;
        qint64 minSize;
        qint64 maxSize;
        bool useAnd; // true = AND, false = OR
    };
    
    explicit SmartSelectionDialog(QWidget* parent = nullptr);
    
    SelectionCriteria getCriteria() const;
    void setCriteria(const SelectionCriteria& criteria);
    
signals:
    void selectionRequested(const SelectionCriteria& criteria);
    
private:
    QComboBox* m_modeCombo;
    QWidget* m_criteriaWidget;
    QLineEdit* m_pathPattern;
    QDateTimeEdit* m_dateFrom;
    QDateTimeEdit* m_dateTo;
    QSpinBox* m_minSize;
    QSpinBox* m_maxSize;
    QRadioButton* m_andRadio;
    QRadioButton* m_orRadio;
};
```

**SelectionPresetManager**
```cpp
class SelectionPresetManager : public QObject {
    Q_OBJECT
public:
    struct SelectionPreset {
        QString name;
        QString description;
        SmartSelectionDialog::SelectionCriteria criteria;
    };
    
    explicit SelectionPresetManager(QObject* parent = nullptr);
    
    void savePreset(const SelectionPreset& preset);
    void deletePreset(const QString& name);
    SelectionPreset loadPreset(const QString& name) const;
    QStringList getPresetNames() const;
    
signals:
    void presetSaved(const QString& name);
    void presetDeleted(const QString& name);
    
private:
    QSettings* m_settings;
    QString getPresetKey(const QString& name) const;
};
```

#### Modified Components

**ResultsWindow** - Add selection enhancements:
```cpp
// Selection history
void recordSelectionState(const QString& description);
void undoSelection();
void redoSelection();

// Smart selection
void showSmartSelectionDialog();
void applySmartSelection(const SmartSelectionDialog::SelectionCriteria& criteria);
void invertSelection();

// Selection presets
void saveSelectionPreset(const QString& name);
void loadSelectionPreset(const QString& name);
void manageSelectionPresets();
```

### 5. Enhanced File Operations

#### New Components

**FileOperationQueue**
```cpp
class FileOperationQueue : public QObject {
    Q_OBJECT
public:
    enum OperationType {
        Delete,
        Move,
        Copy
    };
    
    struct Operation {
        QString id;
        OperationType type;
        QStringList sourcePaths;
        QString destination;
        int filesProcessed;
        int totalFiles;
        qint64 bytesProcessed;
        qint64 totalBytes;
        QDateTime startTime;
        QDateTime endTime;
        QString status; // "queued", "running", "completed", "failed", "cancelled"
        QStringList errors;
    };
    
    explicit FileOperationQueue(QObject* parent = nullptr);
    
    QString queueOperation(const Operation& op);
    void cancelOperation(const QString& id);
    void retryFailedFiles(const QString& id);
    Operation getOperation(const QString& id) const;
    QList<Operation> getAllOperations() const;
    QList<Operation> getActiveOperations() const;
    void clearCompleted();
    
signals:
    void operationQueued(const QString& id);
    void operationStarted(const QString& id);
    void operationProgress(const QString& id, int filesProcessed, qint64 bytesProcessed);
    void operationCompleted(const QString& id);
    void operationFailed(const QString& id, const QString& error);
    void operationCancelled(const QString& id);
    
private:
    QQueue<Operation> m_queue;
    QMap<QString, Operation> m_operations;
    QThreadPool* m_threadPool;
    
    void processNextOperation();
    void executeOperation(Operation& op);
};
```

**FileOperationProgressDialog**
```cpp
class FileOperationProgressDialog : public QDialog {
    Q_OBJECT
public:
    explicit FileOperationProgressDialog(QWidget* parent = nullptr);
    
    void setOperation(const FileOperationQueue::Operation& op);
    void updateProgress(int filesProcessed, qint64 bytesProcessed);
    void setCompleted(bool success, const QStringList& errors);
    
signals:
    void cancelRequested();
    void retryRequested();
    
private:
    QLabel* m_operationLabel;
    QProgressBar* m_fileProgress;
    QProgressBar* m_byteProgress;
    QLabel* m_statusLabel;
    QLabel* m_speedLabel;
    QTextEdit* m_errorLog;
    QPushButton* m_cancelButton;
    QPushButton* m_retryButton;
    QElapsedTimer m_timer;
};
```

**OperationHistoryDialog**
```cpp
class OperationHistoryDialog : public QDialog {
    Q_OBJECT
public:
    explicit OperationHistoryDialog(FileOperationQueue* queue, QWidget* parent = nullptr);
    
    void refreshHistory();
    void showOperationDetails(const QString& id);
    
signals:
    void retryRequested(const QString& id);
    void viewDetailsRequested(const QString& id);
    
private:
    FileOperationQueue* m_queue;
    QTableWidget* m_historyTable;
    QTextEdit* m_detailsView;
    QPushButton* m_retryButton;
    QPushButton* m_clearButton;
    
    void populateTable();
    QString formatOperationType(FileOperationQueue::OperationType type) const;
};
```

#### Modified Components

**FileManager** - Integrate with queue:
```cpp
// Queue integration
void setOperationQueue(FileOperationQueue* queue);
QString queueDeleteOperation(const QStringList& files);
QString queueMoveOperation(const QStringList& files, const QString& destination);

// Progress reporting
signals:
    void operationProgress(const QString& opId, int filesProcessed, qint64 bytesProcessed);
```

**ResultsWindow** - Add operation management:
```cpp
// Operation queue
void showOperationQueue();
void showOperationHistory();
void cancelCurrentOperation();

// Enhanced operations
void deleteSelectedFilesQueued();
void moveSelectedFilesQueued();
```

## Data Models

### Preset Storage

**Scan Presets** (QSettings):
```
presets/scan/
  ├── custom_preset_1/
  │   ├── name
  │   ├── description
  │   ├── targetPaths
  │   ├── detectionMode
  │   ├── minimumFileSize
  │   ├── excludePatterns
  │   └── ...
  └── custom_preset_2/
      └── ...
```

**Selection Presets** (QSettings):
```
presets/selection/
  ├── preset_1/
  │   ├── name
  │   ├── description
  │   ├── mode
  │   ├── criteria
  │   └── ...
  └── preset_2/
      └── ...
```

**Filter Presets** (QSettings):
```
presets/filter/
  ├── preset_1/
  │   ├── name
  │   ├── dateFrom
  │   ├── dateTo
  │   ├── extensions
  │   └── ...
  └── preset_2/
      └── ...
```

### Operation Queue Storage

**In-Memory** (during session):
```cpp
QMap<QString, FileOperationQueue::Operation> m_operations;
```

**Persistent** (QSettings for history):
```
operations/history/
  ├── operation_1/
  │   ├── type
  │   ├── timestamp
  │   ├── filesProcessed
  │   ├── status
  │   └── errors
  └── operation_2/
      └── ...
```

### Thumbnail Cache

**In-Memory Cache**:
```cpp
QCache<QString, QPixmap> m_thumbnailCache;
// Key: filePath + size
// Value: QPixmap thumbnail
// Max size: 100 items (configurable)
```

**Disk Cache** (optional, for persistence):
```
~/.cache/cloneclean/thumbnails/
  ├── [hash1].png
  ├── [hash2].png
  └── ...
```

## Error Handling

### Validation Errors

**Scan Configuration**:
- Empty target paths → Show error dialog, disable Start button
- Invalid exclude patterns → Highlight pattern, show tooltip
- Inaccessible paths → Show warning, allow proceeding with accessible paths

**Selection Criteria**:
- Invalid date range → Show error message, reset to defaults
- Invalid path pattern → Show warning, disable pattern matching
- Conflicting criteria → Show warning, explain conflict

### Operation Errors

**File Operations**:
- Permission denied → Log error, continue with next file
- File not found → Log warning, skip file
- Disk full → Stop operation, show error dialog
- Operation cancelled → Clean up partial changes, restore if possible

**Thumbnail Generation**:
- Corrupt image → Use default icon, log warning
- Unsupported format → Use file type icon
- Generation timeout → Skip thumbnail, use placeholder

### Recovery Strategies

1. **Graceful Degradation**: If thumbnails fail, show file icons
2. **Partial Success**: Complete what's possible, report failures
3. **Retry Logic**: Allow retrying failed operations
4. **State Preservation**: Save operation state for recovery

## Testing Strategy

### Unit Tests

**Component Tests**:
```cpp
// Test exclude pattern validation
void testExcludePatternValidation();
void testExcludePatternMatching();

// Test selection history
void testSelectionHistoryPushPop();
void testSelectionHistoryUndo();
void testSelectionHistoryRedo();

// Test operation queue
void testOperationQueueing();
void testOperationCancellation();
void testOperationRetry();

// Test thumbnail cache
void testThumbnailCaching();
void testThumbnailEviction();
```

### Integration Tests

**UI Integration**:
```cpp
// Test scan configuration flow
void testScanConfigurationWithPresets();
void testScanConfigurationValidation();

// Test results display with thumbnails
void testResultsDisplayWithThumbnails();
void testResultsFiltering();

// Test file operations
void testQueuedFileOperations();
void testOperationProgress();
```

### Manual Testing

**User Workflows**:
1. Create custom scan preset → Save → Load → Verify settings
2. Start scan → Pause → Resume → Complete
3. View results → Apply filters → Select files → Delete
4. Select files → Undo → Redo → Verify selection
5. Queue multiple operations → Monitor progress → View history

**Edge Cases**:
1. Very large result sets (10,000+ files)
2. Many thumbnails (1,000+ images)
3. Long-running operations (1+ hour)
4. Network paths (if supported)
5. Symlink loops
6. Permission issues

### Performance Tests

**Benchmarks**:
```cpp
// Thumbnail generation performance
void benchmarkThumbnailGeneration();

// Filter performance on large datasets
void benchmarkFilteringLargeResults();

// Selection operation performance
void benchmarkSelectionOperations();

// Queue processing performance
void benchmarkOperationQueue();
```

**Targets**:
- Thumbnail generation: < 100ms per image
- Filter application: < 500ms for 10,000 files
- Selection undo/redo: < 50ms
- Operation queueing: < 10ms

## Implementation Notes

### Qt Best Practices

1. **Signal/Slot Connections**: Use new-style connections where possible
2. **Memory Management**: Use Qt parent-child ownership
3. **Threading**: Use QThreadPool for background tasks
4. **Settings**: Use QSettings for persistence
5. **Styling**: Use stylesheets for consistent theming

### Performance Considerations

1. **Lazy Loading**: Load thumbnails on-demand, not all at once
2. **Caching**: Cache expensive operations (thumbnails, filters)
3. **Debouncing**: Debounce real-time updates (estimation, preview)
4. **Pagination**: Consider pagination for very large result sets
5. **Background Processing**: Move heavy operations to background threads

### Accessibility

1. **Keyboard Navigation**: All features accessible via keyboard
2. **Screen Readers**: Proper labels and ARIA attributes
3. **High Contrast**: Support high contrast themes
4. **Font Scaling**: Respect system font size settings

### Internationalization

1. **Translatable Strings**: Use tr() for all user-facing strings
2. **Date/Time Formatting**: Use QLocale for formatting
3. **Number Formatting**: Use QLocale for number formatting
4. **RTL Support**: Test with RTL languages

## Dependencies

### Existing Dependencies
- Qt 6.5+ (Core, Widgets, Gui)
- C++17 or later
- Existing core components (FileScanner, DuplicateDetector, etc.)

### New Dependencies
- None (all features use existing Qt modules)

### Optional Dependencies
- FFmpeg (for video thumbnail generation) - fallback to first frame extraction
- libexif (for EXIF data) - fallback to basic file info

## Migration Path

### Phase 1: Foundation (Week 1)
- Implement base classes (ThumbnailCache, SelectionHistoryManager, FileOperationQueue)
- Add new methods to existing classes
- Create unit tests

### Phase 2: UI Components (Week 2)
- Implement new dialogs and widgets
- Integrate with existing UI
- Add keyboard shortcuts

### Phase 3: Integration (Week 3)
- Wire up all components
- Implement signal/slot connections
- Add persistence

### Phase 4: Polish (Week 4)
- Performance optimization
- Bug fixes
- Documentation
- User testing

## Rollback Plan

If issues arise:
1. **Feature Flags**: Disable individual enhancements via settings
2. **Graceful Degradation**: Fall back to basic functionality
3. **Version Control**: Revert specific commits if needed
4. **User Feedback**: Collect feedback, iterate on design

## Future Enhancements

Potential future improvements:
1. Cloud storage integration
2. Advanced duplicate resolution algorithms
3. Batch preset execution
4. Scheduled scans
5. Plugin system for custom filters
6. Machine learning-based recommendations
