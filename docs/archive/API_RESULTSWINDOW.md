# ResultsWindow API Documentation

**Version:** 1.0  
**Created:** 2025-10-04  
**Component:** Results Dashboard Interface  
**Files:** `src/gui/results_window.h`, `src/gui/results_window.cpp`  

---

## Overview

The `ResultsWindow` class provides a comprehensive interface for displaying and managing duplicate file scan results. It features a professional three-panel layout with advanced file operations, selection management, and safety features.

### Key Features
- **Three-Panel Layout:** Header, Results Tree, Actions Panel
- **Hierarchical Display:** Groups and individual duplicate files
- **Smart Selection:** Automatic recommendations and bulk operations
- **File Operations:** Preview, delete, move, ignore with safety confirmations
- **Real-time Updates:** Live statistics and selection summaries

---

## Class Declaration

```cpp
class ResultsWindow : public QMainWindow
{
    Q_OBJECT

public:
    // Window size constants
    static const QSize MIN_WINDOW_SIZE;      // 800x600
    static const QSize DEFAULT_WINDOW_SIZE;  // 1200x800

    // Constructor and destructor
    explicit ResultsWindow(QWidget* parent = nullptr);
    virtual ~ResultsWindow();

    // Public interface methods
    void displayResults(const ScanResults& results);
    void clearResults();
    void updateProgress(const QString& operation, int percentage);

public slots:
    void refreshResults();
    void exportResults();

signals:
    void windowClosed();
    void fileOperationRequested(const QString& operation, const QStringList& files);
    void resultsUpdated(const ScanResults& results);

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;

    // ... private implementation details
};
```

---

## Data Structures

### DuplicateFile Structure

```cpp
struct DuplicateFile {
    QString filePath;           // Full path to the file
    QString fileName;           // File name only
    QString directory;          // Parent directory
    qint64 fileSize = 0;       // File size in bytes
    QDateTime lastModified;     // Last modification time
    QDateTime created;          // Creation time
    QString hash;               // File hash (SHA-256)
    QPixmap thumbnail;          // Preview thumbnail (for images)
    bool isSelected = false;    // Selection state
    bool isMarkedForDeletion = false;  // Deletion marker
    QString fileType;           // Human-readable file type
    
    // Comparison operator
    bool operator==(const DuplicateFile& other) const;
};
```

### DuplicateGroup Structure

```cpp
struct DuplicateGroup {
    QString groupId;                    // Unique group identifier
    QList<DuplicateFile> files;        // Files in this group
    int fileCount = 0;                  // Number of files
    qint64 totalSize = 0;              // Total size of all files
    QString primaryFile;                // Recommended file to keep
    bool isExpanded = false;            // UI expansion state
    bool hasSelection = false;          // Has selected files
    
    // Helper methods
    void calculateSize();               // Recalculate group statistics
    qint64 getWastedSpace() const;     // Calculate potential savings
};
```

### ScanResults Structure

```cpp
struct ScanResults {
    QString scanPath;                   // Root scan directory
    QDateTime scanTime;                 // When scan was performed
    QString scanDuration;               // Human-readable duration
    int totalFilesScanned = 0;         // Total files processed
    int totalDuplicatesFound = 0;      // Total duplicate files
    qint64 potentialSavings = 0;       // Total reclaimable space
    QList<DuplicateGroup> duplicateGroups;  // All duplicate groups
    
    // Helper methods
    void calculateTotals();             // Recalculate summary statistics
    ScanResults() = default;            // Default constructor
};
```

---

## Public Interface Methods

### Core Display Methods

#### `displayResults(const ScanResults& results)`
**Purpose:** Display scan results in the window  
**Parameters:** 
- `results` - Complete scan results with duplicate groups
**Usage:**
```cpp
ResultsWindow* resultsWindow = new ResultsWindow(this);
ScanResults results = performScan();
resultsWindow->displayResults(results);
resultsWindow->show();
```

#### `clearResults()`
**Purpose:** Clear all displayed results and reset the interface  
**Usage:**
```cpp
resultsWindow->clearResults(); // Resets to empty state
```

#### `updateProgress(const QString& operation, int percentage)`
**Purpose:** Update progress during operations  
**Parameters:**
- `operation` - Description of current operation
- `percentage` - Progress percentage (0-100)
**Usage:**
```cpp
resultsWindow->updateProgress("Deleting files...", 75);
```

---

## Public Slots

### `refreshResults()`
**Purpose:** Refresh the current results display  
**Behavior:** Repopulates the tree widget and updates statistics  
**Usage:** Connected to refresh button or called programmatically

### `exportResults()`
**Purpose:** Export results to external file  
**Behavior:** Opens file dialog and saves results in selected format  
**Supported Formats:** TXT, CSV
**Usage:** Connected to export button

---

## Signals

### `windowClosed()`
**Purpose:** Emitted when the results window is closed  
**Usage:** Connect to handle cleanup in parent window
```cpp
connect(resultsWindow, &ResultsWindow::windowClosed, 
        this, [this]() { 
            // Handle window closed
        });
```

### `fileOperationRequested(const QString& operation, const QStringList& files)`
**Purpose:** Emitted when user requests file operations  
**Parameters:**
- `operation` - Operation type ("delete", "move", "ignore")
- `files` - List of file paths to operate on
**Usage:** Connect to handle actual file operations
```cpp
connect(resultsWindow, &ResultsWindow::fileOperationRequested,
        this, [this](const QString& op, const QStringList& files) {
            handleFileOperation(op, files);
        });
```

### `resultsUpdated(const ScanResults& results)`
**Purpose:** Emitted when results data changes  
**Parameters:**
- `results` - Updated results data
**Usage:** Connect to update external statistics
```cpp
connect(resultsWindow, &ResultsWindow::resultsUpdated,
        mainWindow, &MainWindow::updateResultsStats);
```

---

## Selection Management API

### Selection Methods

#### `getSelectedFiles() const`
**Purpose:** Get list of currently selected files  
**Returns:** `QList<DuplicateFile>` - Selected files  
**Usage:**
```cpp
QList<ResultsWindow::DuplicateFile> selected = resultsWindow->getSelectedFiles();
```

#### `getSelectedFilesSize() const`
**Purpose:** Calculate total size of selected files  
**Returns:** `qint64` - Total size in bytes  

#### `getSelectedFilesCount() const`
**Purpose:** Get count of selected files  
**Returns:** `int` - Number of selected files  

### Selection Actions

#### `selectAllDuplicates()`
**Purpose:** Select all duplicate files (excluding recommended to keep)  

#### `selectNoneFiles()`
**Purpose:** Clear all file selections  

#### `selectRecommended()`
**Purpose:** Select all non-recommended files (smart selection)  

#### `selectBySize(qint64 minSize)`
**Purpose:** Select files above minimum size  
**Parameters:**
- `minSize` - Minimum file size in bytes

#### `selectByType(const QString& fileType)`
**Purpose:** Select files of specific type  
**Parameters:**
- `fileType` - File type ("image", "video", etc.)

---

## File Operations API

### Individual File Operations

#### `deleteSelectedFiles()`
**Purpose:** Delete selected files with confirmation  
**Behavior:** 
- Shows confirmation dialog with file list
- Moves files to system trash (not permanent deletion)
- Updates UI after operation

#### `moveSelectedFiles()`
**Purpose:** Move selected files to different location  
**Behavior:**
- Opens folder selection dialog
- Moves files to selected destination
- Updates UI after operation

#### `ignoreSelectedFiles()`
**Purpose:** Mark files as ignored (exclude from future operations)  

#### `previewSelectedFile()`
**Purpose:** Show preview of selected file  
**Behavior:**
- Currently shows placeholder dialog
- Designed for future image/document preview

#### `openFileLocation()`
**Purpose:** Open file location in system file manager  
**Behavior:** Uses `QDesktopServices::openUrl()` to open folder

#### `copyFilePath()`
**Purpose:** Copy file path to system clipboard  
**Behavior:** Uses `QApplication::clipboard()` to copy path

### Bulk Operations

#### `performBulkDelete()`
**Purpose:** Delete multiple selected files with detailed confirmation  
**Features:**
- Shows file count and total size
- Detailed confirmation dialog
- Safety warnings and information

#### `performBulkMove()`
**Purpose:** Move multiple selected files to single destination  
**Features:**
- Folder selection dialog
- Bulk move with progress indication
- Error handling for failed operations

---

## Utility Methods API

### File Information

#### `formatFileSize(qint64 bytes) const`
**Purpose:** Format file size for display  
**Returns:** Formatted string (e.g., "1.5 GB", "234 MB")  
**Usage:**
```cpp
QString sizeStr = formatFileSize(1536000000); // "1.5 GB"
```

#### `getRecommendedFileToKeep(const DuplicateGroup& group) const`
**Purpose:** Determine which file in group should be kept  
**Algorithm:** Recommends oldest file (likely original)  
**Returns:** File path of recommended file

#### `isImageFile(const QString& filePath) const`
**Purpose:** Check if file is an image  
**Supported Extensions:** jpg, jpeg, png, bmp, gif, tiff, webp  
**Returns:** `bool` - true if image file

#### `isVideoFile(const QString& filePath) const`
**Purpose:** Check if file is a video  
**Supported Extensions:** mp4, avi, mkv, mov, wmv, flv, webm  
**Returns:** `bool` - true if video file

---

## UI Layout Structure

### Three-Panel Layout

#### Header Panel
**Purpose:** Title, summary, and main actions  
**Components:**
- Title label with icon
- Summary statistics
- Action buttons (Refresh, Export, Settings)

#### Results Panel (Main)
**Purpose:** Display duplicate groups and files  
**Components:**
- Filter controls (search, size filter, type filter, sort options)
- Selection controls (select all, recommended, clear)
- Tree widget with hierarchical display
- Selection summary

#### Actions Panel (Right)
**Purpose:** File operations and bulk actions  
**Components:**
- File Actions group (individual operations)
- Bulk Actions group (multi-file operations)
- Safety warnings and status

#### Status Bar
**Purpose:** Real-time statistics and progress  
**Components:**
- Current operation status
- File and group counters
- Progress bar (when applicable)

---

## Integration Example

### Complete Integration with MainWindow

```cpp
class MainWindow : public QMainWindow {
public slots:
    void showScanResults() {
        if (!m_resultsWindow) {
            // Create results window
            m_resultsWindow = new ResultsWindow(this);
            
            // Connect signals
            connect(m_resultsWindow, &ResultsWindow::windowClosed,
                    this, [this]() {
                        // Handle cleanup
                    });
            
            connect(m_resultsWindow, &ResultsWindow::fileOperationRequested,
                    this, [this](const QString& operation, const QStringList& files) {
                        // Handle file operations
                        handleFileOperations(operation, files);
                    });
            
            connect(m_resultsWindow, &ResultsWindow::resultsUpdated,
                    this, [this](const ResultsWindow::ScanResults& results) {
                        // Update main window statistics
                        updateMainWindowStats(results);
                    });
        }
        
        // Show the results window
        m_resultsWindow->show();
        m_resultsWindow->raise();
        m_resultsWindow->activateWindow();
    }

private:
    ResultsWindow* m_resultsWindow = nullptr;
    
    void handleFileOperations(const QString& operation, const QStringList& files) {
        if (operation == "delete") {
            // Use platform-specific trash operations
            for (const QString& file : files) {
                TrashManager::moveToTrash(file);
            }
        } else if (operation == "move") {
            // Handle file moving
        }
        // Update results display after operations
    }
    
    void updateMainWindowStats(const ResultsWindow::ScanResults& results) {
        // Update status bar with latest statistics
        statusBar()->showMessage(
            QString("Found %1 duplicates, %2 potential savings")
            .arg(results.totalDuplicatesFound)
            .arg(formatFileSize(results.potentialSavings))
        );
    }
};
```

---

## Best Practices

### Memory Management
- ResultsWindow uses Qt's parent-child ownership model
- All child widgets are automatically cleaned up
- Large data sets are handled efficiently with lazy loading

### Thread Safety
- All UI operations must be performed on the main thread
- File operations should use separate worker threads
- Progress updates use Qt's signal/slot mechanism for thread safety

### Error Handling
- All file operations include comprehensive error checking
- User-friendly error messages with recovery suggestions
- Graceful degradation when operations fail

### Performance Considerations
- Tree widget uses incremental population for large result sets
- Thumbnails are generated on-demand and cached
- Selection operations are optimized for bulk handling

---

## Future Enhancements

### Planned Features
- Advanced file preview with image thumbnails
- Filter persistence and custom filter creation
- Export to additional formats (JSON, XML)
- Undo/redo system for file operations
- Integration with cloud storage services

### Extensibility Points
- Custom file type handlers
- Plugin system for additional file operations
- Theming and customization options
- Advanced reporting and analytics

---

## See Also

- [MainWindow API Documentation](API_MAINWINDOW.md)
- [Core Engine API Documentation](API_CORE.md)
- [UI Design Specification](UI_DESIGN_SPECIFICATION.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)

---

**Last Updated:** 2025-10-04  
**Next Review:** 2025-10-11  
**Maintained By:** CloneClean Development Team