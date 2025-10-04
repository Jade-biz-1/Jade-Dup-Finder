# DupFinder UI Design Specification

**Version:** 1.0  
**Created:** 2025-10-04  
**Based on:** PRD v1.0, Architecture Design v1.0  
**Target Platform:** Cross-platform (Linux → Windows → macOS)  

---

## Executive Summary

This document defines the complete user interface design for DupFinder, a cross-platform duplicate file finder application. The design focuses on creating an intuitive, modern interface that serves non-technical users while providing powerful functionality for duplicate file management.

### Design Principles
- **Safety First:** Clear visual feedback, confirmation dialogs, no permanent deletion
- **User-Centric:** Simple workflows for "Storage-Conscious Sarah" persona
- **Modern & Clean:** Contemporary design language with Qt6 native look
- **Information Rich:** Compact layouts showing relevant details upfront
- **Cross-Platform:** Native appearance on Linux, Windows, and macOS

---

## 1. Main Window Design

### 1.1 Layout: Modern Dashboard (Selected)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ DupFinder                                            [─] [□] [✕]                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 📁 New Scan    ⚙️ Settings    ❓ Help                    🔔    👤 Free Plan      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─── Quick Actions ────────────────────────────────────────────────────────┐  │
│  │  [🚀 Start Quick Scan]  [📂 Downloads Cleanup]  [📸 Photo Cleanup]     │  │
│  │  [📄 Documents]         [🖥️ Full System Scan]   [⭐ Custom Preset]      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─── Scan History ──────────────────────────────────────────────────────────┐ │
│  │  📅 Today, 2:30 PM     │ 📁 Downloads        │ 127 duplicates │ 2.3 GB   │ │
│  │  📅 Yesterday, 6:15 PM │ 📸 Pictures         │ 89 duplicates  │ 1.8 GB   │ │
│  │  📅 Oct 1, 3:45 PM     │ 🖥️ Full System      │ 341 duplicates │ 5.1 GB   │ │
│  │                                                               [View All →] │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌─── System Overview ───────────────────────────────────────────────────────┐ │
│  │  💾 Total Disk Space: 512 GB    🟢 Available: 127 GB (25%)               │ │
│  │  🗑️ Potential Savings: 8.2 GB   📊 Files Scanned: 45,231                │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Ready │ Files: 0 │ Groups: 0 │ Potential savings: 0 MB                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Implementation Details

#### 1.2.1 Window Properties
- **Minimum Size:** 800x600 pixels
- **Default Size:** 1024x768 pixels
- **Resizable:** Yes, with minimum constraints
- **Theme Support:** Light/Dark mode automatic detection

#### 1.2.2 Component Specifications

**Header Bar (`QWidget` with custom layout):**
- Application title with icon (24x24 px)
- Quick action buttons: New Scan, Settings, Help
- Status indicators: Notifications, Plan type
- Platform-specific window controls

**Quick Actions Panel (`QGroupBox` with `QGridLayout`):**
- 6 preset buttons in 2x3 grid
- Button size: 180x60 pixels
- Icons: 32x32 px with descriptive text
- Hover effects with system colors

**Scan History Panel (`QGroupBox` with `QListWidget`):**
- Last 3 scans with expandable "View All" option
- Custom list items with 4 columns: Date, Type, Results, Savings
- Row height: 32 pixels
- Alternating row colors

**System Overview Panel (`QGroupBox` with custom widget):**
- Disk space visualization with progress bars
- Statistics with icon + text layout
- Color-coded space indicators (green/yellow/red)

**Status Bar (`QStatusBar`):**
- Current operation status
- File/group counters
- Progress information during scans

### 1.3 Qt6 Implementation Structure

```cpp
class MainWindow : public QMainWindow {
    Q_OBJECT
    
private:
    // UI Components
    QWidget* m_centralWidget;
    QVBoxLayout* m_mainLayout;
    
    // Header
    QWidget* m_headerWidget;
    QHBoxLayout* m_headerLayout;
    QPushButton* m_newScanButton;
    QPushButton* m_settingsButton;
    QPushButton* m_helpButton;
    QLabel* m_planIndicator;
    
    // Content Areas
    QuickActionsWidget* m_quickActions;
    ScanHistoryWidget* m_scanHistory;
    SystemOverviewWidget* m_systemOverview;
    
    // Status
    QStatusBar* m_statusBar;
    QProgressBar* m_progressBar;
    
public slots:
    void onNewScanRequested();
    void onPresetSelected(const QString& preset);
    void updateSystemInfo();
    void showScanResults();
};
```

---

## 2. Scan Setup Dialog Design

### 2.1 Layout: Compact Configuration Panel (Selected)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ New Scan Configuration                                               [─] [✕]   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ ┌─ Locations ─────────────────────┐ ┌─ Options ────────────────────────────────┐│
│ │ 📁 Scan Locations:              │ │ Detection: [Smart ▼]                    ││
│ │ ┌─────────────────────────────┐ │ │ Min Size:  [1 MB ▼]                     ││
│ │ │☑️ /home/deepak/Downloads     │ │ │ Max Depth: [Unlimited ▼]               ││
│ │ │☑️ /home/deepak/Pictures      │ │ │                                         ││
│ │ │☐ /home/deepak/Documents     │ │ │ Include:                                ││
│ │ │☐ /home/deepak/Videos        │ │ │ ☑️ Hidden files  ☐ System files         ││
│ │ │☐ /home/deepak/Music         │ │ │ ☑️ Symlinks      ☐ Archives             ││
│ │ │ + Add Folder...             │ │ │                                         ││
│ │ └─────────────────────────────┘ │ │ File Types:                             ││
│ │                                 │ │ ☑️ All  ☐ Images  ☐ Documents           ││
│ │ 📋 Quick Presets:               │ │ ☐ Videos  ☐ Audio  ☐ Archives           ││
│ │ [Downloads] [Photos] [Docs]     │ │                                         ││
│ │ [Media] [Custom] [Full System]  │ │ Exclude Patterns:                       ││
│ │                                 │ │ [*.tmp, *.log, Thumbs.db]               ││
│ └─────────────────────────────────┘ └─────────────────────────────────────────┘│
│                                                                                 │
│ ┌─ Preview & Limits ──────────────────────────────────────────────────────────┐ │
│ │ 📊 Estimated: ~15,000 files, ~45 GB │ 🚫 Free Limit: 10,000 files, 100 GB  │ │
│ │ ⚠️ May exceed free limits            │ [🔒 Upgrade] or [Reduce Scope]        │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│                           [Cancel] [Save as Preset] [▶ Start Scan]              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Details

#### 2.2.1 Dialog Properties
- **Modal Dialog:** `QDialog` with `ApplicationModal`
- **Size:** Fixed at 720x480 pixels
- **Resizable:** No (fixed size for consistency)
- **Parent:** MainWindow

#### 2.2.2 Component Specifications

**Locations Panel (`QGroupBox`):**
- Directory tree widget (`QTreeWidget`) with checkboxes
- Custom folder selection button
- Preset buttons (6 buttons, 2 rows)
- Real-time file count estimates

**Options Panel (`QGroupBox`):**
- Detection mode combo box (`QComboBox`)
- File size spin box (`QSpinBox`) with MB/GB units
- Checkbox options in logical groups
- File type filters with icons
- Pattern exclusion text field (`QLineEdit`)

**Preview Panel (`QGroupBox`):**
- Estimated scan metrics display
- Free plan limit warnings
- Upgrade promotion (premium users)
- Progress indicators during estimation

**Button Bar (`QDialogButtonBox`):**
- Standard dialog buttons: Cancel, Save Preset, Start Scan
- Start Scan button: Primary action styling
- Real-time validation of settings

### 2.3 Qt6 Implementation Structure

```cpp
class ScanSetupDialog : public QDialog {
    Q_OBJECT
    
public:
    struct ScanConfiguration {
        QStringList targetPaths;
        DetectionMode detectionMode;
        qint64 minimumFileSize;
        int maximumDepth;
        QStringList includePatterns;
        QStringList excludePatterns;
        bool includeHidden;
        bool includeSystem;
        bool followSymlinks;
        bool scanArchives;
    };
    
private:
    // UI Components
    QHBoxLayout* m_mainLayout;
    
    // Locations Panel
    QGroupBox* m_locationsGroup;
    QTreeWidget* m_directoryTree;
    QPushButton* m_addFolderButton;
    QWidget* m_presetsWidget;
    
    // Options Panel
    QGroupBox* m_optionsGroup;
    QComboBox* m_detectionMode;
    QSpinBox* m_minimumSize;
    QComboBox* m_maxDepth;
    QCheckBox* m_includeHidden;
    QCheckBox* m_includeSystem;
    QCheckBox* m_followSymlinks;
    QCheckBox* m_scanArchives;
    QLineEdit* m_excludePatterns;
    
    // Preview Panel
    QGroupBox* m_previewGroup;
    QLabel* m_estimateLabel;
    QLabel* m_limitWarning;
    QPushButton* m_upgradeButton;
    
    // Button Bar
    QDialogButtonBox* m_buttonBox;
    
private slots:
    void onDirectorySelectionChanged();
    void onPresetClicked(const QString& preset);
    void onOptionsChanged();
    void updateEstimates();
    void onStartScan();
    void onSavePreset();
    
signals:
    void scanRequested(const ScanConfiguration& config);
    void presetSaved(const QString& name, const ScanConfiguration& config);
};
```

---

## 3. Results Dashboard Design

### 3.1 Layout: Advanced Three-Panel Interface ✅ **IMPLEMENTED**

**📋 STATUS:** Implementation exceeds original specification with advanced features

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🔍 Duplicate Files Found - DupFinder                                  [─][□][✕] │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ┌─ HEADER PANEL ──────────────────────────────────────────────────────────────┐ │
│ │ 🔍 Duplicate Files Results        2 groups found, 3.1 GB potential savings │ │
│ │                                          [🔄 Refresh] [📤 Export] [⚙️ Settings] │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ ┌─ MAIN CONTENT (3-Panel Splitter) ──────────────────────────────────────────┐ │
│ │ ┌─ RESULTS TREE (60%) ─┐ ┌─ DETAILS (25%) ─┐ ┌─ ACTIONS (15%) ──┐        │ │
│ │ │ Filter: [____________] │ │ 📄 File Info     │ │ 📄 File Actions   │        │ │
│ │ │ Size: [All ▼] Type:[▼]│ │ ┌─────────────────┐│ │ 🗑️ Delete File    │        │ │
│ │ │ Sort: [Size ▼] [Clear]│ │ │ No file selected││ │ 📁 Move File      │        │ │
│ │ │ ──────────────────────│ │ │     preview     ││ │ 👁️ Ignore File    │        │ │
│ │ │ ☑️ Select All         │ │ │     area        ││ │ 👀 Preview        │        │ │
│ │ │ [Select Recommended]  │ │ └─────────────────┘│ │ 📂 Open Location  │        │ │
│ │ │ [Clear Selection]     │ │ Name: -           │ │ 📋 Copy Path      │        │ │
│ │ │ Files selected: 0     │ │ Size: -           │ │                   │        │ │
│ │ │ ──────────────────────│ │ Path: -           │ │ ⚡ Bulk Actions   │        │ │
│ │ │ 📁 Group: 2 files     │ │ Modified: -       │ │ 🗑️ Delete Selected│        │ │
│ │ │ ├─☐ vacation.jpg      │ │ Type: -           │ │ 📁 Move Selected  │        │ │
│ │ │ │  2.1 MB Oct 2 6PM   │ │ Hash: -           │ │ 👁️ Ignore Selected │        │ │
│ │ │ └─☑️ vacation_copy.jpg │ │                   │ │                   │        │ │
│ │ │    2.1 MB Oct 1 5PM   │ │ 📁 Group Info     │ │ 0 files selected  │        │ │
│ │ │ (Recommended to keep)  │ │ No group selected │ │                   │        │ │
│ │ │                       │ │                   │ │                   │        │ │
│ │ │ 📁 Group: 3 files     │ │                   │ │                   │        │ │
│ │ │ ├─☐ report.pdf        │ │                   │ │                   │        │ │
│ │ │ │  1.0 MB Oct 3 2PM   │ │                   │ │                   │        │ │
│ │ │ ├─☑️ report_backup.pdf │ │                   │ │                   │        │ │
│ │ │ │  1.0 MB Oct 1 1PM   │ │                   │ │                   │        │ │
│ │ │ └─☑️ report (1).pdf    │ │                   │ │                   │        │ │
│ │ │    1.0 MB Sep 30 9AM  │ │                   │ │                   │        │ │
│ │ └───────────────────────┘ └───────────────────┘ └───────────────────┘        │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 2 groups, 5 files │ Savings: 3.1 GB                              [■■■□□□] 50% │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**🎯 Key Enhancements Implemented:**
- **Professional 3-Panel Layout:** Header + Splitter with Results/Details/Actions
- **Advanced Filtering:** Real-time search, size filters, type filters, sorting
- **Smart Selection System:** Automatic recommendations, bulk selection tools
- **Comprehensive File Operations:** Individual and bulk operations with confirmations
- **Real-time Statistics:** Live updates of selection counts and potential savings
- **Safety Features:** Detailed confirmations and non-destructive operations

### 3.2 Implementation Details

#### 3.2.1 Window Properties
- **Window Type:** `QMainWindow` or `QDialog` (configurable)
- **Size:** Minimum 800x600, default 1024x700
- **Resizable:** Yes, with splitter panels
- **Modal:** No (allows interaction with main window)

#### 3.2.2 Component Specifications

**Toolbar (`QToolBar`):**
- View mode toggle buttons (Groups/List/Tree)
- Sort dropdown (`QComboBox`) with options: Size, Count, Name, Date
- Filter dropdown for file types
- Search field (`QLineEdit`) with live filtering
- All controls with consistent 32px height

**Results Panel (`QScrollArea` with custom widget):**
- Group containers (`QGroupBox`) with collapsible headers
- File list tables (`QTableWidget`) within each group
- Columns: Checkbox (30px), Icon (24px), Path (flexible), Size (80px), Date (100px), Action (120px)
- Alternating row colors for readability
- Custom cell widgets for complex content

**Group Action Buttons:**
- Per-group action buttons: Show Diff, Preview All, Open Locations, Smart Select
- Consistent button styling with icons
- Tooltips for user guidance

**Bottom Action Bar (`QWidget` with layout):**
- Global action buttons: Select All Recommended, Clear Selection
- Primary action: Delete Selected Files (with file count)
- Button hierarchy: Primary (Delete), Secondary (Select/Clear), Tertiary (other)

**Status Bar (`QStatusBar`):**
- Space savings indicator with progress visualization
- Selection counters (files/groups)
- Scan time information
- All information updated in real-time

### 3.3 Qt6 Implementation Structure ✅ **CURRENT IMPLEMENTATION**

**📋 This reflects the actual implementation found in `src/gui/results_window.h`**

```cpp
class ResultsWindow : public QMainWindow {
    Q_OBJECT
    
public:
    // Data structures (actually implemented)
    struct DuplicateFile {
        QString filePath;
        QString fileName;
        QString directory;
        qint64 fileSize = 0;
        QDateTime lastModified;
        QDateTime created;
        QString hash;
        QPixmap thumbnail;
        bool isSelected = false;
        bool isMarkedForDeletion = false;
        QString fileType;
    };
    
    struct DuplicateGroup {
        QString groupId;
        QList<DuplicateFile> files;
        int fileCount = 0;
        qint64 totalSize = 0;
        QString primaryFile;  // Recommended file to keep
        bool isExpanded = false;
        bool hasSelection = false;
    };
    
    struct ScanResults {
        QString scanPath;
        QDateTime scanTime;
        QString scanDuration;
        int totalFilesScanned = 0;
        int totalDuplicatesFound = 0;
        qint64 potentialSavings = 0;
        QList<DuplicateGroup> duplicateGroups;
    };
    
private:
    // Header Panel (implemented)
    QWidget* m_headerPanel;
    QHBoxLayout* m_headerLayout;
    QLabel* m_titleLabel;
    QLabel* m_summaryLabel;
    QPushButton* m_refreshButton;
    QPushButton* m_exportButton;
    QPushButton* m_settingsButton;
    
    // Three-Panel Splitter Layout (implemented)
    QSplitter* m_mainSplitter;
    
    // Results Panel (60%)
    QWidget* m_resultsPanel;
    QVBoxLayout* m_resultsPanelLayout;
    QWidget* m_filterPanel;
    QLineEdit* m_searchFilter;
    QComboBox* m_sizeFilter;
    QComboBox* m_typeFilter;
    QComboBox* m_sortCombo;
    QPushButton* m_clearFiltersButton;
    QTreeWidget* m_resultsTree;
    
    // Details Panel (25%)
    QWidget* m_detailsPanel;
    QTabWidget* m_detailsTabs;
    
    // Actions Panel (15%)
    QWidget* m_actionsPanel;
    QVBoxLayout* m_actionsPanelLayout;
    QGroupBox* m_fileActionsGroup;
    QGroupBox* m_selectionGroup;
    QGroupBox* m_bulkActionsGroup;
    
    // Advanced features (implemented)
    QTimer* m_thumbnailTimer;
    bool m_isProcessingBulkOperation;
    ScanResults m_currentResults;
    
    // Status Bar
    QStatusBar* m_statusBar;
    QLabel* m_spaceSavingsLabel;
    QLabel* m_selectionCountLabel;
    QLabel* m_scanTimeLabel;
    
    // Data
    QList<DuplicateDetector::DuplicateGroup> m_duplicateGroups;
    ViewMode m_currentViewMode;
    SortMode m_currentSortMode;
    QString m_filterCriteria;
    
private slots:
    void onViewModeChanged(ViewMode mode);
    void onSortChanged(SortMode mode);
    void onFilterChanged(const QString& filter);
    void onSearchTextChanged(const QString& text);
    void onFileSelectionChanged();
    void onGroupActionClicked(const QString& groupId, const QString& action);
    void onSelectRecommended();
    void onClearSelection();
    void onDeleteSelected();
    void updateStatusBar();
    void refreshResults();
    
public slots:
    void setDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups);
    void showGroup(const QString& groupId);
    void selectFiles(const QStringList& filePaths);
    
signals:
    void filesSelectedForDeletion(const QStringList& filePaths);
    void filePreviewRequested(const QString& filePath);
    void folderOpenRequested(const QString& folderPath);
    void selectionChanged(int selectedFiles, qint64 spaceSavings);
};

class DuplicateGroupWidget : public QGroupBox {
    Q_OBJECT
    
private:
    QTableWidget* m_fileTable;
    QHBoxLayout* m_actionLayout;
    QPushButton* m_showDiffButton;
    QPushButton* m_previewAllButton;
    QPushButton* m_openLocationsButton;
    QPushButton* m_smartSelectButton;
    
    DuplicateDetector::DuplicateGroup m_groupData;
    
private slots:
    void onCellChanged(int row, int column);
    void onShowDiff();
    void onPreviewAll();
    void onOpenLocations();
    void onSmartSelect();
    
public slots:
    void setGroupData(const DuplicateDetector::DuplicateGroup& group);
    void updateSelectionState();
    
signals:
    void fileSelectionChanged(const QString& filePath, bool selected);
    void actionRequested(const QString& groupId, const QString& action);
};
```

---

## 4. Supporting Dialogs and Components

### 4.1 Progress Dialog

#### Design
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Scanning for Duplicates...                                            [✕]     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                    🔍 Analyzing files in Downloads folder                      │
│                                                                                 │
│ ┌─ Progress ──────────────────────────────────────────────────────────────────┐ │
│ │ Phase: Calculating file hashes                                              │ │
│ │ ████████████████████████████████████░░░░░░░░░░░░ 73% complete              │ │
│ │                                                                             │ │
│ │ Files processed: 8,247 of 11,356                                           │ │
│ │ Current file: ~/Downloads/IMG_2023_vacation_beach_sunset_final_v2.jpg      │ │
│ │ Elapsed time: 2m 34s                                                       │ │
│ │ Estimated remaining: 1m 12s                                                │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ ┌─ Statistics ────────────────────────────────────────────────────────────────┐ │
│ │ Potential duplicates found: 89                                              │ │
│ │ Estimated space savings: 1.8 GB                                            │ │
│ │ Scan speed: 3,247 files/minute                                             │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│                                    [Pause]  [Cancel]                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Implementation
- **Type:** `QProgressDialog` with custom content
- **Size:** Fixed 500x300 pixels
- **Modal:** Yes, but with pause/cancel options
- **Update Frequency:** 4 times per second (250ms intervals)

### 4.2 Confirmation Dialog

#### Design
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Confirm File Deletion                                                  [✕]     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ ⚠️  You're about to move 89 duplicate files to the trash.                      │
│                                                                                 │
│ ┌─ Summary ───────────────────────────────────────────────────────────────────┐ │
│ │ Files to delete: 89                                                         │ │
│ │ Space to free: 1.8 GB                                                      │ │
│ │ Groups affected: 23                                                         │ │
│ │                                                                             │ │
│ │ 📄 Documents: 12 files (245 MB)                                            │ │
│ │ 🖼️ Images: 67 files (1.4 GB)                                               │ │
│ │ 🎵 Audio: 8 files (156 MB)                                                 │ │
│ │ 📦 Other: 2 files (12 MB)                                                  │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ ┌─ Safety Information ────────────────────────────────────────────────────────┐ │
│ │ ✅ Files will be moved to Trash (not permanently deleted)                  │ │
│ │ ✅ You can restore files from Trash if needed                              │ │
│ │ ✅ Original files in safe locations will be kept                           │ │
│ │ ⚠️ This action will be logged for undo capability                          │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│ ☐ Don't show this dialog for small deletions (<10 files)                      │ │
│                                                                                 │
│                           [📋 View File List]  [Cancel]  [🗑️ Move to Trash]   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Implementation
- **Type:** Custom `QDialog` with detailed information
- **Size:** Fixed 520x400 pixels
- **Modal:** Yes, blocking operation until confirmed
- **Expandable:** "View File List" shows detailed file listing

### 4.3 Settings Dialog

#### Design Structure
- **Tab-based interface** with categories: General, Scanning, Performance, Safety, Advanced
- **Consistent layout** with grouped options
- **Real-time validation** with immediate feedback
- **Reset/Default options** for each category

---

## 5. Cross-Platform Design Guidelines

### 5.1 Platform-Specific Adaptations

#### Linux (Primary Platform)
- **Look & Feel:** Follow system theme (GTK/KDE)
- **File Dialogs:** Native Qt file dialogs with bookmark support
- **Icons:** Use system icon theme with fallbacks
- **Keyboard:** Standard Linux shortcuts (Ctrl+O, Ctrl+S, etc.)
- **Desktop Integration:** .desktop files, system tray, notifications

#### Windows (Secondary Platform)
- **Look & Feel:** Windows 10/11 native styling
- **File Dialogs:** Windows native file dialogs
- **Icons:** Windows style icons with proper sizing
- **Keyboard:** Windows shortcuts (Ctrl+N, Alt+F4, etc.)
- **Desktop Integration:** Start Menu, taskbar, Windows notifications

#### macOS (Tertiary Platform)
- **Look & Feel:** macOS native appearance
- **File Dialogs:** macOS native file dialogs
- **Icons:** macOS style icons following HIG
- **Keyboard:** Mac shortcuts (Cmd+N, Cmd+Q, etc.)
- **Desktop Integration:** Dock, menu bar, macOS notifications

### 5.2 Responsive Design Considerations

#### Minimum Screen Sizes
- **Main Window:** 800x600 minimum, 1024x768 optimal
- **Scan Setup:** 720x480 fixed
- **Results Window:** 800x600 minimum, scalable
- **Dialogs:** Platform-appropriate sizing

#### High DPI Support
- **Icons:** Vector-based where possible, multiple sizes (16, 24, 32, 48, 64)
- **Fonts:** System fonts with proper scaling
- **Layout:** Density-independent pixels (dp) for spacing
- **Images:** High-resolution assets for 2x, 3x displays

---

## 6. Implementation Roadmap

### 6.1 Phase 1: Core Windows (Weeks 5-6)
1. **Main Window Implementation**
   - Basic layout and navigation
   - Quick actions and history display
   - System overview integration

2. **Scan Setup Dialog**
   - Directory selection interface
   - Options configuration
   - Preset management

### 6.2 Phase 2: Results Interface (Weeks 7-8)
1. **Results Window Structure**
   - List view with grouping
   - Selection management
   - Action button integration

2. **File Operations UI**
   - Confirmation dialogs
   - Progress indication
   - Status feedback

### 6.3 Phase 3: Polish and Enhancement (Week 8)
1. **Visual Polish**
   - Icon integration
   - Theme support
   - Animation and transitions

2. **Accessibility and Usability**
   - Keyboard navigation
   - Screen reader support
   - Tooltip and help integration

---

## 7. Development Guidelines

### 7.1 Qt6 Best Practices

#### Widget Hierarchy
```cpp
// Preferred widget organization
MainWindow
├── CentralWidget (QWidget)
│   ├── HeaderWidget (QWidget)
│   ├── ContentWidget (QWidget with QVBoxLayout)
│   │   ├── QuickActionsWidget
│   │   ├── ScanHistoryWidget
│   │   └── SystemOverviewWidget
│   └── StatusBar (QStatusBar)
```

#### Signal-Slot Connections
```cpp
// Use modern Qt6 syntax
connect(scanButton, &QPushButton::clicked, 
        this, &MainWindow::onScanRequested);

// Prefer member function pointers over SIGNAL/SLOT macros
connect(fileScanner, &FileScanner::scanProgress,
        progressDialog, &ProgressDialog::updateProgress);
```

#### Layout Management
```cpp
// Use layout managers effectively
auto* layout = new QVBoxLayout(this);
layout->setContentsMargins(12, 12, 12, 12);
layout->setSpacing(8);
layout->addWidget(headerWidget);
layout->addWidget(contentWidget, 1); // Stretch factor
layout->addWidget(statusWidget);
```

### 7.2 Styling and Theming

#### CSS Styling Approach
```cpp
// Use Qt stylesheets sparingly, prefer native styling
QString styleSheet = R"(
    QGroupBox {
        font-weight: bold;
        border: 2px solid gray;
        border-radius: 5px;
        margin-top: 1ex;
        padding-top: 8px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 8px 0 8px;
    }
)";
```

#### Icon Management
```cpp
// Use QIcon::fromTheme for cross-platform icons
QIcon scanIcon = QIcon::fromTheme("edit-find", 
                 QIcon(":/icons/scan.png"));
scanButton->setIcon(scanIcon);
```

### 7.3 Memory Management

#### Widget Ownership
- Use Qt's parent-child ownership model
- Prefer stack allocation for temporary objects
- Use smart pointers for complex ownership scenarios

#### Resource Management
- Load icons and resources lazily
- Implement proper cleanup in destructors
- Use QObject::deleteLater() for safe deletion

---

## 8. Testing and Quality Assurance

### 8.1 UI Testing Strategy

#### Manual Testing Checklist
- [ ] All buttons respond to clicks
- [ ] Keyboard shortcuts work correctly
- [ ] Tab order is logical and complete
- [ ] Window resizing behaves correctly
- [ ] High DPI displays render properly
- [ ] All dialogs are properly modal/modeless
- [ ] Status updates appear in real-time
- [ ] Error messages are user-friendly

#### Automated Testing
```cpp
// Example UI test structure
class MainWindowTest : public QObject {
    Q_OBJECT
    
private slots:
    void testQuickActionButtons();
    void testScanHistoryDisplay();
    void testStatusBarUpdates();
    void testKeyboardShortcuts();
};

void MainWindowTest::testQuickActionButtons() {
    MainWindow window;
    window.show();
    
    auto* scanButton = window.findChild<QPushButton*>("quickScanButton");
    QVERIFY(scanButton != nullptr);
    
    QSignalSpy spy(&window, &MainWindow::scanRequested);
    QTest::mouseClick(scanButton, Qt::LeftButton);
    QCOMPARE(spy.count(), 1);
}
```

### 8.2 Accessibility Requirements

#### Keyboard Navigation
- All interactive elements accessible via Tab/Shift+Tab
- Arrow keys for navigation within lists and trees
- Enter/Space for activation
- Escape for cancellation

#### Screen Reader Support
- Proper labels for all controls
- ARIA attributes where applicable
- Logical reading order
- Status announcements for dynamic content

#### Visual Accessibility
- High contrast mode support
- Minimum font sizes respected
- Color-blind friendly indicators
- Scalable UI elements

---

## 9. Conclusion

This UI specification provides a comprehensive foundation for implementing the DupFinder user interface. The design emphasizes:

- **User-Centric Design:** Simple, intuitive interfaces for non-technical users
- **Safety First:** Clear confirmations and reversible operations
- **Information Rich:** Compact layouts showing relevant details
- **Modern Appeal:** Contemporary design language with Qt6 native styling
- **Cross-Platform:** Native appearance and behavior on all target platforms

The implementation should follow Qt6 best practices, maintain consistent visual language across all components, and provide excellent user experience while supporting the application's core mission of safe, efficient duplicate file management.

---

## Appendices

### A. Icon Requirements
- **File Type Icons:** 16x16, 24x24, 32x32 pixel versions
- **Action Icons:** Standard system icons with custom fallbacks  
- **Status Icons:** Warning, error, success, information indicators
- **Application Icon:** Multiple sizes (16, 24, 32, 48, 64, 128, 256, 512)

### B. Color Palette
- **Primary:** System accent color (platform-dependent)
- **Success:** Green (#4CAF50)
- **Warning:** Orange (#FF9800) 
- **Error:** Red (#F44336)
- **Information:** Blue (#2196F3)
- **Background:** System theme colors
- **Text:** System text colors with appropriate contrast

### C. Typography
- **Headers:** System font, bold, 1.2x base size
- **Body Text:** System font, regular, base size
- **Captions:** System font, regular, 0.9x base size
- **Code/Paths:** Monospace font, regular, 0.95x base size

### D. Spacing Standards
- **Margin:** 12px standard, 8px compact, 16px loose
- **Padding:** 8px standard, 4px compact, 12px loose  
- **Button Height:** 32px standard, 24px compact
- **Line Height:** 1.4x font size for readability

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-04  
**Next Review:** 2025-10-11  
**Approved By:** [Development Team Lead]