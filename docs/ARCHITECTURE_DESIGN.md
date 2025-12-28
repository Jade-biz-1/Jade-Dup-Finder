# CloneClean Architecture Design Document

**Version:** 1.0  
**Created:** 2025-10-03  
**Based on:** Implementation Plan v1.0  

---

## Executive Summary

This document defines the technical architecture for CloneClean, a cross-platform duplicate file finder application. The architecture follows modern C++/Qt6 best practices with a focus on modularity, testability, and maintainability across Linux, Windows, and macOS platforms.

### Architecture Principles
- **Separation of Concerns:** Clear boundaries between core logic, UI, and platform-specific code
- **SOLID Principles:** Single responsibility, open/closed, dependency inversion
- **Qt6 Integration:** Leverage Qt's cross-platform capabilities and signal/slot system
- **Thread Safety:** Multi-threaded design with proper synchronization
- **Scalability:** Handle large datasets efficiently with streaming processing

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer (Qt6 GUI)              │
├─────────────────────────────────────────────────────────────────┤
│                     Business Logic Layer                       │
├─────────────────────────────────────────────────────────────────┤
│                     Core Engine Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                Platform Abstraction Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                     Operating System APIs                      │
└─────────────────────────────────────────────────────────────────┘
```

### Layered Architecture Details

#### 1. Presentation Layer (Qt6 GUI)
- **Responsibility:** User interface, user interaction, visual feedback
- **Technology:** Qt6 Widgets, Qt6 Concurrent for background operations
- **Components:** Main Window, **Advanced Results Window** ✅, Scan Setup Dialog, Custom Widgets, Progress Indicators

#### 2. Business Logic Layer
- **Responsibility:** Application workflow, business rules, validation
- **Technology:** Standard C++17, Qt6 Core classes
- **Components:** Scan Orchestrator, Results Manager, Settings Manager

#### 3. Core Engine Layer
- **Responsibility:** Core algorithms, data processing, duplicate detection
- **Technology:** Standard C++17, Qt6 Core, cryptographic libraries
- **Components:** File Scanner, Hash Calculator, Duplicate Detector, Safety Manager

#### 4. Platform Abstraction Layer
- **Responsibility:** Platform-specific operations with unified interface
- **Technology:** Qt6 Core with platform-specific extensions
- **Components:** File Operations, Trash Manager, System Integration

---

## Core Component Design

### 1. File Scanner Component

```cpp
// High-level interface design
class FileScanner : public QObject {
    Q_OBJECT
    
public:
    struct ScanOptions { /* ... */ };
    struct FileInfo { /* ... */ };
    
    void startScan(const ScanOptions& options);
    void cancelScan();
    
signals:
    void fileFound(const FileInfo& info);
    void scanProgress(int current, int total);
    void scanCompleted();
    
private:
    class ScanWorker;                    // Worker thread implementation
    std::unique_ptr<ScanWorker> m_worker;
    QThreadPool* m_threadPool;
    std::atomic<bool> m_cancelRequested;
};
```

**Design Patterns:**
- **Worker Thread Pattern:** Separate thread for I/O intensive scanning
- **Observer Pattern:** Qt signals/slots for progress notifications
- **Strategy Pattern:** Different scanning strategies (recursive, filtered, etc.)

**Key Responsibilities:**
- Recursive directory traversal using `QDirIterator`
- File filtering based on size, type, and patterns
- Progress reporting and cancellation support
- Memory-efficient processing for large directories

### 2. Hash Calculator Component

```cpp
class HashCalculator : public QObject {
    Q_OBJECT
    
public:
    struct HashResult {
        QString filePath;
        QByteArray hash;
        bool success;
        QString error;
    };
    
    void calculateHash(const QString& filePath);
    void calculateHashes(const QStringList& filePaths);
    
signals:
    void hashCalculated(const HashResult& result);
    void progressUpdate(int completed, int total);
    
private:
    class HashWorker;
    QThreadPool* m_threadPool;
    LRUCache<QString, QByteArray> m_hashCache;
    std::atomic<int> m_pendingTasks;
};
```

**Design Patterns:**
- **Thread Pool Pattern:** Concurrent hash calculation
- **Cache Pattern:** LRU cache for previously calculated hashes
- **Future/Promise Pattern:** Qt's `QFuture` for async operations

**Key Responsibilities:**
- SHA-256 hash calculation using `QCryptographicHash`
- Multi-threaded processing with configurable thread count
- Intelligent caching to avoid recalculating known hashes
- Progressive processing for large files with cancellation support

### 3. Duplicate Detector Component

```cpp
class DuplicateDetector : public QObject {
    Q_OBJECT
    
public:
    enum class DetectionMode {
        Quick,      // Size + filename
        Deep,       // Size + hash
        Media,      // Deep + metadata
        Smart       // Adaptive based on file types
    };
    
    struct DuplicateGroup {
        QByteArray hash;                    // Common hash for the group
        QList<FileScanner::FileInfo> files; // All files in this group
        qint64 wastedSpace;                 // Space that can be reclaimed
        FileScanner::FileInfo recommended;  // Recommended file to keep
    };
    
    void detectDuplicates(const QList<FileScanner::FileInfo>& files, 
                         DetectionMode mode);
    
signals:
    void duplicatesFound(const QList<DuplicateGroup>& groups);
    void detectionProgress(int completed, int total);
    
private:
    class DetectionWorker;
    DetectionMode m_currentMode;
    QHash<qint64, QList<FileScanner::FileInfo>> m_sizeGroups;
    QHash<QByteArray, QList<FileScanner::FileInfo>> m_hashGroups;
};
```

**Design Patterns:**
- **Strategy Pattern:** Different detection algorithms based on mode
- **Command Pattern:** Encapsulate detection operations
- **Template Method Pattern:** Common detection flow with varying steps

**Key Responsibilities:**
- Size-based pre-filtering for performance optimization
- Hash-based duplicate detection with collision handling
- Smart recommendations based on file attributes
- Memory-efficient processing of large file collections

### 4. Safety Manager Component

```cpp
class SafetyManager : public QObject {
    Q_OBJECT
    
public:
    struct OperationRecord {
        QString operationId;
        QDateTime timestamp;
        QString operationType;
        QStringList affectedFiles;
        QString originalLocation;
        bool canUndo;
    };
    
    QString moveToTrash(const QStringList& filePaths);
    bool undoOperation(const QString& operationId);
    QList<OperationRecord> getOperationHistory() const;
    
signals:
    void operationCompleted(const OperationRecord& record);
    void operationFailed(const QString& error);
    
private:
    class TrashOperation;
    QList<OperationRecord> m_operationHistory;
    QHash<QString, std::unique_ptr<TrashOperation>> m_activeOperations;
    
    bool validateOperation(const QStringList& filePaths);
    QString generateOperationId();
};
```

**Design Patterns:**
- **Command Pattern:** Undo/redo operations
- **Memento Pattern:** Store operation state for recovery
- **Chain of Responsibility:** Validation chain before operations

### 5. Results Window Component ✅ **IMPLEMENTED**

```cpp
class ResultsWindow : public QMainWindow {
    Q_OBJECT
    
public:
    struct DuplicateFile {
        QString filePath;
        QString fileName;
        qint64 fileSize;
        QDateTime lastModified;
        QString hash;
        bool isSelected;
        QString fileType;
    };
    
    struct DuplicateGroup {
        QString groupId;
        QList<DuplicateFile> files;
        qint64 totalSize;
        QString primaryFile;  // Recommended to keep
        qint64 getWastedSpace() const;
    };
    
    void displayResults(const ScanResults& results);
    void clearResults();
    
signals:
    void windowClosed();
    void fileOperationRequested(const QString& operation, const QStringList& files);
    void resultsUpdated(const ScanResults& results);
    
private:
    // Advanced 3-panel layout
    QWidget* m_headerPanel;          // Title, summary, action buttons
    QSplitter* m_mainSplitter;       // Results | Details | Actions (60% | 25% | 15%)
    QTreeWidget* m_resultsTree;      // Hierarchical duplicate display
    QTabWidget* m_detailsTabs;       // File info and group details
    QWidget* m_actionsPanel;         // File operations and bulk actions
    
    // Smart selection and operations
    QList<DuplicateFile> getSelectedFiles() const;
    void selectRecommended();        // Smart selection algorithm
    void performBulkOperations(const QString& operation);
    void updateSelectionSummary();   // Real-time statistics
};
```

**Design Patterns:**
- **Model-View Pattern:** Tree widget displays hierarchical duplicate data
- **Observer Pattern:** Qt signals/slots for UI updates and file operations  
- **Command Pattern:** File operations encapsulated as commands with undo capability
- **Strategy Pattern:** Different selection strategies (smart, all, by type, by size)
- **Template Method Pattern:** Common file operation workflow with varying implementations

**Key Responsibilities:**
- **Professional 3-Panel Interface:** Header + Splitter layout for optimal information display
- **Hierarchical Data Display:** Group-based view of duplicate files with expandable tree structure
- **Smart Selection System:** AI-driven recommendations for files to keep vs delete
- **Comprehensive File Operations:** Delete, move, ignore, preview with full system integration
- **Real-time Statistics:** Live updates of selection counts, space savings, and operation progress
- **Safety-First Design:** Detailed confirmations and non-destructive operations

**Advanced Features Implemented:**
- **Smart Recommendations:** Algorithm recommends oldest files (likely originals) to keep
- **Bulk Operations:** Multi-file operations with detailed impact summaries and confirmations
- **System Integration:** File manager integration, clipboard operations, location opening
- **Advanced Filtering:** Real-time search, size filters, type filters, sorting options
- **Selection Management:** Checkbox-based selection with bulk selection tools
- **Progress Tracking:** Real-time updates during operations with cancellation support

**UI Layout Architecture:**
```
┌─ Header Panel (Actions & Summary) ────────────────────────────┐
│ 🔍 Duplicate Files Results    2 groups, 3.1GB potential savings    │
│                                  [Refresh] [Export] [Settings]       │
├─────────────────────────────────────────────────────────────────────────┤
│ Results Tree (60%)     │ Details (25%)  │ Actions (15%)        │
│ ┌─ Filter & Search ─┐   │ ┌─ File Info ────┐   │ ┌─ File Actions ──┐  │
│ │ [____________] │   │ │ Preview Area  │   │ │ Delete File   │  │
│ │ Size:[All▼] Type:│   │ │ Name: file.jpg│   │ │ Move File     │  │
│ └─────────────────┘   │ │ Size: 2.1 MB  │   │ │ Ignore File   │  │
│ ┌─ Selection ───────┐   │ └───────────────┘   │ │ Preview       │  │
│ │ Select All    │   │ ┌─ Group Info ───┐   │ └───────────────┘  │
│ │ [Recommended] │   │ │ Group Summary │   │ ┌─ Bulk Actions ──┐  │
│ └───────────────┘   │ │ 2 files       │   │ │ Delete Selected│  │
│ ┌─ Hierarchical Tree ─┐   │ │ 3.1 GB total  │   │ │ Move Selected  │  │
│ │ 📁 Group: 2 files  │   │ └───────────────┘   │ │ Ignore Selected│  │
│ │ ├─☐ file1.jpg     │   │                 │ └───────────────┘  │
│ │ └─☑️ file2.jpg     │   │                 │ 2 files selected  │
│ └─────────────────┘   │                 │ 1.5 GB savings    │
└─────────────────────┴─────────────────┴───────────────────┘
```

---

## Platform Abstraction Layer

### Cross-Platform Interface Design

```cpp
// Abstract interface for platform-specific operations
class IPlatformFileOperations {
public:
    virtual ~IPlatformFileOperations() = default;
    
    virtual bool moveToTrash(const QString& filePath) = 0;
    virtual bool isSystemDirectory(const QString& path) = 0;
    virtual QStringList getSystemExclusionPaths() = 0;
    virtual bool hasFilePermission(const QString& path, FilePermission perm) = 0;
    virtual QString getTrashLocation() = 0;
};

// Platform-specific implementations
class LinuxFileOperations : public IPlatformFileOperations { /* ... */ };
class WindowsFileOperations : public IPlatformFileOperations { /* ... */ };
class MacOSFileOperations : public IPlatformFileOperations { /* ... */ };

// Factory for creating platform-specific instances
class PlatformFactory {
public:
    static std::unique_ptr<IPlatformFileOperations> createFileOperations();
    static std::unique_ptr<ISystemIntegration> createSystemIntegration();
    static std::unique_ptr<INotificationManager> createNotificationManager();
};
```

**Design Patterns:**
- **Abstract Factory Pattern:** Platform-specific component creation
- **Bridge Pattern:** Separate abstraction from implementation
- **Adapter Pattern:** Adapt platform APIs to common interface

### Platform-Specific Implementations

#### Linux Implementation
- **Trash Operations:** FreeDesktop.org Trash Specification
- **File Systems:** ext4, Btrfs, NTFS support
- **Desktop Integration:** .desktop files, MIME types
- **Notifications:** libnotify integration

#### Windows Implementation  
- **Trash Operations:** Shell API for Recycle Bin
- **File Systems:** NTFS, FAT32 support
- **Desktop Integration:** Registry, Explorer context menus
- **Notifications:** Windows Toast Notifications

#### macOS Implementation
- **Trash Operations:** NSFileManager Trash API
- **File Systems:** APFS, HFS+ support  
- **Desktop Integration:** Finder Services, dock integration
- **Notifications:** NSUserNotificationCenter

---

## Threading Architecture

### Thread Pool Design

```cpp
class ThreadManager {
public:
    enum class ThreadType {
        FileScanning,    // I/O intensive operations
        HashCalculation, // CPU intensive operations
        UIUpdates,       // GUI updates and user interaction
        Background       // Cleanup, maintenance tasks
    };
    
    static ThreadManager& instance();
    
    QThreadPool* getThreadPool(ThreadType type);
    void configureThreadLimits(ThreadType type, int maxThreads);
    void shutdownAll();
    
private:
    QHash<ThreadType, std::unique_ptr<QThreadPool>> m_threadPools;
    void initializeThreadPools();
};
```

### Concurrency Strategy

#### File Scanning Thread
- **Purpose:** Directory traversal and file enumeration
- **Count:** Single thread to avoid filesystem thrashing
- **Communication:** Qt signals for progress updates

#### Hash Calculation Threads
- **Purpose:** Parallel hash computation
- **Count:** CPU cores - 1 (leave one for UI)
- **Communication:** Qt signals and futures for results

#### GUI Thread
- **Purpose:** User interface updates and interaction
- **Count:** Single main thread (Qt requirement)
- **Communication:** Queued signals from worker threads

---

## Data Management

### Caching Strategy

```cpp
template<typename K, typename V>
class LRUCache {
public:
    LRUCache(size_t capacity);
    
    bool contains(const K& key) const;
    V get(const K& key);
    void put(const K& key, const V& value);
    void clear();
    
    size_t size() const { return m_cache.size(); }
    size_t capacity() const { return m_capacity; }
    
private:
    struct CacheNode {
        K key;
        V value;
        std::shared_ptr<CacheNode> prev;
        std::shared_ptr<CacheNode> next;
    };
    
    size_t m_capacity;
    std::unordered_map<K, std::shared_ptr<CacheNode>> m_cache;
    std::shared_ptr<CacheNode> m_head;
    std::shared_ptr<CacheNode> m_tail;
    mutable QReadWriteLock m_lock;
};
```

### Memory Management Strategy

#### File Information Storage
- **Small Files (<1MB):** In-memory storage with full metadata
- **Large Files (>1MB):** Streaming processing, minimal memory footprint
- **Hash Cache:** LRU cache with configurable size (default: 10,000 entries)
- **Result Sets:** Paginated display for large duplicate groups

#### Performance Targets
- **Memory Usage:** Maximum 500MB for typical operations
- **Cache Hit Rate:** >80% for hash operations in repeated scans
- **UI Responsiveness:** <100ms response time for all user interactions

---

## Error Handling and Logging

### Exception Strategy

```cpp
// Custom exception hierarchy
class CloneCleanException : public std::exception {
public:
    explicit CloneCleanException(const QString& message);
    const char* what() const noexcept override;
    
protected:
    QString m_message;
};

class FileAccessException : public CloneCleanException { /* ... */ };
class HashCalculationException : public CloneCleanException { /* ... */ };
class PlatformException : public CloneCleanException { /* ... */ };

// Error handling policy
class ErrorHandler {
public:
    enum class ErrorLevel { Info, Warning, Error, Critical };
    
    static void handleError(const CloneCleanException& ex, ErrorLevel level);
    static void logError(const QString& component, const QString& message);
    static QStringList getRecentErrors();
    
private:
    static QMutex s_logMutex;
    static QStringList s_errorLog;
};
```

### Logging Framework

```cpp
class Logger {
public:
    enum class Level { Debug, Info, Warning, Error };
    
    static void setLevel(Level level);
    static void setOutputFile(const QString& filePath);
    
    template<typename... Args>
    static void debug(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void info(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void warning(const QString& format, Args&&... args);
    
    template<typename... Args>
    static void error(const QString& format, Args&&... args);
    
private:
    static Level s_currentLevel;
    static QMutex s_logMutex;
    static std::unique_ptr<QTextStream> s_logStream;
};
```

---

## Configuration Management

### Settings Architecture

```cpp
class ConfigurationManager : public QObject {
    Q_OBJECT
    
public:
    static ConfigurationManager& instance();
    
    // Scan settings
    void setScanOptions(const FileScanner::ScanOptions& options);
    FileScanner::ScanOptions getScanOptions() const;
    
    // Performance settings
    void setThreadCount(ThreadManager::ThreadType type, int count);
    int getThreadCount(ThreadManager::ThreadType type) const;
    
    // UI settings
    void setUITheme(const QString& theme);
    QString getUITheme() const;
    
    // Cache settings
    void setCacheSize(size_t size);
    size_t getCacheSize() const;
    
signals:
    void settingsChanged();
    
private:
    std::unique_ptr<QSettings> m_settings;
    void loadDefaults();
    void validateSettings();
};
```

### Configuration Schema

```ini
[FileScanning]
DefaultMinimumFileSize=1048576
IncludeHiddenFiles=false
FollowSymlinks=false
ExcludedPatterns=*.tmp,*.log,*.cache

[Performance]
FileScanningThreads=1
HashCalculationThreads=4
MaxMemoryUsage=536870912
CacheSize=10000

[UI]
Theme=auto
ShowThumbnails=true
ProgressUpdateInterval=250
AutoSaveResults=true

[Platform]
TrashIntegration=true
SystemNotifications=true
AutoUpdate=true
```

---

## Security and Privacy

### Data Protection

```cpp
class SecurityManager {
public:
    // Hash validation and integrity
    static bool validateFileIntegrity(const QString& filePath, 
                                     const QByteArray& expectedHash);
    
    // Secure memory handling
    template<typename T>
    class SecureContainer {
    public:
        SecureContainer(size_t size);
        ~SecureContainer();
        
        T* data() { return m_data; }
        size_t size() const { return m_size; }
        
    private:
        T* m_data;
        size_t m_size;
        void secureZero();
    };
    
    // Privacy settings
    static void setDataCollection(bool enabled);
    static bool isDataCollectionEnabled();
    
private:
    static bool s_dataCollectionEnabled;
};
```

### Privacy Principles
- **Local Processing:** All file analysis performed locally
- **Minimal Telemetry:** Only crash reports and basic usage statistics
- **User Consent:** Explicit opt-in for any data collection
- **Secure Deletion:** Secure wiping of sensitive temporary data

---

## Testing Architecture

### Test Strategy

```cpp
// Test fixture base class
class CloneCleanTestBase : public QObject {
    Q_OBJECT
    
protected:
    void SetUp();
    void TearDown();
    
    QString createTestFile(const QString& name, qint64 size);
    QStringList createTestDirectory(const QString& structure);
    void cleanupTestFiles();
    
    QString m_testDataPath;
    QStringList m_createdFiles;
};

// Component-specific test classes
class FileScannerTest : public CloneCleanTestBase {
    Q_OBJECT
    
private slots:
    void testBasicScanning();
    void testFileFiltering();
    void testProgressReporting();
    void testCancellation();
    void testErrorHandling();
};
```

### Test Types

#### Unit Tests
- **Coverage Target:** 85% line coverage
- **Framework:** Qt Test Framework
- **Scope:** Individual components in isolation
- **Mock Objects:** Platform operations, file system

#### Integration Tests  
- **Coverage Target:** 70% of integration scenarios
- **Framework:** Qt Test Framework with test data
- **Scope:** Component interactions and workflows
- **Test Data:** Structured test file hierarchies

#### Performance Tests
- **Framework:** Qt Benchmark
- **Metrics:** Memory usage, processing speed, responsiveness
- **Datasets:** Large file collections (10k, 100k, 1M files)

#### Platform Tests
- **Scope:** Platform-specific operations
- **Environments:** CI/CD pipelines for each OS
- **Validation:** Native API integration

---

## Deployment Architecture

### Build System

```cmake
# Multi-platform build configuration
if(WIN32)
    # Windows-specific settings
    set(PLATFORM_SOURCES src/platform/windows/*)
    target_link_libraries(cloneclean shell32 ole32)
elseif(APPLE)
    # macOS-specific settings  
    set(PLATFORM_SOURCES src/platform/macos/*)
    target_link_libraries(cloneclean ${FOUNDATION_LIB} ${APPKIT_LIB})
elseif(UNIX)
    # Linux-specific settings
    set(PLATFORM_SOURCES src/platform/linux/*)
endif()
```

### Continuous Integration

```yaml
# GitHub Actions workflow structure
name: Cross-Platform Build
on: [push, pull_request]

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - Qt6 installation
      - Build and test
      - Package creation
      
  build-windows:
    runs-on: windows-latest
    steps:
      - MSVC setup
      - Qt6 installation  
      - Build and test
      - Code signing
      
  build-macos:
    runs-on: macos-latest
    steps:
      - Xcode setup
      - Qt6 installation
      - Build and test
      - Notarization
```

### Package Distribution

#### Linux
- **AppImage:** Universal binary for all distributions
- **Debian Package:** .deb for Ubuntu/Debian systems
- **RPM Package:** For Fedora/RHEL systems
- **Snap Package:** Ubuntu Software Center distribution

#### Windows
- **NSIS Installer:** Traditional Windows installer
- **Microsoft Store:** UWP package for store distribution
- **Portable Version:** Self-contained executable

#### macOS
- **DMG Package:** Standard macOS installer
- **Mac App Store:** Sandboxed version for store
- **Homebrew Cask:** Command-line installation

---

## Performance Optimization

### Optimization Strategy

#### Memory Optimization
- **Object Pooling:** Reuse file info objects
- **Lazy Loading:** Load file details on demand
- **Memory Mapping:** For large file operations
- **Smart Pointers:** Automatic memory management

#### CPU Optimization
- **Algorithm Complexity:** O(n log n) for sorting, O(n) for grouping
- **Cache-Friendly Access:** Sequential memory access patterns
- **SIMD Instructions:** Vectorized operations where applicable
- **Thread Affinity:** Pin threads to specific cores

#### I/O Optimization
- **Asynchronous I/O:** Non-blocking file operations
- **Read Ahead:** Prefetch file data for hash calculation
- **Batch Operations:** Group file system calls
- **SSD Optimization:** Different strategies for SSD vs HDD

### Performance Monitoring

```cpp
class PerformanceMonitor {
public:
    struct Metrics {
        std::chrono::milliseconds scanDuration;
        std::chrono::milliseconds hashDuration;
        size_t memoryUsage;
        size_t filesProcessed;
        double throughput;  // files per second
    };
    
    static void startOperation(const QString& operationName);
    static void endOperation(const QString& operationName);
    static Metrics getMetrics(const QString& operationName);
    static void logMetrics();
    
private:
    static QHash<QString, std::chrono::high_resolution_clock::time_point> s_startTimes;
    static QHash<QString, Metrics> s_metrics;
};
```

---

## Future Architecture Considerations

### Scalability Enhancements
- **Distributed Processing:** Multi-machine hash calculation
- **Database Backend:** SQLite for large result sets
- **Plugin Architecture:** Extensible duplicate detection algorithms
- **Cloud Integration:** Cloud storage scanning capabilities

### Advanced Features
- **Machine Learning:** Smart duplicate recommendations
- **Content Analysis:** Semantic duplicate detection
- **Real-time Monitoring:** File system change detection
- **API Interface:** REST API for automation

This architecture design provides a solid foundation for building a robust, scalable, and maintainable duplicate file finder application that can grow and adapt to future requirements while maintaining high quality and performance standards.