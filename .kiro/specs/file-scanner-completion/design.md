# Design Document: FileScanner Completion

## Overview

This design document outlines the technical approach for completing the FileScanner component. The FileScanner is responsible for recursively traversing directories, filtering files based on user criteria, and providing file information to downstream components (HashCalculator and DuplicateDetector).

The current implementation (70% complete) includes:
- Basic recursive directory scanning with QDirIterator
- File size filtering
- Hidden file inclusion/exclusion
- System directory exclusions
- Basic progress reporting
- Scan cancellation

This design adds:
- Pattern-based file filtering (glob and regex)
- Enhanced error handling with specific error types
- Performance optimizations for large-scale scanning
- Comprehensive integration testing

## Architecture

### Component Structure

```
FileScanner
├── Core Scanning Logic (✅ Implemented)
│   ├── QDirIterator-based traversal
│   ├── Async queue processing
│   └── Cancellation support
├── Filtering System (⚠️ Partial)
│   ├── Size-based filtering (✅)
│   ├── Hidden file filtering (✅)
│   ├── System directory filtering (✅)
│   └── Pattern matching (❌ To implement)
├── Error Handling (⚠️ Basic)
│   ├── Basic error logging (✅)
│   └── Specific error types (❌ To implement)
└── Performance (⚠️ Needs optimization)
    ├── Memory management (⚠️)
    ├── Progress batching (⚠️)
    └── Metadata caching (❌ Optional)
```

### Data Flow

```
User Input (Scan Options)
    ↓
FileScanner::startScan()
    ↓
processScanQueue() [Async]
    ↓
scanDirectory() [Per directory]
    ↓
shouldIncludeFile() [Filtering]
    ↓
    ├─→ Pattern matching (NEW)
    ├─→ Size filtering (✅)
    └─→ Hidden file check (✅)
    ↓
Emit fileFound() signal
    ↓
Store in m_scannedFiles
    ↓
Emit scanCompleted()
```

## Components and Interfaces

### 1. Pattern Matching System

#### Interface Extension

```cpp
struct ScanOptions {
    // Existing fields...
    QStringList includePatterns;    // NEW: *.jpg, *.png, etc.
    QStringList excludePatterns;    // NEW: *.tmp, .DS_Store, etc.
    bool caseSensitivePatterns = false;  // NEW
};
```

#### Implementation Methods

```cpp
class FileScanner {
private:
    // Pattern matching methods
    bool matchesIncludePatterns(const QString& fileName) const;
    bool matchesExcludePatterns(const QString& fileName) const;
    bool matchesPattern(const QString& fileName, 
                       const QString& pattern, 
                       bool caseSensitive) const;
    
    // Pattern compilation cache for performance
    mutable QHash<QString, QRegularExpression> m_patternCache;
};
```

#### Pattern Matching Logic

1. **Glob Pattern Support:**
   - Convert glob patterns (*.jpg) to regex using `QRegularExpression::wildcardToRegularExpression()`
   - Cache compiled regex patterns for performance
   - Support multiple patterns with OR logic

2. **Regex Pattern Support:**
   - Direct regex pattern support with proper escaping
   - Validate regex patterns before use
   - Handle invalid patterns gracefully

3. **Matching Priority:**
   - If include patterns specified: file must match at least one include pattern
   - If exclude patterns specified: file must not match any exclude pattern
   - Include patterns take precedence over exclude patterns

### 2. Enhanced Error Handling System

#### Error Type Enumeration

```cpp
enum class ScanError {
    None,
    PermissionDenied,      // Cannot read file/directory
    FileSystemError,       // General I/O error
    NetworkTimeout,        // Network drive timeout
    DiskReadError,         // Bad sectors, disk errors
    PathTooLong,          // Path exceeds system limits
    UnknownError          // Catch-all
};
```

#### Error Information Structure

```cpp
struct ScanErrorInfo {
    ScanError errorType;
    QString filePath;
    QString errorMessage;
    QString systemErrorCode;  // OS-specific error code
    QDateTime timestamp;
};
```

#### Error Handling Strategy

1. **Error Detection:**
   - Wrap file system operations in try-catch blocks
   - Check QFileInfo and QDir error states
   - Monitor QDirIterator for errors

2. **Error Reporting:**
   - Emit `scanError()` signal for each error
   - Accumulate errors in `m_scanErrors` list
   - Provide error summary in scan completion

3. **Error Recovery:**
   - Continue scanning after non-critical errors
   - Skip problematic directories but continue with siblings
   - Implement retry logic for transient errors (network timeouts)

4. **Error Logging:**
   - Log all errors with qWarning() for debugging
   - Include full context (path, error type, system message)
   - Provide error statistics in scan results

### 3. Performance Optimization

#### Memory Management

**Current Issue:** Storing all file information in memory can consume significant RAM for large scans.

**Optimization Strategy:**

1. **Efficient Data Structures:**
   ```cpp
   // Use QVector instead of QList for better memory locality
   QVector<FileInfo> m_scannedFiles;
   
   // Reserve capacity if estimate available
   m_scannedFiles.reserve(estimatedFileCount);
   ```

2. **String Optimization:**
   ```cpp
   // Use QString sharing and implicit sharing
   // Avoid unnecessary string copies
   // Use QStringView for temporary string operations
   ```

3. **Optional Streaming Mode:**
   ```cpp
   // For very large scans, emit files immediately without storing
   bool m_streamingMode = false;  // Don't store all files in memory
   ```

#### Progress Batching

**Current Issue:** Emitting progress for every file can cause UI lag.

**Optimization:**

```cpp
// Batch progress updates
static const int PROGRESS_BATCH_SIZE = 100;

if (m_filesProcessed % PROGRESS_BATCH_SIZE == 0) {
    emit scanProgress(m_filesProcessed, -1, filePath);
}
```

#### Metadata Caching (Optional)

**Use Case:** Repeated scans of same directories.

**Implementation:**

```cpp
struct CachedFileInfo {
    QString filePath;
    qint64 fileSize;
    QDateTime lastModified;
    QDateTime cachedAt;
};

QHash<QString, CachedFileInfo> m_metadataCache;
bool m_enableMetadataCache = false;
```

**Cache Strategy:**
- Cache file metadata (size, modified time) by path
- Invalidate cache entries if file modified time changed
- Optional feature, disabled by default
- Configurable cache size limit

## Data Models

### ScanOptions (Extended)

```cpp
struct ScanOptions {
    QStringList targetPaths;
    qint64 minimumFileSize = 1024 * 1024;  // 1MB default
    qint64 maximumFileSize = 0;            // 0 = no limit
    bool includeHiddenFiles = false;
    bool scanSystemDirectories = false;
    
    // NEW: Pattern matching
    QStringList includePatterns;           // Only include matching files
    QStringList excludePatterns;           // Exclude matching files
    bool caseSensitivePatterns = false;
    
    // NEW: Performance options
    bool enableMetadataCache = false;
    bool streamingMode = false;            // Don't store all files
    int progressBatchSize = 100;           // Progress update frequency
};
```

### ScanStatistics (New)

```cpp
struct ScanStatistics {
    int totalFilesScanned = 0;
    int totalDirectoriesScanned = 0;
    qint64 totalBytesScanned = 0;
    int filesFiltered = 0;                 // Files excluded by filters
    int errorsEncountered = 0;
    QTime scanDuration;
    double filesPerSecond = 0.0;
    qint64 peakMemoryUsage = 0;
};
```

## Error Handling

### Error Scenarios and Responses

| Error Type | Detection | Response | User Impact |
|------------|-----------|----------|-------------|
| Permission Denied | QFileInfo::isReadable() | Log, emit error, continue | File/dir skipped, scan continues |
| Network Timeout | QDir operation timeout | Retry once, then skip | Directory skipped after retry |
| Disk Read Error | QFile::error() | Log, emit error, continue | File skipped, scan continues |
| Path Too Long | Path length check | Log, emit error, continue | File skipped, scan continues |
| Invalid Pattern | Regex compilation | Warn user, disable pattern | Pattern ignored, scan continues |

### Error Signal

```cpp
signals:
    void scanError(ScanError errorType, 
                  const QString& path, 
                  const QString& description);
    void scanErrorSummary(int totalErrors, 
                         const QList<ScanErrorInfo>& errors);
```

## Testing Strategy

### Unit Tests

1. **Pattern Matching Tests:**
   - Test glob patterns (*.jpg, *.*)
   - Test regex patterns
   - Test case sensitivity
   - Test invalid patterns
   - Test pattern combinations

2. **Error Handling Tests:**
   - Test permission denied scenarios
   - Test invalid paths
   - Test network drive simulation
   - Test error accumulation

3. **Performance Tests:**
   - Test memory usage with 100k files
   - Test scan speed benchmarks
   - Test progress batching
   - Test metadata caching

### Integration Tests

1. **FileScanner → HashCalculator:**
   - Verify file list format compatibility
   - Test signal/slot connections
   - Test cancellation propagation

2. **FileScanner → DuplicateDetector:**
   - Verify FileInfo structure compatibility
   - Test end-to-end workflow
   - Test large dataset handling

3. **End-to-End Tests:**
   - Full scan workflow with real directories
   - Test with various file system types
   - Test cross-platform compatibility

### Performance Benchmarks

```cpp
// Target metrics
- Scan rate: >= 1,000 files/minute on SSD
- Memory usage: <= 100MB for 100,000 files
- Progress update latency: <= 100ms
- Pattern matching overhead: <= 5% of scan time
```

## Implementation Plan

### Phase 1: Pattern Matching (4 hours)
1. Add pattern fields to ScanOptions
2. Implement pattern matching methods
3. Add pattern cache for performance
4. Update shouldIncludeFile() logic
5. Write unit tests

### Phase 2: Error Handling (3 hours)
1. Define error types and structures
2. Add error detection in scan methods
3. Implement error signals
4. Add error accumulation and reporting
5. Write error handling tests

### Phase 3: Performance Optimization (5 hours)
1. Implement progress batching
2. Optimize data structures
3. Add optional metadata caching
4. Add optional streaming mode
5. Write performance benchmarks

### Phase 4: Integration Testing (3 hours)
1. Test with HashCalculator integration
2. Test with DuplicateDetector integration
3. End-to-end workflow tests
4. Cross-platform testing
5. Performance regression tests

## Dependencies

- Qt 6.x (QRegularExpression, QDirIterator, QFileInfo)
- Existing FileScanner implementation (70% complete)
- HashCalculator component (for integration testing)
- DuplicateDetector component (for integration testing)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Regex performance overhead | Medium | Cache compiled patterns, limit pattern complexity |
| Memory usage with large scans | High | Implement streaming mode, optimize data structures |
| Cross-platform path handling | Medium | Use Qt's cross-platform APIs, test on all platforms |
| Error handling complexity | Low | Clear error types, comprehensive testing |

## Success Criteria

1. ✅ Pattern matching works with glob and regex patterns
2. ✅ Error handling covers all common error scenarios
3. ✅ Memory usage stays below 100MB for 100k files
4. ✅ Scan rate achieves 1,000+ files/minute on SSD
5. ✅ Integration tests pass with HashCalculator and DuplicateDetector
6. ✅ Code coverage reaches 90%+
7. ✅ All existing tests continue to pass
