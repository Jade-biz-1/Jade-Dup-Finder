# FileScanner Performance Tuning Guide

This guide provides detailed information on optimizing FileScanner performance for various use cases and system configurations.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Performance Targets](#performance-targets)
- [Memory Optimization](#memory-optimization)
- [CPU Optimization](#cpu-optimization)
- [I/O Optimization](#io-optimization)
- [Progress Update Optimization](#progress-update-optimization)
- [Pattern Matching Optimization](#pattern-matching-optimization)
- [Metadata Caching](#metadata-caching)
- [Benchmarking](#benchmarking)
- [Platform-Specific Considerations](#platform-specific-considerations)

## Performance Overview

The FileScanner is designed to efficiently scan large directory structures with minimal resource usage. Performance characteristics depend on:

- **File system type** (SSD vs HDD, local vs network)
- **Number of files** (thousands vs millions)
- **Directory depth** (flat vs deeply nested)
- **File sizes** (many small files vs few large files)
- **Filtering complexity** (simple size filters vs complex patterns)

### Key Performance Metrics

1. **Scan Rate**: Files processed per second
2. **Memory Usage**: RAM consumed during scanning
3. **CPU Usage**: Processor utilization
4. **I/O Throughput**: Disk read operations per second
5. **Progress Latency**: Time between progress updates

## Performance Targets

The FileScanner is designed to meet these performance targets:

| Metric | Target | Typical |
|--------|--------|---------|
| Scan Rate (SSD) | ≥ 1,000 files/min | 2,000-5,000 files/min |
| Scan Rate (HDD) | ≥ 500 files/min | 800-1,500 files/min |
| Memory Usage (100k files) | ≤ 100 MB | 50-80 MB |
| Memory Usage (1M files) | ≤ 1 GB | 500-800 MB |
| Progress Update Latency | ≤ 100 ms | 10-50 ms |
| Pattern Matching Overhead | ≤ 5% | 2-3% |

### Measuring Performance

Use the scan statistics to measure actual performance:

```cpp
connect(&scanner, &FileScanner::scanStatistics,
        [](const FileScanner::ScanStatistics& stats) {
    qDebug() << "=== Performance Metrics ===";
    qDebug() << "Scan rate:" << stats.filesPerSecond << "files/sec";
    qDebug() << "Duration:" << (stats.scanDurationMs / 1000.0) << "seconds";
    qDebug() << "Memory:" << (stats.peakMemoryUsage / 1024.0 / 1024.0) << "MB";
});
```

## Memory Optimization

### Problem: High Memory Usage

Storing file information for millions of files can consume significant RAM.

### Solution 1: Streaming Mode

Process files as they're found without storing them all in memory:

```cpp
FileScanner::ScanOptions options;
options.streamingMode = true;  // Don't store all files

connect(&scanner, &FileScanner::fileFound,
        [](const FileScanner::FileInfo& file) {
    // Process file immediately
    processFile(file);
    // File info is not stored in scanner
});

scanner.startScan(options);
```

**Benefits**:
- Constant memory usage regardless of file count
- Suitable for millions of files
- Lower memory footprint

**Trade-offs**:
- Cannot call `getScannedFiles()` after scan
- Must process files in real-time

**Use When**:
- Scanning > 100,000 files
- Memory is constrained
- Files can be processed immediately

### Solution 2: Pre-allocate Memory

If you know the approximate file count, pre-allocate memory:

```cpp
FileScanner::ScanOptions options;
options.estimatedFileCount = 50000;  // Hint: expect ~50k files

scanner.startScan(options);
```

**Benefits**:
- Reduces memory reallocations
- Improves performance by 10-20%
- No trade-offs

**Use When**:
- You know approximate file count
- Not using streaming mode

### Solution 3: Filter Early

Apply filters to reduce the number of files stored:

```cpp
FileScanner::ScanOptions options;
options.minimumFileSize = 1024 * 1024;  // Only files > 1MB
options.includePatterns = {"*.jpg", "*.png"};  // Only images

scanner.startScan(options);
```

**Benefits**:
- Fewer files stored in memory
- Faster scanning (less processing)
- Lower memory usage

### Memory Usage Calculation

Approximate memory per file:

```cpp
// FileInfo structure size
sizeof(QString) * 3 +  // filePath, fileName, directory (~150 bytes)
sizeof(qint64) +       // fileSize (8 bytes)
sizeof(QDateTime) +    // lastModified (8 bytes)
// Total: ~170 bytes per file

// For 100,000 files:
// 100,000 * 170 bytes = 17 MB (file info)
// + QVector overhead (~20%)
// + Qt string data (~2x for QString)
// ≈ 50-60 MB total
```

### Monitoring Memory Usage

```cpp
#include <QProcess>

qint64 getCurrentMemoryUsage() {
    #ifdef Q_OS_LINUX
    QFile file("/proc/self/status");
    if (file.open(QIODevice::ReadOnly)) {
        while (!file.atEnd()) {
            QString line = file.readLine();
            if (line.startsWith("VmRSS:")) {
                QStringList parts = line.split(QRegularExpression("\\s+"));
                if (parts.size() >= 2) {
                    return parts[1].toLongLong() * 1024;  // Convert KB to bytes
                }
            }
        }
    }
    #endif
    return 0;
}

// Monitor during scan
qint64 startMemory = getCurrentMemoryUsage();
scanner.startScan(options);
// ... after scan ...
qint64 endMemory = getCurrentMemoryUsage();
qDebug() << "Memory used:" << ((endMemory - startMemory) / 1024.0 / 1024.0) << "MB";
```

## CPU Optimization

### Pattern Matching Optimization

Pattern matching is cached to minimize CPU overhead:

```cpp
// Patterns are compiled once and cached
QHash<QString, QRegularExpression> m_patternCache;

QRegularExpression FileScanner::compilePattern(const QString& pattern, bool caseSensitive) const {
    QString cacheKey = pattern + (caseSensitive ? ":cs" : ":ci");
    
    if (m_patternCache.contains(cacheKey)) {
        return m_patternCache[cacheKey];  // Return cached
    }
    
    // Compile and cache
    QRegularExpression regex = QRegularExpression::wildcardToRegularExpression(pattern);
    if (!caseSensitive) {
        regex.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
    }
    m_patternCache[cacheKey] = regex;
    return regex;
}
```

**Best Practices**:
- Use simple glob patterns when possible (faster than regex)
- Avoid overly complex regex patterns
- Reuse the same scanner instance for multiple scans (cache persists)

### Minimize String Operations

```cpp
// Good: Use QStringView for temporary operations
bool matchesPattern(const QString& fileName, const QString& pattern) {
    QStringView fileView(fileName);
    // Operations on view are faster
}

// Good: Use implicit sharing
QString filePath = fileInfo.filePath();  // Shared, not copied

// Bad: Unnecessary string copies
QString filePath = QString(fileInfo.filePath());  // Explicit copy
```

## I/O Optimization

### File System Considerations

Different file systems have different performance characteristics:

| File System | Scan Rate | Notes |
|-------------|-----------|-------|
| SSD (ext4) | 3,000-5,000 files/min | Best performance |
| SSD (NTFS) | 2,500-4,000 files/min | Good performance |
| HDD (ext4) | 800-1,500 files/min | Limited by seek time |
| Network (SMB) | 200-800 files/min | Limited by network |
| Network (NFS) | 300-1,000 files/min | Better than SMB |

### Optimizing for HDDs

HDDs are limited by seek time. Optimize by:

1. **Minimize random access**:
```cpp
// FileScanner already uses QDirIterator which reads sequentially
// No additional optimization needed
```

2. **Reduce metadata queries**:
```cpp
// Enable metadata caching for repeated scans
options.enableMetadataCache = true;
```

### Optimizing for Network Drives

Network drives have high latency. Optimize by:

1. **Increase progress batch size**:
```cpp
options.progressBatchSize = 500;  // Reduce signal overhead
```

2. **Handle timeouts gracefully**:
```cpp
// FileScanner automatically retries network timeouts
// No additional configuration needed
```

3. **Consider local caching**:
```cpp
options.enableMetadataCache = true;
options.metadataCacheSizeLimit = 50000;
```

## Progress Update Optimization

### Problem: UI Lag from Frequent Updates

Emitting progress for every file can cause UI lag, especially with fast scans.

### Solution: Batch Progress Updates

```cpp
FileScanner::ScanOptions options;
options.progressBatchSize = 100;  // Default: update every 100 files

// For very fast scans (SSD, many small files):
options.progressBatchSize = 500;  // Update every 500 files

// For slow scans (HDD, network):
options.progressBatchSize = 50;   // Update every 50 files
```

### Progress Update Frequency Guidelines

| Scan Speed | Files/sec | Recommended Batch Size | Update Frequency |
|------------|-----------|------------------------|------------------|
| Very Fast | > 1,000 | 500-1,000 | ~1-2 updates/sec |
| Fast | 500-1,000 | 200-500 | ~2-5 updates/sec |
| Medium | 100-500 | 100-200 | ~1-5 updates/sec |
| Slow | < 100 | 50-100 | ~1-2 updates/sec |

### Measuring Progress Overhead

```cpp
QElapsedTimer timer;
int progressCount = 0;
qint64 progressTime = 0;

connect(&scanner, &FileScanner::scanProgress,
        [&]() {
    timer.start();
    // Your progress handling code
    updateProgressBar();
    progressTime += timer.nsecsElapsed();
    progressCount++;
});

connect(&scanner, &FileScanner::scanCompleted, [&]() {
    double avgProgressTime = progressTime / 1000000.0 / progressCount;  // ms
    qDebug() << "Average progress update time:" << avgProgressTime << "ms";
});
```

## Pattern Matching Optimization

### Pattern Complexity

Different pattern types have different performance characteristics:

| Pattern Type | Example | Performance | Overhead |
|--------------|---------|-------------|----------|
| Simple glob | `*.jpg` | Fastest | < 1% |
| Multi-char glob | `IMG_????.jpg` | Fast | 1-2% |
| Character class | `photo[0-9].png` | Fast | 1-2% |
| Simple regex | `.*\.jpg$` | Medium | 2-3% |
| Complex regex | `^IMG_\d{4}_(A|B)\.jpg$` | Slower | 5-10% |

### Optimization Tips

1. **Use glob patterns when possible**:
```cpp
// Good: Simple glob
options.includePatterns = {"*.jpg", "*.png"};

// Avoid: Regex when glob works
options.includePatterns = {".*\\.jpg$", ".*\\.png$"};
```

2. **Combine patterns efficiently**:
```cpp
// Good: Multiple simple patterns
options.includePatterns = {"*.jpg", "*.jpeg", "*.png"};

// Avoid: One complex regex
options.includePatterns = {".*\\.(jpg|jpeg|png)$"};
```

3. **Use exclude patterns for common cases**:
```cpp
// Excluding is often faster than including
options.excludePatterns = {"*.tmp", "*.bak", "*~"};
```

### Pattern Matching Benchmark

```cpp
void benchmarkPatternMatching() {
    FileScanner scanner;
    
    // Test 1: No patterns
    FileScanner::ScanOptions options1;
    options1.targetPaths = {"/test/path"};
    QElapsedTimer timer;
    timer.start();
    scanner.startScan(options1);
    // wait for completion...
    qint64 timeNoPatterns = timer.elapsed();
    
    // Test 2: With patterns
    FileScanner::ScanOptions options2;
    options2.targetPaths = {"/test/path"};
    options2.includePatterns = {"*.jpg", "*.png"};
    timer.restart();
    scanner.startScan(options2);
    // wait for completion...
    qint64 timeWithPatterns = timer.elapsed();
    
    double overhead = ((timeWithPatterns - timeNoPatterns) * 100.0) / timeNoPatterns;
    qDebug() << "Pattern matching overhead:" << overhead << "%";
}
```

## Metadata Caching

### When to Use Caching

Metadata caching is beneficial when:
- Scanning the same directories repeatedly
- Files don't change frequently
- Scan speed is critical

### Enabling Caching

```cpp
FileScanner::ScanOptions options;
options.enableMetadataCache = true;
options.metadataCacheSizeLimit = 20000;  // Cache up to 20k files

scanner.startScan(options);
```

### Cache Behavior

1. **First scan**: Builds cache (no performance benefit)
2. **Subsequent scans**: Uses cache (significant speedup)
3. **Cache invalidation**: Automatic based on file modification time

### Cache Performance

| Scenario | First Scan | Cached Scan | Speedup |
|----------|------------|-------------|---------|
| 10,000 files (no changes) | 10 sec | 2 sec | 5x |
| 10,000 files (10% changed) | 10 sec | 3 sec | 3.3x |
| 10,000 files (all changed) | 10 sec | 10 sec | 1x |

### Cache Management

```cpp
// Clear cache when you know files have changed
scanner.clearMetadataCache();

// Or let it auto-invalidate based on modification time
// (no action needed)
```

### Cache Size Tuning

```cpp
// Small cache (memory constrained)
options.metadataCacheSizeLimit = 5000;

// Medium cache (default)
options.metadataCacheSizeLimit = 10000;

// Large cache (plenty of RAM)
options.metadataCacheSizeLimit = 50000;

// Unlimited cache (use with caution)
options.metadataCacheSizeLimit = -1;
```

## Benchmarking

### Creating Performance Tests

```cpp
class FileScannerBenchmark {
public:
    void runBenchmark(const QString& path) {
        FileScanner scanner;
        
        connect(&scanner, &FileScanner::scanStatistics,
                this, &FileScannerBenchmark::recordResults);
        
        // Warm-up run
        FileScanner::ScanOptions options;
        options.targetPaths = {path};
        scanner.startScan(options);
        // wait...
        
        // Benchmark runs
        for (int i = 0; i < 5; ++i) {
            scanner.startScan(options);
            // wait...
        }
        
        printResults();
    }
    
private:
    void recordResults(const FileScanner::ScanStatistics& stats) {
        m_results.append(stats);
    }
    
    void printResults() {
        qDebug() << "=== Benchmark Results ===";
        
        double avgRate = 0;
        qint64 avgDuration = 0;
        
        for (const auto& stats : m_results) {
            avgRate += stats.filesPerSecond;
            avgDuration += stats.scanDurationMs;
        }
        
        avgRate /= m_results.size();
        avgDuration /= m_results.size();
        
        qDebug() << "Average scan rate:" << avgRate << "files/sec";
        qDebug() << "Average duration:" << (avgDuration / 1000.0) << "seconds";
    }
    
    QList<FileScanner::ScanStatistics> m_results;
};
```

### Performance Regression Testing

```cpp
void testPerformanceRegression() {
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths = {"/test/data"};  // Known test dataset
    
    QElapsedTimer timer;
    timer.start();
    scanner.startScan(options);
    // wait for completion...
    qint64 duration = timer.elapsed();
    
    // Assert performance target
    const qint64 MAX_DURATION_MS = 10000;  // 10 seconds
    QVERIFY2(duration < MAX_DURATION_MS,
             QString("Scan took %1ms, expected < %2ms")
             .arg(duration).arg(MAX_DURATION_MS).toUtf8());
}
```

## Platform-Specific Considerations

### Linux

**Optimizations**:
- Use ext4 for best performance
- Avoid scanning `/proc`, `/sys` (set `scanSystemDirectories = false`)
- Consider using `ionice` for background scans

**Example**:
```bash
# Run with lower I/O priority
ionice -c 3 ./your_app
```

### Windows

**Optimizations**:
- NTFS performs well, avoid FAT32
- Exclude `C:\Windows\WinSxS` (very large)
- Handle long paths (> 260 chars)

**Example**:
```cpp
options.excludePatterns = {
    "C:/Windows/WinSxS/*",
    "C:/Windows/Installer/*",
    "C:/$Recycle.Bin/*"
};
```

### macOS

**Optimizations**:
- APFS performs well
- Exclude `.Spotlight-V100`, `.fseventsd`
- Handle case-insensitive file system

**Example**:
```cpp
options.excludePatterns = {
    ".Spotlight-V100/*",
    ".fseventsd/*",
    ".Trashes/*",
    ".DS_Store"
};
```

## Performance Checklist

Use this checklist to optimize FileScanner performance:

- [ ] Enable streaming mode for > 100k files
- [ ] Provide estimated file count when known
- [ ] Use appropriate progress batch size
- [ ] Apply filters early (size, patterns)
- [ ] Use simple glob patterns instead of complex regex
- [ ] Enable metadata caching for repeated scans
- [ ] Exclude system directories
- [ ] Exclude temporary/cache directories
- [ ] Test with realistic data
- [ ] Measure actual performance with statistics
- [ ] Monitor memory usage
- [ ] Profile if performance is below targets

## See Also

- [FileScanner API Documentation](API_FILESCANNER.md)
- [Usage Examples](FILESCANNER_EXAMPLES.md)
- [Error Handling Guide](FILESCANNER_ERROR_HANDLING.md)
- [Integration Examples](FILESCANNER_INTEGRATION.md)
