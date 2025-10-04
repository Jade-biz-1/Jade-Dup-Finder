#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QTime>
#include <QThread>
#include <QFuture>
#include <QHash>
#include <QList>
#include <QMutex>
#include <QCryptographicHash>
#include <QThreadPool>
#include <QRunnable>
#include <QAtomicInt>
#include <QSemaphore>
#include <QQueue>
#include <QWaitCondition>
#include <QFile>
#include <QSharedPointer>

/**
 * @brief HashCalculator - High-performance SHA-256 file hash calculator
 * 
 * Features:
 * - Multi-threaded concurrent hash calculation
 * - Progressive hashing for large files with progress reporting
 * - LRU cache for hash results with configurable size
 * - Memory-efficient processing regardless of file size
 * - Cancellation support for long operations
 * - Robust error handling and recovery
 * 
 * Usage:
 * ```cpp
 * HashCalculator calc;
 * connect(&calc, &HashCalculator::hashCompleted, this, &MyClass::onHashReady);
 * calc.calculateFileHash("/path/to/file.txt");
 * ```
 */
class HashCalculator : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Result structure for hash calculation
     */
    struct HashResult {
        QString filePath;           // Full path to the file
        QString hash;               // SHA-256 hash (lowercase hex)
        qint64 fileSize;           // File size in bytes
        QDateTime calculated;       // When hash was calculated
        bool fromCache;            // True if result came from cache
        QString errorMessage;       // Error description if calculation failed
        bool success = true;        // True if hash calculation succeeded
        
        // Convenience constructor
        HashResult(const QString& path = QString()) 
            : filePath(path), fileSize(0), fromCache(false), success(false) {}
    };
    
    /**
     * @brief Configuration options for hash calculation
     */
    struct HashOptions {
        int threadPoolSize = QThread::idealThreadCount();     // Number of concurrent threads
        qint64 largeFileThreshold = 100 * 1024 * 1024;      // 100MB - threshold for progressive hashing
        qint64 chunkSize = 64 * 1024;                        // 64KB - chunk size for file reading
        int progressUpdateInterval = 1024 * 1024;            // 1MB - progress update frequency
        bool enableCaching = true;                           // Enable hash result caching
        int maxCacheSize = 10000;                           // Maximum cache entries
        int maxRetryAttempts = 3;                           // Retry attempts for transient errors
        int retryDelayMs = 100;                             // Delay between retries
        
        // Advanced thread pool options
        bool enableDynamicThreads = true;                   // Enable dynamic thread count adjustment
        int minThreads = 1;                                 // Minimum threads to keep active
        int maxThreads = QThread::idealThreadCount() * 2;   // Maximum threads allowed
        int threadIdleTimeout = 30000;                      // Thread idle timeout in milliseconds
        bool enableWorkStealing = true;                     // Enable work-stealing between threads
        int workStealingThreshold = 5;                      // Minimum queue size before stealing
        
        // Batch processing options
        bool enableBatchProcessing = true;                  // Enable intelligent batching
        int batchSize = 10;                                 // Files per batch
        int smallFileBatchSize = 50;                        // Larger batches for small files
        qint64 smallFileThreshold = 1024 * 1024;           // 1MB threshold for small files
        int mediumFileBatchSize = 25;                       // Medium files batch size
        qint64 mediumFileThreshold = 10 * 1024 * 1024;     // 10MB threshold for medium files
        int largeFileBatchSize = 5;                         // Large files batch size
        bool enableParallelBatches = true;                  // Process batches in parallel
        int maxConcurrentBatches = 3;                       // Maximum concurrent batches
        bool enableSizeBasedGrouping = true;                // Group by file size categories
        bool enableAdaptiveBatching = true;                 // Adapt batch sizes based on performance
        
        // I/O optimization options
        bool enableIOOptimizations = true;                  // Enable all I/O optimizations
        bool enableMemoryMapping = true;                    // Use memory mapping for large files
        qint64 memoryMapThreshold = 50 * 1024 * 1024;      // 50MB threshold
        bool enableReadAhead = true;                        // Enable read-ahead caching
        qint64 readAheadSize = 1024 * 1024;                // 1MB read-ahead
        bool enableAsyncIO = true;                          // Enable async I/O operations
        int maxConcurrentReads = 4;                         // Max concurrent reads
        bool enableBufferPooling = true;                    // Use buffer pool
    };
    
    /**
     * @brief Enhanced progress information with detailed metrics
     */
    struct ProgressInfo {
        QString filePath;
        qint64 bytesProcessed;
        qint64 totalBytes;
        int percentComplete;
        
        // Timing information
        QDateTime startTime;
        QDateTime lastUpdateTime;
        qint64 elapsedTimeMs;
        qint64 estimatedTimeRemainingMs;
        
        // Speed and throughput metrics
        double instantaneousSpeed;   // MB/s - current speed
        double averageSpeed;         // MB/s - overall average
        double peakSpeed;            // MB/s - highest recorded speed
        double speedVariance;        // Speed variation coefficient
        
        // Queue and thread information
        int queueDepth;             // Files waiting to be processed
        int activeThreads;          // Currently processing threads
        int totalThreads;           // Total available threads
        double threadUtilization;   // Percentage of threads active
        
        // Resource utilization
        double cpuUtilization;      // Estimated CPU usage percentage
        double ioUtilization;       // I/O subsystem utilization
        qint64 memoryUsage;         // Current memory usage in bytes
        
        // Advanced ETA calculations
        double etaAccuracy;         // Confidence in ETA estimate (0.0-1.0)
        QStringList etaFactors;     // Factors affecting ETA calculation
        
        // Performance trends
        QString performanceTrend;   // "improving", "stable", "degrading"
        double trendStrength;       // How strong the trend is (0.0-1.0)
        
        ProgressInfo() : bytesProcessed(0), totalBytes(0), percentComplete(0),
                        elapsedTimeMs(0), estimatedTimeRemainingMs(0),
                        instantaneousSpeed(0.0), averageSpeed(0.0), peakSpeed(0.0), speedVariance(0.0),
                        queueDepth(0), activeThreads(0), totalThreads(0), threadUtilization(0.0),
                        cpuUtilization(0.0), ioUtilization(0.0), memoryUsage(0),
                        etaAccuracy(0.0), trendStrength(0.0) {}
    };
    
    /**
     * @brief Batch processing information and statistics
     */
    struct BatchInfo {
        int batchId;
        QStringList filePaths;
        qint64 totalSize;
        int priority;
        QDateTime created;
        QDateTime started;
        QDateTime completed;
        bool isSmallFileBatch;
        double processingTime;       // milliseconds
        double averageFileSpeed;     // files/second
        
        BatchInfo() : batchId(-1), totalSize(0), priority(0), isSmallFileBatch(false), 
                     processingTime(0.0), averageFileSpeed(0.0) {}
    };
    
    /**
     * @brief Adaptive chunk sizing based on system performance
     */
    struct AdaptiveChunkConfig {
        qint64 baseChunkSize = 64 * 1024;           // 64KB default
        qint64 minChunkSize = 16 * 1024;            // 16KB minimum
        qint64 maxChunkSize = 2 * 1024 * 1024;      // 2MB maximum
        double targetThroughput = 50.0;             // Target MB/s
        double adaptationRate = 0.1;                // How quickly to adapt (0.0-1.0)
        bool enableAdaptation = true;               // Enable adaptive sizing
        
        // Runtime adaptation state
        qint64 currentChunkSize = baseChunkSize;
        double recentThroughput = 0.0;
        QElapsedTimer performanceTimer;
        int adaptationSamples = 0;
    };
    
    /**
     * @brief I/O optimization configuration and statistics
     */
    struct IOOptimizationConfig {
        // Main I/O optimization toggle
        bool enabled = true;
        
        // Memory-mapped file settings
        bool memoryMappingEnabled = true;            // Use memory mapping for large files
        qint64 memoryMappingThreshold = 50 * 1024 * 1024; // 50MB threshold for memory mapping
        qint64 maxMemoryMapSize = 500 * 1024 * 1024;  // 500MB max memory map size
        
        // Read-ahead caching
        bool readAheadEnabled = true;                // Enable read-ahead optimization
        qint64 readAheadBufferSize = 64 * 1024;     // 64KB read-ahead buffer
        int readAheadBlocks = 4;                     // Number of blocks to read ahead
        
        // Asynchronous I/O
        bool asyncIOEnabled = true;                  // Enable asynchronous I/O operations
        int maxConcurrentReads = 4;                  // Maximum concurrent read operations
        qint64 asyncIOThreshold = 10 * 1024 * 1024;  // 10MB threshold for async I/O
        
        // I/O buffer management
        bool bufferPoolEnabled = true;               // Use buffer pool for efficiency
        int bufferPoolSize = 16;                     // Number of buffers in pool
        qint64 bufferSize = 256 * 1024;             // 256KB per buffer
        
        // Direct I/O (bypassing OS cache)
        bool directIOEnabled = false;                // Enable direct I/O (Linux O_DIRECT)
        bool sequentialAdviceEnabled = true;        // Use POSIX_FADV_SEQUENTIAL
    };
    
    /**
     * @brief Real-time throughput monitoring with sliding window
     */
    struct ThroughputMonitor {
        int windowSizeMs = 5000;                    // 5-second sliding window
        QList<QPair<qint64, double>> dataPoints;   // timestamp, throughput pairs
        double currentThroughput = 0.0;             // MB/s
        double averageThroughput = 0.0;             // MB/s
        double peakThroughput = 0.0;                // MB/s
        double minThroughput = 0.0;                 // MB/s
        double throughputVariance = 0.0;            // Variance in throughput
        
        // Trend analysis
        double shortTermTrend = 0.0;                // 1-second trend
        double longTermTrend = 0.0;                 // 30-second trend
        QString trendDirection = "stable";          // "increasing", "decreasing", "stable"
        
        // Bandwidth utilization
        double maxBandwidth = 1000.0;               // MB/s - theoretical max
        double bandwidthUtilization = 0.0;          // Percentage of max bandwidth
        
        void addDataPoint(double throughput) {
            qint64 now = QDateTime::currentMSecsSinceEpoch();
            dataPoints.append({now, throughput});
            
            // Remove old data points outside the window
            qint64 cutoff = now - windowSizeMs;
            while (!dataPoints.isEmpty() && dataPoints.first().first < cutoff) {
                dataPoints.removeFirst();
            }
            
            updateMetrics();
        }
        
        void updateMetrics();
        void calculateTrends();
    };
    
    /**
     * @brief Performance histogram for statistical analysis
     */
    struct PerformanceHistogram {
        QMap<QString, QList<double>> metrics;       // metric_name -> values
        int maxSamples = 1000;                      // Maximum samples to keep
        
        void addSample(const QString& metricName, double value) {
            if (!metrics.contains(metricName)) {
                metrics[metricName] = QList<double>();
            }
            
            QList<double>& samples = metrics[metricName];
            samples.append(value);
            
            // Keep only recent samples
            while (samples.size() > maxSamples) {
                samples.removeFirst();
            }
        }
        
        double getPercentile(const QString& metricName, double percentile) const;
        double getMean(const QString& metricName) const;
        double getStandardDeviation(const QString& metricName) const;
        QPair<double, double> getRange(const QString& metricName) const;
    };
    
    /**
     * @brief Advanced ETA prediction engine
     */
    struct ETAPredictionEngine {
        // Historical performance data
        QList<double> recentSpeeds;                 // Recent speed measurements
        QList<qint64> speedTimestamps;              // Timestamps for speed measurements
        
        // Weighted moving averages
        double shortTermAverage = 0.0;              // 30-second average
        double mediumTermAverage = 0.0;             // 5-minute average  
        double longTermAverage = 0.0;               // 30-minute average
        
        // Queue analysis
        int pendingFiles = 0;
        qint64 pendingBytes = 0;
        double averageFileSize = 0.0;
        
        // Confidence factors
        double speedStability = 1.0;                // How stable the speed is
        double queuePredictability = 1.0;           // How predictable the queue is
        double systemLoad = 0.0;                    // Current system load impact
        
        qint64 calculateETA(qint64 remainingBytes, int remainingFiles);
        double calculateConfidence() const;
        QStringList getInfluencingFactors() const;
        void updateWeightedAverages();
        void calculateSpeedStability();
    };
    
    /**
     * @brief Asynchronous I/O operation tracking
     */
    struct AsyncIOOperation {
        int operationId;
        QString filePath;
        qint64 offset;
        qint64 size;
        QElapsedTimer timer;
        QFuture<QByteArray> future;
        bool completed = false;
        
        AsyncIOOperation(int id, const QString& path, qint64 off, qint64 sz) 
            : operationId(id), filePath(path), offset(off), size(sz) {
            timer.start();
        }
    };

    explicit HashCalculator(QObject* parent = nullptr);
    ~HashCalculator();
    
    // Configuration
    void setOptions(const HashOptions& options);
    HashOptions getOptions() const;
    
    // Main hash calculation interface
    void calculateFileHash(const QString& filePath);
    void calculateFileHashes(const QStringList& filePaths);
    QString calculateFileHashSync(const QString& filePath);  // Synchronous version
    
    // Advanced batch processing
    void calculateFileHashesBatch(const QStringList& filePaths, int priority = 0);
    void calculateFileHashesOptimized(const QStringList& filePaths);
    void calculateFileHashesIntelligent(const QStringList& filePaths);
    void calculateFileHashesParallel(const QList<QStringList>& batches);
    
    // Batch management and monitoring
    QList<BatchInfo> getActiveBatches() const;
    int getActiveBatchCount() const;
    void cancelBatch(int batchId);
    BatchInfo getBatchInfo(int batchId) const;
    
    // Operation control
    void cancelAll();
    void cancelFile(const QString& filePath);
    bool isProcessing() const;
    bool isProcessingFile(const QString& filePath) const;
    
    // Cache management
    void clearCache();
    int getCacheSize() const;
    double getCacheHitRate() const;
    void setCacheEnabled(bool enabled);
    
    // Thread pool management
    void setThreadPoolSize(int size);
    int getThreadPoolSize() const;
    void setDynamicThreadsEnabled(bool enabled);
    bool isDynamicThreadsEnabled() const;
    
    // Statistics
    struct Statistics {
        int totalHashesCalculated = 0;
        int cacheHits = 0;
        int cacheMisses = 0;
        qint64 totalBytesProcessed = 0;
        double averageSpeed = 0.0;  // MB/s
        QTime totalProcessingTime;
        
        // Thread pool statistics
        int activeThreads = 0;
        int peakThreads = 0;
        int tasksQueued = 0;
        int tasksCompleted = 0;
        double threadUtilization = 0.0;  // Percentage
        int workStealingEvents = 0;
        
        // Batch processing statistics
        int batchesProcessed = 0;
        double averageBatchTime = 0.0;  // milliseconds
        int smallFileBatchesOptimized = 0;
        int mediumFileBatchesProcessed = 0;
        int largeFileBatchesProcessed = 0;
        int parallelBatchesExecuted = 0;
        double batchThroughput = 0.0;    // batches/second
        double averageFilesPerBatch = 0.0;
        qint64 totalBatchSizeProcessed = 0;
        
        // Adaptive processing statistics
        int chunkSizeAdaptations = 0;
        qint64 optimalChunkSize = 64 * 1024; // Current optimal chunk size
        double adaptiveThroughputGain = 0.0;  // Percentage improvement
        
        // I/O optimization statistics
        int ioErrors = 0;
        int memoryMappedReads = 0;
        qint64 memoryMappedBytes = 0;
        int memoryMappingFallbacks = 0;
        int readAheadOperations = 0;
        qint64 readAheadBytes = 0;
        int asyncIOOperations = 0;
        qint64 asyncIOBytes = 0;
        int bufferHits = 0;
        int bufferMisses = 0;
        qint64 totalIOTime = 0;              // milliseconds
        double averageIOSpeed = 0.0;         // MB/s
        
        // Enhanced performance metrics (HC-002d)
        double instantaneousThroughput = 0.0;    // Current MB/s
        double peakThroughput = 0.0;             // Highest recorded MB/s
        double throughputVariance = 0.0;         // Variance in throughput
        QString performanceTrend = "stable";     // "improving", "stable", "degrading"
        double trendStrength = 0.0;              // Trend strength (0.0-1.0)
        
        // Resource utilization metrics
        double cpuUtilization = 0.0;             // Estimated CPU usage %
        double memoryUtilization = 0.0;          // Memory usage %
        qint64 peakMemoryUsage = 0;              // Peak memory usage in bytes
        double ioWaitTime = 0.0;                 // Average I/O wait time %
        
        // Queue and concurrency metrics
        int maxQueueDepth = 0;                   // Highest queue depth recorded
        double averageQueueDepth = 0.0;          // Average queue depth
        double queueEfficiency = 1.0;            // Queue utilization efficiency
        int threadContentions = 0;               // Thread contention events
        
        // ETA and prediction accuracy
        double etaAccuracy = 0.0;                // Historical ETA accuracy (0.0-1.0)
        int etaPredictions = 0;                   // Number of ETA predictions made
        double averageETAError = 0.0;            // Average ETA error in seconds
        
        // Performance percentiles
        double throughputP50 = 0.0;              // 50th percentile throughput
        double throughputP90 = 0.0;              // 90th percentile throughput
        double throughputP99 = 0.0;              // 99th percentile throughput
        double processingTimeP50 = 0.0;          // 50th percentile processing time
        double processingTimeP90 = 0.0;          // 90th percentile processing time
        double processingTimeP99 = 0.0;          // 99th percentile processing time
        
        // File size distribution metrics
        qint64 smallFileCount = 0;               // Files < 1MB
        qint64 mediumFileCount = 0;              // Files 1MB-100MB
        qint64 largeFileCount = 0;               // Files > 100MB
        double averageFileSize = 0.0;            // Average file size in MB
        qint64 largestFileProcessed = 0;         // Largest file size processed
        
        // Error and retry metrics
        int totalErrors = 0;                     // Total errors encountered
        int retryAttempts = 0;                   // Total retry attempts
        double errorRate = 0.0;                  // Error rate as percentage
        QMap<QString, int> errorTypeFrequency;   // Error types and their frequency
    };
    Statistics getStatistics() const;
    void resetStatistics();
    
    // Advanced monitoring and progress reporting (HC-002d)
    double getCurrentThroughput() const;  // MB/s
    int getQueuedTaskCount() const;
    
    // Enhanced progress reporting
    ProgressInfo getDetailedProgress() const;
    ThroughputMonitor getThroughputMonitor() const;
    PerformanceHistogram getPerformanceHistogram() const;
    ETAPredictionEngine getETAPredictionEngine() const;
    
    // Real-time metrics
    double getInstantaneousThroughput() const;
    double getThroughputVariance() const;
    QString getPerformanceTrend() const;
    double getTrendStrength() const;
    
    // Resource utilization monitoring
    double getCPUUtilization() const;
    double getMemoryUtilization() const;
    double getIOUtilization() const;
    qint64 getCurrentMemoryUsage() const;
    
    // Queue and concurrency analysis
    int getCurrentQueueDepth() const;
    double getAverageQueueDepth() const;
    double getQueueEfficiency() const;
    int getActiveThreadCount() const;
    
    // ETA and prediction methods
    qint64 getEnhancedETA(qint64 remainingBytes, int remainingFiles) const;
    double getETAAccuracy() const;
    QStringList getETAInfluencingFactors() const;
    
    // Statistical analysis
    double getThroughputPercentile(double percentile) const;
    double getProcessingTimePercentile(double percentile) const;
    QPair<double, double> getThroughputRange() const;
    QMap<QString, double> getPerformanceSummary() const;
    
    // Adaptive chunk sizing
    void setAdaptiveChunkSizing(bool enabled);
    bool isAdaptiveChunkSizingEnabled() const;
    qint64 getOptimalChunkSize() const;
    void resetChunkSizeAdaptation();
    
    // I/O optimization controls
    void setIOOptimizations(bool enabled);
    bool isIOOptimizationsEnabled() const;
    void setMemoryMappingEnabled(bool enabled);
    void setReadAheadEnabled(bool enabled);
    void setAsyncIOEnabled(bool enabled);
    IOOptimizationConfig getIOConfig() const;

signals:
    /**
     * @brief Emitted when a file hash calculation is completed
     * @param result Hash result with file path, hash, and metadata
     */
    void hashCompleted(const HashResult& result);
    
    /**
     * @brief Emitted periodically during large file processing
     * @param progress Current progress information
     */
    void hashProgress(const ProgressInfo& progress);
    
    /**
     * @brief Emitted when an error occurs during hash calculation
     * @param filePath File that caused the error
     * @param error Error description
     */
    void hashError(const QString& filePath, const QString& error);
    
    /**
     * @brief Emitted when a hash operation is cancelled
     * @param filePath File path that was cancelled
     */
    void hashCancelled(const QString& filePath);
    
    /**
     * @brief Emitted when all pending operations complete
     */
    void allOperationsComplete();
    
    /**
     * @brief Emitted when a batch processing starts
     * @param batchInfo Information about the batch being processed
     */
    void batchStarted(const BatchInfo& batchInfo);
    
    /**
     * @brief Emitted when a batch processing completes
     * @param batchInfo Information about the completed batch
     */
    void batchCompleted(const BatchInfo& batchInfo);
    
    /**
     * @brief Emitted when chunk size is adapted for better performance
     * @param oldSize Previous chunk size
     * @param newSize New optimized chunk size
     * @param throughputGain Percentage improvement in throughput
     */
    void chunkSizeAdapted(qint64 oldSize, qint64 newSize, double throughputGain);

private slots:
    void onHashJobFinished();
    
private:
    // Core hash calculation methods
    QString calculateFileHashInternal(const QString& filePath, bool updateCache = true);
    QString calculateChunkedHash(const QString& filePath, qint64 fileSize);
    QString formatHashResult(const QByteArray& hash);
    
    // Large file handling
    QString hashLargeFile(const QString& filePath, qint64 fileSize);
    QString hashFileWithAdaptiveChunking(const QString& filePath);
    void emitProgress(const QString& filePath, qint64 processed, qint64 total, const QTime& startTime);
    
    // Error handling and retry logic
    bool shouldRetry(const QString& errorString) const;
    QString getReadableError(const QString& filePath, const QString& systemError) const;
    
    // Advanced thread pool methods
    void initializeThreadPool();
    void shutdownThreadPool();
    void adjustThreadPoolSize();
    void updateThreadPoolStatistics();
    
    // Task management
    int createHashTask(const QString& filePath, int priority = 0);
    void onTaskCompleted(int taskId, const HashResult& result);
    void onTaskFailed(int taskId, const QString& error);
    
    // Batch processing
    QList<QStringList> createOptimalBatches(const QStringList& filePaths);
    void processBatch(const QStringList& batch, int batchId);
    QList<QStringList> createIntelligentBatches(const QStringList& filePaths);
    QList<QStringList> createSizeGroupedBatches(const QStringList& filePaths);
    void processBatchWithAdaptation(const QStringList& batch, int batchId);
    
    // Adaptive chunk sizing
    void adaptChunkSize(double currentThroughput);
    qint64 calculateOptimalChunkSize(qint64 fileSize, double targetThroughput);
    void updateAdaptiveConfig(double throughput, qint64 bytesProcessed, qint64 timeElapsed);
    
    // I/O optimization methods
    QByteArray readFileOptimized(const QString& filePath, qint64 size);
    QByteArray readFileMemoryMapped(const QString& filePath);
    QByteArray readFileWithReadAhead(const QString& filePath, qint64 size);
    QFuture<QByteArray> readFileAsync(const QString& filePath, qint64 size);
    QByteArray getBufferFromPool(qint64 size);
    void returnBufferToPool(const QByteArray& buffer);
    void initializeBufferPool();
    void cleanupIOOptimizations();
    
    // Enhanced progress reporting methods (HC-002d)
    void updateThroughputMonitor(double throughput);
    void updatePerformanceHistogram(const QString& metricName, double value);
    void updateETAPrediction(qint64 processedBytes, qint64 totalBytes);
    void calculatePerformanceTrend();
    void updateResourceUtilization();
    double estimateCPUUtilization() const;
    qint64 getCurrentMemoryUsageInternal() const;
    double calculateThroughputVariance() const;
    void analyzeQueueEfficiency();
    void updateETAAccuracy(qint64 actualTime, qint64 predictedTime);
    ProgressInfo createDetailedProgressInfo() const;
    
    // Cache implementation
    class HashCache {
    public:
        struct CacheEntry {
            QString filePath;
            QString hash;
            QDateTime fileModified;
            QDateTime cached;
            qint64 fileSize;
            
            CacheEntry() : fileSize(0) {}
            CacheEntry(const QString& path, const QString& h, const QDateTime& modified, qint64 size)
                : filePath(path), hash(h), fileModified(modified), cached(QDateTime::currentDateTime()), fileSize(size) {}
        };
        
        HashCache(int maxSize = 10000);
        ~HashCache() = default;
        
        bool hasValidHash(const QString& filePath, const QDateTime& lastModified, qint64 fileSize);
        QString getHash(const QString& filePath);
        void putHash(const QString& filePath, const QString& hash, const QDateTime& lastModified, qint64 fileSize);
        void clear();
        int size() const;
        double hitRate() const;
        void resetStatistics();
        
    private:
        void evictLRU();
        void updateLRU(const QString& filePath);
        
        QHash<QString, CacheEntry> m_cache;
        QList<QString> m_lruOrder;
        int m_maxSize;
        mutable QMutex m_mutex;
        
        // Statistics
        int m_hits = 0;
        int m_misses = 0;
    };
    
    // Forward declarations for advanced thread management
    class HashTask;
    class WorkStealingThreadPool;
    class TaskBatch;
    
    // Member variables
    HashOptions m_options;
    HashCache* m_cache;
    
    // Legacy concurrent support (for compatibility)
    QHash<QString, QFuture<void>> m_activeJobs;
    QMutex m_jobsMutex;
    
    // Advanced thread pool management
    WorkStealingThreadPool* m_threadPool;
    QAtomicInt m_nextTaskId;
    QHash<int, HashTask*> m_activeTasks;
    QMutex m_tasksMutex;
    
    // Statistics and monitoring
    Statistics m_statistics;
    mutable QMutex m_statsMutex;
    QElapsedTimer m_performanceTimer;
    
    // Control flags
    bool m_cancelAllRequested = false;
    QAtomicInt m_activeTaskCount;
    
    // Batch processing management
    QHash<int, BatchInfo> m_activeBatches;
    QMutex m_batchesMutex;
    QAtomicInt m_nextBatchId;
    QSemaphore m_batchSemaphore;  // Limit concurrent batches
    
    // Adaptive chunk sizing
    AdaptiveChunkConfig m_chunkConfig;
    QMutex m_chunkConfigMutex;
    
    // I/O optimization
    IOOptimizationConfig m_ioConfig;
    QMutex m_ioMutex;
    QMap<QString, QSharedPointer<QFile>> m_memoryMappedFiles;
    QQueue<QByteArray> m_bufferPool;
    int m_maxBufferPoolSize;
    QAtomicInt m_activeAsyncOperations;
    QSemaphore m_asyncIOSemaphore;
    QThreadPool m_ioThreadPool;
    
    // Enhanced progress reporting and monitoring (HC-002d)
    ThroughputMonitor m_throughputMonitor;
    PerformanceHistogram m_performanceHistogram;
    ETAPredictionEngine m_etaPredictionEngine;
    QMutex m_progressMutex;
    
    // Real-time metrics tracking
    QElapsedTimer m_sessionTimer;
    QList<double> m_recentThroughputs;
    QList<qint64> m_recentTimestamps;
    int m_maxRecentSamples = 100;
    
    // Resource monitoring
    QElapsedTimer m_resourceTimer;
    double m_lastCpuCheck = 0.0;
    qint64 m_lastMemoryCheck = 0;
    int m_resourceCheckInterval = 1000; // milliseconds
};

// Q_DECLARE_METATYPE for use with Qt's signal/slot system
Q_DECLARE_METATYPE(HashCalculator::HashResult)
Q_DECLARE_METATYPE(HashCalculator::ProgressInfo)
Q_DECLARE_METATYPE(HashCalculator::BatchInfo)
