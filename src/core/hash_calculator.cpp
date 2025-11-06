#include "hash_calculator.h"
#include "logger.h"
#include <QFile>
#include <QFileInfo>

#include <QDir>
#include <QCoreApplication>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QElapsedTimer>
#include <QThread>
#include <QByteArrayView>
#include <QRandomGenerator>
#include <QTimer>
#include <memory>
#include <vector>

// GPU acceleration includes
#ifdef HAS_CUDA
#include "gpu/cuda_hash_calculator.h"
#endif
#ifdef HAS_OPENCL
#include "gpu/opencl_hash_calculator.h"
#endif
#include "gpu/gpu_hash_calculator.h"
#include "gpu/gpu_detector.h"
#include "gpu/gpu_config.h"

// Forward declarations for advanced thread management classes
class HashCalculator::HashTask {
public:
    int taskId;
    QString filePath;
    int priority;
    QDateTime created;
    QElapsedTimer executionTimer;
    
    HashTask(int id, const QString& path, int prio = 0) 
        : taskId(id), filePath(path), priority(prio), created(QDateTime::currentDateTime()) {}
    
    bool operator<(const HashTask& other) const {
        // Higher priority first, then FIFO for same priority
        if (priority != other.priority) {
            return priority < other.priority;
        }
        return created > other.created;
    }
};

class HashCalculator::WorkStealingThreadPool {
public:
    struct WorkerThread {
        QThread* thread;
        QQueue<HashTask*> localQueue;
        std::unique_ptr<QMutex> localMutex;
        std::unique_ptr<QWaitCondition> waitCondition;
        QAtomicInt isActive;
        QElapsedTimer idleTimer;
        int tasksProcessed = 0;
        
        WorkerThread() : thread(nullptr), 
                        localMutex(std::make_unique<QMutex>()), 
                        waitCondition(std::make_unique<QWaitCondition>()), 
                        isActive(0) {}
        
        // Make non-copyable but movable
        WorkerThread(const WorkerThread&) = delete;
        WorkerThread& operator=(const WorkerThread&) = delete;
        
        WorkerThread(WorkerThread&& other) noexcept
            : thread(other.thread)
            , localQueue(std::move(other.localQueue))
            , localMutex(std::move(other.localMutex))
            , waitCondition(std::move(other.waitCondition))
            , isActive(other.isActive.loadAcquire())
            , idleTimer(std::move(other.idleTimer))
            , tasksProcessed(other.tasksProcessed)
        {
            other.thread = nullptr;
            other.tasksProcessed = 0;
        }
        
        WorkerThread& operator=(WorkerThread&& other) noexcept
        {
            if (this != &other) {
                thread = other.thread;
                localQueue = std::move(other.localQueue);
                localMutex = std::move(other.localMutex);
                waitCondition = std::move(other.waitCondition);
                isActive.storeRelease(other.isActive.loadAcquire());
                idleTimer = std::move(other.idleTimer);
                tasksProcessed = other.tasksProcessed;
                
                other.thread = nullptr;
                other.tasksProcessed = 0;
            }
            return *this;
        }
    };
    
    WorkStealingThreadPool(HashCalculator* parent, const HashCalculator::HashOptions& options) 
        : m_parent(parent), m_options(options), m_shutdown(false) {
        
        // Initialize workers vector with move construction
        m_workers.reserve(m_options.threadPoolSize);
        for (int i = 0; i < m_options.threadPoolSize; ++i) {
            m_workers.emplace_back();
        }
        m_activeWorkers = 0;
        
        // Initialize worker threads
        for (int i = 0; i < m_options.threadPoolSize; ++i) {
            initializeWorker(i);
        }
        
        // Start monitoring timer if dynamic threads enabled
        if (m_options.enableDynamicThreads) {
            m_monitorTimer = new QTimer();
            QObject::connect(m_monitorTimer, &QTimer::timeout, [this]() { monitorAndAdjust(); });
            m_monitorTimer->start(5000); // Monitor every 5 seconds
        }
    }
    
    ~WorkStealingThreadPool() {
        shutdown();
    }
    
    void submitTask(HashTask* task) {
        if (m_shutdown) {
            delete task;
            return;
        }
        
        // Find worker with smallest queue or round-robin
        int targetWorker = findBestWorker();
        
        QMutexLocker locker(m_workers[targetWorker].localMutex.get());
        m_workers[targetWorker].localQueue.enqueue(task);
        m_workers[targetWorker].waitCondition->wakeOne();
        
        m_queuedTasks.fetchAndAddOrdered(1);
    }
    
    void shutdown() {
        m_shutdown = true;
        
        if (m_monitorTimer) {
            m_monitorTimer->stop();
            delete m_monitorTimer;
            m_monitorTimer = nullptr;
        }
        
        // Wake up all workers and wait for them to finish
        for (auto& worker : m_workers) {
            worker.waitCondition->wakeAll();
            if (worker.thread && worker.thread->isRunning()) {
                worker.thread->quit();
                if (!worker.thread->wait(3000)) {
                    worker.thread->terminate();
                    worker.thread->wait(1000);
                }
                delete worker.thread;
                worker.thread = nullptr;
            }
        }
    }
    
    int getActiveThreads() const {
        return m_activeWorkers.loadAcquire();
    }
    
    int getQueuedTasks() const {
        return m_queuedTasks.loadAcquire();
    }
    
private:
    HashCalculator* m_parent;
    HashCalculator::HashOptions m_options;
    std::vector<WorkerThread> m_workers;
    QAtomicInt m_activeWorkers;
    QAtomicInt m_queuedTasks;
    QAtomicInt m_shutdown;
    QTimer* m_monitorTimer = nullptr;
    
    void initializeWorker(int workerId) {
        WorkerThread& worker = m_workers[workerId];
        worker.thread = QThread::create([this, workerId]() { workerLoop(workerId); });
        worker.thread->start();
        m_activeWorkers.fetchAndAddOrdered(1);
    }
    
    void workerLoop(int workerId) {
        WorkerThread& worker = m_workers[workerId];
        
        while (!m_shutdown.loadAcquire()) {
            HashTask* task = getNextTask(workerId);
            
            if (task) {
                worker.isActive.storeRelease(1);
                task->executionTimer.start();
                
                // Execute the hash calculation
                executeHashTask(task);
                
                worker.tasksProcessed++;
                m_queuedTasks.fetchAndSubOrdered(1);
                delete task;
                
                worker.isActive.storeRelease(0);
            } else {
                // No work available, wait or check for work stealing
                QMutexLocker locker(worker.localMutex.get());
                if (worker.localQueue.isEmpty()) {
                    worker.idleTimer.start();
                    worker.waitCondition->wait(worker.localMutex.get(), 1000); // Wait up to 1 second
                }
            }
        }
    }
    
    HashTask* getNextTask(int workerId) {
        WorkerThread& worker = m_workers[workerId];
        
        // Try local queue first
        {
            QMutexLocker locker(worker.localMutex.get());
            if (!worker.localQueue.isEmpty()) {
                return worker.localQueue.dequeue();
            }
        }
        
        // Try work stealing if enabled
        if (m_options.enableWorkStealing) {
            return stealWork(workerId);
        }
        
        return nullptr;
    }
    
    HashTask* stealWork(int thiefId) {
        // Try to steal from a random worker with enough tasks
        for (int attempts = 0; attempts < static_cast<int>(m_workers.size()); ++attempts) {
            int victimId = static_cast<int>(QRandomGenerator::global()->bounded(static_cast<qint64>(m_workers.size())));
            if (victimId == thiefId) continue;
            
            WorkerThread& victim = m_workers[victimId];
            QMutexLocker locker(victim.localMutex.get());
            
            if (victim.localQueue.size() >= m_options.workStealingThreshold) {
                // Steal from the end (LIFO for cache locality)
                HashTask* stolen = victim.localQueue.takeLast();
                if (stolen) {
                    // Update statistics
                    QMutexLocker statsLocker(&m_parent->m_statsMutex);
                    m_parent->m_statistics.workStealingEvents++;
                    return stolen;
                }
            }
        }
        
        return nullptr;
    }
    
    int findBestWorker() {
        int bestWorker = 0;
        int minQueueSize = INT_MAX;
        
        for (int i = 0; i < static_cast<int>(m_workers.size()); ++i) {
            QMutexLocker locker(m_workers[i].localMutex.get());
            int queueSize = static_cast<int>(m_workers[i].localQueue.size());
            if (queueSize < minQueueSize) {
                minQueueSize = queueSize;
                bestWorker = i;
                if (queueSize == 0) break; // Found idle worker
            }
        }
        
        return bestWorker;
    }
    
    void executeHashTask(HashTask* task) {
        HashCalculator::HashResult result(task->filePath);
        
        try {
            QString hash = m_parent->calculateFileHashInternal(task->filePath, true);
            if (!hash.isEmpty()) {
                result.hash = hash;
                result.success = true;
                result.calculated = QDateTime::currentDateTime();
                
                QFileInfo info(task->filePath);
                result.fileSize = info.size();
                
                // Update statistics
                QMutexLocker statsLocker(&m_parent->m_statsMutex);
                m_parent->m_statistics.tasksCompleted++;
                m_parent->m_statistics.totalHashesCalculated++;
                m_parent->m_statistics.totalBytesProcessed += result.fileSize;
                
            } else {
                result.success = false;
                result.errorMessage = "Hash calculation failed";
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = QString("Exception: %1").arg(e.what());
        } catch (...) {
            result.success = false;
            result.errorMessage = "Unknown exception occurred";
        }
        
        // Emit result
        QMetaObject::invokeMethod(m_parent, "hashCompleted", Qt::QueuedConnection,
                                Q_ARG(HashCalculator::HashResult, result));
    }
    
    void monitorAndAdjust() {
        if (m_shutdown.loadAcquire()) return;
        
        int activeThreads = 0;
        int totalQueueSize = 0;
        
        // Count active threads and total queue size
        for (const auto& worker : m_workers) {
            if (worker.isActive.loadAcquire()) {
                activeThreads++;
            }
            QMutexLocker locker(worker.localMutex.get());
            totalQueueSize += static_cast<int>(worker.localQueue.size());
        }
        
        // Update statistics
        {
            QMutexLocker locker(&m_parent->m_statsMutex);
            m_parent->m_statistics.activeThreads = activeThreads;
            m_parent->m_statistics.peakThreads = qMax(m_parent->m_statistics.peakThreads, activeThreads);
            m_parent->m_statistics.tasksQueued = totalQueueSize;
            
            if (m_workers.size() > 0) {
                m_parent->m_statistics.threadUtilization = static_cast<double>(activeThreads) / static_cast<double>(m_workers.size()) * 100.0;
            }
        }
        
        // Dynamic thread adjustment logic
        if (totalQueueSize > m_workers.size() * 2 && m_workers.size() < m_options.maxThreads) {
            // Add more threads if queue is backing up
        int newWorkerId = static_cast<int>(m_workers.size());
            m_workers.resize(newWorkerId + 1);
            initializeWorker(newWorkerId);
            LOG_DEBUG(LogCategories::HASH, QString("Added thread, now %1 threads").arg(m_workers.size()));
            
        } else if (activeThreads < m_workers.size() / 2 && m_workers.size() > m_options.minThreads) {
            // Remove idle threads (implementation would need thread-safe removal)
            // For now, just log the opportunity
            // TODO: Could remove threads, utilization low
            // qDebug() << "HashCalculator: Could remove threads, utilization low";  // Commented out - too verbose
        }
    }
};

class HashCalculator::TaskBatch {
public:
    int batchId;
    QStringList filePaths;
    QDateTime created;
    QElapsedTimer processingTimer;
    bool isSmallFileBatch;
    
    TaskBatch(int id, const QStringList& paths, bool smallFiles = false) 
        : batchId(id), filePaths(paths), created(QDateTime::currentDateTime()), isSmallFileBatch(smallFiles) {}
};

// HashCalculator Implementation - Enhanced with Advanced Thread Pool

HashCalculator::HashCalculator(QObject* parent)
    : QObject(parent)
    , m_cache(nullptr)
    , m_threadPool(nullptr)
    , m_nextTaskId(1)
    , m_cancelAllRequested(false)
    , m_activeTaskCount(0)
    , m_nextBatchId(1)
    , m_batchSemaphore(3)  // Initialize with default value, will be updated by setOptions
{
    Logger::instance()->debug(LogCategories::HASH, "HashCalculator created");
    
    // Initialize cache
    m_cache = new HashCache(m_options.maxCacheSize);
    
    // Initialize advanced thread pool
    initializeThreadPool();
    
    // Register meta types for signal/slot system
    qRegisterMetaType<HashCalculator::HashResult>("HashCalculator::HashResult");
    qRegisterMetaType<HashCalculator::ProgressInfo>("HashCalculator::ProgressInfo");
    qRegisterMetaType<HashCalculator::BatchInfo>("HashCalculator::BatchInfo");
    
    // Start performance monitoring
    m_performanceTimer.start();
    
    // Initialize adaptive chunk configuration
    m_chunkConfig.currentChunkSize = m_options.chunkSize;
    m_chunkConfig.performanceTimer.start();
    
    // Initialize I/O optimization
    m_maxBufferPoolSize = 10;
    m_activeAsyncOperations = 0;
    m_ioThreadPool.setMaxThreadCount(qMin(4, QThread::idealThreadCount()));
    if (m_options.enableIOOptimizations) {
        initializeBufferPool();
    }
    
    // Initialize GPU acceleration
    initializeGPU();
    
    LOG_INFO(LogCategories::HASH, QString("HashCalculator initialized with advanced thread pool: %1 threads").arg(m_options.threadPoolSize));
    LOG_INFO(LogCategories::HASH, QString("Batch processing enabled with %1 concurrent batches").arg(m_options.maxConcurrentBatches));
    LOG_INFO(LogCategories::HASH, QString("I/O optimizations %1").arg(m_options.enableIOOptimizations ? "enabled" : "disabled"));
}

HashCalculator::~HashCalculator()
{
    // Cancel all pending operations
    cancelAll();
    
    // Shutdown advanced thread pool
    shutdownThreadPool();
    
    // Cleanup I/O optimizations
    if (m_ioConfig.enabled) {
        cleanupIOOptimizations();
    }
    
    // Wait for active jobs to finish (with timeout) - legacy support
    QElapsedTimer timer;
    timer.start();
    while (!m_activeJobs.isEmpty() && timer.elapsed() < 5000) {
        QCoreApplication::processEvents();
        QThread::msleep(10);
    }
    
    delete m_cache;
    LOG_DEBUG(LogCategories::HASH, "HashCalculator destroyed");
}

void HashCalculator::setOptions(const HashOptions& options)
{
    m_options = options;
    
    // Update cache size if needed
    if (m_cache && m_cache->size() > options.maxCacheSize) {
        m_cache->clear();
    }
    
    if (!options.enableCaching) {
        m_cache->clear();
    }
    
    // Update batch semaphore capacity
    // Note: QSemaphore doesn't have a direct way to change capacity
    // This would require recreating the semaphore, but for now we'll note this limitation
    if (options.maxConcurrentBatches != 3) {
        LOG_WARNING(LogCategories::HASH, "Batch concurrency setting requires restart to take effect");
    }
}

HashCalculator::HashOptions HashCalculator::getOptions() const
{
    return m_options;
}

void HashCalculator::calculateFileHash(const QString& filePath)
{
    if (filePath.isEmpty()) {
        emit hashError(filePath, "Empty file path provided");
        return;
    }
    
    // Check if already processing this file
    {
        QMutexLocker locker(&m_jobsMutex);
        if (m_activeJobs.contains(filePath)) {
            LOG_DEBUG(LogCategories::HASH, QString("File already being processed: %1").arg(filePath));
            return;
        }
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Starting hash calculation for: %1").arg(filePath));
    
    // Submit job to thread pool
    QFuture<void> future = QtConcurrent::run([this, filePath]() {
        HashResult result(filePath);
        
        try {
            // Check cache first
            QFileInfo fileInfo(filePath);
            if (!fileInfo.exists()) {
                result.success = false;
                result.errorMessage = "File does not exist";
                emit hashCompleted(result);
                return;
            }
            
            result.fileSize = fileInfo.size();
            
            // Try cache if enabled
            if (m_options.enableCaching && 
                m_cache->hasValidHash(filePath, fileInfo.lastModified(), result.fileSize)) {
                result.hash = m_cache->getHash(filePath);
                result.fromCache = true;
                result.success = true;
                result.calculated = QDateTime::currentDateTime();
                
                // Update statistics
                {
                    QMutexLocker statsLocker(&m_statsMutex);
                    m_statistics.cacheHits++;
                }
                
                LOG_DEBUG(LogCategories::HASH, QString("Cache hit for: %1").arg(filePath));
                emit hashCompleted(result);
                return;
            }
            
            // Calculate hash
            QString hash = calculateFileHashInternal(filePath, true);
            if (!hash.isEmpty()) {
                result.hash = hash;
                result.success = true;
                result.calculated = QDateTime::currentDateTime();
                result.fromCache = false;
                
                // Update statistics
                {
                    QMutexLocker statsLocker(&m_statsMutex);
                    m_statistics.totalHashesCalculated++;
                    m_statistics.totalBytesProcessed += result.fileSize;
                    m_statistics.cacheMisses++;
                }
                
                LOG_DEBUG(LogCategories::HASH, QString("Hash calculated for: %1 -> %2...").arg(filePath, hash.left(16)));
            } else {
                result.success = false;
                result.errorMessage = "Failed to calculate hash";
            }
            
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = QString("Exception: %1").arg(e.what());
        } catch (...) {
            result.success = false;
            result.errorMessage = "Unknown exception occurred";
        }
        
        emit hashCompleted(result);
    });
    
    // Store future for tracking
    {
        QMutexLocker locker(&m_jobsMutex);
        m_activeJobs[filePath] = future;
    }
    
    // Set up watcher for cleanup
    auto* watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcher<void>::finished, [this, watcher, filePath]() {
        // Remove from active jobs
        {
            QMutexLocker locker(&m_jobsMutex);
            m_activeJobs.remove(filePath);
        }
        
        // Check if all operations complete
        if (m_activeJobs.isEmpty()) {
            emit allOperationsComplete();
        }
        
        watcher->deleteLater();
    });
    
    watcher->setFuture(future);
}

void HashCalculator::calculateFileHashes(const QStringList& filePaths)
{
    Logger::instance()->info(LogCategories::HASH, QString("Starting batch hash calculation for %1 files").arg(filePaths.size()));
    
    for (const QString& filePath : filePaths) {
        calculateFileHash(filePath);
    }
}

QString HashCalculator::calculateFileHashSync(const QString& filePath)
{
    if (filePath.isEmpty()) {
        return QString();
    }
    
    // Check cache first
    QFileInfo fileInfo(filePath);
    if (!fileInfo.exists()) {
        return QString();
    }
    
    if (m_options.enableCaching && 
        m_cache->hasValidHash(filePath, fileInfo.lastModified(), fileInfo.size())) {
        return m_cache->getHash(filePath);
    }
    
    // Calculate hash synchronously
    return calculateFileHashInternal(filePath, true);
}

QString HashCalculator::calculateFileHashInternal(const QString& filePath, bool updateCache)
{
    QFileInfo fileInfo(filePath);
    if (!fileInfo.exists() || !fileInfo.isReadable()) {
        return QString();
    }
    
    qint64 fileSize = fileInfo.size();
    
    // Handle empty files
    if (fileSize == 0) {
        QCryptographicHash hasher(QCryptographicHash::Sha256);
        return formatHashResult(hasher.result());
    }
    
    // Try GPU acceleration for suitable files
    if (isGPUEnabled() && fileSize <= 100 * 1024 * 1024) { // Limit to 100MB for GPU
        QString gpuHash = calculateFileHashGPU(filePath);
        if (!gpuHash.isEmpty()) {
            // Update cache if successful and enabled
            if (updateCache && m_options.enableCaching) {
                m_cache->putHash(filePath, gpuHash, fileInfo.lastModified(), fileSize);
            }
            return gpuHash;
        }
        // Fall back to CPU if GPU fails
    }
    
    // Use progressive hashing for large files
    if (fileSize > m_options.largeFileThreshold) {
        return hashLargeFile(filePath, fileSize);
    }
    
    // Standard chunked hashing for smaller files
    QString hash = calculateChunkedHash(filePath, fileSize);
    
    // Update cache if successful and enabled
    if (!hash.isEmpty() && updateCache && m_options.enableCaching) {
        m_cache->putHash(filePath, hash, fileInfo.lastModified(), fileSize);
    }
    
    return hash;
}

QString HashCalculator::calculateChunkedHash(const QString& filePath, qint64 /* fileSize */)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "HashCalculator: Cannot open file for reading:" << filePath << file.errorString();
        return QString();
    }
    
    QCryptographicHash hasher(QCryptographicHash::Sha256);
    QByteArray buffer(static_cast<int>(m_options.chunkSize), 0);
    
    qint64 totalRead = 0;
    while (!file.atEnd() && !m_cancelAllRequested) {
        qint64 bytesRead = file.read(buffer.data(), m_options.chunkSize);
        if (bytesRead < 0) {
            qWarning() << "HashCalculator: Error reading file:" << filePath << file.errorString();
            return QString();
        }
        
        if (bytesRead > 0) {
            hasher.addData(QByteArrayView(buffer.constData(), bytesRead));
            totalRead += bytesRead;
        }
    }
    
    file.close();
    
    if (m_cancelAllRequested) {
        return QString();
    }
    
    return formatHashResult(hasher.result());
}

QString HashCalculator::hashLargeFile(const QString& filePath, qint64 fileSize)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "HashCalculator: Cannot open large file:" << filePath << file.errorString();
        return QString();
    }
    
    QCryptographicHash hasher(QCryptographicHash::Sha256);
    QByteArray buffer(static_cast<int>(m_options.chunkSize), 0);
    
    QElapsedTimer timer;
    timer.start();
    
    qint64 totalRead = 0;
    qint64 lastProgressUpdate = 0;
    
    while (!file.atEnd() && !m_cancelAllRequested) {
        qint64 bytesRead = file.read(buffer.data(), m_options.chunkSize);
        if (bytesRead < 0) {
            qWarning() << "HashCalculator: Error reading large file:" << filePath << file.errorString();
            return QString();
        }
        
        if (bytesRead > 0) {
            hasher.addData(QByteArrayView(buffer.constData(), bytesRead));
            totalRead += bytesRead;
            
            // Emit progress updates
            if (totalRead - lastProgressUpdate >= m_options.progressUpdateInterval) {
                emitProgress(filePath, totalRead, fileSize, QTime::fromMSecsSinceStartOfDay(static_cast<int>(timer.elapsed())));
                lastProgressUpdate = totalRead;
                
                // Process events to keep UI responsive
                QCoreApplication::processEvents();
            }
        }
    }
    
    file.close();
    
    if (m_cancelAllRequested) {
        return QString();
    }
    
    // Final progress update
    emitProgress(filePath, totalRead, fileSize, QTime::fromMSecsSinceStartOfDay(static_cast<int>(timer.elapsed())));
    
    return formatHashResult(hasher.result());
}

void HashCalculator::emitProgress(const QString& filePath, qint64 processed, qint64 total, const QTime& startTime)
{
    ProgressInfo progress;
    progress.filePath = filePath;
    progress.bytesProcessed = processed;
    progress.totalBytes = total;
    progress.percentComplete = total > 0 ? static_cast<int>((processed * 100) / total) : 0;
    
    // Calculate processing speed and estimated time remaining
    qint64 elapsedMs = QTime(0, 0).msecsTo(startTime);
    if (elapsedMs > 0) {
        progress.instantaneousSpeed = (static_cast<double>(processed) / 1024.0 / 1024.0) / (static_cast<double>(elapsedMs) / 1000.0); // MB/s
        
        if (processed > 0) {
            qint64 remainingBytes = total - processed;
            qint64 estimatedRemainingMs = static_cast<qint64>((static_cast<double>(remainingBytes) / static_cast<double>(processed)) * static_cast<double>(elapsedMs));
            progress.estimatedTimeRemainingMs = estimatedRemainingMs;
        }
    }
    
    emit hashProgress(progress);
}

QString HashCalculator::formatHashResult(const QByteArray& hash)
{
    return hash.toHex().toLower();
}

void HashCalculator::cancelAll()
{
    Logger::instance()->info(LogCategories::HASH, "Cancelling all hash operations");
    m_cancelAllRequested = true;
    
    // Cancel all active futures
    {
        QMutexLocker locker(&m_jobsMutex);
        for (auto it = m_activeJobs.begin(); it != m_activeJobs.end(); ++it) {
            it.value().cancel();
            emit hashCancelled(it.key());
        }
        m_activeJobs.clear();
    }
    
    emit allOperationsComplete();
    m_cancelAllRequested = false;
}

void HashCalculator::cancelFile(const QString& filePath)
{
    QMutexLocker locker(&m_jobsMutex);
    auto it = m_activeJobs.find(filePath);
    if (it != m_activeJobs.end()) {
        it.value().cancel();
        m_activeJobs.erase(it);
        emit hashCancelled(filePath);
        LOG_DEBUG(LogCategories::HASH, QString("Cancelled hash calculation for: %1").arg(filePath));
    }
}

bool HashCalculator::isProcessing() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_jobsMutex));
    return !m_activeJobs.isEmpty();
}

bool HashCalculator::isProcessingFile(const QString& filePath) const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_jobsMutex));
    return m_activeJobs.contains(filePath);
}

void HashCalculator::clearCache()
{
    if (m_cache) {
        m_cache->clear();
        LOG_DEBUG(LogCategories::HASH, "Cache cleared");
    }
}

int HashCalculator::getCacheSize() const
{
    return m_cache ? m_cache->size() : 0;
}

double HashCalculator::getCacheHitRate() const
{
    return m_cache ? m_cache->hitRate() : 0.0;
}

void HashCalculator::setCacheEnabled(bool enabled)
{
    m_options.enableCaching = enabled;
    if (!enabled && m_cache) {
        m_cache->clear();
    }
}

HashCalculator::Statistics HashCalculator::getStatistics() const
{
    QMutexLocker locker(&m_statsMutex);
    return m_statistics;
}

void HashCalculator::resetStatistics()
{
    {
        QMutexLocker locker(&m_statsMutex);
        m_statistics = Statistics();
    }
    
    if (m_cache) {
        m_cache->resetStatistics();
    }
    
    LOG_DEBUG(LogCategories::HASH, "Statistics reset");
}

void HashCalculator::onHashJobFinished()
{
    // This slot can be used for additional cleanup if needed
    // Currently handled by QFutureWatcher in calculateFileHash
}

// HashCache Implementation

HashCalculator::HashCache::HashCache(int maxSize)
    : m_maxSize(maxSize)
    , m_hits(0)
    , m_misses(0)
{
    LOG_DEBUG(LogCategories::HASH, QString("HashCache initialized with max size: %1").arg(maxSize));
}

bool HashCalculator::HashCache::hasValidHash(const QString& filePath, const QDateTime& lastModified, qint64 fileSize)
{
    QMutexLocker locker(&m_mutex);
    
    auto it = m_cache.find(filePath);
    if (it == m_cache.end()) {
        m_misses++;
        return false;
    }
    
    const CacheEntry& entry = it.value();
    
    // Check if cache entry is still valid (file not modified)
    if (entry.fileModified != lastModified || entry.fileSize != fileSize) {
        // Remove invalid entry
        m_lruOrder.removeAll(filePath);
        m_cache.erase(it);
        m_misses++;
        return false;
    }
    
    m_hits++;
    updateLRU(filePath);
    return true;
}

QString HashCalculator::HashCache::getHash(const QString& filePath)
{
    QMutexLocker locker(&m_mutex);
    auto it = m_cache.find(filePath);
    if (it != m_cache.end()) {
        updateLRU(filePath);
        return it.value().hash;
    }
    return QString();
}

void HashCalculator::HashCache::putHash(const QString& filePath, const QString& hash, const QDateTime& lastModified, qint64 fileSize)
{
    QMutexLocker locker(&m_mutex);
    
    // Remove existing entry if present
    auto it = m_cache.find(filePath);
    if (it != m_cache.end()) {
        m_lruOrder.removeAll(filePath);
        m_cache.erase(it);
    }
    
    // Check if we need to evict
    while (m_cache.size() >= m_maxSize) {
        evictLRU();
    }
    
    // Add new entry
    CacheEntry entry(filePath, hash, lastModified, fileSize);
    m_cache[filePath] = entry;
    m_lruOrder.append(filePath);
}

void HashCalculator::HashCache::clear()
{
    QMutexLocker locker(&m_mutex);
    m_cache.clear();
    m_lruOrder.clear();
    LOG_DEBUG(LogCategories::HASH, "HashCache cleared");
}

int HashCalculator::HashCache::size() const
{
    QMutexLocker locker(&m_mutex);
    return static_cast<int>(m_cache.size());
}

double HashCalculator::HashCache::hitRate() const
{
    QMutexLocker locker(&m_mutex);
    int total = m_hits + m_misses;
    return total > 0 ? static_cast<double>(m_hits) / total : 0.0;
}

void HashCalculator::HashCache::resetStatistics()
{
    QMutexLocker locker(&m_mutex);
    m_hits = 0;
    m_misses = 0;
}

void HashCalculator::HashCache::evictLRU()
{
    // Assumes mutex is already locked
    if (m_lruOrder.isEmpty()) {
        return;
    }
    
    QString oldestPath = m_lruOrder.takeFirst();
    m_cache.remove(oldestPath);
}

void HashCalculator::HashCache::updateLRU(const QString& filePath)
{
    // Assumes mutex is already locked
    m_lruOrder.removeAll(filePath);
    m_lruOrder.append(filePath);
}

// HC-002a: Advanced Thread Pool Management Implementation

void HashCalculator::initializeThreadPool()
{
    if (m_threadPool) {
        shutdownThreadPool();
    }
    
    m_threadPool = new WorkStealingThreadPool(this, m_options);
    LOG_DEBUG(LogCategories::HASH, QString("Advanced thread pool initialized with %1 worker threads").arg(m_options.threadPoolSize));
}

void HashCalculator::shutdownThreadPool()
{
    if (m_threadPool) {
        m_threadPool->shutdown();
        delete m_threadPool;
        m_threadPool = nullptr;
        LOG_DEBUG(LogCategories::HASH, "Thread pool shutdown complete");
    }
    
    // Clean up active tasks
    {
        QMutexLocker locker(&m_tasksMutex);
        for (auto it = m_activeTasks.begin(); it != m_activeTasks.end(); ++it) {
            delete it.value();
        }
        m_activeTasks.clear();
    }
}

void HashCalculator::adjustThreadPoolSize()
{
    if (!m_threadPool) return;
    
    // This could be implemented to dynamically resize the thread pool
    // For now, the WorkStealingThreadPool handles dynamic adjustment internally
    updateThreadPoolStatistics();
}

void HashCalculator::updateThreadPoolStatistics()
{
    if (!m_threadPool) return;
    
    QMutexLocker locker(&m_statsMutex);
    m_statistics.activeThreads = m_threadPool->getActiveThreads();
    m_statistics.tasksQueued = m_threadPool->getQueuedTasks();
}

void HashCalculator::calculateFileHashesBatch(const QStringList& filePaths, int priority)
{
    Logger::instance()->debug(LogCategories::HASH, QString("Processing batch of %1 files with priority %2").arg(filePaths.size()).arg(priority));
    
    if (!m_threadPool) {
        // Fallback to legacy method
        calculateFileHashes(filePaths);
        return;
    }
    
    // Create and submit tasks with priority
    for (const QString& filePath : filePaths) {
        int taskId = createHashTask(filePath, priority);
        if (taskId > 0) {
            HashTask* task;
            {
                QMutexLocker locker(&m_tasksMutex);
                task = m_activeTasks.value(taskId);
            }
            if (task) {
                m_threadPool->submitTask(task);
                m_activeTaskCount.fetchAndAddOrdered(1);
            }
        }
    }
}

void HashCalculator::calculateFileHashesOptimized(const QStringList& filePaths)
{
    Logger::instance()->debug(LogCategories::HASH, QString("Optimized processing for %1 files").arg(filePaths.size()));
    
    if (!m_options.enableBatchProcessing || !m_threadPool) {
        calculateFileHashes(filePaths);
        return;
    }
    
    // Create optimal batches based on file sizes
    QList<QStringList> batches = createOptimalBatches(filePaths);
    
    LOG_DEBUG(LogCategories::HASH, QString("Created %1 optimized batches").arg(batches.size()));
    
    // Process each batch
    int batchId = 0;
    for (const QStringList& batch : batches) {
        processBatch(batch, batchId++);
    }
}

int HashCalculator::createHashTask(const QString& filePath, int priority)
{
    if (filePath.isEmpty()) {
        return -1;
    }
    
    int taskId = m_nextTaskId.fetchAndAddOrdered(1);
    HashTask* task = new HashTask(taskId, filePath, priority);
    
    {
        QMutexLocker locker(&m_tasksMutex);
        m_activeTasks[taskId] = task;
    }
    
    return taskId;
}

void HashCalculator::onTaskCompleted(int taskId, const HashResult& result)
{
    {
        QMutexLocker locker(&m_tasksMutex);
        auto it = m_activeTasks.find(taskId);
        if (it != m_activeTasks.end()) {
            delete it.value();
            m_activeTasks.erase(it);
        }
    }
    
    m_activeTaskCount.fetchAndSubOrdered(1);
    
    // Emit completion signal
    emit hashCompleted(result);
    
    // Check if all operations complete
    if (m_activeTaskCount.loadAcquire() == 0) {
        emit allOperationsComplete();
    }
}

void HashCalculator::onTaskFailed(int taskId, const QString& error)
{
    QString filePath;
    {
        QMutexLocker locker(&m_tasksMutex);
        auto it = m_activeTasks.find(taskId);
        if (it != m_activeTasks.end()) {
            filePath = it.value()->filePath;
            delete it.value();
            m_activeTasks.erase(it);
        }
    }
    
    m_activeTaskCount.fetchAndSubOrdered(1);
    
    emit hashError(filePath, error);
}

QList<QStringList> HashCalculator::createOptimalBatches(const QStringList& filePaths)
{
    QList<QStringList> batches;
    QStringList smallFiles;
    QStringList largeFiles;
    
    // Separate files by size for optimal batching
    for (const QString& filePath : filePaths) {
        QFileInfo info(filePath);
        if (info.exists()) {
            if (info.size() < m_options.smallFileThreshold) {
                smallFiles.append(filePath);
            } else {
                largeFiles.append(filePath);
            }
        }
    }
    
    // Create larger batches for small files
    while (!smallFiles.isEmpty()) {
        int batchSize = qMin(m_options.smallFileBatchSize, static_cast<int>(smallFiles.size()));
        QStringList batch = smallFiles.mid(0, batchSize);
        smallFiles = smallFiles.mid(batchSize);
        batches.append(batch);
    }
    
    // Create smaller batches for large files
    while (!largeFiles.isEmpty()) {
        int batchSize = qMin(m_options.batchSize, static_cast<int>(largeFiles.size()));
        QStringList batch = largeFiles.mid(0, batchSize);
        largeFiles = largeFiles.mid(batchSize);
        batches.append(batch);
    }
    
    return batches;
}

void HashCalculator::processBatch(const QStringList& batch, int batchId)
{
    QElapsedTimer batchTimer;
    batchTimer.start();
    
    // Determine if this is a small file batch for optimization
    bool isSmallFileBatch = true;
    for (const QString& filePath : batch) {
        QFileInfo info(filePath);
        if (info.exists() && info.size() >= m_options.smallFileThreshold) {
            isSmallFileBatch = false;
            break;
        }
    }
    
    // Submit batch with appropriate priority
    int priority = isSmallFileBatch ? 1 : 0; // Small files get higher priority
    calculateFileHashesBatch(batch, priority);
    
    // Update batch statistics
    {
        QMutexLocker locker(&m_statsMutex);
        m_statistics.batchesProcessed++;
        if (isSmallFileBatch) {
            m_statistics.smallFileBatchesOptimized++;
        }
        
        // Update average batch time (moving average)
        double elapsedMs = static_cast<double>(batchTimer.elapsed());
        if (m_statistics.batchesProcessed == 1) {
            m_statistics.averageBatchTime = elapsedMs;
        } else {
            m_statistics.averageBatchTime = (m_statistics.averageBatchTime * 0.9) + (elapsedMs * 0.1);
        }
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Processed batch %1 with %2 files (%3) in %4ms")
              .arg(batchId)
              .arg(batch.size())
              .arg(isSmallFileBatch ? "small" : "large")
              .arg(batchTimer.elapsed()));
}

// New API methods

void HashCalculator::setThreadPoolSize(int size)
{
    if (size < 1) size = 1;
    if (size > QThread::idealThreadCount() * 4) size = QThread::idealThreadCount() * 4;
    
    m_options.threadPoolSize = size;
    
    if (m_threadPool) {
        // Reinitialize thread pool with new size
        initializeThreadPool();
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Thread pool size set to %1").arg(size));
}

int HashCalculator::getThreadPoolSize() const
{
    return m_options.threadPoolSize;
}

void HashCalculator::setDynamicThreadsEnabled(bool enabled)
{
    m_options.enableDynamicThreads = enabled;
    
    if (m_threadPool) {
        // Reinitialize to apply new settings
        initializeThreadPool();
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Dynamic threads %1").arg(enabled ? "enabled" : "disabled"));
}

bool HashCalculator::isDynamicThreadsEnabled() const
{
    return m_options.enableDynamicThreads;
}

double HashCalculator::getCurrentThroughput() const
{
    QMutexLocker locker(&m_statsMutex);
    
    if (m_performanceTimer.elapsed() == 0) {
        return 0.0;
    }
    
    double elapsedSeconds = static_cast<double>(m_performanceTimer.elapsed()) / 1000.0;
    double throughputMBps = (static_cast<double>(m_statistics.totalBytesProcessed) / 1024.0 / 1024.0) / elapsedSeconds;
    
    return throughputMBps;
}

int HashCalculator::getQueuedTaskCount() const
{
    if (m_threadPool) {
        return m_threadPool->getQueuedTasks();
    }
    return 0;
}

// HC-002b: Advanced Batch Processing Implementation

void HashCalculator::calculateFileHashesIntelligent(const QStringList& filePaths)
{
    Logger::instance()->debug(LogCategories::HASH, QString("Intelligent processing for %1 files").arg(filePaths.size()));
    
    if (!m_options.enableBatchProcessing) {
        calculateFileHashes(filePaths);
        return;
    }
    
    // Create intelligent batches based on file characteristics
    QList<QStringList> batches;
    
    if (m_options.enableSizeBasedGrouping) {
        batches = createSizeGroupedBatches(filePaths);
    } else {
        batches = createIntelligentBatches(filePaths);
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Created %1 intelligent batches").arg(batches.size()));
    
    // Process batches (parallel or sequential based on configuration)
    if (m_options.enableParallelBatches && batches.size() > 1) {
        calculateFileHashesParallel(batches);
    } else {
        // Sequential processing
        for (int i = 0; i < batches.size(); ++i) {
            processBatchWithAdaptation(batches[i], i);
        }
    }
}

void HashCalculator::calculateFileHashesParallel(const QList<QStringList>& batches)
{
    Logger::instance()->debug(LogCategories::HASH, QString("Parallel processing of %1 batches").arg(batches.size()));
    
    // Submit all batches for concurrent processing
    QList<QFuture<void>> batchFutures;
    
    for (int i = 0; i < batches.size(); ++i) {
        const QStringList& batch = batches[i];
        
        // Acquire semaphore to limit concurrent batches
        m_batchSemaphore.acquire();
        
        QFuture<void> future = QtConcurrent::run([this, batch, i]() {
            processBatchWithAdaptation(batch, i);
            m_batchSemaphore.release();
        });
        
        batchFutures.append(future);
    }
    
    // Update statistics
    {
        QMutexLocker locker(&m_statsMutex);
        m_statistics.parallelBatchesExecuted += static_cast<int>(batches.size());
    }
    
    // Wait for all batches to complete (optional - they run asynchronously)
    // for (auto& future : batchFutures) {
    //     future.waitForFinished();
    // }
}

QList<QStringList> HashCalculator::createIntelligentBatches(const QStringList& filePaths)
{
    QList<QStringList> batches;
    
    // Simple intelligent batching: group by directory for cache locality
    QHash<QString, QStringList> directoryGroups;
    
    for (const QString& filePath : filePaths) {
        QFileInfo info(filePath);
        QString directory = info.absolutePath();
        directoryGroups[directory].append(filePath);
    }
    
    // Create batches from directory groups
    for (auto it = directoryGroups.begin(); it != directoryGroups.end(); ++it) {
        const QStringList& dirFiles = it.value();
        
        // Split large directory groups into multiple batches
        int batchSize = m_options.batchSize;
        for (int i = 0; i < dirFiles.size(); i += batchSize) {
            QStringList batch = dirFiles.mid(i, batchSize);
            if (!batch.isEmpty()) {
                batches.append(batch);
            }
        }
    }
    
    return batches;
}

QList<QStringList> HashCalculator::createSizeGroupedBatches(const QStringList& filePaths)
{
    LOG_DEBUG(LogCategories::HASH, QString("Creating size-grouped batches for %1 files").arg(filePaths.size()));
    
    QStringList smallFiles;
    QStringList mediumFiles;
    QStringList largeFiles;
    QStringList hugeFiles;
    
    // Categorize files by size
    for (const QString& filePath : filePaths) {
        QFileInfo info(filePath);
        if (!info.exists()) continue;
        
        qint64 size = info.size();
        if (size < m_options.smallFileThreshold) {
            smallFiles.append(filePath);
        } else if (size < m_options.mediumFileThreshold) {
            mediumFiles.append(filePath);
        } else if (size < m_options.largeFileThreshold) {
            largeFiles.append(filePath);
        } else {
            hugeFiles.append(filePath);
        }
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("File categorization - Small: %1, Medium: %2, Large: %3, Huge: %4")
              .arg(smallFiles.size())
              .arg(mediumFiles.size())
              .arg(largeFiles.size())
              .arg(hugeFiles.size()));
    
    QList<QStringList> batches;
    
    // Create batches for each category with appropriate sizes
    auto createBatchesForCategory = [](const QStringList& files, int batchSize) {
        QList<QStringList> categoryBatches;
        for (int i = 0; i < files.size(); i += batchSize) {
            QStringList batch = files.mid(i, batchSize);
            if (!batch.isEmpty()) {
                categoryBatches.append(batch);
            }
        }
        return categoryBatches;
    };
    
    // Small files: larger batches for efficiency
    batches.append(createBatchesForCategory(smallFiles, m_options.smallFileBatchSize));
    
    // Medium files: moderate batch sizes
    batches.append(createBatchesForCategory(mediumFiles, m_options.mediumFileBatchSize));
    
    // Large files: smaller batches to avoid memory issues
    batches.append(createBatchesForCategory(largeFiles, m_options.largeFileBatchSize));
    
    // Huge files: process individually
    for (const QString& hugeFile : hugeFiles) {
        batches.append(QStringList() << hugeFile);
    }
    
    // Flatten the list
    QList<QStringList> flatBatches;
    for (const auto& batchGroup : batches) {
        flatBatches.append(batchGroup);
    }

    return flatBatches;
}

// GPU-related stub implementations
void HashCalculator::initializeGPU() {
#ifdef HAS_CUDA
    try {
        m_cudaCalculator = std::make_unique<GPU::CUDAHashCalculator>();
        if (m_cudaCalculator->initialize()) {
            m_gpuAvailable = true;
            LOG_INFO(LogCategories::HASH, "CUDA GPU acceleration initialized successfully");
        } else {
            m_gpuAvailable = false;
            LOG_WARNING(LogCategories::HASH, "CUDA GPU acceleration failed to initialize");
        }
    } catch (const std::exception& e) {
        m_gpuAvailable = false;
        LOG_WARNING(LogCategories::HASH, QString("CUDA GPU acceleration initialization error: %1").arg(e.what()));
    }
#else
    m_gpuAvailable = false;
    LOG_INFO(LogCategories::HASH, "CUDA support not compiled in");
#endif
}

QString HashCalculator::calculateFileHashGPU(const QString& filePath) {
#ifdef HAS_CUDA
    if (!m_gpuAvailable || !m_cudaCalculator) {
        return QString();
    }

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return QString();
    }

    QByteArray fileData = file.readAll();
    file.close();

    if (fileData.isEmpty()) {
        return QString();
    }

    try {
        // Convert to vector<uint8_t>
        std::vector<uint8_t> inputData(fileData.begin(), fileData.end());

        // Compute hash on GPU
        auto hashResult = m_cudaCalculator->computeHash(inputData);

        // Convert back to hex string
        QString hashString;
        for (uint8_t byte : hashResult) {
            hashString += QString("%1").arg(byte, 2, 16, QChar('0'));
        }

        return hashString;
    } catch (const std::exception& e) {
        LOG_WARNING(LogCategories::HASH, QString("GPU hash calculation failed for %1: %2").arg(filePath, e.what()));
        return QString();
    }
#else
    Q_UNUSED(filePath);
    return QString();
#endif
}

bool HashCalculator::isGPUEnabled() const {
    return m_gpuAvailable;
}

// Buffer pool stub implementations
void HashCalculator::initializeBufferPool() {
    // TODO: Implement buffer pool initialization
}

void HashCalculator::cleanupIOOptimizations() {
    // TODO: Implement IO optimizations cleanup
}

// Batch processing stub implementation
void HashCalculator::processBatchWithAdaptation(const QStringList& batch, int threadId) {
    // TODO: Implement adaptive batch processing
    Q_UNUSED(batch);
    Q_UNUSED(threadId);
}

