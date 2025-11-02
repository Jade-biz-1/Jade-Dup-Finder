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

void HashCalculator::processBatchWithAdaptation(const QStringList& batch, int /* batchId */)
{
    QElapsedTimer batchTimer;
    batchTimer.start();
    
    // Create and initialize batch info
    BatchInfo batchInfo;
    batchInfo.batchId = m_nextBatchId.fetchAndAddOrdered(1);
    batchInfo.filePaths = batch;
    batchInfo.priority = 0;
    batchInfo.created = QDateTime::currentDateTime();
    batchInfo.started = QDateTime::currentDateTime();
    
    // Calculate batch characteristics
    qint64 totalSize = 0;
    int smallFileCount = 0;
    
    for (const QString& filePath : batch) {
        QFileInfo info(filePath);
        if (info.exists()) {
            totalSize += info.size();
            if (info.size() < m_options.smallFileThreshold) {
                smallFileCount++;
            }
        }
    }
    
    batchInfo.totalSize = totalSize;
    batchInfo.isSmallFileBatch = (smallFileCount > batch.size() / 2);
    
    // Store active batch info
    {
        QMutexLocker locker(&m_batchesMutex);
        m_activeBatches[batchInfo.batchId] = batchInfo;
    }
    
    emit batchStarted(batchInfo);
    
    LOG_DEBUG(LogCategories::HASH, QString("Starting batch %1 with %2 files (%3)")
              .arg(batchInfo.batchId)
              .arg(batch.size())
              .arg(batchInfo.isSmallFileBatch ? "small" : "mixed/large"));
    
    // Process each file in the batch with adaptive chunk sizing
    QElapsedTimer adaptiveTimer;
    adaptiveTimer.start();
    qint64 bytesProcessed = 0;
    
    for (const QString& filePath : batch) {
        if (m_cancelAllRequested) break;
        
        QFileInfo info(filePath);
        if (!info.exists()) continue;
        
        // Use adaptive chunk sizing for this file
        QString hash;
        if (m_chunkConfig.enableAdaptation && info.size() > m_options.largeFileThreshold) {
            hash = hashFileWithAdaptiveChunking(filePath);
        } else {
            hash = calculateFileHashInternal(filePath, true);
        }
        
        if (!hash.isEmpty()) {
            HashResult result(filePath);
            result.hash = hash;
            result.success = true;
            result.calculated = QDateTime::currentDateTime();
            result.fileSize = info.size();
            result.fromCache = false;
            
            bytesProcessed += info.size();
            emit hashCompleted(result);
        }
    }
    
    // Update adaptive configuration based on batch performance
    double batchTime = static_cast<double>(batchTimer.elapsed());
    if (batchTime > 0) {
        double throughput = (static_cast<double>(bytesProcessed) / 1024.0 / 1024.0) / (batchTime / 1000.0);
        updateAdaptiveConfig(throughput, bytesProcessed, static_cast<qint64>(batchTime));
    }
    
    // Finalize batch info
    batchInfo.completed = QDateTime::currentDateTime();
    batchInfo.processingTime = batchTime;
    batchInfo.averageFileSpeed = static_cast<double>(batch.size()) / (batchTime / 1000.0);
    
    // Update statistics
    {
        QMutexLocker locker(&m_statsMutex);
        m_statistics.batchesProcessed++;
        m_statistics.totalBatchSizeProcessed += totalSize;
        
        if (batchInfo.isSmallFileBatch) {
            m_statistics.smallFileBatchesOptimized++;
        }
        
        // Update average batch time (exponential moving average)
        if (m_statistics.batchesProcessed == 1) {
            m_statistics.averageBatchTime = batchTime;
            m_statistics.averageFilesPerBatch = static_cast<double>(batch.size());
        } else {
            m_statistics.averageBatchTime = (m_statistics.averageBatchTime * 0.8) + (batchTime * 0.2);
            m_statistics.averageFilesPerBatch = (m_statistics.averageFilesPerBatch * 0.8) + (batch.size() * 0.2);
        }
        
        // Calculate batch throughput
        double elapsedSeconds = m_performanceTimer.elapsed() / 1000.0;
        if (elapsedSeconds > 0) {
            m_statistics.batchThroughput = m_statistics.batchesProcessed / elapsedSeconds;
        }
    }
    
    // Remove from active batches and emit completion
    {
        QMutexLocker locker(&m_batchesMutex);
        m_activeBatches.remove(batchInfo.batchId);
    }
    
    emit batchCompleted(batchInfo);
    
    LOG_DEBUG(LogCategories::HASH, QString("Completed batch %1 in %2ms (%3 files/sec)")
              .arg(batchInfo.batchId)
              .arg(batchTime)
              .arg(QString::number(batchInfo.averageFileSpeed, 'f', 2)));
}

QString HashCalculator::hashFileWithAdaptiveChunking(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "HashCalculator: Cannot open file for adaptive hashing:" << filePath << file.errorString();
        return QString();
    }
    
    QFileInfo info(filePath);
    qint64 fileSize = info.size();
    
    QCryptographicHash hasher(QCryptographicHash::Sha256);
    
    // Use current adaptive chunk size
    qint64 chunkSize;
    {
        QMutexLocker locker(&m_chunkConfigMutex);
        chunkSize = m_chunkConfig.currentChunkSize;
    }
    
    QByteArray buffer(static_cast<int>(chunkSize), 0);
    QElapsedTimer timer;
    timer.start();
    
    qint64 totalRead = 0;
    qint64 lastProgressUpdate = 0;
    
    while (!file.atEnd() && !m_cancelAllRequested) {
        qint64 bytesRead = file.read(buffer.data(), chunkSize);
        if (bytesRead < 0) {
            qWarning() << "HashCalculator: Error reading file with adaptive chunking:" << filePath << file.errorString();
            return QString();
        }
        
        if (bytesRead > 0) {
            hasher.addData(QByteArrayView(buffer.constData(), bytesRead));
            totalRead += bytesRead;
            
            // Emit progress updates for large files
            if (totalRead - lastProgressUpdate >= m_options.progressUpdateInterval) {
                emitProgress(filePath, totalRead, fileSize, QTime::fromMSecsSinceStartOfDay(static_cast<int>(timer.elapsed())));
                lastProgressUpdate = totalRead;
                
                QCoreApplication::processEvents();
            }
        }
    }
    
    file.close();
    
    if (m_cancelAllRequested) {
        return QString();
    }
    
    // Update adaptive configuration based on this file's performance
    double elapsedTime = timer.elapsed();
    if (elapsedTime > 0) {
        double throughput = (static_cast<double>(totalRead) / 1024.0 / 1024.0) / (elapsedTime / 1000.0);
        adaptChunkSize(throughput);
    }
    
    return formatHashResult(hasher.result());
}

void HashCalculator::adaptChunkSize(double currentThroughput)
{
    QMutexLocker locker(&m_chunkConfigMutex);
    
    if (!m_chunkConfig.enableAdaptation) {
        return;
    }
    
    m_chunkConfig.adaptationSamples++;
    
    // Update recent throughput (exponential moving average)
    if (m_chunkConfig.adaptationSamples == 1) {
        m_chunkConfig.recentThroughput = currentThroughput;
    } else {
        double alpha = m_chunkConfig.adaptationRate;
        m_chunkConfig.recentThroughput = (1.0 - alpha) * m_chunkConfig.recentThroughput + alpha * currentThroughput;
    }
    
    // Only adapt after we have some samples
    if (m_chunkConfig.adaptationSamples < 3) {
        return;
    }
    
    qint64 oldChunkSize = m_chunkConfig.currentChunkSize;
    qint64 newChunkSize = oldChunkSize;
    
    // Adapt chunk size based on throughput vs target
    double throughputRatio = m_chunkConfig.recentThroughput / m_chunkConfig.targetThroughput;
    
    if (throughputRatio < 0.8) {
        // Throughput too low, try smaller chunks for better responsiveness
        newChunkSize = qMax(m_chunkConfig.minChunkSize, static_cast<qint64>(oldChunkSize * 0.8));
    } else if (throughputRatio > 1.2) {
        // Throughput high, try larger chunks for better efficiency
        newChunkSize = qMin(m_chunkConfig.maxChunkSize, static_cast<qint64>(oldChunkSize * 1.2));
    }
    
    if (newChunkSize != oldChunkSize) {
        m_chunkConfig.currentChunkSize = newChunkSize;
        
        double throughputGain = ((m_chunkConfig.recentThroughput - m_chunkConfig.targetThroughput) / m_chunkConfig.targetThroughput) * 100.0;
        
        // Update statistics
        QMutexLocker statsLocker(&m_statsMutex);
        m_statistics.chunkSizeAdaptations++;
        m_statistics.optimalChunkSize = newChunkSize;
        m_statistics.adaptiveThroughputGain = throughputGain;
        statsLocker.unlock();
        
        emit chunkSizeAdapted(oldChunkSize, newChunkSize, throughputGain);
        
        LOG_DEBUG(LogCategories::HASH, QString("Adapted chunk size from %1 to %2 (%3% gain)")
                  .arg(oldChunkSize)
                  .arg(newChunkSize)
                  .arg(QString::number(throughputGain, 'f', 1)));
    }
}

qint64 HashCalculator::calculateOptimalChunkSize(qint64 fileSize, double /* targetThroughput */)
{
    // Calculate optimal chunk size based on file size and target throughput
    // This is a heuristic that can be improved with more sophisticated algorithms
    
    qint64 optimalSize = m_chunkConfig.baseChunkSize;
    
    if (fileSize < 1024 * 1024) {
        // Small files: use smaller chunks
        optimalSize = m_chunkConfig.minChunkSize;
    } else if (fileSize > 100 * 1024 * 1024) {
        // Large files: use larger chunks for efficiency
        optimalSize = qMin(m_chunkConfig.maxChunkSize, fileSize / 100);
    } else {
        // Medium files: scale chunk size with file size
        double scaleFactor = static_cast<double>(fileSize) / (10 * 1024 * 1024);
        optimalSize = static_cast<qint64>(m_chunkConfig.baseChunkSize * scaleFactor);
    }
    
    return qBound(m_chunkConfig.minChunkSize, optimalSize, m_chunkConfig.maxChunkSize);
}

void HashCalculator::updateAdaptiveConfig(double throughput, qint64 bytesProcessed, qint64 timeElapsed)
{
    Q_UNUSED(bytesProcessed)
    Q_UNUSED(timeElapsed)
    
    QMutexLocker locker(&m_chunkConfigMutex);
    
    // Update the adaptive configuration with performance data
    if (m_chunkConfig.enableAdaptation) {
        adaptChunkSize(throughput);
    }
    
    // Could add more sophisticated adaptation logic here
    // such as learning from historical performance patterns
}

// Batch management and monitoring methods

QList<HashCalculator::BatchInfo> HashCalculator::getActiveBatches() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_batchesMutex));
    return m_activeBatches.values();
}

int HashCalculator::getActiveBatchCount() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_batchesMutex));
    return static_cast<int>(m_activeBatches.size());
}

void HashCalculator::cancelBatch(int batchId)
{
    QMutexLocker locker(&m_batchesMutex);
    auto it = m_activeBatches.find(batchId);
    if (it != m_activeBatches.end()) {
        // Mark for cancellation - actual cancellation handled by worker threads checking m_cancelAllRequested
        LOG_DEBUG(LogCategories::HASH, QString("Batch %1 marked for cancellation").arg(batchId));
        m_activeBatches.erase(it);
    }
}

HashCalculator::BatchInfo HashCalculator::getBatchInfo(int batchId) const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_batchesMutex));
    return m_activeBatches.value(batchId, BatchInfo());
}

// Adaptive chunk sizing public methods

void HashCalculator::setAdaptiveChunkSizing(bool enabled)
{
    QMutexLocker locker(&m_chunkConfigMutex);
    m_chunkConfig.enableAdaptation = enabled;
    
    if (enabled) {
        m_chunkConfig.performanceTimer.restart();
        m_chunkConfig.adaptationSamples = 0;
        LOG_DEBUG(LogCategories::HASH, "Adaptive chunk sizing enabled");
    } else {
        m_chunkConfig.currentChunkSize = m_chunkConfig.baseChunkSize;
        LOG_DEBUG(LogCategories::HASH, "Adaptive chunk sizing disabled, reset to base size");
    }
}

bool HashCalculator::isAdaptiveChunkSizingEnabled() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_chunkConfigMutex));
    return m_chunkConfig.enableAdaptation;
}

qint64 HashCalculator::getOptimalChunkSize() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_chunkConfigMutex));
    return m_chunkConfig.currentChunkSize;
}

void HashCalculator::resetChunkSizeAdaptation()
{
    QMutexLocker locker(&m_chunkConfigMutex);
    m_chunkConfig.currentChunkSize = m_chunkConfig.baseChunkSize;
    m_chunkConfig.recentThroughput = 0.0;
    m_chunkConfig.adaptationSamples = 0;
    m_chunkConfig.performanceTimer.restart();
    
    LOG_DEBUG(LogCategories::HASH, QString("Chunk size adaptation reset to base size %1").arg(m_chunkConfig.baseChunkSize));
}

// I/O Optimization Implementation (HC-002c)

void HashCalculator::setIOOptimizations(bool enabled)
{
    QMutexLocker locker(&m_ioMutex);
    m_ioConfig.enabled = enabled;
    
    if (enabled) {
        initializeBufferPool();
        LOG_DEBUG(LogCategories::HASH, "I/O optimizations enabled");
    } else {
        cleanupIOOptimizations();
        LOG_DEBUG(LogCategories::HASH, "I/O optimizations disabled");
    }
}

bool HashCalculator::isIOOptimizationsEnabled() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_ioMutex));
    return m_ioConfig.enabled;
}

void HashCalculator::setMemoryMappingEnabled(bool enabled)
{
    QMutexLocker locker(&m_ioMutex);
    m_ioConfig.memoryMappingEnabled = enabled;
    
    if (!enabled) {
        // Clean up existing memory mapped files
        m_memoryMappedFiles.clear();
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Memory mapping %1").arg(enabled ? "enabled" : "disabled"));
}

void HashCalculator::setReadAheadEnabled(bool enabled)
{
    QMutexLocker locker(&m_ioMutex);
    m_ioConfig.readAheadEnabled = enabled;
    LOG_DEBUG(LogCategories::HASH, QString("Read-ahead %1").arg(enabled ? "enabled" : "disabled"));
}

void HashCalculator::setAsyncIOEnabled(bool enabled)
{
    QMutexLocker locker(&m_ioMutex);
    m_ioConfig.asyncIOEnabled = enabled;
    
    if (enabled) {
        m_ioThreadPool.setMaxThreadCount(qMin(4, QThread::idealThreadCount()));
    } else {
        m_ioThreadPool.waitForDone(3000);
    }
    
    LOG_DEBUG(LogCategories::HASH, QString("Async I/O %1").arg(enabled ? "enabled" : "disabled"));
}

HashCalculator::IOOptimizationConfig HashCalculator::getIOConfig() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_ioMutex));
    return m_ioConfig;
}

QByteArray HashCalculator::readFileOptimized(const QString& filePath, qint64 size)
{
    QMutexLocker locker(&m_ioMutex);
    
    if (!m_ioConfig.enabled) {
        // Fallback to standard file reading
        locker.unlock();
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly)) {
            return QByteArray();
        }
        return file.readAll();
    }
    
    // Choose optimal reading strategy based on file size and configuration
    if (m_ioConfig.memoryMappingEnabled && size >= m_ioConfig.memoryMappingThreshold) {
        locker.unlock();
        return readFileMemoryMapped(filePath);
    } else if (m_ioConfig.asyncIOEnabled && size >= m_ioConfig.asyncIOThreshold) {
        locker.unlock();
        QFuture<QByteArray> future = readFileAsync(filePath, size);
        return future.result();
    } else if (m_ioConfig.readAheadEnabled) {
        locker.unlock();
        return readFileWithReadAhead(filePath, size);
    } else {
        locker.unlock();
        // Standard buffered reading
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly)) {
            return QByteArray();
        }
        return file.readAll();
    }
}

QByteArray HashCalculator::readFileMemoryMapped(const QString& filePath)
{
    QElapsedTimer timer;
    timer.start();
    
    QMutexLocker locker(&m_ioMutex);
    
    // Check if file is already memory mapped
    QSharedPointer<QFile> mappedFile = m_memoryMappedFiles.value(filePath);
    if (!mappedFile) {
        mappedFile = QSharedPointer<QFile>(new QFile(filePath));
        if (!mappedFile->open(QIODevice::ReadOnly)) {
            QMutexLocker statsLocker(&m_statsMutex);
            m_statistics.ioErrors++;
            return QByteArray();
        }
        m_memoryMappedFiles[filePath] = mappedFile;
    }
    
    qint64 fileSize = mappedFile->size();
    uchar* mappedData = mappedFile->map(0, fileSize);
    
    if (!mappedData) {
        // Fallback to regular reading
        mappedFile->seek(0);
        QByteArray data = mappedFile->readAll();
        
        QMutexLocker statsLocker(&m_statsMutex);
        m_statistics.memoryMappingFallbacks++;
        m_statistics.totalIOTime += timer.elapsed();
        
        return data;
    }
    
    // Copy mapped data to QByteArray
    QByteArray data(reinterpret_cast<const char*>(mappedData), fileSize);
    mappedFile->unmap(mappedData);
    
    // Update statistics
    {
        QMutexLocker statsLocker(&m_statsMutex);
        m_statistics.memoryMappedReads++;
        m_statistics.memoryMappedBytes += fileSize;
        m_statistics.totalIOTime += timer.elapsed();
        m_statistics.averageIOSpeed = m_statistics.memoryMappedBytes / qMax(1.0, m_statistics.totalIOTime / 1000.0);
    }
    
    return data;
}

QByteArray HashCalculator::readFileWithReadAhead(const QString& filePath, qint64 size)
{
    QElapsedTimer timer;
    timer.start();
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMutexLocker statsLocker(&m_statsMutex);
        m_statistics.ioErrors++;
        return QByteArray();
    }
    
    // Use optimized buffer size for read-ahead
    qint64 bufferSize = m_ioConfig.readAheadBufferSize;
    if (size < bufferSize) {
        bufferSize = size;
    }
    
    QByteArray buffer = getBufferFromPool(bufferSize);
    QByteArray data;
    data.reserve(size);
    
    qint64 totalRead = 0;
    while (totalRead < size && !file.atEnd()) {
        qint64 toRead = qMin(bufferSize, size - totalRead);
        qint64 bytesRead = file.read(buffer.data(), toRead);
        
        if (bytesRead <= 0) {
            break;
        }
        
        data.append(buffer.constData(), bytesRead);
        totalRead += bytesRead;
        
        // Simulate read-ahead by reading slightly more than requested
        if (totalRead < size && file.bytesAvailable() > 0) {
            qint64 readAheadSize = qMin(static_cast<qint64>(bufferSize * 0.1), file.bytesAvailable());
            if (readAheadSize > 0) {
                file.read(readAheadSize); // Discard read-ahead data for simulation
            }
        }
    }
    
    returnBufferToPool(buffer);
    
    // Update statistics
    {
        QMutexLocker statsLocker(&m_statsMutex);
        m_statistics.readAheadOperations++;
        m_statistics.readAheadBytes += totalRead;
        m_statistics.totalIOTime += timer.elapsed();
        m_statistics.bufferHits += (buffer.size() > 0 ? 1 : 0);
    }
    
    return data;
}

QFuture<QByteArray> HashCalculator::readFileAsync(const QString& filePath, qint64 size)
{
    return QtConcurrent::run(&m_ioThreadPool, [this, filePath, size]() -> QByteArray {
        QElapsedTimer timer;
        timer.start();
        
        m_activeAsyncOperations.fetchAndAddOrdered(1);
        
        // Acquire semaphore to limit concurrent async operations
        m_asyncIOSemaphore.acquire();
        
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly)) {
            m_asyncIOSemaphore.release();
            m_activeAsyncOperations.fetchAndSubOrdered(1);
            
            QMutexLocker statsLocker(&m_statsMutex);
            m_statistics.ioErrors++;
            return QByteArray();
        }
        
        // Use buffer pool for async operations
        QByteArray buffer = getBufferFromPool(size);
        QByteArray data = file.readAll();
        returnBufferToPool(buffer);
        
        m_asyncIOSemaphore.release();
        m_activeAsyncOperations.fetchAndSubOrdered(1);
        
        // Update statistics
        {
            QMutexLocker statsLocker(&m_statsMutex);
            m_statistics.asyncIOOperations++;
            m_statistics.asyncIOBytes += data.size();
            m_statistics.totalIOTime += timer.elapsed();
        }
        
        return data;
    });
}

QByteArray HashCalculator::getBufferFromPool(qint64 size)
{
    QMutexLocker locker(&m_ioMutex);
    
    // Try to reuse a buffer from the pool
    for (auto it = m_bufferPool.begin(); it != m_bufferPool.end(); ++it) {
        if (it->size() >= size) {
            QByteArray buffer = *it;
            m_bufferPool.erase(it);
            
            QMutexLocker statsLocker(&m_statsMutex);
            m_statistics.bufferHits++;
            
            return buffer;
        }
    }
    
    // Create new buffer if none available
    QMutexLocker statsLocker(&m_statsMutex);
    m_statistics.bufferMisses++;
    
    return QByteArray(size, 0);
}

void HashCalculator::returnBufferToPool(const QByteArray& buffer)
{
    QMutexLocker locker(&m_ioMutex);
    
    // Only keep buffers in pool if not at capacity
    if (m_bufferPool.size() < m_maxBufferPoolSize && buffer.size() >= 1024) {
        m_bufferPool.enqueue(buffer);
    }
    
    // Keep pool size reasonable
    while (m_bufferPool.size() > m_maxBufferPoolSize) {
        m_bufferPool.dequeue();
    }
}

void HashCalculator::initializeBufferPool()
{
    QMutexLocker locker(&m_ioMutex);
    
    m_maxBufferPoolSize = 10;
    m_bufferPool.clear();
    
    // Pre-allocate some common buffer sizes
    QList<qint64> commonSizes = { 4096, 8192, 16384, 32768, 65536 };
    for (qint64 size : commonSizes) {
        if (m_bufferPool.size() < m_maxBufferPoolSize) {
            m_bufferPool.enqueue(QByteArray(size, 0));
        }
    }
    
    // Initialize semaphore for async operations
    while (m_asyncIOSemaphore.available() > 0) {
        m_asyncIOSemaphore.tryAcquire(m_asyncIOSemaphore.available());
    }
    
    int maxAsyncOps = qMin(8, QThread::idealThreadCount());
    m_asyncIOSemaphore.release(maxAsyncOps);
    
    LOG_DEBUG(LogCategories::HASH, QString("Buffer pool initialized with %1 buffers").arg(m_bufferPool.size()));
    LOG_DEBUG(LogCategories::HASH, QString("Async I/O semaphore initialized with %1 permits").arg(maxAsyncOps));
}

void HashCalculator::cleanupIOOptimizations()
{
    QMutexLocker locker(&m_ioMutex);
    
    // Clear memory mapped files
    m_memoryMappedFiles.clear();
    
    // Clear buffer pool
    m_bufferPool.clear();
    
    // Wait for async operations to complete
    m_ioThreadPool.waitForDone(3000);
    
    LOG_DEBUG(LogCategories::HASH, "I/O optimizations cleaned up");
}

// Enhanced Progress Reporting Implementation (HC-002d)

void HashCalculator::ThroughputMonitor::updateMetrics() {
    if (dataPoints.isEmpty()) {
        currentThroughput = 0.0;
        averageThroughput = 0.0;
        return;
    }
    
    // Current throughput is the latest data point
    currentThroughput = dataPoints.last().second;
    
    // Calculate average throughput
    double sum = 0.0;
    for (const auto& point : dataPoints) {
        sum += point.second;
    }
    averageThroughput = sum / dataPoints.size();
    
    // Update peak and min
    peakThroughput = qMax(peakThroughput, currentThroughput);
    if (minThroughput == 0.0) {
        minThroughput = currentThroughput;
    } else {
        minThroughput = qMin(minThroughput, currentThroughput);
    }
    
    // Calculate variance
    double varianceSum = 0.0;
    for (const auto& point : dataPoints) {
        double diff = point.second - averageThroughput;
        varianceSum += diff * diff;
    }
    throughputVariance = dataPoints.size() > 1 ? varianceSum / (dataPoints.size() - 1) : 0.0;
    
    // Calculate trends
    calculateTrends();
    
    // Update bandwidth utilization
    bandwidthUtilization = (currentThroughput / maxBandwidth) * 100.0;
}

void HashCalculator::ThroughputMonitor::calculateTrends() {
    if (dataPoints.size() < 3) {
        shortTermTrend = 0.0;
        longTermTrend = 0.0;
        trendDirection = "stable";
        return;
    }
    
    // Short-term trend (last 3 points)
    int shortTermCount = qMin(3, dataPoints.size());
    double shortTermSum = 0.0;
    for (int i = dataPoints.size() - shortTermCount; i < dataPoints.size(); ++i) {
        shortTermSum += dataPoints[i].second;
    }
    shortTermTrend = shortTermSum / shortTermCount;
    
    // Long-term trend (all points)
    double longTermSum = 0.0;
    for (const auto& point : dataPoints) {
        longTermSum += point.second;
    }
    longTermTrend = longTermSum / dataPoints.size();
    
    // Determine trend direction
    double trendDifference = shortTermTrend - longTermTrend;
    if (qAbs(trendDifference) < averageThroughput * 0.05) { // 5% threshold
        trendDirection = "stable";
    } else if (trendDifference > 0) {
        trendDirection = "increasing";
    } else {
        trendDirection = "decreasing";
    }
}

double HashCalculator::PerformanceHistogram::getPercentile(const QString& metricName, double percentile) const {
    if (!metrics.contains(metricName) || metrics[metricName].isEmpty()) {
        return 0.0;
    }
    
    QList<double> values = metrics[metricName];
    std::sort(values.begin(), values.end());
    
    int index = static_cast<int>((percentile / 100.0) * (values.size() - 1));
    index = qBound(0, index, values.size() - 1);
    
    return values[index];
}

double HashCalculator::PerformanceHistogram::getMean(const QString& metricName) const {
    if (!metrics.contains(metricName) || metrics[metricName].isEmpty()) {
        return 0.0;
    }
    
    const QList<double>& values = metrics[metricName];
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    
    return sum / values.size();
}

double HashCalculator::PerformanceHistogram::getStandardDeviation(const QString& metricName) const {
    if (!metrics.contains(metricName) || metrics[metricName].size() < 2) {
        return 0.0;
    }
    
    const QList<double>& values = metrics[metricName];
    double mean = getMean(metricName);
    
    double varianceSum = 0.0;
    for (double value : values) {
        double diff = value - mean;
        varianceSum += diff * diff;
    }
    
    double variance = varianceSum / (values.size() - 1);
    return qSqrt(variance);
}

QPair<double, double> HashCalculator::PerformanceHistogram::getRange(const QString& metricName) const {
    if (!metrics.contains(metricName) || metrics[metricName].isEmpty()) {
        return {0.0, 0.0};
    }
    
    const QList<double>& values = metrics[metricName];
    double min = *std::min_element(values.begin(), values.end());
    double max = *std::max_element(values.begin(), values.end());
    
    return {min, max};
}

qint64 HashCalculator::ETAPredictionEngine::calculateETA(qint64 remainingBytes, int remainingFiles) {
    if (remainingBytes <= 0 || recentSpeeds.isEmpty()) {
        return 0;
    }
    
    // Update weighted averages
    updateWeightedAverages();
    
    // Choose the most appropriate average based on data availability and stability
    double predictiveSpeed = shortTermAverage;
    if (recentSpeeds.size() > 10 && speedStability > 0.7) {
        // Use medium-term average for more stable predictions
        predictiveSpeed = mediumTermAverage;
    } else if (recentSpeeds.size() > 50 && speedStability > 0.9) {
        // Use long-term average for very stable systems
        predictiveSpeed = longTermAverage;
    }
    
    // Adjust for queue analysis
    double queueFactor = 1.0;
    if (remainingFiles > 0 && averageFileSize > 0) {
        // Estimate based on file size distribution
        double estimatedRemainingSize = remainingFiles * averageFileSize;
        if (qAbs(estimatedRemainingSize - remainingBytes) / remainingBytes > 0.2) {
            // Significant discrepancy - adjust prediction
            queueFactor = estimatedRemainingSize / remainingBytes;
        }
    }
    
    // Account for system load
    double loadFactor = 1.0 + (systemLoad * 0.5); // Up to 50% slowdown under high load
    
    // Calculate ETA in milliseconds
    double speedMBps = predictiveSpeed;
    if (speedMBps <= 0) {
        return 0;
    }
    
    double remainingMB = remainingBytes / (1024.0 * 1024.0);
    double etaSeconds = (remainingMB / speedMBps) * queueFactor * loadFactor;
    
    return static_cast<qint64>(etaSeconds * 1000); // Convert to milliseconds
}

void HashCalculator::ETAPredictionEngine::updateWeightedAverages() {
    if (recentSpeeds.isEmpty()) {
        return;
    }
    
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    
    // Short-term average (last 30 seconds)
    double shortTermSum = 0.0;
    int shortTermCount = 0;
    for (int i = recentSpeeds.size() - 1; i >= 0; --i) {
        if (now - speedTimestamps[i] <= 30000) { // 30 seconds
            shortTermSum += recentSpeeds[i];
            shortTermCount++;
        } else {
            break;
        }
    }
    shortTermAverage = shortTermCount > 0 ? shortTermSum / shortTermCount : 0.0;
    
    // Medium-term average (last 5 minutes)
    double mediumTermSum = 0.0;
    int mediumTermCount = 0;
    for (int i = recentSpeeds.size() - 1; i >= 0; --i) {
        if (now - speedTimestamps[i] <= 300000) { // 5 minutes
            mediumTermSum += recentSpeeds[i];
            mediumTermCount++;
        } else {
            break;
        }
    }
    mediumTermAverage = mediumTermCount > 0 ? mediumTermSum / mediumTermCount : shortTermAverage;
    
    // Long-term average (all data)
    double longTermSum = 0.0;
    for (double speed : recentSpeeds) {
        longTermSum += speed;
    }
    longTermAverage = longTermSum / recentSpeeds.size();
    
    // Calculate speed stability
    calculateSpeedStability();
}

void HashCalculator::ETAPredictionEngine::calculateSpeedStability() {
    if (recentSpeeds.size() < 2) {
        speedStability = 1.0;
        return;
    }
    
    // Calculate coefficient of variation
    double mean = longTermAverage;
    if (mean <= 0) {
        speedStability = 0.0;
        return;
    }
    
    double varianceSum = 0.0;
    for (double speed : recentSpeeds) {
        double diff = speed - mean;
        varianceSum += diff * diff;
    }
    
    double variance = varianceSum / (recentSpeeds.size() - 1);
    double stdDev = qSqrt(variance);
    double coefficientOfVariation = stdDev / mean;
    
    // Convert coefficient of variation to stability (inverse relationship)
    // Lower CV = higher stability
    speedStability = 1.0 / (1.0 + coefficientOfVariation);
    speedStability = qBound(0.0, speedStability, 1.0);
}

double HashCalculator::ETAPredictionEngine::calculateConfidence() const {
    double confidence = 1.0;
    
    // Reduce confidence for low speed stability
    confidence *= speedStability;
    
    // Reduce confidence for low queue predictability
    confidence *= queuePredictability;
    
    // Reduce confidence for high system load
    confidence *= (1.0 - systemLoad);
    
    // Reduce confidence if we have limited data
    if (recentSpeeds.size() < 5) {
        confidence *= recentSpeeds.size() / 5.0;
    }
    
    return qBound(0.0, confidence, 1.0);
}

QStringList HashCalculator::ETAPredictionEngine::getInfluencingFactors() const {
    QStringList factors;
    
    if (speedStability < 0.7) {
        factors << "Variable processing speed";
    }
    
    if (queuePredictability < 0.7) {
        factors << "Unpredictable file queue";
    }
    
    if (systemLoad > 0.3) {
        factors << "High system load";
    }
    
    if (recentSpeeds.size() < 10) {
        factors << "Limited historical data";
    }
    
    if (pendingFiles > 1000) {
        factors << "Large file queue";
    }
    
    if (factors.isEmpty()) {
        factors << "Stable conditions";
    }
    
    return factors;
}

// HashCalculator Enhanced Progress Reporting Methods

void HashCalculator::updateThroughputMonitor(double throughput) {
    QMutexLocker locker(&m_progressMutex);
    m_throughputMonitor.addDataPoint(throughput);
    
    // Update recent throughputs for variance calculation
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    m_recentThroughputs.append(throughput);
    m_recentTimestamps.append(now);
    
    // Keep only recent samples
    while (m_recentThroughputs.size() > m_maxRecentSamples) {
        m_recentThroughputs.removeFirst();
        m_recentTimestamps.removeFirst();
    }
    
    // Update statistics
    QMutexLocker statsLocker(&m_statsMutex);
    m_statistics.instantaneousThroughput = throughput;
    m_statistics.peakThroughput = qMax(m_statistics.peakThroughput, throughput);
    
    // Update performance histogram
    m_performanceHistogram.addSample("throughput", throughput);
}

void HashCalculator::updatePerformanceHistogram(const QString& metricName, double value) {
    QMutexLocker locker(&m_progressMutex);
    m_performanceHistogram.addSample(metricName, value);
}

void HashCalculator::updateETAPrediction(qint64 processedBytes, qint64 totalBytes) {
    QMutexLocker locker(&m_progressMutex);
    
    // Update ETA engine with current processing data
    double currentSpeed = getCurrentThroughput();
    if (currentSpeed > 0) {
        qint64 now = QDateTime::currentMSecsSinceEpoch();
        m_etaPredictionEngine.recentSpeeds.append(currentSpeed);
        m_etaPredictionEngine.speedTimestamps.append(now);
        
        // Keep reasonable amount of data
        if (m_etaPredictionEngine.recentSpeeds.size() > 200) {
            m_etaPredictionEngine.recentSpeeds.removeFirst();
            m_etaPredictionEngine.speedTimestamps.removeFirst();
        }
        
        // Update queue information
        m_etaPredictionEngine.pendingBytes = totalBytes - processedBytes;
        m_etaPredictionEngine.pendingFiles = getQueuedTaskCount();
        
        if (m_statistics.totalHashesCalculated > 0) {
            m_etaPredictionEngine.averageFileSize = 
                static_cast<double>(m_statistics.totalBytesProcessed) / m_statistics.totalHashesCalculated;
        }
    }
}

void HashCalculator::calculatePerformanceTrend() {
    QMutexLocker locker(&m_progressMutex);
    
    if (m_recentThroughputs.size() < 5) {
        QMutexLocker statsLocker(&m_statsMutex);
        m_statistics.performanceTrend = "insufficient_data";
        m_statistics.trendStrength = 0.0;
        return;
    }
    
    // Calculate trend using linear regression over recent throughputs
    int n = m_recentThroughputs.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (int i = 0; i < n; ++i) {
        double x = i; // Time index
        double y = m_recentThroughputs[i];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
    }
    
    double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    double avgThroughput = sumY / n;
    
    // Normalize slope to get trend strength
    double trendStrength = qAbs(slope) / avgThroughput;
    trendStrength = qBound(0.0, trendStrength, 1.0);
    
    QString trend;
    if (qAbs(slope) < avgThroughput * 0.02) { // 2% threshold
        trend = "stable";
    } else if (slope > 0) {
        trend = "improving";
    } else {
        trend = "degrading";
    }
    
    // Update statistics
    QMutexLocker statsLocker(&m_statsMutex);
    m_statistics.performanceTrend = trend;
    m_statistics.trendStrength = trendStrength;
    
    // Update throughput variance
    m_statistics.throughputVariance = calculateThroughputVariance();
}

void HashCalculator::updateResourceUtilization() {
    if (m_resourceTimer.elapsed() < m_resourceCheckInterval) {
        return; // Don't check too frequently
    }
    
    m_resourceTimer.restart();
    
    QMutexLocker locker(&m_statsMutex);
    
    // Estimate CPU utilization based on thread activity
    m_statistics.cpuUtilization = estimateCPUUtilization();
    
    // Update memory usage
    qint64 currentMemory = getCurrentMemoryUsageInternal();
    m_statistics.peakMemoryUsage = qMax(m_statistics.peakMemoryUsage, currentMemory);
    
    // Estimate memory utilization (assuming 8GB system memory as baseline)
    qint64 estimatedSystemMemory = 8LL * 1024 * 1024 * 1024; // 8GB
    m_statistics.memoryUtilization = (static_cast<double>(currentMemory) / estimatedSystemMemory) * 100.0;
    
    // Update I/O utilization based on recent I/O operations
    if (m_statistics.totalIOTime > 0 && m_performanceTimer.elapsed() > 0) {
        double ioRatio = static_cast<double>(m_statistics.totalIOTime) / m_performanceTimer.elapsed();
        m_statistics.ioWaitTime = qMin(100.0, ioRatio * 100.0);
    }
}

double HashCalculator::estimateCPUUtilization() const {
    // Estimate CPU utilization based on thread activity and processing speed
    int activeThreads = m_statistics.activeThreads;
    int totalThreads = m_options.threadPoolSize;
    
    if (totalThreads == 0) {
        return 0.0;
    }
    
    // Base utilization on thread utilization
    double baseUtilization = (static_cast<double>(activeThreads) / totalThreads) * 100.0;
    
    // Adjust based on current throughput vs. peak throughput
    if (m_statistics.peakThroughput > 0 && m_statistics.instantaneousThroughput > 0) {
        double performanceRatio = m_statistics.instantaneousThroughput / m_statistics.peakThroughput;
        baseUtilization *= performanceRatio;
    }
    
    return qBound(0.0, baseUtilization, 100.0);
}

qint64 HashCalculator::getCurrentMemoryUsageInternal() const {
    // Estimate memory usage based on active operations and cache size
    qint64 estimatedMemory = 0;
    
    // Cache memory
    estimatedMemory += getCacheSize() * 1024; // Rough estimate: 1KB per cache entry
    
    // Buffer pool memory
    estimatedMemory += m_maxBufferPoolSize * 256 * 1024; // Assume 256KB per buffer
    
    // Active task memory
    int activeTasks = m_activeTaskCount.loadAcquire();
    estimatedMemory += activeTasks * m_options.chunkSize; // One chunk per active task
    
    // Memory mapped files (if any)
    QMutexLocker ioLocker(&const_cast<QMutex&>(m_ioMutex));
    estimatedMemory += m_statistics.memoryMappedBytes;
    
    return estimatedMemory;
}

double HashCalculator::calculateThroughputVariance() const {
    if (m_recentThroughputs.size() < 2) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double throughput : m_recentThroughputs) {
        sum += throughput;
    }
    double mean = sum / m_recentThroughputs.size();
    
    double varianceSum = 0.0;
    for (double throughput : m_recentThroughputs) {
        double diff = throughput - mean;
        varianceSum += diff * diff;
    }
    
    return varianceSum / (m_recentThroughputs.size() - 1);
}

void HashCalculator::analyzeQueueEfficiency() {
    QMutexLocker locker(&m_statsMutex);
    
    int currentQueue = getQueuedTaskCount();
    int activeThreads = m_statistics.activeThreads;
    int totalThreads = m_options.threadPoolSize;
    
    // Update queue depth statistics
    m_statistics.maxQueueDepth = qMax(m_statistics.maxQueueDepth, currentQueue);
    
    // Calculate average queue depth (exponential moving average)
    if (m_statistics.averageQueueDepth == 0.0) {
        m_statistics.averageQueueDepth = currentQueue;
    } else {
        m_statistics.averageQueueDepth = (m_statistics.averageQueueDepth * 0.9) + (currentQueue * 0.1);
    }
    
    // Calculate queue efficiency
    // Efficiency is high when:
    // 1. Queue is not too large (avoiding memory overhead)
    // 2. Threads are well-utilized
    // 3. Queue is not empty while threads are idle
    
    double efficiency = 1.0;
    
    // Penalize very large queues
    if (currentQueue > totalThreads * 10) {
        efficiency *= 0.8;
    }
    
    // Penalize empty queues when threads could be working
    if (currentQueue == 0 && activeThreads < totalThreads) {
        efficiency *= 0.5;
    }
    
    // Reward good thread utilization
    if (totalThreads > 0) {
        double threadUtilization = static_cast<double>(activeThreads) / totalThreads;
        efficiency *= threadUtilization;
    }
    
    m_statistics.queueEfficiency = qBound(0.0, efficiency, 1.0);
}

void HashCalculator::updateETAAccuracy(qint64 actualTime, qint64 predictedTime) {
    QMutexLocker locker(&m_statsMutex);
    
    if (predictedTime <= 0) {
        return; // Invalid prediction
    }
    
    // Calculate prediction error
    double error = qAbs(actualTime - predictedTime) / static_cast<double>(predictedTime);
    
    // Update ETA accuracy using exponential moving average
    double accuracy = 1.0 - qMin(1.0, error); // Convert error to accuracy
    
    if (m_statistics.etaPredictions == 0) {
        m_statistics.etaAccuracy = accuracy;
        m_statistics.averageETAError = error;
    } else {
        m_statistics.etaAccuracy = (m_statistics.etaAccuracy * 0.8) + (accuracy * 0.2);
        m_statistics.averageETAError = (m_statistics.averageETAError * 0.8) + (error * 0.2);
    }
    
    m_statistics.etaPredictions++;
}

HashCalculator::ProgressInfo HashCalculator::createDetailedProgressInfo() const {
    ProgressInfo info;
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    QMutexLocker statsLocker(const_cast<QMutex*>(&m_statsMutex));
    
    // Basic progress information
    info.bytesProcessed = m_statistics.totalBytesProcessed;
    // Note: totalBytes would need to be tracked separately for overall progress
    info.percentComplete = 0; // This would need overall job context
    
    // Timing information
    info.startTime = QDateTime::currentDateTime().addMSecs(-m_performanceTimer.elapsed());
    info.lastUpdateTime = QDateTime::currentDateTime();
    info.elapsedTimeMs = m_performanceTimer.elapsed();
    
    // Speed metrics
    info.instantaneousSpeed = m_statistics.instantaneousThroughput;
    info.averageSpeed = m_statistics.averageSpeed;
    info.peakSpeed = m_statistics.peakThroughput;
    info.speedVariance = m_statistics.throughputVariance;
    
    // Thread and queue information
    info.queueDepth = getQueuedTaskCount();
    info.activeThreads = m_statistics.activeThreads;
    info.totalThreads = m_options.threadPoolSize;
    info.threadUtilization = m_statistics.threadUtilization;
    
    // Resource utilization
    info.cpuUtilization = m_statistics.cpuUtilization;
    info.ioUtilization = m_statistics.ioWaitTime;
    info.memoryUsage = getCurrentMemoryUsageInternal();
    
    // ETA information
    info.etaAccuracy = m_statistics.etaAccuracy;
    info.etaFactors = m_etaPredictionEngine.getInfluencingFactors();
    
    // Performance trends
    info.performanceTrend = m_statistics.performanceTrend;
    info.trendStrength = m_statistics.trendStrength;
    
    return info;
}

// Public API Methods for Enhanced Progress Reporting

HashCalculator::ProgressInfo HashCalculator::getDetailedProgress() const {
    return createDetailedProgressInfo();
}

HashCalculator::ThroughputMonitor HashCalculator::getThroughputMonitor() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_throughputMonitor;
}

HashCalculator::PerformanceHistogram HashCalculator::getPerformanceHistogram() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_performanceHistogram;
}

HashCalculator::ETAPredictionEngine HashCalculator::getETAPredictionEngine() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_etaPredictionEngine;
}

double HashCalculator::getInstantaneousThroughput() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.instantaneousThroughput;
}

double HashCalculator::getThroughputVariance() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.throughputVariance;
}

QString HashCalculator::getPerformanceTrend() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.performanceTrend;
}

double HashCalculator::getTrendStrength() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.trendStrength;
}

double HashCalculator::getCPUUtilization() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.cpuUtilization;
}

double HashCalculator::getMemoryUtilization() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.memoryUtilization;
}

double HashCalculator::getIOUtilization() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.ioWaitTime;
}

qint64 HashCalculator::getCurrentMemoryUsage() const {
    return getCurrentMemoryUsageInternal();
}

int HashCalculator::getCurrentQueueDepth() const {
    return getQueuedTaskCount();
}

double HashCalculator::getAverageQueueDepth() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.averageQueueDepth;
}

double HashCalculator::getQueueEfficiency() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.queueEfficiency;
}

int HashCalculator::getActiveThreadCount() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.activeThreads;
}

qint64 HashCalculator::getEnhancedETA(qint64 remainingBytes, int remainingFiles) const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return const_cast<ETAPredictionEngine&>(m_etaPredictionEngine).calculateETA(remainingBytes, remainingFiles);
}

double HashCalculator::getETAAccuracy() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_statsMutex));
    return m_statistics.etaAccuracy;
}

QStringList HashCalculator::getETAInfluencingFactors() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_etaPredictionEngine.getInfluencingFactors();
}

double HashCalculator::getThroughputPercentile(double percentile) const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_performanceHistogram.getPercentile("throughput", percentile);
}

double HashCalculator::getProcessingTimePercentile(double percentile) const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_performanceHistogram.getPercentile("processing_time", percentile);
}

QPair<double, double> HashCalculator::getThroughputRange() const {
    QMutexLocker locker(const_cast<QMutex*>(&m_progressMutex));
    return m_performanceHistogram.getRange("throughput");
}

QMap<QString, double> HashCalculator::getPerformanceSummary() const {
    QMutexLocker statsLocker(const_cast<QMutex*>(&m_statsMutex));
    QMutexLocker progressLocker(const_cast<QMutex*>(&m_progressMutex));
    
    QMap<QString, double> summary;
    
    // Throughput metrics
    summary["current_throughput"] = m_statistics.instantaneousThroughput;
    summary["average_throughput"] = m_statistics.averageSpeed;
    summary["peak_throughput"] = m_statistics.peakThroughput;
    summary["throughput_variance"] = m_statistics.throughputVariance;
    
    // Resource utilization
    summary["cpu_utilization"] = m_statistics.cpuUtilization;
    summary["memory_utilization"] = m_statistics.memoryUtilization;
    summary["io_utilization"] = m_statistics.ioWaitTime;
    
    // Thread and queue metrics
    summary["thread_utilization"] = m_statistics.threadUtilization;
    summary["queue_efficiency"] = m_statistics.queueEfficiency;
    summary["average_queue_depth"] = m_statistics.averageQueueDepth;
    
    // ETA and prediction metrics
    summary["eta_accuracy"] = m_statistics.etaAccuracy;
    summary["trend_strength"] = m_statistics.trendStrength;
    
    // Performance percentiles
    summary["throughput_p50"] = m_performanceHistogram.getPercentile("throughput", 50.0);
    summary["throughput_p90"] = m_performanceHistogram.getPercentile("throughput", 90.0);
    summary["throughput_p99"] = m_performanceHistogram.getPercentile("throughput", 99.0);
    
    // I/O optimization metrics
    summary["memory_mapped_efficiency"] = m_statistics.memoryMappedReads > 0 ? 
        static_cast<double>(m_statistics.memoryMappedBytes) / (m_statistics.memoryMappedReads * 1024.0 * 1024.0) : 0.0;
    summary["buffer_pool_hit_rate"] = (m_statistics.bufferHits + m_statistics.bufferMisses) > 0 ?
        static_cast<double>(m_statistics.bufferHits) / (m_statistics.bufferHits + m_statistics.bufferMisses) : 0.0;
    summary["async_io_efficiency"] = m_statistics.asyncIOOperations > 0 ?
        static_cast<double>(m_statistics.asyncIOBytes) / (m_statistics.asyncIOOperations * 1024.0 * 1024.0) : 0.0;
    
    return summary;
}
