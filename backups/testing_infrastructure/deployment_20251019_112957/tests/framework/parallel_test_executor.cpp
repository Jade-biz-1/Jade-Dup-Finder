#include "parallel_test_executor.h"
#include "test_harness.h"
#include "test_environment.h"
#include <QCoreApplication>
#include <QDebug>
#include <QRandomGenerator>
#include <QtConcurrent>
#include <algorithm>

// TestWorker Implementation
TestWorker::TestWorker(int workerId, QObject* parent)
    : QObject(parent)
    , m_workerId(workerId)
    , m_timeoutTimer(new QTimer(this))
{
    m_timeoutTimer->setSingleShot(true);
    connect(m_timeoutTimer, &QTimer::timeout, this, &TestWorker::onTimeout);
}

TestWorker::~TestWorker() {
    stop();
}

void TestWorker::executeTest(const TestExecutionContext& context) {
    QMutexLocker locker(&m_mutex);
    
    if (m_busy || m_stopRequested) {
        emit workerError(m_workerId, "Worker is busy or stopped");
        return;
    }
    
    m_busy = true;
    m_currentTest = QString("%1::%2").arg(context.suiteName, context.testName);
    
    locker.unlock();
    
    emit testStarted(m_workerId, context.suiteName, context.testName);
    
    // Start timeout timer
    m_timeoutTimer->start(context.resources.estimatedDurationMs * 2); // 2x estimated time
    
    QElapsedTimer executionTimer;
    executionTimer.start();
    
    bool testPassed = false;
    QString errorMessage;
    
    try {
        // Setup test environment with resource isolation
        setupTestEnvironment(context);
        
        // Acquire required resources
        acquireResources(context.resources);
        
        // Execute the actual test
        if (context.suite) {
            context.suite->setUp();
            testPassed = context.suite->runTest(context.testName);
            context.suite->tearDown();
            
            // Get any error messages from the suite
            TestResults results = context.suite->getResults();
            if (!results.failures.isEmpty()) {
                errorMessage = results.failures.last().errorMessage;
            }
        } else {
            errorMessage = "Test suite is null";
        }
        
    } catch (const std::exception& e) {
        testPassed = false;
        errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        testPassed = false;
        errorMessage = "Unknown exception occurred";
    }
    
    // Cleanup
    try {
        releaseResources(context.resources);
        cleanupTestEnvironment(context);
    } catch (...) {
        qWarning() << "Error during test cleanup for worker" << m_workerId;
    }
    
    m_timeoutTimer->stop();
    qint64 duration = executionTimer.elapsed();
    
    locker.relock();
    m_busy = false;
    m_currentTest.clear();
    locker.unlock();
    
    if (testPassed) {
        emit testCompleted(m_workerId, context.suiteName, context.testName, true, duration);
    } else {
        emit testFailed(m_workerId, context.suiteName, context.testName, errorMessage);
        emit testCompleted(m_workerId, context.suiteName, context.testName, false, duration);
    }
    
    emit workerReady(m_workerId);
}

void TestWorker::stop() {
    QMutexLocker locker(&m_mutex);
    m_stopRequested = true;
    m_timeoutTimer->stop();
    
    // Release any acquired resources
    releaseResources(TestResourceRequirements());
}

void TestWorker::onTimeout() {
    QMutexLocker locker(&m_mutex);
    if (m_busy) {
        emit testFailed(m_workerId, "", m_currentTest, "Test execution timeout");
        m_busy = false;
        m_currentTest.clear();
        emit workerReady(m_workerId);
    }
}

void TestWorker::setupTestEnvironment(const TestExecutionContext& context) {
    if (context.environment) {
        // Create isolated test environment for this worker
        QString workerTempDir = QString("worker_%1").arg(m_workerId);
        // Environment setup would be implemented based on specific requirements
    }
}

void TestWorker::cleanupTestEnvironment(const TestExecutionContext& context) {
    if (context.environment) {
        // Cleanup isolated environment
        // Implementation would clean up worker-specific resources
    }
}

bool TestWorker::checkResourceAvailability(const TestResourceRequirements& requirements) {
    // Check if required resources are available
    // This would integrate with the resource manager
    return true; // Simplified for now
}

void TestWorker::acquireResources(const TestResourceRequirements& requirements) {
    // Acquire resources needed for test execution
    if (requirements.requiresFileSystem) {
        m_acquiredResources << "filesystem";
    }
    if (requirements.requiresNetwork) {
        m_acquiredResources << "network";
    }
    if (requirements.requiresDisplay) {
        m_acquiredResources << "display";
    }
}

void TestWorker::releaseResources(const TestResourceRequirements& requirements) {
    // Release all acquired resources
    m_acquiredResources.clear();
}

// TestLoadBalancer Implementation
TestLoadBalancer::TestLoadBalancer(QObject* parent)
    : QObject(parent)
{
}

void TestLoadBalancer::addWorker(int workerId, const QStringList& capabilities) {
    QMutexLocker locker(&m_mutex);
    
    WorkerInfo info;
    info.workerId = workerId;
    info.capabilities = capabilities;
    info.lastActivityTimer.start();
    
    m_workers[workerId] = info;
    
    emit loadBalancingUpdated();
}

void TestLoadBalancer::removeWorker(int workerId) {
    QMutexLocker locker(&m_mutex);
    m_workers.remove(workerId);
    emit loadBalancingUpdated();
}

void TestLoadBalancer::updateWorkerLoad(int workerId, double load) {
    QMutexLocker locker(&m_mutex);
    if (m_workers.contains(workerId)) {
        m_workers[workerId].currentLoad = load;
        m_workers[workerId].lastActivityTimer.restart();
    }
}

int TestLoadBalancer::assignTest(const TestExecutionContext& context) {
    QMutexLocker locker(&m_mutex);
    
    int selectedWorker = -1;
    
    switch (m_strategy) {
        case Strategy::RoundRobin:
            selectedWorker = selectWorkerRoundRobin();
            break;
        case Strategy::LeastBusy:
            selectedWorker = selectWorkerLeastBusy();
            break;
        case Strategy::ResourceAware:
            selectedWorker = selectWorkerResourceAware(context.resources);
            break;
        case Strategy::PriorityBased:
            selectedWorker = selectWorkerPriorityBased(context.priority);
            break;
        case Strategy::Adaptive:
            selectedWorker = selectWorkerAdaptive(context);
            break;
    }
    
    if (selectedWorker != -1 && m_workers.contains(selectedWorker)) {
        m_workers[selectedWorker].available = false;
        m_workers[selectedWorker].currentLoad += 1.0;
        emit workerAssigned(selectedWorker, context);
    }
    
    return selectedWorker;
}

void TestLoadBalancer::reportTestCompletion(int workerId, qint64 durationMs) {
    QMutexLocker locker(&m_mutex);
    
    if (m_workers.contains(workerId)) {
        WorkerInfo& worker = m_workers[workerId];
        worker.available = true;
        worker.currentLoad = qMax(0.0, worker.currentLoad - 1.0);
        worker.totalExecutionTime += durationMs;
        worker.completedTests++;
        worker.lastActivityTimer.restart();
        
        // Update statistics
        m_stats.completedTests++;
        m_stats.totalExecutionTimeMs += durationMs;
        if (m_stats.completedTests > 0) {
            m_stats.averageExecutionTimeMs = m_stats.totalExecutionTimeMs / m_stats.completedTests;
        }
    }
}

void TestLoadBalancer::resetStats() {
    QMutexLocker locker(&m_mutex);
    m_stats = ParallelExecutionStats();
}

int TestLoadBalancer::selectWorkerRoundRobin() {
    QList<int> availableWorkers;
    for (auto it = m_workers.begin(); it != m_workers.end(); ++it) {
        if (it.value().available) {
            availableWorkers << it.key();
        }
    }
    
    if (availableWorkers.isEmpty()) {
        return -1;
    }
    
    std::sort(availableWorkers.begin(), availableWorkers.end());
    int selectedWorker = availableWorkers[m_nextWorkerIndex % availableWorkers.size()];
    m_nextWorkerIndex++;
    
    return selectedWorker;
}

int TestLoadBalancer::selectWorkerLeastBusy() {
    int bestWorker = -1;
    double lowestLoad = std::numeric_limits<double>::max();
    
    for (auto it = m_workers.begin(); it != m_workers.end(); ++it) {
        if (it.value().available && it.value().currentLoad < lowestLoad) {
            lowestLoad = it.value().currentLoad;
            bestWorker = it.key();
        }
    }
    
    return bestWorker;
}

int TestLoadBalancer::selectWorkerResourceAware(const TestResourceRequirements& requirements) {
    int bestWorker = -1;
    double bestScore = -1.0;
    
    for (auto it = m_workers.begin(); it != m_workers.end(); ++it) {
        if (!it.value().available) {
            continue;
        }
        
        if (!canWorkerHandleTest(it.value(), TestExecutionContext())) {
            continue;
        }
        
        // Calculate resource compatibility score
        double score = 1.0 - it.value().currentLoad * 0.5;
        
        // Bonus for having required capabilities
        if (requirements.requiresDisplay && it.value().capabilities.contains("display")) {
            score += 0.2;
        }
        if (requirements.requiresNetwork && it.value().capabilities.contains("network")) {
            score += 0.2;
        }
        if (requirements.requiresFileSystem && it.value().capabilities.contains("filesystem")) {
            score += 0.2;
        }
        
        if (score > bestScore) {
            bestScore = score;
            bestWorker = it.key();
        }
    }
    
    return bestWorker;
}

int TestLoadBalancer::selectWorkerPriorityBased(TestPriority priority) {
    // For high priority tests, prefer workers with lower load
    // For low priority tests, use any available worker
    
    if (priority >= TestPriority::High) {
        return selectWorkerLeastBusy();
    } else {
        return selectWorkerRoundRobin();
    }
}

int TestLoadBalancer::selectWorkerAdaptive(const TestExecutionContext& context) {
    int bestWorker = -1;
    double bestScore = -1.0;
    
    for (auto it = m_workers.begin(); it != m_workers.end(); ++it) {
        if (!it.value().available) {
            continue;
        }
        
        double score = calculateWorkerScore(it.value(), context);
        if (score > bestScore) {
            bestScore = score;
            bestWorker = it.key();
        }
    }
    
    return bestWorker;
}

bool TestLoadBalancer::canWorkerHandleTest(const WorkerInfo& worker, const TestExecutionContext& context) {
    // Check if worker has conflicting resources
    for (const QString& resource : context.resources.conflictingResources) {
        if (worker.activeResources.contains(resource)) {
            return false;
        }
    }
    
    return true;
}

double TestLoadBalancer::calculateWorkerScore(const WorkerInfo& worker, const TestExecutionContext& context) {
    double score = 1.0;
    
    // Load factor (lower load = higher score)
    score -= worker.currentLoad * 0.3;
    
    // Priority factor
    if (context.priority == TestPriority::Critical) {
        score += 0.3;
    } else if (context.priority == TestPriority::High) {
        score += 0.2;
    }
    
    // Resource compatibility
    int compatibleResources = 0;
    if (context.resources.requiresDisplay && worker.capabilities.contains("display")) {
        compatibleResources++;
    }
    if (context.resources.requiresNetwork && worker.capabilities.contains("network")) {
        compatibleResources++;
    }
    if (context.resources.requiresFileSystem && worker.capabilities.contains("filesystem")) {
        compatibleResources++;
    }
    
    score += compatibleResources * 0.1;
    
    // Performance history (workers with better average performance get higher scores)
    if (worker.completedTests > 0) {
        double avgTime = (double)worker.totalExecutionTime / worker.completedTests;
        double expectedTime = context.resources.estimatedDurationMs;
        if (expectedTime > 0 && avgTime < expectedTime) {
            score += 0.1; // Bonus for fast workers
        }
    }
    
    return qMax(0.0, score);
}

// TestResourceManager Implementation
TestResourceManager::TestResourceManager(QObject* parent)
    : QObject(parent)
{
    // Initialize common resource pools
    createResourcePool("filesystem", 4);
    createResourcePool("network", 2);
    createResourcePool("display", 1);
}

bool TestResourceManager::acquireResource(const QString& resourceName, int workerId, int timeoutMs) {
    QMutexLocker locker(&m_mutex);
    
    if (!m_resources.contains(resourceName)) {
        // Create resource if it doesn't exist
        ResourceInfo info;
        info.name = resourceName;
        info.exclusive = true;
        info.maxConcurrent = 1;
        m_resources[resourceName] = info;
    }
    
    ResourceInfo& resource = m_resources[resourceName];
    
    // Check if resource is available
    if (resource.owners.size() < resource.maxConcurrent) {
        resource.owners.append(workerId);
        m_workerResources[workerId].append(resourceName);
        resource.acquisitionCount++;
        
        emit resourceAcquired(resourceName, workerId);
        return true;
    }
    
    // Resource is not available
    if (timeoutMs > 0) {
        resource.waitingQueue.enqueue(workerId);
        // In a real implementation, we would wait for the resource to become available
        // For now, we'll just return false
    }
    
    return false;
}

void TestResourceManager::releaseResource(const QString& resourceName, int workerId) {
    QMutexLocker locker(&m_mutex);
    
    if (!m_resources.contains(resourceName)) {
        return;
    }
    
    ResourceInfo& resource = m_resources[resourceName];
    resource.owners.removeAll(workerId);
    m_workerResources[workerId].removeAll(resourceName);
    
    emit resourceReleased(resourceName, workerId);
    
    // Check if there are waiting workers
    if (!resource.waitingQueue.isEmpty()) {
        int waitingWorker = resource.waitingQueue.dequeue();
        if (acquireResource(resourceName, waitingWorker, 0)) {
            // Successfully assigned to waiting worker
        }
    }
}

void TestResourceManager::releaseAllResources(int workerId) {
    QMutexLocker locker(&m_mutex);
    
    QStringList resources = m_workerResources.value(workerId);
    for (const QString& resourceName : resources) {
        releaseResource(resourceName, workerId);
    }
    
    m_workerResources.remove(workerId);
}

bool TestResourceManager::isResourceAvailable(const QString& resourceName) const {
    QMutexLocker locker(&m_mutex);
    
    if (!m_resources.contains(resourceName)) {
        return true; // Non-existent resources are considered available
    }
    
    const ResourceInfo& resource = m_resources[resourceName];
    return resource.owners.size() < resource.maxConcurrent;
}

QStringList TestResourceManager::getAvailableResources() const {
    QMutexLocker locker(&m_mutex);
    
    QStringList available;
    for (auto it = m_resources.begin(); it != m_resources.end(); ++it) {
        if (it.value().owners.size() < it.value().maxConcurrent) {
            available << it.key();
        }
    }
    
    return available;
}

QStringList TestResourceManager::getResourceOwners(const QString& resourceName) const {
    QMutexLocker locker(&m_mutex);
    
    if (!m_resources.contains(resourceName)) {
        return QStringList();
    }
    
    QStringList owners;
    for (int workerId : m_resources[resourceName].owners) {
        owners << QString::number(workerId);
    }
    
    return owners;
}

void TestResourceManager::createResourcePool(const QString& poolName, int maxConcurrent) {
    QMutexLocker locker(&m_mutex);
    
    ResourcePool pool;
    pool.name = poolName;
    pool.maxConcurrent = maxConcurrent;
    m_resourcePools[poolName] = pool;
}

bool TestResourceManager::acquireFromPool(const QString& poolName, int workerId) {
    QMutexLocker locker(&m_mutex);
    
    if (!m_resourcePools.contains(poolName)) {
        return false;
    }
    
    ResourcePool& pool = m_resourcePools[poolName];
    if (pool.currentUsage < pool.maxConcurrent) {
        pool.currentUsage++;
        pool.users.append(workerId);
        return true;
    }
    
    return false;
}

void TestResourceManager::releaseToPool(const QString& poolName, int workerId) {
    QMutexLocker locker(&m_mutex);
    
    if (!m_resourcePools.contains(poolName)) {
        return;
    }
    
    ResourcePool& pool = m_resourcePools[poolName];
    if (pool.users.removeAll(workerId) > 0) {
        pool.currentUsage = qMax(0, pool.currentUsage - 1);
    }
}

QMap<QString, int> TestResourceManager::getResourceUtilization() const {
    QMutexLocker locker(&m_mutex);
    
    QMap<QString, int> utilization;
    for (auto it = m_resources.begin(); it != m_resources.end(); ++it) {
        utilization[it.key()] = it.value().owners.size();
    }
    
    return utilization;
}

void TestResourceManager::resetStatistics() {
    QMutexLocker locker(&m_mutex);
    
    for (auto it = m_resources.begin(); it != m_resources.end(); ++it) {
        it.value().totalAcquisitionTime = 0;
        it.value().acquisitionCount = 0;
    }
}

// ParallelTestExecutor Implementation
ParallelTestExecutor::ParallelTestExecutor(QObject* parent)
    : QObject(parent)
    , m_loadBalancer(std::make_unique<TestLoadBalancer>(this))
    , m_resourceManager(std::make_unique<TestResourceManager>(this))
    , m_progressTimer(new QTimer(this))
{
    // Connect load balancer signals
    connect(m_loadBalancer.get(), &TestLoadBalancer::workerAssigned,
            this, [this](int workerId, const TestExecutionContext& context) {
                // Worker assignment handled in scheduleNextTest
            });
    
    // Setup progress timer
    m_progressTimer->setInterval(1000); // Update every second
    connect(m_progressTimer, &QTimer::timeout, this, &ParallelTestExecutor::updateExecutionStats);
}

ParallelTestExecutor::~ParallelTestExecutor() {
    shutdownWorkers();
}

void ParallelTestExecutor::setMaxWorkers(int maxWorkers) {
    if (maxWorkers > 0 && maxWorkers <= 32) { // Reasonable limit
        m_maxWorkers = maxWorkers;
        
        if (!m_executionActive) {
            // Reinitialize workers if not currently executing
            shutdownWorkers();
            initializeWorkers();
        }
    }
}

void ParallelTestExecutor::setLoadBalancingStrategy(TestLoadBalancer::Strategy strategy) {
    m_loadBalancer->setStrategy(strategy);
}

void ParallelTestExecutor::setResourceIsolation(bool enabled) {
    m_resourceIsolation = enabled;
}

void ParallelTestExecutor::setRetryPolicy(int maxRetries, int retryDelayMs) {
    m_maxRetries = qMax(0, maxRetries);
    m_retryDelayMs = qMax(0, retryDelayMs);
}

bool ParallelTestExecutor::executeTests(const QList<TestExecutionContext>& tests) {
    if (m_executionActive) {
        qWarning() << "Test execution already in progress";
        return false;
    }
    
    if (tests.isEmpty()) {
        qWarning() << "No tests to execute";
        return false;
    }
    
    // Initialize execution state
    m_executionActive = true;
    m_stopRequested = false;
    m_testQueue.clear();
    m_runningTests.clear();
    m_completedTests.clear();
    m_failedTests.clear();
    
    // Queue all tests
    for (const TestExecutionContext& test : tests) {
        m_testQueue.enqueue(test);
    }
    
    // Initialize statistics
    m_executionStats = ParallelExecutionStats();
    m_executionStats.totalTests = tests.size();
    m_executionStats.queuedTests = tests.size();
    
    // Initialize workers
    initializeWorkers();
    
    // Start execution
    emit executionStarted(tests.size());
    m_executionTimer.start();
    m_progressTimer->start();
    
    // Schedule initial tests
    scheduleNextTest();
    
    return true;
}

bool ParallelTestExecutor::executeTestsWithPriority(const QMap<TestPriority, QList<TestExecutionContext>>& prioritizedTests) {
    QList<TestExecutionContext> allTests;
    
    // Add tests in priority order (Critical -> High -> Normal -> Low)
    QList<TestPriority> priorities = {TestPriority::Critical, TestPriority::High, TestPriority::Normal, TestPriority::Low};
    
    for (TestPriority priority : priorities) {
        if (prioritizedTests.contains(priority)) {
            allTests.append(prioritizedTests[priority]);
        }
    }
    
    return executeTests(allTests);
}

void ParallelTestExecutor::stopExecution() {
    m_stopRequested = true;
    
    // Stop all workers
    for (auto& worker : m_workers) {
        worker->stop();
    }
    
    m_progressTimer->stop();
    
    // Update final statistics
    updateExecutionStats();
    
    m_executionActive = false;
    emit executionCompleted(m_executionStats);
}

void ParallelTestExecutor::addWorker(const QStringList& capabilities) {
    int workerId = m_workers.size();
    
    // Create worker thread
    QThread* workerThread = new QThread(this);
    m_workerThreads[workerId] = workerThread;
    
    // Create worker
    auto worker = std::make_unique<TestWorker>(workerId);
    worker->moveToThread(workerThread);
    
    // Connect worker signals
    connect(worker.get(), &TestWorker::testStarted, this, &ParallelTestExecutor::onTestStarted);
    connect(worker.get(), &TestWorker::testCompleted, this, &ParallelTestExecutor::onTestCompleted);
    connect(worker.get(), &TestWorker::testFailed, this, &ParallelTestExecutor::onTestFailed);
    connect(worker.get(), &TestWorker::workerReady, this, &ParallelTestExecutor::onWorkerReady);
    connect(worker.get(), &TestWorker::workerError, this, &ParallelTestExecutor::onWorkerError);
    
    m_workers[workerId] = std::move(worker);
    
    // Add to load balancer
    m_loadBalancer->addWorker(workerId, capabilities);
    
    // Start worker thread
    workerThread->start();
}

void ParallelTestExecutor::removeWorker(int workerId) {
    if (!m_workers.contains(workerId)) {
        return;
    }
    
    // Stop worker
    m_workers[workerId]->stop();
    
    // Stop and cleanup thread
    if (m_workerThreads.contains(workerId)) {
        m_workerThreads[workerId]->quit();
        m_workerThreads[workerId]->wait(5000);
        delete m_workerThreads[workerId];
        m_workerThreads.remove(workerId);
    }
    
    // Remove from load balancer
    m_loadBalancer->removeWorker(workerId);
    
    // Remove worker
    m_workers.remove(workerId);
}

int ParallelTestExecutor::getWorkerCount() const {
    return m_workers.size();
}

QList<int> ParallelTestExecutor::getActiveWorkers() const {
    QList<int> activeWorkers;
    for (auto it = m_workers.begin(); it != m_workers.end(); ++it) {
        if (it.value()->isBusy()) {
            activeWorkers << it.key();
        }
    }
    return activeWorkers;
}

ParallelExecutionStats ParallelTestExecutor::getExecutionStats() const {
    QMutexLocker locker(&m_statsMutex);
    return m_executionStats;
}

QMap<QString, int> ParallelTestExecutor::getResourceUtilization() const {
    return m_resourceManager->getResourceUtilization();
}

bool ParallelTestExecutor::isExecutionComplete() const {
    return !m_executionActive && m_testQueue.isEmpty() && m_runningTests.isEmpty();
}

double ParallelTestExecutor::getProgress() const {
    if (m_executionStats.totalTests == 0) {
        return 0.0;
    }
    
    return (double)m_executionStats.completedTests / m_executionStats.totalTests * 100.0;
}

void ParallelTestExecutor::enableResultAggregation(bool enabled) {
    m_resultAggregation = enabled;
}

QMap<QString, QVariant> ParallelTestExecutor::getAggregatedResults() const {
    QMutexLocker locker(&m_resultsMutex);
    return m_aggregatedResults;
}

// Private slots
void ParallelTestExecutor::onTestStarted(int workerId, const QString& suiteName, const QString& testName) {
    emit testStarted(suiteName, testName, workerId);
    emit workerStatusChanged(workerId, true);
    
    QMutexLocker locker(&m_statsMutex);
    m_executionStats.runningTests++;
    m_executionStats.queuedTests--;
    m_executionStats.peakConcurrency = qMax(m_executionStats.peakConcurrency, m_executionStats.runningTests);
}

void ParallelTestExecutor::onTestCompleted(int workerId, const QString& suiteName, const QString& testName, bool passed, qint64 durationMs) {
    emit testCompleted(suiteName, testName, passed, durationMs);
    emit workerStatusChanged(workerId, false);
    
    // Update statistics
    {
        QMutexLocker locker(&m_statsMutex);
        m_executionStats.runningTests--;
        m_executionStats.completedTests++;
        
        if (!passed) {
            m_executionStats.failedTests++;
        }
        
        m_executionStats.totalExecutionTimeMs += durationMs;
        if (m_executionStats.completedTests > 0) {
            m_executionStats.averageExecutionTimeMs = m_executionStats.totalExecutionTimeMs / m_executionStats.completedTests;
        }
    }
    
    // Report to load balancer
    m_loadBalancer->reportTestCompletion(workerId, durationMs);
    
    // Remove from running tests
    if (m_runningTests.contains(workerId)) {
        TestExecutionContext context = m_runningTests[workerId];
        m_runningTests.remove(workerId);
        
        if (passed) {
            m_completedTests.append(context);
        } else {
            // Check if we should retry
            if (context.retryCount < m_maxRetries) {
                context.retryCount++;
                QTimer::singleShot(m_retryDelayMs, this, [this, context]() {
                    retryFailedTest(context);
                });
            } else {
                m_failedTests.append(context);
            }
        }
    }
    
    // Aggregate results if enabled
    if (m_resultAggregation) {
        aggregateTestResults();
    }
    
    // Schedule next test
    scheduleNextTest();
    
    // Check if execution is complete
    if (isExecutionComplete()) {
        m_executionActive = false;
        m_progressTimer->stop();
        updateExecutionStats();
        emit executionCompleted(m_executionStats);
    }
    
    emit progressUpdated(m_executionStats.completedTests, m_executionStats.totalTests);
}

void ParallelTestExecutor::onTestFailed(int workerId, const QString& suiteName, const QString& testName, const QString& error) {
    emit testFailed(suiteName, testName, error);
}

void ParallelTestExecutor::onWorkerReady(int workerId) {
    // Worker is ready for next test
    scheduleNextTest();
}

void ParallelTestExecutor::onWorkerError(int workerId, const QString& error) {
    qWarning() << "Worker" << workerId << "error:" << error;
    
    // Remove failed test from running tests
    if (m_runningTests.contains(workerId)) {
        TestExecutionContext context = m_runningTests[workerId];
        m_runningTests.remove(workerId);
        
        // Retry the test on a different worker
        if (context.retryCount < m_maxRetries) {
            context.retryCount++;
            m_testQueue.enqueue(context);
        } else {
            m_failedTests.append(context);
        }
    }
    
    scheduleNextTest();
}

// Private methods
void ParallelTestExecutor::initializeWorkers() {
    // Create workers up to the maximum
    for (int i = m_workers.size(); i < m_maxWorkers; ++i) {
        QStringList capabilities = {"filesystem", "network"};
        if (i == 0) {
            capabilities << "display"; // Only first worker gets display capability
        }
        addWorker(capabilities);
    }
}

void ParallelTestExecutor::shutdownWorkers() {
    // Stop all workers
    QList<int> workerIds = m_workers.keys();
    for (int workerId : workerIds) {
        removeWorker(workerId);
    }
}

void ParallelTestExecutor::scheduleNextTest() {
    if (m_stopRequested || m_testQueue.isEmpty()) {
        return;
    }
    
    // Find available worker
    int workerId = m_loadBalancer->assignTest(m_testQueue.head());
    if (workerId == -1) {
        return; // No available workers
    }
    
    if (!m_workers.contains(workerId)) {
        return; // Worker doesn't exist
    }
    
    // Get next test
    TestExecutionContext context = m_testQueue.dequeue();
    m_runningTests[workerId] = context;
    
    // Execute test on worker
    QMetaObject::invokeMethod(m_workers[workerId].get(), "executeTest", 
                             Qt::QueuedConnection, Q_ARG(TestExecutionContext, context));
}

void ParallelTestExecutor::retryFailedTest(const TestExecutionContext& context) {
    if (!m_stopRequested) {
        m_testQueue.enqueue(context);
        
        QMutexLocker locker(&m_statsMutex);
        m_executionStats.retriedTests++;
        
        scheduleNextTest();
    }
}

void ParallelTestExecutor::updateExecutionStats() {
    QMutexLocker locker(&m_statsMutex);
    
    // Calculate parallel efficiency
    if (m_executionTimer.isValid() && m_executionStats.totalExecutionTimeMs > 0) {
        qint64 wallClockTime = m_executionTimer.elapsed();
        if (wallClockTime > 0) {
            double theoreticalSpeedup = (double)m_executionStats.totalExecutionTimeMs / wallClockTime;
            double actualSpeedup = qMin(theoreticalSpeedup, (double)m_maxWorkers);
            m_executionStats.parallelEfficiency = (actualSpeedup / m_maxWorkers) * 100.0;
        }
    }
    
    // Update resource utilization
    m_executionStats.resourceUtilization = m_resourceManager->getResourceUtilization();
}

void ParallelTestExecutor::aggregateTestResults() {
    QMutexLocker locker(&m_resultsMutex);
    
    // Aggregate results from completed tests
    // This would collect metrics, performance data, etc.
    m_aggregatedResults["completed_tests"] = m_completedTests.size();
    m_aggregatedResults["failed_tests"] = m_failedTests.size();
    m_aggregatedResults["average_duration"] = m_executionStats.averageExecutionTimeMs;
    m_aggregatedResults["parallel_efficiency"] = m_executionStats.parallelEfficiency;
}

#include "parallel_test_executor.moc"