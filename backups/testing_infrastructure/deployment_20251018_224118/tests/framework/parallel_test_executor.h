#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QSemaphore>
#include <QQueue>
#include <QTimer>
#include <QElapsedTimer>
#include <QAtomicInt>
#include <QThreadPool>
#include <memory>
#include <functional>

// Forward declarations
class TestSuite;
class TestEnvironment;

/**
 * @brief Test execution priority levels for load balancing
 */
enum class TestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

/**
 * @brief Resource requirements for test isolation
 */
struct TestResourceRequirements {
    bool requiresFileSystem = false;
    bool requiresNetwork = false;
    bool requiresDisplay = false;
    bool requiresExclusiveAccess = false;
    int memoryMB = 100;
    int estimatedDurationMs = 5000;
    QStringList conflictingResources;
};

/**
 * @brief Individual test execution context
 */
struct TestExecutionContext {
    QString suiteName;
    QString testName;
    TestPriority priority = TestPriority::Normal;
    TestResourceRequirements resources;
    std::shared_ptr<TestSuite> suite;
    std::shared_ptr<TestEnvironment> environment;
    QElapsedTimer executionTimer;
    int retryCount = 0;
    int maxRetries = 2;
};

/**
 * @brief Test execution statistics and metrics
 */
struct ParallelExecutionStats {
    int totalTests = 0;
    int completedTests = 0;
    int runningTests = 0;
    int queuedTests = 0;
    int failedTests = 0;
    int retriedTests = 0;
    qint64 totalExecutionTimeMs = 0;
    qint64 averageExecutionTimeMs = 0;
    double parallelEfficiency = 0.0; // Actual vs theoretical speedup
    int peakConcurrency = 0;
    QMap<QString, int> resourceUtilization;
};

/**
 * @brief Thread-safe test worker for executing individual tests
 */
class TestWorker : public QObject {
    Q_OBJECT

public:
    explicit TestWorker(int workerId, QObject* parent = nullptr);
    ~TestWorker();

    int workerId() const { return m_workerId; }
    bool isBusy() const { return m_busy; }
    QString currentTest() const { return m_currentTest; }

public slots:
    void executeTest(const TestExecutionContext& context);
    void stop();

signals:
    void testStarted(int workerId, const QString& suiteName, const QString& testName);
    void testCompleted(int workerId, const QString& suiteName, const QString& testName, bool passed, qint64 durationMs);
    void testFailed(int workerId, const QString& suiteName, const QString& testName, const QString& error);
    void workerReady(int workerId);
    void workerError(int workerId, const QString& error);

private slots:
    void onTimeout();

private:
    void setupTestEnvironment(const TestExecutionContext& context);
    void cleanupTestEnvironment(const TestExecutionContext& context);
    bool checkResourceAvailability(const TestResourceRequirements& requirements);
    void acquireResources(const TestResourceRequirements& requirements);
    void releaseResources(const TestResourceRequirements& requirements);

    int m_workerId;
    bool m_busy = false;
    bool m_stopRequested = false;
    QString m_currentTest;
    QTimer* m_timeoutTimer;
    QMutex m_mutex;
    QStringList m_acquiredResources;
};

/**
 * @brief Intelligent load balancer for distributing tests across workers
 */
class TestLoadBalancer : public QObject {
    Q_OBJECT

public:
    explicit TestLoadBalancer(QObject* parent = nullptr);

    // Load balancing strategies
    enum class Strategy {
        RoundRobin,
        LeastBusy,
        ResourceAware,
        PriorityBased,
        Adaptive
    };

    void setStrategy(Strategy strategy) { m_strategy = strategy; }
    Strategy getStrategy() const { return m_strategy; }

    // Worker management
    void addWorker(int workerId, const QStringList& capabilities = {});
    void removeWorker(int workerId);
    void updateWorkerLoad(int workerId, double load);

    // Test assignment
    int assignTest(const TestExecutionContext& context);
    void reportTestCompletion(int workerId, qint64 durationMs);

    // Statistics
    ParallelExecutionStats getStats() const { return m_stats; }
    void resetStats();

signals:
    void workerAssigned(int workerId, const TestExecutionContext& context);
    void loadBalancingUpdated();

private:
    struct WorkerInfo {
        int workerId;
        bool available = true;
        double currentLoad = 0.0;
        QStringList capabilities;
        QStringList activeResources;
        qint64 totalExecutionTime = 0;
        int completedTests = 0;
        QElapsedTimer lastActivityTimer;
    };

    int selectWorkerRoundRobin();
    int selectWorkerLeastBusy();
    int selectWorkerResourceAware(const TestResourceRequirements& requirements);
    int selectWorkerPriorityBased(TestPriority priority);
    int selectWorkerAdaptive(const TestExecutionContext& context);

    bool canWorkerHandleTest(const WorkerInfo& worker, const TestExecutionContext& context);
    double calculateWorkerScore(const WorkerInfo& worker, const TestExecutionContext& context);

    Strategy m_strategy = Strategy::Adaptive;
    QMap<int, WorkerInfo> m_workers;
    ParallelExecutionStats m_stats;
    int m_nextWorkerIndex = 0;
    QMutex m_mutex;
};

/**
 * @brief Resource manager for test isolation and conflict resolution
 */
class TestResourceManager : public QObject {
    Q_OBJECT

public:
    explicit TestResourceManager(QObject* parent = nullptr);

    // Resource management
    bool acquireResource(const QString& resourceName, int workerId, int timeoutMs = 5000);
    void releaseResource(const QString& resourceName, int workerId);
    void releaseAllResources(int workerId);

    // Resource queries
    bool isResourceAvailable(const QString& resourceName) const;
    QStringList getAvailableResources() const;
    QStringList getResourceOwners(const QString& resourceName) const;

    // Resource pools
    void createResourcePool(const QString& poolName, int maxConcurrent);
    bool acquireFromPool(const QString& poolName, int workerId);
    void releaseToPool(const QString& poolName, int workerId);

    // Statistics
    QMap<QString, int> getResourceUtilization() const;
    void resetStatistics();

signals:
    void resourceAcquired(const QString& resourceName, int workerId);
    void resourceReleased(const QString& resourceName, int workerId);
    void resourceConflict(const QString& resourceName, int requestingWorker, int owningWorker);

private:
    struct ResourceInfo {
        QString name;
        bool exclusive = false;
        int maxConcurrent = 1;
        QList<int> owners;
        QQueue<int> waitingQueue;
        qint64 totalAcquisitionTime = 0;
        int acquisitionCount = 0;
    };

    struct ResourcePool {
        QString name;
        int maxConcurrent;
        int currentUsage = 0;
        QList<int> users;
        QQueue<int> waitingQueue;
    };

    mutable QMutex m_mutex;
    QMap<QString, ResourceInfo> m_resources;
    QMap<QString, ResourcePool> m_resourcePools;
    QMap<int, QStringList> m_workerResources; // workerId -> resources
};

/**
 * @brief Advanced parallel test execution framework
 */
class ParallelTestExecutor : public QObject {
    Q_OBJECT

public:
    explicit ParallelTestExecutor(QObject* parent = nullptr);
    ~ParallelTestExecutor();

    // Configuration
    void setMaxWorkers(int maxWorkers);
    void setLoadBalancingStrategy(TestLoadBalancer::Strategy strategy);
    void setResourceIsolation(bool enabled);
    void setRetryPolicy(int maxRetries, int retryDelayMs = 1000);

    // Test execution
    bool executeTests(const QList<TestExecutionContext>& tests);
    bool executeTestsWithPriority(const QMap<TestPriority, QList<TestExecutionContext>>& prioritizedTests);
    void stopExecution();

    // Worker management
    void addWorker(const QStringList& capabilities = {});
    void removeWorker(int workerId);
    int getWorkerCount() const;
    QList<int> getActiveWorkers() const;

    // Monitoring
    ParallelExecutionStats getExecutionStats() const;
    QMap<QString, int> getResourceUtilization() const;
    bool isExecutionComplete() const;
    double getProgress() const;

    // Results aggregation
    void enableResultAggregation(bool enabled);
    QMap<QString, QVariant> getAggregatedResults() const;

signals:
    void executionStarted(int totalTests);
    void executionCompleted(const ParallelExecutionStats& stats);
    void testStarted(const QString& suiteName, const QString& testName, int workerId);
    void testCompleted(const QString& suiteName, const QString& testName, bool passed, qint64 durationMs);
    void testFailed(const QString& suiteName, const QString& testName, const QString& error);
    void progressUpdated(int completed, int total);
    void workerStatusChanged(int workerId, bool busy);
    void resourceConflictDetected(const QString& resourceName, const QStringList& conflictingTests);

private slots:
    void onTestStarted(int workerId, const QString& suiteName, const QString& testName);
    void onTestCompleted(int workerId, const QString& suiteName, const QString& testName, bool passed, qint64 durationMs);
    void onTestFailed(int workerId, const QString& suiteName, const QString& testName, const QString& error);
    void onWorkerReady(int workerId);
    void onWorkerError(int workerId, const QString& error);

private:
    void initializeWorkers();
    void shutdownWorkers();
    void scheduleNextTest();
    void retryFailedTest(const TestExecutionContext& context);
    void updateExecutionStats();
    void aggregateTestResults();

    // Configuration
    int m_maxWorkers = 4;
    bool m_resourceIsolation = true;
    int m_maxRetries = 2;
    int m_retryDelayMs = 1000;
    bool m_resultAggregation = true;

    // Execution state
    QQueue<TestExecutionContext> m_testQueue;
    QMap<int, TestExecutionContext> m_runningTests; // workerId -> context
    QList<TestExecutionContext> m_completedTests;
    QList<TestExecutionContext> m_failedTests;
    bool m_executionActive = false;
    bool m_stopRequested = false;

    // Components
    QMap<int, std::unique_ptr<TestWorker>> m_workers;
    QMap<int, QThread*> m_workerThreads;
    std::unique_ptr<TestLoadBalancer> m_loadBalancer;
    std::unique_ptr<TestResourceManager> m_resourceManager;

    // Statistics and monitoring
    ParallelExecutionStats m_executionStats;
    QElapsedTimer m_executionTimer;
    QTimer* m_progressTimer;
    QMutex m_statsMutex;

    // Result aggregation
    QMap<QString, QVariant> m_aggregatedResults;
    QMutex m_resultsMutex;
};