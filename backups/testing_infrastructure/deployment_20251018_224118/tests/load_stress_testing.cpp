#include "load_stress_testing.h"
#include "performance_benchmark.h"
#include "test_data_generator.h"
#include <QApplication>
#include <QWidget>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QJsonArray>
#include <QJsonDocument>
#include <QCryptographicHash>
#include <QThread>
#include <QProcess>
#include <QSysInfo>
#include <QDebug>
#include <QtMath>
#include <QRandomGenerator>
#include <QDirIterator>
#include <QTemporaryDir>
#include <algorithm>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#elif defined(Q_OS_LINUX)
#include <unistd.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#elif defined(Q_OS_MAC)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

LoadStressTesting::LoadStressTesting(QObject* parent)
    : QObject(parent)
    , m_maxConcurrentThreads(QThread::idealThreadCount())
    , m_resourceMonitoringInterval(1000)
    , m_performanceBenchmark(nullptr)
    , m_testDataGenerator(nullptr)
    , m_resourceTimer(new QTimer(this))
    , m_resourceMonitoringActive(false)
    , m_threadPool(new QThreadPool(this))
    , m_activeOperations(0)
{
    // Configure thread pool
    m_threadPool->setMaxThreadCount(m_maxConcurrentThreads);
    
    // Setup resource monitoring timer
    connect(m_resourceTimer, &QTimer::timeout, this, &LoadStressTesting::onResourceMonitoringTimer);
    
    qDebug() << "LoadStressTesting initialized with" << m_maxConcurrentThreads << "max threads";
}

LoadStressTesting::~LoadStressTesting() {
    // Wait for all operations to complete
    m_threadPool->waitForDone(30000); // 30 second timeout
    
    if (m_resourceMonitoringActive) {
        m_resourceTimer->stop();
    }
}

void LoadStressTesting::setPerformanceBenchmark(PerformanceBenchmark* benchmark) {
    m_performanceBenchmark = benchmark;
}

void LoadStressTesting::setTestDataGenerator(TestDataGenerator* generator) {
    m_testDataGenerator = generator;
}

void LoadStressTesting::setMaxConcurrentThreads(int maxThreads) {
    m_maxConcurrentThreads = maxThreads;
    m_threadPool->setMaxThreadCount(maxThreads);
}

void LoadStressTesting::setResourceMonitoringInterval(int intervalMs) {
    m_resourceMonitoringInterval = intervalMs;
}

bool LoadStressTesting::runLoadTest(const LoadTestConfig& config, std::function<void()> operation) {
    if (!operation) {
        qWarning() << "Invalid operation function for load test:" << config.name;
        return false;
    }
    
    emit loadTestStarted(config.name);
    
    // Convert operation to indexed operation
    auto indexedOperation = [operation](int) { operation(); };
    
    LoadTestResult result = executeLoadTest(config, indexedOperation);
    recordLoadTestResult(result);
    
    emit loadTestCompleted(config.name, result);
    return result.completedSuccessfully;
}

bool LoadStressTesting::runConcurrentUserTest(int userCount, int operationsPerUser, std::function<void(int)> userOperation) {
    LoadTestConfig config;
    config.name = QString("concurrent_users_%1_users_%2_ops").arg(userCount).arg(operationsPerUser);
    config.type = LoadTestType::ConcurrentUsers;
    config.concurrentThreads = userCount;
    config.totalOperations = userCount * operationsPerUser;
    config.description = QString("Concurrent user test with %1 users performing %2 operations each").arg(userCount).arg(operationsPerUser);
    
    return runLoadTest(config, [userOperation, operationsPerUser]() {
        static thread_local int userId = 0;
        for (int i = 0; i < operationsPerUser; ++i) {
            userOperation(userId);
        }
        userId++;
    });
}

bool LoadStressTesting::runHighVolumeFileTest(int fileCount, qint64 fileSize, const QString& testDirectory) {
    LoadTestConfig config;
    config.name = QString("high_volume_files_%1_files_%2_bytes").arg(fileCount).arg(fileSize);
    config.type = LoadTestType::HighVolumeFiles;
    config.totalOperations = fileCount;
    config.description = QString("High volume file test with %1 files of %2 bytes each").arg(fileCount).arg(fileSize);
    
    // Create test directory
    QDir().mkpath(testDirectory);
    
    return runLoadTest(config, [this, testDirectory, fileSize]() {
        static QAtomicInt fileCounter(0);
        int fileIndex = fileCounter.fetchAndAddAcquire(1);
        
        QString fileName = QString("load_test_file_%1.dat").arg(fileIndex);
        QString filePath = QDir(testDirectory).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(fileSize, 'L');
            file.write(data);
        }
    });
}

bool LoadStressTesting::runLargeFileSizeTest(qint64 maxFileSize, const QString& testDirectory) {
    LoadTestConfig config;
    config.name = QString("large_file_size_%1_bytes").arg(maxFileSize);
    config.type = LoadTestType::LargeFileSize;
    config.totalOperations = 10; // Test with 10 large files
    config.description = QString("Large file size test with files up to %1 bytes").arg(maxFileSize);
    
    QDir().mkpath(testDirectory);
    
    return runLoadTest(config, [this, testDirectory, maxFileSize]() {
        static QAtomicInt fileCounter(0);
        int fileIndex = fileCounter.fetchAndAddAcquire(1);
        
        // Vary file sizes from 10% to 100% of maxFileSize
        qint64 fileSize = (maxFileSize / 10) * (1 + (fileIndex % 10));
        
        QString fileName = QString("large_file_%1.dat").arg(fileIndex);
        QString filePath = QDir(testDirectory).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            // Write in chunks to avoid memory issues
            const qint64 chunkSize = 1024 * 1024; // 1MB chunks
            QByteArray chunk(chunkSize, 'L');
            
            qint64 remaining = fileSize;
            while (remaining > 0) {
                qint64 writeSize = qMin(remaining, chunkSize);
                if (writeSize < chunkSize) {
                    chunk.resize(writeSize);
                }
                file.write(chunk);
                remaining -= writeSize;
            }
        }
    });
}

bool LoadStressTesting::runDeepDirectoryTest(int maxDepth, int filesPerLevel, const QString& testDirectory) {
    LoadTestConfig config;
    config.name = QString("deep_directory_%1_depth_%2_files").arg(maxDepth).arg(filesPerLevel);
    config.type = LoadTestType::DeepDirectories;
    config.totalOperations = maxDepth * filesPerLevel;
    config.description = QString("Deep directory test with %1 levels and %2 files per level").arg(maxDepth).arg(filesPerLevel);
    
    return runLoadTest(config, [this, testDirectory, maxDepth, filesPerLevel]() {
        if (!createDeepDirectoryStructure(testDirectory, maxDepth, filesPerLevel)) {
            throw std::runtime_error("Failed to create deep directory structure");
        }
        
        // Scan the created structure
        QDirIterator iterator(testDirectory, QDir::Files, QDirIterator::Subdirectories);
        int fileCount = 0;
        while (iterator.hasNext()) {
            iterator.next();
            fileCount++;
        }
    });
}

bool LoadStressTesting::runWideDirectoryTest(int filesPerDirectory, int directoryCount, const QString& testDirectory) {
    LoadTestConfig config;
    config.name = QString("wide_directory_%1_files_%2_dirs").arg(filesPerDirectory).arg(directoryCount);
    config.type = LoadTestType::WideDirectories;
    config.totalOperations = filesPerDirectory * directoryCount;
    config.description = QString("Wide directory test with %1 directories containing %2 files each").arg(directoryCount).arg(filesPerDirectory);
    
    return runLoadTest(config, [this, testDirectory, filesPerDirectory, directoryCount]() {
        if (!createWideDirectoryStructure(testDirectory, filesPerDirectory, directoryCount)) {
            throw std::runtime_error("Failed to create wide directory structure");
        }
        
        // Scan all directories
        QDir baseDir(testDirectory);
        QStringList subdirs = baseDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        
        for (const QString& subdir : subdirs) {
            QDir dir(baseDir.absoluteFilePath(subdir));
            QStringList files = dir.entryList(QDir::Files);
            // Process files (simulate work)
            for (const QString& file : files) {
                QFileInfo info(dir.absoluteFilePath(file));
                info.size(); // Access file info
            }
        }
    });
}

bool LoadStressTesting::runSustainedLoadTest(qint64 durationMs, int operationsPerSecond, std::function<void()> operation) {
    LoadTestConfig config;
    config.name = QString("sustained_load_%1ms_%2ops").arg(durationMs).arg(operationsPerSecond);
    config.type = LoadTestType::SustainedLoad;
    config.durationMs = durationMs;
    config.operationsPerSecond = operationsPerSecond;
    config.totalOperations = (durationMs / 1000) * operationsPerSecond;
    config.description = QString("Sustained load test for %1ms at %2 operations per second").arg(durationMs).arg(operationsPerSecond);
    
    return runLoadTest(config, operation);
}

bool LoadStressTesting::runBurstLoadTest(int burstCount, int operationsPerBurst, qint64 burstIntervalMs, std::function<void()> operation) {
    LoadTestConfig config;
    config.name = QString("burst_load_%1_bursts_%2_ops_%3ms").arg(burstCount).arg(operationsPerBurst).arg(burstIntervalMs);
    config.type = LoadTestType::BurstLoad;
    config.totalOperations = burstCount * operationsPerBurst;
    config.description = QString("Burst load test with %1 bursts of %2 operations every %3ms").arg(burstCount).arg(operationsPerBurst).arg(burstIntervalMs);
    
    return runLoadTest(config, [operation, operationsPerBurst, burstIntervalMs]() {
        static QElapsedTimer lastBurst;
        static QAtomicInt burstOperationCount(0);
        
        // Wait for burst interval if needed
        if (lastBurst.isValid() && lastBurst.elapsed() < burstIntervalMs) {
            QThread::msleep(burstIntervalMs - lastBurst.elapsed());
        }
        
        // Execute burst operations
        for (int i = 0; i < operationsPerBurst; ++i) {
            operation();
        }
        
        burstOperationCount.fetchAndAddAcquire(operationsPerBurst);
        lastBurst.restart();
    });
}

bool LoadStressTesting::runGradualRampTest(int startThreads, int endThreads, qint64 rampDurationMs, std::function<void()> operation) {
    LoadTestConfig config;
    config.name = QString("gradual_ramp_%1_to_%2_threads_%3ms").arg(startThreads).arg(endThreads).arg(rampDurationMs);
    config.type = LoadTestType::GradualRamp;
    config.concurrentThreads = startThreads;
    config.rampUpTimeMs = rampDurationMs;
    config.durationMs = rampDurationMs * 2; // Ramp up + steady state
    config.description = QString("Gradual ramp test from %1 to %2 threads over %3ms").arg(startThreads).arg(endThreads).arg(rampDurationMs);
    
    // Custom implementation for gradual ramping
    emit loadTestStarted(config.name);
    
    LoadTestResult result;
    result.testName = config.name;
    result.testType = config.type;
    result.startTime = QDateTime::currentDateTime();
    result.concurrentThreadsUsed = startThreads;
    
    QElapsedTimer testTimer;
    testTimer.start();
    
    QAtomicInt successCount(0);
    QAtomicInt failureCount(0);
    QList<double> responseTimes;
    QMutex responseTimesMutex;
    
    // Start resource monitoring
    m_resourceMonitoringActive = true;
    m_resourceTimer->start(m_resourceMonitoringInterval);
    
    // Gradually increase thread count
    QElapsedTimer rampTimer;
    rampTimer.start();
    
    int currentThreads = startThreads;
    while (rampTimer.elapsed() < rampDurationMs && currentThreads < endThreads) {
        // Calculate how many threads to add
        double progress = static_cast<double>(rampTimer.elapsed()) / rampDurationMs;
        int targetThreads = startThreads + static_cast<int>((endThreads - startThreads) * progress);
        
        // Add threads if needed
        while (currentThreads < targetThreads) {
            LoadTestWorker* worker = new LoadTestWorker(operation, &successCount, &failureCount, &responseTimes, &responseTimesMutex);
            m_threadPool->start(worker);
            currentThreads++;
        }
        
        QThread::msleep(100); // Check every 100ms
    }
    
    // Wait for operations to complete
    m_threadPool->waitForDone();
    
    // Stop resource monitoring
    m_resourceMonitoringActive = false;
    m_resourceTimer->stop();
    
    result.endTime = QDateTime::currentDateTime();
    result.actualDurationMs = testTimer.elapsed();
    result.successfulOperations = successCount.loadAcquire();
    result.failedOperations = failureCount.loadAcquire();
    result.totalOperations = result.successfulOperations + result.failedOperations;
    result.completedSuccessfully = result.failedOperations == 0;
    
    if (result.actualDurationMs > 0) {
        result.operationsPerSecond = (result.totalOperations * 1000.0) / result.actualDurationMs;
    }
    
    // Calculate response time statistics
    if (!responseTimes.isEmpty()) {
        result.responseTimes = responseTimes;
        result.averageResponseTime = calculateMean(responseTimes);
        result.minResponseTime = *std::min_element(responseTimes.begin(), responseTimes.end());
        result.maxResponseTime = *std::max_element(responseTimes.begin(), responseTimes.end());
        result.percentile95ResponseTime = calculatePercentile(responseTimes, 95.0);
        result.percentile99ResponseTime = calculatePercentile(responseTimes, 99.0);
    }
    
    recordLoadTestResult(result);
    emit loadTestCompleted(config.name, result);
    
    return result.completedSuccessfully;
}bool Loa
dStressTesting::runStressTest(const StressTestConfig& config, std::function<void()> operation) {
    if (!operation) {
        qWarning() << "Invalid operation function for stress test:" << config.name;
        return false;
    }
    
    emit stressTestStarted(config.name);
    
    StressTestResult result = executeStressTest(config, operation);
    recordStressTestResult(result);
    
    emit stressTestCompleted(config.name, result);
    return result.completedSuccessfully;
}

bool LoadStressTesting::runStressToFailureTest(std::function<void()> operation, const StressTestConfig& config) {
    StressTestConfig stressConfig = config;
    stressConfig.name = "stress_to_failure";
    stressConfig.stopOnFirstFailure = false;
    stressConfig.stopOnResourceLimit = true;
    stressConfig.description = "Stress test that pushes system until failure point";
    
    return runStressTest(stressConfig, operation);
}

bool LoadStressTesting::runMemoryStressTest(qint64 maxMemoryMB, std::function<void()> operation) {
    StressTestConfig config;
    config.name = QString("memory_stress_%1MB").arg(maxMemoryMB);
    config.memoryLimitMB = maxMemoryMB;
    config.stopOnResourceLimit = true;
    config.description = QString("Memory stress test with %1MB limit").arg(maxMemoryMB);
    
    return runStressTest(config, operation);
}

bool LoadStressTesting::runCpuStressTest(double maxCpuPercent, std::function<void()> operation) {
    StressTestConfig config;
    config.name = QString("cpu_stress_%1_percent").arg(maxCpuPercent);
    config.cpuLimitPercent = maxCpuPercent;
    config.stopOnResourceLimit = true;
    config.description = QString("CPU stress test with %1% limit").arg(maxCpuPercent);
    
    return runStressTest(config, operation);
}

bool LoadStressTesting::runConcurrencyStressTest(int maxConcurrentOperations, std::function<void()> operation) {
    StressTestConfig config;
    config.name = QString("concurrency_stress_%1_ops").arg(maxConcurrentOperations);
    config.maxConcurrentOperations = maxConcurrentOperations;
    config.description = QString("Concurrency stress test with %1 concurrent operations").arg(maxConcurrentOperations);
    
    return runStressTest(config, operation);
}

bool LoadStressTesting::runResourceExhaustionTest(const StressTestConfig& config, std::function<void()> operation) {
    StressTestConfig exhaustionConfig = config;
    exhaustionConfig.name = "resource_exhaustion";
    exhaustionConfig.stopOnResourceLimit = true;
    exhaustionConfig.description = "Resource exhaustion test to find system limits";
    
    return runStressTest(exhaustionConfig, operation);
}

bool LoadStressTesting::runScalabilityTest(const ScalabilityTestConfig& config, std::function<void(int, qint64, int)> operation) {
    if (!operation) {
        qWarning() << "Invalid operation function for scalability test:" << config.name;
        return false;
    }
    
    emit scalabilityTestStarted(config.name);
    
    ScalabilityTestResult result = executeScalabilityTest(config, operation);
    recordScalabilityTestResult(result);
    
    emit scalabilityTestCompleted(config.name, result);
    return result.completedSuccessfully;
}

bool LoadStressTesting::runFileCountScalabilityTest(const QList<int>& fileCounts, std::function<void(int)> operation) {
    ScalabilityTestConfig config;
    config.name = "file_count_scalability";
    config.fileCounts = fileCounts;
    config.description = QString("File count scalability test with counts: %1").arg(QStringList(fileCounts.begin(), fileCounts.end()).join(", "));
    
    return runScalabilityTest(config, [operation](int fileCount, qint64, int) {
        operation(fileCount);
    });
}

bool LoadStressTesting::runFileSizeScalabilityTest(const QList<qint64>& fileSizes, std::function<void(qint64)> operation) {
    ScalabilityTestConfig config;
    config.name = "file_size_scalability";
    config.fileSizes = fileSizes;
    config.description = "File size scalability test";
    
    return runScalabilityTest(config, [operation](int, qint64 fileSize, int) {
        operation(fileSize);
    });
}

bool LoadStressTesting::runThreadScalabilityTest(const QList<int>& threadCounts, std::function<void(int)> operation) {
    ScalabilityTestConfig config;
    config.name = "thread_scalability";
    config.threadCounts = threadCounts;
    config.description = QString("Thread scalability test with counts: %1").arg(QStringList(threadCounts.begin(), threadCounts.end()).join(", "));
    
    return runScalabilityTest(config, [operation](int, qint64, int threadCount) {
        operation(threadCount);
    });
}

bool LoadStressTesting::runDuplicateDetectionLoadTest(const QString& testDirectory, int fileCount, double duplicateRatio) {
    LoadTestConfig config;
    config.name = QString("duplicate_detection_load_%1_files_%2_ratio").arg(fileCount).arg(duplicateRatio);
    config.type = LoadTestType::MixedWorkload;
    config.totalOperations = fileCount;
    config.description = QString("Duplicate detection load test with %1 files and %2 duplicate ratio").arg(fileCount).arg(duplicateRatio);
    
    // Create test files with duplicates
    QDir().mkpath(testDirectory);
    
    return runLoadTest(config, [this, testDirectory, fileCount, duplicateRatio]() {
        if (m_testDataGenerator) {
            m_testDataGenerator->createDuplicateTestSet(testDirectory, fileCount, 10240, duplicateRatio);
        }
        
        // Simulate duplicate detection
        QDir dir(testDirectory);
        QStringList files = dir.entryList(QDir::Files);
        QMap<QString, QStringList> hashGroups;
        
        for (const QString& fileName : files) {
            QString filePath = dir.absoluteFilePath(fileName);
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(QCryptographicHash::Md5);
                hash.addData(&file);
                QString hashString = hash.result().toHex();
                hashGroups[hashString].append(filePath);
            }
        }
        
        // Count duplicates
        int duplicateGroups = 0;
        for (auto it = hashGroups.begin(); it != hashGroups.end(); ++it) {
            if (it.value().size() > 1) {
                duplicateGroups++;
            }
        }
    });
}

bool LoadStressTesting::runHashCalculationLoadTest(const QStringList& filePaths, int concurrentHashers) {
    LoadTestConfig config;
    config.name = QString("hash_calculation_load_%1_files_%2_hashers").arg(filePaths.size()).arg(concurrentHashers);
    config.type = LoadTestType::ConcurrentUsers;
    config.concurrentThreads = concurrentHashers;
    config.totalOperations = filePaths.size();
    config.description = QString("Hash calculation load test with %1 concurrent hashers processing %2 files").arg(concurrentHashers).arg(filePaths.size());
    
    return runLoadTest(config, [filePaths]() {
        static QAtomicInt fileIndex(0);
        int index = fileIndex.fetchAndAddAcquire(1);
        
        if (index < filePaths.size()) {
            QString filePath = filePaths[index];
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(QCryptographicHash::Sha256);
                hash.addData(&file);
                hash.result(); // Force calculation
            }
        }
    });
}

bool LoadStressTesting::runFileScanningLoadTest(const QString& rootDirectory, int concurrentScanners) {
    LoadTestConfig config;
    config.name = QString("file_scanning_load_%1_scanners").arg(concurrentScanners);
    config.type = LoadTestType::ConcurrentUsers;
    config.concurrentThreads = concurrentScanners;
    config.totalOperations = concurrentScanners * 10; // Each scanner does 10 scans
    config.description = QString("File scanning load test with %1 concurrent scanners").arg(concurrentScanners);
    
    return runLoadTest(config, [rootDirectory]() {
        QDirIterator iterator(rootDirectory, QDir::Files, QDirIterator::Subdirectories);
        int fileCount = 0;
        while (iterator.hasNext()) {
            iterator.next();
            fileCount++;
            
            // Simulate some processing time
            if (fileCount % 100 == 0) {
                QThread::msleep(1);
            }
        }
    });
}

bool LoadStressTesting::runUILoadTest(QWidget* mainWindow, int concurrentUIOperations) {
    if (!mainWindow) {
        return false;
    }
    
    LoadTestConfig config;
    config.name = QString("ui_load_%1_operations").arg(concurrentUIOperations);
    config.type = LoadTestType::UIResponsiveness;
    config.concurrentThreads = 1; // UI operations must be on main thread
    config.totalOperations = concurrentUIOperations;
    config.description = QString("UI load test with %1 operations").arg(concurrentUIOperations);
    
    return runLoadTest(config, [mainWindow]() {
        // Simulate UI operations
        mainWindow->update();
        QApplication::processEvents();
        
        // Simulate user interactions
        QTest::qWait(1); // Small delay to simulate real user interaction timing
    });
}

LoadStressTesting::LoadTestResult LoadStressTesting::executeLoadTest(const LoadTestConfig& config, std::function<void(int)> operation) {
    LoadTestResult result;
    result.testName = config.name;
    result.testType = config.type;
    result.startTime = QDateTime::currentDateTime();
    result.concurrentThreadsUsed = config.concurrentThreads;
    
    QElapsedTimer testTimer;
    testTimer.start();
    
    QAtomicInt successCount(0);
    QAtomicInt failureCount(0);
    QList<double> responseTimes;
    QMutex responseTimesMutex;
    
    // Start resource monitoring
    m_resourceMonitoringActive = true;
    m_resourceTimer->start(m_resourceMonitoringInterval);
    
    // Execute operations
    if (config.type == LoadTestType::SustainedLoad) {
        // Sustained load with controlled rate
        QElapsedTimer sustainedTimer;
        sustainedTimer.start();
        
        qint64 operationInterval = 1000 / config.operationsPerSecond; // ms between operations
        int operationCount = 0;
        
        while (sustainedTimer.elapsed() < config.durationMs && operationCount < config.totalOperations) {
            QElapsedTimer operationTimer;
            operationTimer.start();
            
            try {
                operation(operationCount);
                successCount.fetchAndAddAcquire(1);
            } catch (const std::exception& e) {
                failureCount.fetchAndAddAcquire(1);
                if (config.failOnError) {
                    result.errorMessage = e.what();
                    break;
                }
            }
            
            qint64 operationTime = operationTimer.elapsed();
            {
                QMutexLocker locker(&responseTimesMutex);
                responseTimes.append(operationTime);
            }
            
            operationCount++;
            
            // Wait for next operation if needed
            qint64 elapsed = operationTimer.elapsed();
            if (elapsed < operationInterval) {
                QThread::msleep(operationInterval - elapsed);
            }
            
            emit testProgress(config.name, operationCount, config.totalOperations);
        }
    } else {
        // Concurrent execution
        QList<LoadTestWorker*> workers;
        
        for (int i = 0; i < config.totalOperations; ++i) {
            LoadTestWorker* worker = new LoadTestWorker(
                [operation, i]() { operation(i); },
                &successCount,
                &failureCount,
                &responseTimes,
                &responseTimesMutex
            );
            
            workers.append(worker);
            m_threadPool->start(worker);
            
            // Control concurrency
            while (m_threadPool->activeThreadCount() >= config.concurrentThreads) {
                QThread::msleep(1);
            }
            
            emit testProgress(config.name, i + 1, config.totalOperations);
        }
        
        // Wait for all operations to complete
        m_threadPool->waitForDone(config.timeoutMs);
    }
    
    // Stop resource monitoring
    m_resourceMonitoringActive = false;
    m_resourceTimer->stop();
    
    result.endTime = QDateTime::currentDateTime();
    result.actualDurationMs = testTimer.elapsed();
    result.successfulOperations = successCount.loadAcquire();
    result.failedOperations = failureCount.loadAcquire();
    result.totalOperations = result.successfulOperations + result.failedOperations;
    result.completedSuccessfully = result.failedOperations == 0 && !result.errorMessage.isEmpty() == false;
    
    if (result.actualDurationMs > 0) {
        result.operationsPerSecond = (result.totalOperations * 1000.0) / result.actualDurationMs;
    }
    
    // Calculate response time statistics
    if (!responseTimes.isEmpty()) {
        result.responseTimes = responseTimes;
        result.averageResponseTime = calculateMean(responseTimes);
        result.minResponseTime = *std::min_element(responseTimes.begin(), responseTimes.end());
        result.maxResponseTime = *std::max_element(responseTimes.begin(), responseTimes.end());
        result.percentile95ResponseTime = calculatePercentile(responseTimes, 95.0);
        result.percentile99ResponseTime = calculatePercentile(responseTimes, 99.0);
    }
    
    // Collect resource usage
    result.peakMemoryUsage = getCurrentMemoryUsage();
    result.averageCpuUsage = getCurrentCpuUsage();
    result.peakCpuUsage = result.averageCpuUsage; // Simplified for now
    
    return result;
}LoadS
tressTesting::StressTestResult LoadStressTesting::executeStressTest(const StressTestConfig& config, std::function<void()> operation) {
    StressTestResult result;
    result.testName = config.name;
    result.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer testTimer;
    testTimer.start();
    
    // Start resource monitoring
    m_resourceMonitoringActive = true;
    m_resourceTimer->start(m_resourceMonitoringInterval);
    
    int operationCount = 0;
    int failureCount = 0;
    int concurrentOperations = 0;
    
    while (testTimer.elapsed() < config.maxTestDurationMs) {
        // Check resource limits
        if (config.stopOnResourceLimit && !checkResourceLimits(config, result)) {
            result.failureReason = "Resource limit exceeded";
            break;
        }
        
        // Check failure threshold
        if (failureCount >= config.failureThreshold) {
            result.failureReason = QString("Failure threshold exceeded (%1 failures)").arg(failureCount);
            break;
        }
        
        // Execute operation if we haven't hit concurrency limit
        if (concurrentOperations < config.maxConcurrentOperations) {
            LoadTestWorker* worker = new LoadTestWorker(
                operation,
                nullptr, // We'll track success/failure differently for stress tests
                nullptr,
                nullptr,
                nullptr
            );
            
            connect(worker, &LoadTestWorker::operationCompleted, this, [&](qint64 responseTime, bool success) {
                concurrentOperations--;
                if (!success) {
                    failureCount++;
                    result.errorMessages.append(QString("Operation failed at %1ms").arg(testTimer.elapsed()));
                }
            });
            
            m_threadPool->start(worker);
            concurrentOperations++;
            operationCount++;
            
            // Update maximum concurrent operations reached
            result.maxConcurrentOperationsReached = qMax(result.maxConcurrentOperationsReached, concurrentOperations);
        }
        
        // Monitor resource usage
        monitorResourceUsage(result);
        
        // Small delay to prevent tight loop
        QThread::msleep(1);
        
        emit testProgress(config.name, operationCount, config.maxConcurrentOperations);
    }
    
    // Wait for remaining operations to complete
    m_threadPool->waitForDone(5000); // 5 second timeout
    
    // Stop resource monitoring
    m_resourceMonitoringActive = false;
    m_resourceTimer->stop();
    
    result.endTime = QDateTime::currentDateTime();
    result.actualDurationMs = testTimer.elapsed();
    result.totalFailures = failureCount;
    result.completedSuccessfully = failureCount < config.failureThreshold && result.failureReason.isEmpty();
    
    // Check if we hit time limit
    if (testTimer.elapsed() >= config.maxTestDurationMs) {
        result.hitTimeLimit = true;
        if (result.failureReason.isEmpty()) {
            result.failureReason = "Time limit reached";
        }
    }
    
    return result;
}

bool LoadStressTesting::checkResourceLimits(const StressTestConfig& config, StressTestResult& result) {
    qint64 currentMemoryMB = getCurrentMemoryUsage() / (1024 * 1024);
    double currentCpuPercent = getCurrentCpuUsage();
    
    if (currentMemoryMB > config.memoryLimitMB) {
        result.hitMemoryLimit = true;
        result.peakMemoryUsageMB = currentMemoryMB;
        emit resourceLimitReached(config.name, "memory", currentMemoryMB, config.memoryLimitMB);
        return false;
    }
    
    if (currentCpuPercent > config.cpuLimitPercent) {
        result.hitCpuLimit = true;
        result.peakCpuUsagePercent = currentCpuPercent;
        emit resourceLimitReached(config.name, "cpu", currentCpuPercent, config.cpuLimitPercent);
        return false;
    }
    
    return true;
}

void LoadStressTesting::monitorResourceUsage(StressTestResult& result) {
    qint64 currentMemoryMB = getCurrentMemoryUsage() / (1024 * 1024);
    double currentCpuPercent = getCurrentCpuUsage();
    
    result.peakMemoryUsageMB = qMax(result.peakMemoryUsageMB, currentMemoryMB);
    result.peakCpuUsagePercent = qMax(result.peakCpuUsagePercent, currentCpuPercent);
    
    // Store in resource metrics
    result.resourceMetrics["current_memory_mb"] = currentMemoryMB;
    result.resourceMetrics["current_cpu_percent"] = currentCpuPercent;
    result.resourceMetrics["active_threads"] = getCurrentActiveThreads();
}

LoadStressTesting::ScalabilityTestResult LoadStressTesting::executeScalabilityTest(const ScalabilityTestConfig& config, std::function<void(int, qint64, int)> operation) {
    ScalabilityTestResult result;
    result.testName = config.name;
    result.startTime = QDateTime::currentDateTime();
    
    QList<double> timeValues, memoryValues, throughputValues;
    QList<double> inputValues;
    
    // Test different configurations
    for (int fileCount : config.fileCounts) {
        for (qint64 fileSize : config.fileSizes) {
            for (int threadCount : config.threadCounts) {
                QString configName = QString("%1_files_%2_size_%3_threads").arg(fileCount).arg(fileSize).arg(threadCount);
                
                // Run multiple iterations for this configuration
                QList<double> iterationTimes, iterationMemory, iterationThroughput;
                
                for (int iteration = 0; iteration < config.iterationsPerConfiguration; ++iteration) {
                    QElapsedTimer iterationTimer;
                    iterationTimer.start();
                    
                    qint64 memoryBefore = getCurrentMemoryUsage();
                    
                    try {
                        operation(fileCount, fileSize, threadCount);
                    } catch (const std::exception& e) {
                        result.completedSuccessfully = false;
                        qWarning() << "Scalability test iteration failed:" << e.what();
                        continue;
                    }
                    
                    qint64 iterationTime = iterationTimer.elapsed();
                    qint64 memoryAfter = getCurrentMemoryUsage();
                    qint64 memoryUsed = memoryAfter - memoryBefore;
                    
                    iterationTimes.append(iterationTime);
                    iterationMemory.append(memoryUsed);
                    
                    if (iterationTime > 0) {
                        double throughput = (fileCount * 1000.0) / iterationTime; // files per second
                        iterationThroughput.append(throughput);
                    }
                    
                    emit testProgress(config.name, iteration + 1, config.iterationsPerConfiguration);
                }
                
                // Calculate averages for this configuration
                if (!iterationTimes.isEmpty()) {
                    LoadTestResult configResult;
                    configResult.testName = configName;
                    configResult.averageResponseTime = calculateMean(iterationTimes);
                    configResult.peakMemoryUsage = calculateMean(iterationMemory);
                    configResult.operationsPerSecond = calculateMean(iterationThroughput);
                    configResult.completedSuccessfully = true;
                    
                    result.results[configName] = configResult;
                    
                    // Collect data for scaling analysis
                    double inputComplexity = fileCount * fileSize * threadCount;
                    inputValues.append(inputComplexity);
                    timeValues.append(configResult.averageResponseTime);
                    memoryValues.append(configResult.peakMemoryUsage);
                    throughputValues.append(configResult.operationsPerSecond);
                }
            }
        }
    }
    
    result.endTime = QDateTime::currentDateTime();
    
    // Analyze scaling behavior
    if (inputValues.size() > 1) {
        if (config.measureTimeScaling) {
            result.scalingFactors["time"] = calculateScalingFactor(timeValues, inputValues);
            result.linearTimeScaling = qAbs(result.scalingFactors["time"] - 1.0) < 0.2; // Within 20% of linear
        }
        
        if (config.measureMemoryScaling) {
            result.scalingFactors["memory"] = calculateScalingFactor(memoryValues, inputValues);
            result.linearMemoryScaling = qAbs(result.scalingFactors["memory"] - 1.0) < 0.2;
        }
        
        if (config.measureThroughputScaling) {
            result.scalingFactors["throughput"] = calculateScalingFactor(throughputValues, inputValues);
            result.linearThroughputScaling = qAbs(result.scalingFactors["throughput"] - 1.0) < 0.2;
        }
        
        result.scalingAnalysis = analyzeScalingBehavior(result.scalingFactors);
        result.completedSuccessfully = true;
    }
    
    return result;
}

double LoadStressTesting::calculateScalingFactor(const QList<double>& values, const QList<double>& inputs) const {
    if (values.size() != inputs.size() || values.size() < 2) {
        return 0.0;
    }
    
    // Simple linear regression to find scaling factor
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    int n = values.size();
    
    for (int i = 0; i < n; ++i) {
        double x = qLn(inputs[i]); // Log of input complexity
        double y = qLn(values[i]); // Log of measured value
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
    }
    
    // Calculate slope (scaling factor)
    double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
}

QString LoadStressTesting::analyzeScalingBehavior(const QMap<QString, double>& scalingFactors) const {
    QStringList analysis;
    
    for (auto it = scalingFactors.begin(); it != scalingFactors.end(); ++it) {
        QString metric = it.key();
        double factor = it.value();
        
        QString behavior;
        if (qAbs(factor - 1.0) < 0.1) {
            behavior = "linear";
        } else if (factor < 1.0) {
            behavior = "sub-linear (better than linear)";
        } else if (factor > 1.0 && factor < 2.0) {
            behavior = "super-linear";
        } else if (qAbs(factor - 2.0) < 0.1) {
            behavior = "quadratic";
        } else {
            behavior = QString("exponential (factor: %1)").arg(factor, 0, 'f', 2);
        }
        
        analysis.append(QString("%1: %2").arg(metric, behavior));
    }
    
    return analysis.join("; ");
}

void LoadStressTesting::onResourceMonitoringTimer() {
    if (!m_resourceMonitoringActive) {
        return;
    }
    
    m_currentResourceUsage = collectResourceMetrics();
    emit resourceMonitoringUpdate(m_currentResourceUsage);
}

QMap<QString, QVariant> LoadStressTesting::collectResourceMetrics() const {
    QMap<QString, QVariant> metrics;
    
    metrics["memory_usage"] = getCurrentMemoryUsage();
    metrics["cpu_usage"] = getCurrentCpuUsage();
    metrics["active_threads"] = getCurrentActiveThreads();
    metrics["timestamp"] = QDateTime::currentDateTime();
    
    return metrics;
}

qint64 LoadStressTesting::getCurrentMemoryUsage() const {
#ifdef Q_OS_WIN
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#elif defined(Q_OS_LINUX)
    QFile file("/proc/self/status");
    if (file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        while (stream.readLineInto(&line)) {
            if (line.startsWith("VmRSS:")) {
                QStringList parts = line.split(QRegExp("\\s+"));
                if (parts.size() >= 2) {
                    return parts[1].toLongLong() * 1024; // Convert KB to bytes
                }
            }
        }
    }
#elif defined(Q_OS_MAC)
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

double LoadStressTesting::getCurrentCpuUsage() const {
    // Simplified CPU usage - in real implementation would measure actual CPU usage
    return QRandomGenerator::global()->bounded(100.0);
}

int LoadStressTesting::getCurrentActiveThreads() const {
    return m_threadPool->activeThreadCount();
}

bool LoadStressTesting::createTestFileStructure(const QString& baseDirectory, int fileCount, qint64 fileSize) {
    QDir().mkpath(baseDirectory);
    
    for (int i = 0; i < fileCount; ++i) {
        QString fileName = QString("test_file_%1.dat").arg(i);
        QString filePath = QDir(baseDirectory).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (!file.open(QIODevice::WriteOnly)) {
            return false;
        }
        
        QByteArray data(fileSize, 'T');
        if (file.write(data) != fileSize) {
            return false;
        }
    }
    
    return true;
}

bool LoadStressTesting::createDeepDirectoryStructure(const QString& baseDirectory, int depth, int filesPerLevel) {
    QString currentDir = baseDirectory;
    QDir().mkpath(currentDir);
    
    for (int level = 0; level < depth; ++level) {
        // Create files at this level
        for (int file = 0; file < filesPerLevel; ++file) {
            QString fileName = QString("level_%1_file_%2.txt").arg(level).arg(file);
            QString filePath = QDir(currentDir).absoluteFilePath(fileName);
            
            QFile testFile(filePath);
            if (testFile.open(QIODevice::WriteOnly)) {
                testFile.write(QString("Level %1 File %2 Content").arg(level).arg(file).toUtf8());
            }
        }
        
        // Create subdirectory for next level
        if (level < depth - 1) {
            currentDir = QDir(currentDir).absoluteFilePath(QString("level_%1").arg(level + 1));
            QDir().mkpath(currentDir);
        }
    }
    
    return true;
}

bool LoadStressTesting::createWideDirectoryStructure(const QString& baseDirectory, int filesPerDirectory, int directoryCount) {
    QDir().mkpath(baseDirectory);
    
    for (int dir = 0; dir < directoryCount; ++dir) {
        QString dirName = QString("dir_%1").arg(dir);
        QString dirPath = QDir(baseDirectory).absoluteFilePath(dirName);
        QDir().mkpath(dirPath);
        
        for (int file = 0; file < filesPerDirectory; ++file) {
            QString fileName = QString("file_%1.txt").arg(file);
            QString filePath = QDir(dirPath).absoluteFilePath(fileName);
            
            QFile testFile(filePath);
            if (testFile.open(QIODevice::WriteOnly)) {
                testFile.write(QString("Directory %1 File %2 Content").arg(dir).arg(file).toUtf8());
            }
        }
    }
    
    return true;
}

void LoadStressTesting::cleanupTestDirectory(const QString& directory) {
    QDir dir(directory);
    if (dir.exists()) {
        dir.removeRecursively();
    }
}//
 Result management methods
QList<LoadStressTesting::LoadTestResult> LoadStressTesting::getLoadTestResults() const {
    QMutexLocker locker(&m_resultsMutex);
    return m_loadTestResults;
}

QList<LoadStressTesting::StressTestResult> LoadStressTesting::getStressTestResults() const {
    QMutexLocker locker(&m_resultsMutex);
    return m_stressTestResults;
}

QList<LoadStressTesting::ScalabilityTestResult> LoadStressTesting::getScalabilityTestResults() const {
    QMutexLocker locker(&m_resultsMutex);
    return m_scalabilityTestResults;
}

LoadStressTesting::LoadTestResult LoadStressTesting::getLoadTestResult(const QString& testName) const {
    QMutexLocker locker(&m_resultsMutex);
    
    for (const LoadTestResult& result : m_loadTestResults) {
        if (result.testName == testName) {
            return result;
        }
    }
    
    return LoadTestResult(); // Return empty result if not found
}

LoadStressTesting::StressTestResult LoadStressTesting::getStressTestResult(const QString& testName) const {
    QMutexLocker locker(&m_resultsMutex);
    
    for (const StressTestResult& result : m_stressTestResults) {
        if (result.testName == testName) {
            return result;
        }
    }
    
    return StressTestResult(); // Return empty result if not found
}

LoadStressTesting::ScalabilityTestResult LoadStressTesting::getScalabilityTestResult(const QString& testName) const {
    QMutexLocker locker(&m_resultsMutex);
    
    for (const ScalabilityTestResult& result : m_scalabilityTestResults) {
        if (result.testName == testName) {
            return result;
        }
    }
    
    return ScalabilityTestResult(); // Return empty result if not found
}

void LoadStressTesting::recordLoadTestResult(const LoadTestResult& result) {
    QMutexLocker locker(&m_resultsMutex);
    m_loadTestResults.append(result);
}

void LoadStressTesting::recordStressTestResult(const StressTestResult& result) {
    QMutexLocker locker(&m_resultsMutex);
    m_stressTestResults.append(result);
}

void LoadStressTesting::recordScalabilityTestResult(const ScalabilityTestResult& result) {
    QMutexLocker locker(&m_resultsMutex);
    m_scalabilityTestResults.append(result);
}

void LoadStressTesting::clearResults() {
    QMutexLocker locker(&m_resultsMutex);
    m_loadTestResults.clear();
    m_stressTestResults.clear();
    m_scalabilityTestResults.clear();
}

// Reporting methods
QJsonObject LoadStressTesting::generateLoadTestReport() const {
    QMutexLocker locker(&m_resultsMutex);
    
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["test_type"] = "load_testing";
    report["platform"] = detectPlatform();
    
    QJsonArray resultsArray;
    for (const LoadTestResult& result : m_loadTestResults) {
        QJsonObject resultObj;
        resultObj["test_name"] = result.testName;
        resultObj["test_type"] = formatLoadTestType(result.testType);
        resultObj["start_time"] = result.startTime.toString(Qt::ISODate);
        resultObj["end_time"] = result.endTime.toString(Qt::ISODate);
        resultObj["duration_ms"] = result.actualDurationMs;
        resultObj["total_operations"] = result.totalOperations;
        resultObj["successful_operations"] = result.successfulOperations;
        resultObj["failed_operations"] = result.failedOperations;
        resultObj["operations_per_second"] = result.operationsPerSecond;
        resultObj["average_response_time"] = result.averageResponseTime;
        resultObj["min_response_time"] = result.minResponseTime;
        resultObj["max_response_time"] = result.maxResponseTime;
        resultObj["percentile_95_response_time"] = result.percentile95ResponseTime;
        resultObj["percentile_99_response_time"] = result.percentile99ResponseTime;
        resultObj["peak_memory_usage"] = result.peakMemoryUsage;
        resultObj["average_cpu_usage"] = result.averageCpuUsage;
        resultObj["peak_cpu_usage"] = result.peakCpuUsage;
        resultObj["concurrent_threads_used"] = result.concurrentThreadsUsed;
        resultObj["completed_successfully"] = result.completedSuccessfully;
        resultObj["error_message"] = result.errorMessage;
        
        resultsArray.append(resultObj);
    }
    report["results"] = resultsArray;
    
    return report;
}

QJsonObject LoadStressTesting::generateStressTestReport() const {
    QMutexLocker locker(&m_resultsMutex);
    
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["test_type"] = "stress_testing";
    report["platform"] = detectPlatform();
    
    QJsonArray resultsArray;
    for (const StressTestResult& result : m_stressTestResults) {
        QJsonObject resultObj;
        resultObj["test_name"] = result.testName;
        resultObj["start_time"] = result.startTime.toString(Qt::ISODate);
        resultObj["end_time"] = result.endTime.toString(Qt::ISODate);
        resultObj["duration_ms"] = result.actualDurationMs;
        resultObj["max_concurrent_operations_reached"] = result.maxConcurrentOperationsReached;
        resultObj["max_file_size_processed"] = result.maxFileSizeProcessed;
        resultObj["max_file_count_processed"] = result.maxFileCountProcessed;
        resultObj["max_directory_depth_reached"] = result.maxDirectoryDepthReached;
        resultObj["peak_memory_usage_mb"] = result.peakMemoryUsageMB;
        resultObj["peak_cpu_usage_percent"] = result.peakCpuUsagePercent;
        resultObj["hit_memory_limit"] = result.hitMemoryLimit;
        resultObj["hit_cpu_limit"] = result.hitCpuLimit;
        resultObj["hit_time_limit"] = result.hitTimeLimit;
        resultObj["total_failures"] = result.totalFailures;
        resultObj["failure_reason"] = result.failureReason;
        resultObj["completed_successfully"] = result.completedSuccessfully;
        
        QJsonArray errorArray;
        for (const QString& error : result.errorMessages) {
            errorArray.append(error);
        }
        resultObj["error_messages"] = errorArray;
        
        resultsArray.append(resultObj);
    }
    report["results"] = resultsArray;
    
    return report;
}

QJsonObject LoadStressTesting::generateScalabilityTestReport() const {
    QMutexLocker locker(&m_resultsMutex);
    
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["test_type"] = "scalability_testing";
    report["platform"] = detectPlatform();
    
    QJsonArray resultsArray;
    for (const ScalabilityTestResult& result : m_scalabilityTestResults) {
        QJsonObject resultObj;
        resultObj["test_name"] = result.testName;
        resultObj["start_time"] = result.startTime.toString(Qt::ISODate);
        resultObj["end_time"] = result.endTime.toString(Qt::ISODate);
        resultObj["linear_time_scaling"] = result.linearTimeScaling;
        resultObj["linear_memory_scaling"] = result.linearMemoryScaling;
        resultObj["linear_throughput_scaling"] = result.linearThroughputScaling;
        resultObj["scaling_analysis"] = result.scalingAnalysis;
        resultObj["completed_successfully"] = result.completedSuccessfully;
        
        QJsonObject scalingFactorsObj;
        for (auto it = result.scalingFactors.begin(); it != result.scalingFactors.end(); ++it) {
            scalingFactorsObj[it.key()] = it.value();
        }
        resultObj["scaling_factors"] = scalingFactorsObj;
        
        resultsArray.append(resultObj);
    }
    report["results"] = resultsArray;
    
    return report;
}

QJsonObject LoadStressTesting::generateComprehensiveReport() const {
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["test_type"] = "comprehensive_load_stress_testing";
    report["platform"] = detectPlatform();
    
    report["load_tests"] = generateLoadTestReport()["results"];
    report["stress_tests"] = generateStressTestReport()["results"];
    report["scalability_tests"] = generateScalabilityTestReport()["results"];
    
    // Summary statistics
    QJsonObject summary;
    summary["total_load_tests"] = m_loadTestResults.size();
    summary["total_stress_tests"] = m_stressTestResults.size();
    summary["total_scalability_tests"] = m_scalabilityTestResults.size();
    
    int successfulLoadTests = 0;
    for (const LoadTestResult& result : m_loadTestResults) {
        if (result.completedSuccessfully) successfulLoadTests++;
    }
    summary["successful_load_tests"] = successfulLoadTests;
    
    int successfulStressTests = 0;
    for (const StressTestResult& result : m_stressTestResults) {
        if (result.completedSuccessfully) successfulStressTests++;
    }
    summary["successful_stress_tests"] = successfulStressTests;
    
    int successfulScalabilityTests = 0;
    for (const ScalabilityTestResult& result : m_scalabilityTestResults) {
        if (result.completedSuccessfully) successfulScalabilityTests++;
    }
    summary["successful_scalability_tests"] = successfulScalabilityTests;
    
    report["summary"] = summary;
    
    return report;
}

bool LoadStressTesting::exportResults(const QString& filePath, const QString& format) const {
    QJsonObject report = generateComprehensiveReport();
    
    if (format.toLower() == "json") {
        QJsonDocument doc(report);
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(doc.toJson());
            return true;
        }
    }
    
    return false;
}

// Utility methods
QString LoadStressTesting::formatLoadTestType(LoadTestType type) const {
    switch (type) {
        case LoadTestType::ConcurrentUsers: return "Concurrent Users";
        case LoadTestType::HighVolumeFiles: return "High Volume Files";
        case LoadTestType::LargeFileSize: return "Large File Size";
        case LoadTestType::DeepDirectories: return "Deep Directories";
        case LoadTestType::WideDirectories: return "Wide Directories";
        case LoadTestType::MixedWorkload: return "Mixed Workload";
        case LoadTestType::SustainedLoad: return "Sustained Load";
        case LoadTestType::BurstLoad: return "Burst Load";
        case LoadTestType::GradualRamp: return "Gradual Ramp";
        case LoadTestType::StressToFailure: return "Stress to Failure";
        default: return "Unknown";
    }
}

QString LoadStressTesting::formatDuration(qint64 milliseconds) const {
    if (milliseconds < 1000) {
        return QString("%1 ms").arg(milliseconds);
    } else if (milliseconds < 60000) {
        return QString("%1.%2 s").arg(milliseconds / 1000).arg((milliseconds % 1000) / 100);
    } else {
        int minutes = milliseconds / 60000;
        int seconds = (milliseconds % 60000) / 1000;
        return QString("%1m %2s").arg(minutes).arg(seconds);
    }
}

QString LoadStressTesting::formatThroughput(double operationsPerSecond) const {
    if (operationsPerSecond < 1000) {
        return QString("%1 ops/sec").arg(operationsPerSecond, 0, 'f', 2);
    } else if (operationsPerSecond < 1000000) {
        return QString("%1K ops/sec").arg(operationsPerSecond / 1000.0, 0, 'f', 2);
    } else {
        return QString("%1M ops/sec").arg(operationsPerSecond / 1000000.0, 0, 'f', 2);
    }
}

QString LoadStressTesting::formatMemoryUsage(qint64 bytes) const {
    const QStringList units = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = bytes;
    
    while (size >= 1024.0 && unitIndex < units.size() - 1) {
        size /= 1024.0;
        unitIndex++;
    }
    
    return QString("%1 %2").arg(size, 0, 'f', 2).arg(units[unitIndex]);
}

QString LoadStressTesting::detectPlatform() const {
    return QString("%1_%2_%3")
        .arg(QSysInfo::kernelType())
        .arg(QSysInfo::currentCpuArchitecture())
        .arg(QSysInfo::productVersion());
}

qint64 LoadStressTesting::getAvailableMemory() const {
#ifdef Q_OS_LINUX
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
#elif defined(Q_OS_WIN)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullTotalPhys;
    }
#elif defined(Q_OS_MAC)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        return memsize;
    }
#endif
    return 0;
}

int LoadStressTesting::getAvailableCpuCores() const {
    return QThread::idealThreadCount();
}

// Statistical helper methods
double LoadStressTesting::calculateMean(const QList<double>& values) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    
    return sum / values.size();
}

double LoadStressTesting::calculatePercentile(QList<double> values, double percentile) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    std::sort(values.begin(), values.end());
    
    double index = (percentile / 100.0) * (values.size() - 1);
    int lowerIndex = qFloor(index);
    int upperIndex = qCeil(index);
    
    if (lowerIndex == upperIndex) {
        return values[lowerIndex];
    }
    
    double weight = index - lowerIndex;
    return values[lowerIndex] * (1.0 - weight) + values[upperIndex] * weight;
}

double LoadStressTesting::calculateStandardDeviation(const QList<double>& values, double mean) const {
    if (values.size() <= 1) {
        return 0.0;
    }
    
    double sumSquaredDiffs = 0.0;
    for (double value : values) {
        double diff = value - mean;
        sumSquaredDiffs += diff * diff;
    }
    
    return qSqrt(sumSquaredDiffs / (values.size() - 1));
}

// LoadTestWorker implementation
LoadTestWorker::LoadTestWorker(std::function<void()> operation, QAtomicInt* successCount, QAtomicInt* failureCount, QList<double>* responseTimes, QMutex* mutex)
    : m_operation(operation)
    , m_successCount(successCount)
    , m_failureCount(failureCount)
    , m_responseTimes(responseTimes)
    , m_mutex(mutex)
{
    setAutoDelete(true);
}

void LoadTestWorker::run() {
    QElapsedTimer timer;
    timer.start();
    
    bool success = true;
    try {
        m_operation();
    } catch (const std::exception& e) {
        success = false;
        qWarning() << "Load test worker operation failed:" << e.what();
    } catch (...) {
        success = false;
        qWarning() << "Load test worker operation failed with unknown exception";
    }
    
    qint64 responseTime = timer.elapsed();
    
    if (m_successCount && m_failureCount) {
        if (success) {
            m_successCount->fetchAndAddAcquire(1);
        } else {
            m_failureCount->fetchAndAddAcquire(1);
        }
    }
    
    if (m_responseTimes && m_mutex) {
        QMutexLocker locker(m_mutex);
        m_responseTimes->append(responseTime);
    }
    
    emit operationCompleted(responseTime, success);
}

#include "load_stress_testing.moc"