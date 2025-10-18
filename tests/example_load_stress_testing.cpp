#include <QtTest>
#include <QApplication>
#include <QWidget>
#include <QDir>
#include <QTemporaryDir>
#include <QThread>
#include "load_stress_testing.h"
#include "performance_benchmark.h"
#include "test_data_generator.h"

/**
 * @brief Example load and stress testing for DupFinder
 * 
 * This class demonstrates how to use the LoadStressTesting framework
 * to validate system performance under various load conditions and
 * stress scenarios.
 */
class ExampleLoadStressTesting : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Load testing examples
    void testConcurrentUsers();
    void testHighVolumeFiles();
    void testLargeFileSize();
    void testDeepDirectories();
    void testWideDirectories();
    void testSustainedLoad();
    void testBurstLoad();
    void testGradualRamp();
    
    // Stress testing examples
    void testStressToFailure();
    void testMemoryStress();
    void testCpuStress();
    void testConcurrencyStress();
    void testResourceExhaustion();
    
    // Scalability testing examples
    void testFileCountScalability();
    void testFileSizeScalability();
    void testThreadScalability();
    void testComprehensiveScalability();
    
    // DupFinder-specific load tests
    void testDuplicateDetectionLoad();
    void testHashCalculationLoad();
    void testFileScanningLoad();
    void testUILoad();
    
    // Performance validation tests
    void testLoadPerformanceRequirements();
    void testStressLimits();
    void testScalingBehavior();
    
    // Reporting and analysis tests
    void testReportGeneration();
    void testResultExport();

private:
    LoadStressTesting* m_loadStressTesting;
    PerformanceBenchmark* m_performanceBenchmark;
    TestDataGenerator* m_testDataGenerator;
    QTemporaryDir* m_testDir;
    QString m_testPath;
};

void ExampleLoadStressTesting::initTestCase() {
    // Initialize load and stress testing framework
    m_loadStressTesting = new LoadStressTesting(this);
    m_performanceBenchmark = new PerformanceBenchmark(this);
    m_testDataGenerator = new TestDataGenerator(this);
    
    // Connect frameworks
    m_loadStressTesting->setPerformanceBenchmark(m_performanceBenchmark);
    m_loadStressTesting->setTestDataGenerator(m_testDataGenerator);
    
    // Create temporary directory for test files
    m_testDir = new QTemporaryDir();
    QVERIFY(m_testDir->isValid());
    m_testPath = m_testDir->path();
    
    // Configure load testing settings
    m_loadStressTesting->setMaxConcurrentThreads(QThread::idealThreadCount());
    m_loadStressTesting->setResourceMonitoringInterval(500); // 500ms monitoring interval
    
    qDebug() << "Load and stress testing suite initialized";
    qDebug() << "Test directory:" << m_testPath;
    qDebug() << "Available CPU cores:" << QThread::idealThreadCount();
}

void ExampleLoadStressTesting::cleanupTestCase() {
    // Export comprehensive results
    QString reportPath = QDir(m_testPath).absoluteFilePath("load_stress_report.json");
    m_loadStressTesting->exportResults(reportPath);
    qDebug() << "Load and stress testing report exported to:" << reportPath;
    
    // Cleanup
    delete m_testDir;
    qDebug() << "Load and stress testing suite completed";
}

void ExampleLoadStressTesting::testConcurrentUsers() {
    qDebug() << "Testing concurrent users load...";
    
    // Test with different user counts
    QList<int> userCounts = {5, 10, 20, 50};
    int operationsPerUser = 10;
    
    for (int userCount : userCounts) {
        bool success = m_loadStressTesting->runConcurrentUserTest(userCount, operationsPerUser, [this](int userId) {
            // Simulate user operation
            QString userFile = QString("user_%1_operation.txt").arg(userId);
            QString filePath = QDir(m_testPath).absoluteFilePath(userFile);
            
            QFile file(filePath);
            if (file.open(QIODevice::WriteOnly | QIODevice::Append)) {
                file.write(QString("User %1 operation at %2\n")
                          .arg(userId)
                          .arg(QDateTime::currentDateTime().toString()).toUtf8());
            }
            
            // Simulate processing time
            QThread::msleep(QRandomGenerator::global()->bounded(10, 50));
        });
        
        QVERIFY2(success, QString("Concurrent user test failed for %1 users").arg(userCount).toUtf8().constData());
        
        // Verify performance requirements
        QString testName = QString("concurrent_users_%1_users_%2_ops").arg(userCount).arg(operationsPerUser);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        QVERIFY(result.successfulOperations > 0);
        
        qDebug() << QString("Concurrent users (%1): %2 ops/sec, avg response: %3ms")
                    .arg(userCount)
                    .arg(result.operationsPerSecond, 0, 'f', 2)
                    .arg(result.averageResponseTime, 0, 'f', 2);
    }
}

void ExampleLoadStressTesting::testHighVolumeFiles() {
    qDebug() << "Testing high volume files load...";
    
    QString testSubDir = "high_volume_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Test with increasing file counts
    QList<QPair<int, qint64>> testCases = {
        {100, 1024},     // 100 files of 1KB each
        {500, 2048},     // 500 files of 2KB each
        {1000, 1024},    // 1000 files of 1KB each
        {2000, 512}      // 2000 files of 512B each
    };
    
    for (const auto& testCase : testCases) {
        int fileCount = testCase.first;
        qint64 fileSize = testCase.second;
        
        bool success = m_loadStressTesting->runHighVolumeFileTest(fileCount, fileSize, testPath);
        QVERIFY2(success, QString("High volume file test failed for %1 files of %2 bytes")
                 .arg(fileCount).arg(fileSize).toUtf8().constData());
        
        QString testName = QString("high_volume_files_%1_files_%2_bytes").arg(fileCount).arg(fileSize);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("High volume files (%1 x %2B): %3 ops/sec")
                    .arg(fileCount)
                    .arg(fileSize)
                    .arg(result.operationsPerSecond, 0, 'f', 2);
        
        // Cleanup for next test
        QDir(testPath).removeRecursively();
    }
}

void ExampleLoadStressTesting::testLargeFileSize() {
    qDebug() << "Testing large file size load...";
    
    QString testSubDir = "large_file_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Test with different large file sizes
    QList<qint64> fileSizes = {
        1024 * 1024,      // 1MB
        10 * 1024 * 1024, // 10MB
        50 * 1024 * 1024  // 50MB
    };
    
    for (qint64 fileSize : fileSizes) {
        bool success = m_loadStressTesting->runLargeFileSizeTest(fileSize, testPath);
        QVERIFY2(success, QString("Large file size test failed for %1 bytes")
                 .arg(fileSize).toUtf8().constData());
        
        QString testName = QString("large_file_size_%1_bytes").arg(fileSize);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("Large file size (%1): %2 ops/sec, avg response: %3ms")
                    .arg(m_loadStressTesting->formatMemoryUsage(fileSize))
                    .arg(result.operationsPerSecond, 0, 'f', 2)
                    .arg(result.averageResponseTime, 0, 'f', 2);
        
        // Cleanup for next test
        QDir(testPath).removeRecursively();
    }
}

void ExampleLoadStressTesting::testDeepDirectories() {
    qDebug() << "Testing deep directories load...";
    
    QString testSubDir = "deep_directory_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Test with different directory depths
    QList<QPair<int, int>> testCases = {
        {5, 10},   // 5 levels deep, 10 files per level
        {10, 5},   // 10 levels deep, 5 files per level
        {15, 3}    // 15 levels deep, 3 files per level
    };
    
    for (const auto& testCase : testCases) {
        int depth = testCase.first;
        int filesPerLevel = testCase.second;
        
        bool success = m_loadStressTesting->runDeepDirectoryTest(depth, filesPerLevel, testPath);
        QVERIFY2(success, QString("Deep directory test failed for depth %1 with %2 files per level")
                 .arg(depth).arg(filesPerLevel).toUtf8().constData());
        
        QString testName = QString("deep_directory_%1_depth_%2_files").arg(depth).arg(filesPerLevel);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("Deep directories (%1 levels, %2 files/level): %3ms")
                    .arg(depth)
                    .arg(filesPerLevel)
                    .arg(result.averageResponseTime, 0, 'f', 2);
        
        // Cleanup for next test
        QDir(testPath).removeRecursively();
    }
}

void ExampleLoadStressTesting::testWideDirectories() {
    qDebug() << "Testing wide directories load...";
    
    QString testSubDir = "wide_directory_test";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Test with different directory widths
    QList<QPair<int, int>> testCases = {
        {50, 10},   // 10 directories with 50 files each
        {100, 5},   // 5 directories with 100 files each
        {200, 3}    // 3 directories with 200 files each
    };
    
    for (const auto& testCase : testCases) {
        int filesPerDir = testCase.first;
        int dirCount = testCase.second;
        
        bool success = m_loadStressTesting->runWideDirectoryTest(filesPerDir, dirCount, testPath);
        QVERIFY2(success, QString("Wide directory test failed for %1 files per directory with %2 directories")
                 .arg(filesPerDir).arg(dirCount).toUtf8().constData());
        
        QString testName = QString("wide_directory_%1_files_%2_dirs").arg(filesPerDir).arg(dirCount);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("Wide directories (%1 files/dir, %2 dirs): %3ms")
                    .arg(filesPerDir)
                    .arg(dirCount)
                    .arg(result.averageResponseTime, 0, 'f', 2);
        
        // Cleanup for next test
        QDir(testPath).removeRecursively();
    }
}

void ExampleLoadStressTesting::testSustainedLoad() {
    qDebug() << "Testing sustained load...";
    
    // Test sustained load for different durations and rates
    QList<QPair<qint64, int>> testCases = {
        {10000, 10},  // 10 seconds at 10 ops/sec
        {15000, 20},  // 15 seconds at 20 ops/sec
        {20000, 5}    // 20 seconds at 5 ops/sec
    };
    
    for (const auto& testCase : testCases) {
        qint64 duration = testCase.first;
        int opsPerSec = testCase.second;
        
        bool success = m_loadStressTesting->runSustainedLoadTest(duration, opsPerSec, [this]() {
            // Simulate sustained operation
            QString fileName = QString("sustained_%1.tmp").arg(QRandomGenerator::global()->bounded(1000));
            QString filePath = QDir(m_testPath).absoluteFilePath(fileName);
            
            QFile file(filePath);
            if (file.open(QIODevice::WriteOnly)) {
                file.write("Sustained load test data");
            }
            
            // Cleanup immediately
            QFile::remove(filePath);
        });
        
        QVERIFY2(success, QString("Sustained load test failed for %1ms at %2 ops/sec")
                 .arg(duration).arg(opsPerSec).toUtf8().constData());
        
        QString testName = QString("sustained_load_%1ms_%2ops").arg(duration).arg(opsPerSec);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("Sustained load (%1ms, %2 ops/sec): actual %3 ops/sec")
                    .arg(duration)
                    .arg(opsPerSec)
                    .arg(result.operationsPerSecond, 0, 'f', 2);
    }
}

void ExampleLoadStressTesting::testBurstLoad() {
    qDebug() << "Testing burst load...";
    
    // Test burst load patterns
    int burstCount = 5;
    int operationsPerBurst = 20;
    qint64 burstInterval = 2000; // 2 seconds between bursts
    
    bool success = m_loadStressTesting->runBurstLoadTest(burstCount, operationsPerBurst, burstInterval, [this]() {
        // Simulate burst operation
        QString fileName = QString("burst_%1.tmp").arg(QRandomGenerator::global()->bounded(10000));
        QString filePath = QDir(m_testPath).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write("Burst load test data");
        }
        
        // Cleanup immediately
        QFile::remove(filePath);
    });
    
    QVERIFY2(success, "Burst load test failed");
    
    QString testName = QString("burst_load_%1_bursts_%2_ops_%3ms").arg(burstCount).arg(operationsPerBurst).arg(burstInterval);
    auto result = m_loadStressTesting->getLoadTestResult(testName);
    QVERIFY(result.completedSuccessfully);
    
    qDebug() << QString("Burst load (%1 bursts, %2 ops/burst): %3 ops/sec")
                .arg(burstCount)
                .arg(operationsPerBurst)
                .arg(result.operationsPerSecond, 0, 'f', 2);
}

void ExampleLoadStressTesting::testGradualRamp() {
    qDebug() << "Testing gradual ramp load...";
    
    int startThreads = 2;
    int endThreads = 10;
    qint64 rampDuration = 5000; // 5 seconds
    
    bool success = m_loadStressTesting->runGradualRampTest(startThreads, endThreads, rampDuration, [this]() {
        // Simulate ramped operation
        QString fileName = QString("ramp_%1_%2.tmp")
                          .arg(QThread::currentThreadId())
                          .arg(QRandomGenerator::global()->bounded(1000));
        QString filePath = QDir(m_testPath).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write("Gradual ramp test data");
        }
        
        // Simulate some processing time
        QThread::msleep(QRandomGenerator::global()->bounded(10, 100));
        
        // Cleanup
        QFile::remove(filePath);
    });
    
    QVERIFY2(success, "Gradual ramp test failed");
    
    QString testName = QString("gradual_ramp_%1_to_%2_threads_%3ms").arg(startThreads).arg(endThreads).arg(rampDuration);
    auto result = m_loadStressTesting->getLoadTestResult(testName);
    QVERIFY(result.completedSuccessfully);
    
    qDebug() << QString("Gradual ramp (%1 to %2 threads): %3 ops/sec")
                .arg(startThreads)
                .arg(endThreads)
                .arg(result.operationsPerSecond, 0, 'f', 2);
}v
oid ExampleLoadStressTesting::testStressToFailure() {
    qDebug() << "Testing stress to failure...";
    
    LoadStressTesting::StressTestConfig config;
    config.name = "stress_to_failure_test";
    config.maxConcurrentOperations = 1000;
    config.memoryLimitMB = 512; // 512MB limit
    config.cpuLimitPercent = 90.0;
    config.maxTestDurationMs = 30000; // 30 seconds max
    config.stopOnResourceLimit = true;
    config.description = "Stress test to find system failure point";
    
    bool success = m_loadStressTesting->runStressToFailureTest([this]() {
        // Memory-intensive operation
        static thread_local QList<QByteArray> memoryBuffers;
        
        // Allocate memory in chunks
        for (int i = 0; i < 10; ++i) {
            memoryBuffers.append(QByteArray(1024 * 1024, 'S')); // 1MB chunks
        }
        
        // CPU-intensive operation
        double result = 0.0;
        for (int i = 0; i < 100000; ++i) {
            result += qSqrt(i) * qSin(i);
        }
        
        // Occasionally clear some memory to simulate realistic usage
        if (QRandomGenerator::global()->bounded(100) < 10) {
            memoryBuffers.clear();
        }
    }, config);
    
    // Note: This test is expected to hit resource limits, so success might be false
    auto result = m_loadStressTesting->getStressTestResult("stress_to_failure");
    QVERIFY(!result.testName.isEmpty());
    
    qDebug() << QString("Stress to failure: %1, peak memory: %2MB, peak CPU: %3%")
                .arg(result.failureReason)
                .arg(result.peakMemoryUsageMB)
                .arg(result.peakCpuUsagePercent, 0, 'f', 1);
    
    qDebug() << QString("Max concurrent operations reached: %1").arg(result.maxConcurrentOperationsReached);
}

void ExampleLoadStressTesting::testMemoryStress() {
    qDebug() << "Testing memory stress...";
    
    qint64 memoryLimitMB = 256; // 256MB limit
    
    bool success = m_loadStressTesting->runMemoryStressTest(memoryLimitMB, [this]() {
        static thread_local QList<QByteArray> buffers;
        
        // Allocate memory progressively
        int bufferSize = QRandomGenerator::global()->bounded(100000, 1000000); // 100KB to 1MB
        buffers.append(QByteArray(bufferSize, 'M'));
        
        // Simulate some work with the memory
        if (!buffers.isEmpty()) {
            QByteArray& buffer = buffers.last();
            for (int i = 0; i < qMin(1000, buffer.size()); ++i) {
                buffer[i] = static_cast<char>(i % 256);
            }
        }
        
        // Occasionally free some memory
        if (buffers.size() > 50 && QRandomGenerator::global()->bounded(100) < 20) {
            buffers.removeFirst();
        }
    });
    
    auto result = m_loadStressTesting->getStressTestResult(QString("memory_stress_%1MB").arg(memoryLimitMB));
    QVERIFY(!result.testName.isEmpty());
    
    qDebug() << QString("Memory stress (%1MB limit): peak usage %2MB, hit limit: %3")
                .arg(memoryLimitMB)
                .arg(result.peakMemoryUsageMB)
                .arg(result.hitMemoryLimit ? "Yes" : "No");
}

void ExampleLoadStressTesting::testCpuStress() {
    qDebug() << "Testing CPU stress...";
    
    double cpuLimitPercent = 85.0;
    
    bool success = m_loadStressTesting->runCpuStressTest(cpuLimitPercent, [this]() {
        // CPU-intensive calculations
        double result = 0.0;
        int iterations = QRandomGenerator::global()->bounded(50000, 200000);
        
        for (int i = 0; i < iterations; ++i) {
            result += qSqrt(i) * qCos(i) * qSin(i * 0.1);
            
            // Add some complexity
            if (i % 1000 == 0) {
                result = qPow(result, 0.99); // Prevent overflow
            }
        }
        
        // Use result to prevent optimization
        static volatile double globalResult = 0.0;
        globalResult = result;
    });
    
    auto result = m_loadStressTesting->getStressTestResult(QString("cpu_stress_%1_percent").arg(cpuLimitPercent));
    QVERIFY(!result.testName.isEmpty());
    
    qDebug() << QString("CPU stress (%1% limit): peak usage %2%, hit limit: %3")
                .arg(cpuLimitPercent)
                .arg(result.peakCpuUsagePercent, 0, 'f', 1)
                .arg(result.hitCpuLimit ? "Yes" : "No");
}

void ExampleLoadStressTesting::testConcurrencyStress() {
    qDebug() << "Testing concurrency stress...";
    
    int maxConcurrentOps = 100;
    
    bool success = m_loadStressTesting->runConcurrencyStressTest(maxConcurrentOps, [this]() {
        // Simulate concurrent file operations
        QString fileName = QString("concurrent_%1_%2.tmp")
                          .arg(QThread::currentThreadId())
                          .arg(QRandomGenerator::global()->bounded(100000));
        QString filePath = QDir(m_testPath).absoluteFilePath(fileName);
        
        // Create file
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(QRandomGenerator::global()->bounded(1000, 10000), 'C');
            file.write(data);
        }
        
        // Simulate processing time
        QThread::msleep(QRandomGenerator::global()->bounded(10, 100));
        
        // Read file back
        if (file.open(QIODevice::ReadOnly)) {
            QByteArray readData = file.readAll();
            // Process data (simulate work)
            volatile int checksum = 0;
            for (char byte : readData) {
                checksum += byte;
            }
        }
        
        // Cleanup
        QFile::remove(filePath);
    });
    
    auto result = m_loadStressTesting->getStressTestResult(QString("concurrency_stress_%1_ops").arg(maxConcurrentOps));
    QVERIFY(!result.testName.isEmpty());
    
    qDebug() << QString("Concurrency stress (%1 max ops): reached %2 concurrent, failures: %3")
                .arg(maxConcurrentOps)
                .arg(result.maxConcurrentOperationsReached)
                .arg(result.totalFailures);
}

void ExampleLoadStressTesting::testResourceExhaustion() {
    qDebug() << "Testing resource exhaustion...";
    
    LoadStressTesting::StressTestConfig config;
    config.name = "resource_exhaustion";
    config.maxConcurrentOperations = 200;
    config.memoryLimitMB = 128;
    config.cpuLimitPercent = 80.0;
    config.maxTestDurationMs = 20000; // 20 seconds
    config.stopOnResourceLimit = true;
    config.description = "Resource exhaustion test";
    
    bool success = m_loadStressTesting->runResourceExhaustionTest(config, [this]() {
        // Combined resource-intensive operation
        static thread_local QList<QByteArray> memoryPool;
        
        // Memory allocation
        int allocSize = QRandomGenerator::global()->bounded(50000, 500000);
        memoryPool.append(QByteArray(allocSize, 'R'));
        
        // CPU work
        double cpuWork = 0.0;
        for (int i = 0; i < 10000; ++i) {
            cpuWork += qSqrt(i) * qTan(i * 0.01);
        }
        
        // File I/O
        QString fileName = QString("resource_%1.tmp").arg(QRandomGenerator::global()->bounded(10000));
        QString filePath = QDir(m_testPath).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(QByteArray(1000, 'E'));
        }
        
        // Cleanup some resources occasionally
        if (memoryPool.size() > 20) {
            memoryPool.removeFirst();
        }
        
        QFile::remove(filePath);
    });
    
    auto result = m_loadStressTesting->getStressTestResult("resource_exhaustion");
    QVERIFY(!result.testName.isEmpty());
    
    qDebug() << QString("Resource exhaustion: %1")
                .arg(result.failureReason.isEmpty() ? "Completed within limits" : result.failureReason);
    qDebug() << QString("Peak memory: %1MB, peak CPU: %2%, max concurrent: %3")
                .arg(result.peakMemoryUsageMB)
                .arg(result.peakCpuUsagePercent, 0, 'f', 1)
                .arg(result.maxConcurrentOperationsReached);
}

void ExampleLoadStressTesting::testFileCountScalability() {
    qDebug() << "Testing file count scalability...";
    
    QList<int> fileCounts = {10, 50, 100, 500, 1000};
    
    bool success = m_loadStressTesting->runFileCountScalabilityTest(fileCounts, [this](int fileCount) {
        QString testSubDir = QString("scalability_files_%1").arg(fileCount);
        QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
        
        // Create files
        QDir().mkpath(testPath);
        for (int i = 0; i < fileCount; ++i) {
            QString fileName = QString("scale_file_%1.dat").arg(i);
            QString filePath = QDir(testPath).absoluteFilePath(fileName);
            
            QFile file(filePath);
            if (file.open(QIODevice::WriteOnly)) {
                file.write(QByteArray(1024, 'S')); // 1KB files
            }
        }
        
        // Process files (simulate duplicate detection)
        QDir dir(testPath);
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
        
        // Cleanup
        QDir(testPath).removeRecursively();
    });
    
    QVERIFY2(success, "File count scalability test failed");
    
    auto result = m_loadStressTesting->getScalabilityTestResult("file_count_scalability");
    QVERIFY(result.completedSuccessfully);
    
    qDebug() << QString("File count scalability: %1").arg(result.scalingAnalysis);
    qDebug() << QString("Linear time scaling: %1").arg(result.linearTimeScaling ? "Yes" : "No");
}

void ExampleLoadStressTesting::testFileSizeScalability() {
    qDebug() << "Testing file size scalability...";
    
    QList<qint64> fileSizes = {
        1024,           // 1KB
        10 * 1024,      // 10KB
        100 * 1024,     // 100KB
        1024 * 1024,    // 1MB
        5 * 1024 * 1024 // 5MB
    };
    
    bool success = m_loadStressTesting->runFileSizeScalabilityTest(fileSizes, [this](qint64 fileSize) {
        QString fileName = QString("scale_size_%1.dat").arg(fileSize);
        QString filePath = QDir(m_testPath).absoluteFilePath(fileName);
        
        // Create file of specified size
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            const qint64 chunkSize = 64 * 1024; // 64KB chunks
            QByteArray chunk(qMin(fileSize, chunkSize), 'S');
            
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
        
        // Process file (calculate hash)
        if (file.open(QIODevice::ReadOnly)) {
            QCryptographicHash hash(QCryptographicHash::Sha256);
            hash.addData(&file);
            hash.result(); // Force calculation
        }
        
        // Cleanup
        QFile::remove(filePath);
    });
    
    QVERIFY2(success, "File size scalability test failed");
    
    auto result = m_loadStressTesting->getScalabilityTestResult("file_size_scalability");
    QVERIFY(result.completedSuccessfully);
    
    qDebug() << QString("File size scalability: %1").arg(result.scalingAnalysis);
    qDebug() << QString("Linear memory scaling: %1").arg(result.linearMemoryScaling ? "Yes" : "No");
}

void ExampleLoadStressTesting::testThreadScalability() {
    qDebug() << "Testing thread scalability...";
    
    QList<int> threadCounts = {1, 2, 4, 8, QThread::idealThreadCount()};
    
    bool success = m_loadStressTesting->runThreadScalabilityTest(threadCounts, [this](int threadCount) {
        // Set thread pool size
        m_loadStressTesting->setMaxConcurrentThreads(threadCount);
        
        // Run concurrent operations
        QAtomicInt operationCount(0);
        QList<QThread*> threads;
        
        for (int i = 0; i < threadCount; ++i) {
            QThread* thread = QThread::create([this, &operationCount]() {
                for (int j = 0; j < 100; ++j) {
                    // CPU-intensive work
                    double result = 0.0;
                    for (int k = 0; k < 10000; ++k) {
                        result += qSqrt(k) * qSin(k * 0.1);
                    }
                    
                    operationCount.fetchAndAddAcquire(1);
                    
                    // Use result to prevent optimization
                    static volatile double globalResult = 0.0;
                    globalResult = result;
                }
            });
            
            threads.append(thread);
            thread->start();
        }
        
        // Wait for all threads to complete
        for (QThread* thread : threads) {
            thread->wait();
            delete thread;
        }
        
        // Restore original thread count
        m_loadStressTesting->setMaxConcurrentThreads(QThread::idealThreadCount());
    });
    
    QVERIFY2(success, "Thread scalability test failed");
    
    auto result = m_loadStressTesting->getScalabilityTestResult("thread_scalability");
    QVERIFY(result.completedSuccessfully);
    
    qDebug() << QString("Thread scalability: %1").arg(result.scalingAnalysis);
    qDebug() << QString("Linear throughput scaling: %1").arg(result.linearThroughputScaling ? "Yes" : "No");
}

void ExampleLoadStressTesting::testComprehensiveScalability() {
    qDebug() << "Testing comprehensive scalability...";
    
    LoadStressTesting::ScalabilityTestConfig config;
    config.name = "comprehensive_scalability";
    config.fileCounts = {10, 50, 100};
    config.fileSizes = {1024, 10240, 102400}; // 1KB, 10KB, 100KB
    config.threadCounts = {1, 2, 4};
    config.iterationsPerConfiguration = 2;
    config.maxDurationPerTest = 60000; // 1 minute per test
    config.description = "Comprehensive scalability analysis";
    
    bool success = m_loadStressTesting->runScalabilityTest(config, [this](int fileCount, qint64 fileSize, int threadCount) {
        QString testSubDir = QString("comprehensive_%1_%2_%3").arg(fileCount).arg(fileSize).arg(threadCount);
        QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
        
        // Set thread count
        m_loadStressTesting->setMaxConcurrentThreads(threadCount);
        
        // Create and process files
        QDir().mkpath(testPath);
        
        for (int i = 0; i < fileCount; ++i) {
            QString fileName = QString("comp_file_%1.dat").arg(i);
            QString filePath = QDir(testPath).absoluteFilePath(fileName);
            
            QFile file(filePath);
            if (file.open(QIODevice::WriteOnly)) {
                file.write(QByteArray(fileSize, 'C'));
            }
        }
        
        // Process all files
        QDir dir(testPath);
        QStringList files = dir.entryList(QDir::Files);
        
        for (const QString& fileName : files) {
            QString filePath = dir.absoluteFilePath(fileName);
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(QCryptographicHash::Md5);
                hash.addData(&file);
                hash.result();
            }
        }
        
        // Cleanup
        QDir(testPath).removeRecursively();
        
        // Restore thread count
        m_loadStressTesting->setMaxConcurrentThreads(QThread::idealThreadCount());
    });
    
    QVERIFY2(success, "Comprehensive scalability test failed");
    
    auto result = m_loadStressTesting->getScalabilityTestResult("comprehensive_scalability");
    QVERIFY(result.completedSuccessfully);
    
    qDebug() << QString("Comprehensive scalability completed with %1 configurations").arg(result.results.size());
    qDebug() << QString("Scaling analysis: %1").arg(result.scalingAnalysis);
}void Ex
ampleLoadStressTesting::testDuplicateDetectionLoad() {
    qDebug() << "Testing duplicate detection load...";
    
    QString testSubDir = "duplicate_detection_load";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Test with different scenarios
    QList<QPair<QPair<int, double>, QString>> testCases = {
        {{100, 0.2}, "low_duplicates"},     // 100 files, 20% duplicates
        {{200, 0.5}, "medium_duplicates"},  // 200 files, 50% duplicates
        {{500, 0.8}, "high_duplicates"}     // 500 files, 80% duplicates
    };
    
    for (const auto& testCase : testCases) {
        int fileCount = testCase.first.first;
        double duplicateRatio = testCase.first.second;
        QString scenario = testCase.second;
        
        QString scenarioPath = QDir(testPath).absoluteFilePath(scenario);
        
        bool success = m_loadStressTesting->runDuplicateDetectionLoadTest(scenarioPath, fileCount, duplicateRatio);
        QVERIFY2(success, QString("Duplicate detection load test failed for %1").arg(scenario).toUtf8().constData());
        
        QString testName = QString("duplicate_detection_load_%1_files_%2_ratio").arg(fileCount).arg(duplicateRatio);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("Duplicate detection (%1): %2 ops/sec, avg response: %3ms")
                    .arg(scenario)
                    .arg(result.operationsPerSecond, 0, 'f', 2)
                    .arg(result.averageResponseTime, 0, 'f', 2);
        
        // Cleanup
        QDir(scenarioPath).removeRecursively();
    }
}

void ExampleLoadStressTesting::testHashCalculationLoad() {
    qDebug() << "Testing hash calculation load...";
    
    // Create test files for hash calculation
    QString testSubDir = "hash_calculation_load";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    QDir().mkpath(testPath);
    
    // Create files of various sizes
    QStringList testFiles;
    QList<qint64> fileSizes = {1024, 10240, 102400, 1048576}; // 1KB to 1MB
    
    for (int i = 0; i < 100; ++i) {
        qint64 fileSize = fileSizes[i % fileSizes.size()];
        QString fileName = QString("hash_test_%1.dat").arg(i);
        QString filePath = QDir(testPath).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(fileSize, static_cast<char>('A' + (i % 26)));
            file.write(data);
            testFiles.append(filePath);
        }
    }
    
    // Test with different numbers of concurrent hashers
    QList<int> concurrentHashers = {1, 2, 4, 8};
    
    for (int hashers : concurrentHashers) {
        bool success = m_loadStressTesting->runHashCalculationLoadTest(testFiles, hashers);
        QVERIFY2(success, QString("Hash calculation load test failed with %1 hashers").arg(hashers).toUtf8().constData());
        
        QString testName = QString("hash_calculation_load_%1_files_%2_hashers").arg(testFiles.size()).arg(hashers);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("Hash calculation (%1 hashers): %2 ops/sec")
                    .arg(hashers)
                    .arg(result.operationsPerSecond, 0, 'f', 2);
    }
    
    // Cleanup
    QDir(testPath).removeRecursively();
}

void ExampleLoadStressTesting::testFileScanningLoad() {
    qDebug() << "Testing file scanning load...";
    
    // Create a complex directory structure for scanning
    QString testSubDir = "file_scanning_load";
    QString testPath = QDir(m_testPath).absoluteFilePath(testSubDir);
    
    // Create nested structure with many files
    if (m_testDataGenerator) {
        m_testDataGenerator->createNestedDirectoryStructure(testPath, 4, 5, 20); // 4 levels, 5 dirs per level, 20 files per dir
    }
    
    // Test with different numbers of concurrent scanners
    QList<int> concurrentScanners = {1, 2, 4};
    
    for (int scanners : concurrentScanners) {
        bool success = m_loadStressTesting->runFileScanningLoadTest(testPath, scanners);
        QVERIFY2(success, QString("File scanning load test failed with %1 scanners").arg(scanners).toUtf8().constData());
        
        QString testName = QString("file_scanning_load_%1_scanners").arg(scanners);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("File scanning (%1 scanners): %2 ops/sec")
                    .arg(scanners)
                    .arg(result.operationsPerSecond, 0, 'f', 2);
    }
    
    // Cleanup
    QDir(testPath).removeRecursively();
}

void ExampleLoadStressTesting::testUILoad() {
    qDebug() << "Testing UI load...";
    
    // Create a test widget
    QWidget testWidget;
    testWidget.resize(800, 600);
    testWidget.show();
    
    // Wait for widget to be displayed
    QTest::qWaitForWindowExposed(&testWidget);
    
    // Test with different numbers of UI operations
    QList<int> operationCounts = {50, 100, 200};
    
    for (int operations : operationCounts) {
        bool success = m_loadStressTesting->runUILoadTest(&testWidget, operations);
        QVERIFY2(success, QString("UI load test failed with %1 operations").arg(operations).toUtf8().constData());
        
        QString testName = QString("ui_load_%1_operations").arg(operations);
        auto result = m_loadStressTesting->getLoadTestResult(testName);
        QVERIFY(result.completedSuccessfully);
        
        qDebug() << QString("UI load (%1 operations): %2 ops/sec, avg response: %3ms")
                    .arg(operations)
                    .arg(result.operationsPerSecond, 0, 'f', 2)
                    .arg(result.averageResponseTime, 0, 'f', 2);
    }
}

void ExampleLoadStressTesting::testLoadPerformanceRequirements() {
    qDebug() << "Testing load performance requirements...";
    
    // Define performance requirements
    struct PerformanceRequirement {
        QString testPattern;
        double maxResponseTimeMs;
        double minThroughputOpsPerSec;
        QString description;
    };
    
    QList<PerformanceRequirement> requirements = {
        {"concurrent_users_*", 1000.0, 5.0, "Concurrent users should respond within 1 second"},
        {"high_volume_files_*", 5000.0, 10.0, "High volume file operations should complete within 5 seconds"},
        {"hash_calculation_*", 2000.0, 20.0, "Hash calculations should maintain good throughput"},
        {"file_scanning_*", 3000.0, 15.0, "File scanning should be efficient"}
    };
    
    auto loadResults = m_loadStressTesting->getLoadTestResults();
    
    for (const PerformanceRequirement& req : requirements) {
        bool foundMatchingTest = false;
        
        for (const auto& result : loadResults) {
            QRegExp pattern(req.testPattern);
            pattern.setPatternSyntax(QRegExp::Wildcard);
            
            if (pattern.exactMatch(result.testName)) {
                foundMatchingTest = true;
                
                // Check response time requirement
                QVERIFY2(result.averageResponseTime <= req.maxResponseTimeMs,
                        QString("Test %1: Average response time %2ms exceeds requirement %3ms")
                        .arg(result.testName)
                        .arg(result.averageResponseTime)
                        .arg(req.maxResponseTimeMs).toUtf8().constData());
                
                // Check throughput requirement
                QVERIFY2(result.operationsPerSecond >= req.minThroughputOpsPerSec,
                        QString("Test %1: Throughput %2 ops/sec below requirement %3 ops/sec")
                        .arg(result.testName)
                        .arg(result.operationsPerSecond)
                        .arg(req.minThroughputOpsPerSec).toUtf8().constData());
                
                qDebug() << QString("✓ %1: Response %2ms (req: <%3ms), Throughput %4 ops/sec (req: >%5)")
                            .arg(result.testName)
                            .arg(result.averageResponseTime, 0, 'f', 1)
                            .arg(req.maxResponseTimeMs)
                            .arg(result.operationsPerSecond, 0, 'f', 1)
                            .arg(req.minThroughputOpsPerSec);
            }
        }
        
        QVERIFY2(foundMatchingTest, QString("No test found matching pattern: %1").arg(req.testPattern).toUtf8().constData());
    }
}

void ExampleLoadStressTesting::testStressLimits() {
    qDebug() << "Testing stress limits...";
    
    auto stressResults = m_loadStressTesting->getStressTestResults();
    
    for (const auto& result : stressResults) {
        qDebug() << QString("Stress test: %1").arg(result.testName);
        
        // Verify that stress tests pushed the system appropriately
        if (result.testName.contains("memory")) {
            QVERIFY2(result.peakMemoryUsageMB > 0, "Memory stress test should show memory usage");
            qDebug() << QString("  Peak memory: %1MB").arg(result.peakMemoryUsageMB);
        }
        
        if (result.testName.contains("cpu")) {
            QVERIFY2(result.peakCpuUsagePercent > 0, "CPU stress test should show CPU usage");
            qDebug() << QString("  Peak CPU: %1%").arg(result.peakCpuUsagePercent, 0, 'f', 1);
        }
        
        if (result.testName.contains("concurrency")) {
            QVERIFY2(result.maxConcurrentOperationsReached > 0, "Concurrency stress test should show concurrent operations");
            qDebug() << QString("  Max concurrent operations: %1").arg(result.maxConcurrentOperationsReached);
        }
        
        // Check if test hit expected limits
        if (result.hitMemoryLimit || result.hitCpuLimit || result.hitTimeLimit) {
            qDebug() << QString("  Hit limits: Memory=%1, CPU=%2, Time=%3")
                        .arg(result.hitMemoryLimit ? "Yes" : "No")
                        .arg(result.hitCpuLimit ? "Yes" : "No")
                        .arg(result.hitTimeLimit ? "Yes" : "No");
        }
        
        if (!result.failureReason.isEmpty()) {
            qDebug() << QString("  Failure reason: %1").arg(result.failureReason);
        }
    }
}

void ExampleLoadStressTesting::testScalingBehavior() {
    qDebug() << "Testing scaling behavior...";
    
    auto scalabilityResults = m_loadStressTesting->getScalabilityTestResults();
    
    for (const auto& result : scalabilityResults) {
        qDebug() << QString("Scalability test: %1").arg(result.testName);
        
        QVERIFY2(result.completedSuccessfully, QString("Scalability test %1 should complete successfully").arg(result.testName).toUtf8().constData());
        
        // Analyze scaling factors
        for (auto it = result.scalingFactors.begin(); it != result.scalingFactors.end(); ++it) {
            QString metric = it.key();
            double factor = it.value();
            
            qDebug() << QString("  %1 scaling factor: %2").arg(metric).arg(factor, 0, 'f', 2);
            
            // Verify scaling factors are reasonable (not exponential)
            QVERIFY2(factor < 3.0, QString("Scaling factor for %1 (%2) should not be exponential")
                    .arg(metric).arg(factor).toUtf8().constData());
        }
        
        qDebug() << QString("  Scaling analysis: %1").arg(result.scalingAnalysis);
        
        // Check for linear scaling where expected
        if (result.testName.contains("file_count")) {
            // File count should generally scale linearly with time
            qDebug() << QString("  Linear time scaling: %1").arg(result.linearTimeScaling ? "Yes" : "No");
        }
        
        if (result.testName.contains("thread")) {
            // Thread scaling should improve throughput
            qDebug() << QString("  Linear throughput scaling: %1").arg(result.linearThroughputScaling ? "Yes" : "No");
        }
    }
}

void ExampleLoadStressTesting::testReportGeneration() {
    qDebug() << "Testing report generation...";
    
    // Generate individual reports
    QJsonObject loadReport = m_loadStressTesting->generateLoadTestReport();
    QVERIFY(!loadReport.isEmpty());
    QVERIFY(loadReport.contains("results"));
    
    QJsonObject stressReport = m_loadStressTesting->generateStressTestReport();
    QVERIFY(!stressReport.isEmpty());
    QVERIFY(stressReport.contains("results"));
    
    QJsonObject scalabilityReport = m_loadStressTesting->generateScalabilityTestReport();
    QVERIFY(!scalabilityReport.isEmpty());
    QVERIFY(scalabilityReport.contains("results"));
    
    // Generate comprehensive report
    QJsonObject comprehensiveReport = m_loadStressTesting->generateComprehensiveReport();
    QVERIFY(!comprehensiveReport.isEmpty());
    QVERIFY(comprehensiveReport.contains("load_tests"));
    QVERIFY(comprehensiveReport.contains("stress_tests"));
    QVERIFY(comprehensiveReport.contains("scalability_tests"));
    QVERIFY(comprehensiveReport.contains("summary"));
    
    // Verify summary statistics
    QJsonObject summary = comprehensiveReport["summary"].toObject();
    QVERIFY(summary["total_load_tests"].toInt() > 0);
    QVERIFY(summary["successful_load_tests"].toInt() >= 0);
    
    qDebug() << QString("Report generation successful:");
    qDebug() << QString("  Load tests: %1 total, %2 successful")
                .arg(summary["total_load_tests"].toInt())
                .arg(summary["successful_load_tests"].toInt());
    qDebug() << QString("  Stress tests: %1 total, %2 successful")
                .arg(summary["total_stress_tests"].toInt())
                .arg(summary["successful_stress_tests"].toInt());
    qDebug() << QString("  Scalability tests: %1 total, %2 successful")
                .arg(summary["total_scalability_tests"].toInt())
                .arg(summary["successful_scalability_tests"].toInt());
}

void ExampleLoadStressTesting::testResultExport() {
    qDebug() << "Testing result export...";
    
    // Export results to file
    QString exportPath = QDir(m_testPath).absoluteFilePath("load_stress_export.json");
    bool success = m_loadStressTesting->exportResults(exportPath, "json");
    QVERIFY2(success, "Failed to export load and stress test results");
    
    // Verify file exists and has content
    QFile exportFile(exportPath);
    QVERIFY(exportFile.exists());
    QVERIFY(exportFile.open(QIODevice::ReadOnly));
    
    QByteArray exportData = exportFile.readAll();
    QVERIFY(!exportData.isEmpty());
    
    // Verify it's valid JSON
    QJsonDocument doc = QJsonDocument::fromJson(exportData);
    QVERIFY(!doc.isNull());
    
    QJsonObject exportedReport = doc.object();
    QVERIFY(exportedReport.contains("timestamp"));
    QVERIFY(exportedReport.contains("load_tests"));
    QVERIFY(exportedReport.contains("stress_tests"));
    QVERIFY(exportedReport.contains("scalability_tests"));
    
    qDebug() << QString("Results exported successfully to: %1 (%2)")
                .arg(exportPath)
                .arg(m_loadStressTesting->formatMemoryUsage(exportData.size()));
}

QTEST_MAIN(ExampleLoadStressTesting)
#include "example_load_stress_testing.moc"