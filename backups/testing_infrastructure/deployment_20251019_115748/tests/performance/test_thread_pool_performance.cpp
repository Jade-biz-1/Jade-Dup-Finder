#include "test_thread_pool_performance.h"
#include <QSignalSpy>
#include <QElapsedTimer>
#include <QtConcurrent>

using namespace PerformanceTest;

ThreadPoolPerformanceTest::ThreadPoolPerformanceTest(QObject* parent)
    : PerformanceTestCase(parent) {
    
    // Configure test workload options
    m_workloadOptions.fileCount = m_config.testFileCount;
    m_workloadOptions.totalSize = m_config.testDataSize;
    m_workloadOptions.fileExtensions = {"bin", "dat", "tmp"};
    m_workloadOptions.createDuplicates = false; // Focus on processing speed
}

ThreadPoolPerformanceTest::~ThreadPoolPerformanceTest() {
}

void ThreadPoolPerformanceTest::initTestCase() {
    PerformanceTestCase::initTestCase();
    
    qDebug() << "Initializing thread pool performance tests...";
    
    // Generate test workload
    if (!dataGenerator()->generateTestFiles(m_workloadOptions)) {
        QFAIL("Failed to generate test workload");
    }
    
    m_testFiles = dataGenerator()->getGeneratedFiles();
    qDebug() << "Generated" << m_testFiles.size() << "test files for thread pool benchmarks";
    
    // Establish single-thread baseline
    measureSingleThreadBaseline();
}

void ThreadPoolPerformanceTest::cleanupTestCase() {
    m_calculator.reset();
    PerformanceTestCase::cleanupTestCase();
}

void ThreadPoolPerformanceTest::runPerformanceTests() {
    qDebug() << "\n=== Thread Pool Performance Tests ===";
    
    // Core functionality benchmarks
    benchmarkBasicThreadPool();
    benchmarkWorkStealingEfficiency();
    benchmarkDynamicThreadScaling();
    benchmarkPriorityScheduling();
    benchmarkThreadUtilization();
    benchmarkLoadBalancing();
    
    // Comparative analysis
    compareWithStandardThreadPool();
    compareDifferentThreadCounts();
    benchmarkScalabilityLimits();
    
    // Stress tests
    stressTestHighConcurrency();
    stressTestVariableLoad();
    stressTestMemoryPressure();
    
    // Run all registered benchmarks
    if (!benchmarkRunner()->runAllBenchmarks()) {
        QFAIL("Thread pool performance benchmarks failed");
    }
    
    // Analyze results
    analyzeThreadPoolPerformance();
}

void ThreadPoolPerformanceTest::benchmarkBasicThreadPool() {
    qDebug() << "Benchmarking basic thread pool performance...";
    
    benchmarkRunner()->registerBenchmark("BasicThreadPool_StandardLoad", "threadpool", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.baselineThreadCount, true);
        
        QElapsedTimer timer;
        timer.start();
        
        // Process moderate workload
        QStringList moderateFiles = m_testFiles.mid(0, qMin(500, m_testFiles.size()));
        m_calculator->calculateFileHashesOptimized(moderateFiles);
        
        // Wait for completion
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(30000); // 30 second timeout
        
        qint64 elapsedMs = timer.elapsed();
        double throughputMBps = (static_cast<double>(m_config.testDataSize) / 2.0 / 1024.0 / 1024.0) / (static_cast<double>(elapsedMs) / 1000.0);
        
        // Store baseline for comparisons
        m_baselines.basicHashingTime = static_cast<double>(elapsedMs);
        
        qDebug() << "Basic thread pool throughput:" << throughputMBps << "MB/s";
        return completed;
    });
}

void ThreadPoolPerformanceTest::benchmarkWorkStealingEfficiency() {
    qDebug() << "Benchmarking work-stealing efficiency...";
    
    benchmarkRunner()->registerBenchmark("WorkStealing_UnbalancedLoad", "threadpool", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.baselineThreadCount, true);
        
        // Create unbalanced workload (mix of small and large files)
        QStringList unbalancedFiles;
        
        // Add some large files first
        int largeFileCount = static_cast<int>(m_testFiles.size()) / 4;
        unbalancedFiles.append(m_testFiles.mid(0, largeFileCount));
        
        // Add many small files
        unbalancedFiles.append(m_testFiles.mid(largeFileCount));
        
        QElapsedTimer timer;
        timer.start();
        
        m_calculator->calculateFileHashesOptimized(unbalancedFiles);
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(45000);
        
        // Verify work stealing occurred
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        bool workStealingEffective = verifyWorkStealingOccurred(stats);
        
        qDebug() << "Work stealing events:" << stats.workStealingEvents;
        qDebug() << "Thread utilization:" << stats.threadUtilization << "%";
        
        return completed && workStealingEffective;
    });
    
    benchmarkRunner()->registerBenchmark("WorkStealing_BalancedLoad", "threadpool", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.baselineThreadCount, true);
        
        // Balanced workload should show minimal work stealing
        QStringList balancedFiles = m_testFiles.mid(0, qMin(400, m_testFiles.size()));
        
        m_calculator->calculateFileHashesOptimized(balancedFiles);
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(30000);
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        qDebug() << "Work stealing events (balanced):" << stats.workStealingEvents;
        
        return completed;
    });
}

void ThreadPoolPerformanceTest::benchmarkDynamicThreadScaling() {
    qDebug() << "Benchmarking dynamic thread scaling...";
    
    benchmarkRunner()->registerBenchmark("DynamicScaling_LoadIncrease", "threadpool", [this]() -> bool {
        // Start with minimum threads
        setupHashCalculatorWithThreadPool(2, true);
        m_calculator->setDynamicThreadsEnabled(true);
        
        QElapsedTimer timer;
        timer.start();
        
        // Submit increasing load
        int batchSize = 100;
        for (int i = 0; i < m_testFiles.size(); i += batchSize) {
            QStringList batch = m_testFiles.mid(i, batchSize);
            m_calculator->calculateFileHashesOptimized(batch);
            
            // Small delay to allow thread scaling
            QThread::msleep(100);
        }
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(60000);
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        bool scalingOccurred = verifyDynamicScaling(stats);
        
        qDebug() << "Peak threads used:" << stats.peakThreads;
        qDebug() << "Final thread count:" << m_calculator->getActiveThreadCount();
        
        return completed && scalingOccurred;
    });
}

void ThreadPoolPerformanceTest::benchmarkPriorityScheduling() {
    qDebug() << "Benchmarking priority-based task scheduling...";
    
    benchmarkRunner()->registerBenchmark("PriorityScheduling_MixedPriority", "threadpool", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.baselineThreadCount, true);
        
        QElapsedTimer timer;
        timer.start();
        
        // Submit low priority tasks first
        QStringList lowPriorityFiles = m_testFiles.mid(0, m_testFiles.size() / 2);
        m_calculator->calculateFileHashesBatch(lowPriorityFiles, 0); // Low priority
        
        // Small delay
        QThread::msleep(50);
        
        // Submit high priority tasks
        QStringList highPriorityFiles = m_testFiles.mid(m_testFiles.size() / 2);
        m_calculator->calculateFileHashesBatch(highPriorityFiles, 2); // High priority
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(45000);
        
        // Verify that high priority tasks were processed efficiently
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        double efficiency = calculateEfficiencyScore(stats);
        
        qDebug() << "Priority scheduling efficiency:" << efficiency;
        
        return completed && efficiency > 0.7; // 70% efficiency threshold
    });
}

void ThreadPoolPerformanceTest::benchmarkThreadUtilization() {
    qDebug() << "Benchmarking thread utilization optimization...";
    
    benchmarkRunner()->registerBenchmark("ThreadUtilization_OptimalLoad", "threadpool", [this]() -> bool {
        setupHashCalculatorWithThreadPool(QThread::idealThreadCount(), true);
        
        // Create optimal workload (balanced, sufficient tasks)
        QStringList optimalFiles = m_testFiles;
        
        QElapsedTimer timer;
        timer.start();
        
        m_calculator->calculateFileHashesOptimized(optimalFiles);
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(60000);
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "Thread utilization:" << stats.threadUtilization << "%";
        qDebug() << "Average queue depth:" << stats.averageQueueDepth;
        qDebug() << "Queue efficiency:" << stats.queueEfficiency;
        
        // Good utilization should be >80%
        return completed && stats.threadUtilization > 80.0;
    });
}

void ThreadPoolPerformanceTest::benchmarkLoadBalancing() {
    qDebug() << "Benchmarking load balancing effectiveness...";
    
    benchmarkRunner()->registerBenchmark("LoadBalancing_VariableFileSizes", "threadpool", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.baselineThreadCount, true);
        
        // Generate files with highly variable sizes for load balancing test
        PerformanceTest::TestDataGenerator::DataGenerationOptions variableOptions;
        variableOptions.fileCount = 200;
        variableOptions.fileSizes.clear();
        
        // Mix of tiny, small, medium and large files
        for (int i = 0; i < 200; ++i) {
            qint64 size;
            int category = i % 4;
            switch (category) {
                case 0: size = 1024; break;                    // 1KB
                case 1: size = 50 * 1024; break;               // 50KB
                case 2: size = 5 * 1024 * 1024; break;         // 5MB
                case 3: size = 20 * 1024 * 1024; break;        // 20MB
            }
            variableOptions.fileSizes.append(size);
        }
        
        if (!dataGenerator()->generateTestFiles(variableOptions)) {
            return false;
        }
        
        QStringList variableFiles = dataGenerator()->getGeneratedFiles();
        
        QElapsedTimer timer;
        timer.start();
        
        m_calculator->calculateFileHashesOptimized(variableFiles);
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(60000);
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "Load balancing - Thread utilization:" << stats.threadUtilization << "%";
        qDebug() << "Work stealing events:" << stats.workStealingEvents;
        qDebug() << "Batch throughput:" << stats.batchThroughput;
        
        // Good load balancing should maintain high utilization even with variable loads
        return completed && stats.threadUtilization > 75.0;
    });
}

void ThreadPoolPerformanceTest::compareWithStandardThreadPool() {
    qDebug() << "Comparing with standard thread pool implementation...";
    
    // This would require a standard Qt thread pool implementation for comparison
    // For now, we'll compare with single-threaded performance
    benchmarkRunner()->registerBenchmark("Comparison_SingleThread", "comparison", [this]() -> bool {
        HashCalculator singleThreadCalculator;
        HashCalculator::HashOptions options;
        options.threadPoolSize = 1;
        options.enableBatchProcessing = false;
        options.enableDynamicThreads = false;
        singleThreadCalculator.setOptions(options);
        
        QStringList comparisonFiles = m_testFiles.mid(0, qMin(200, m_testFiles.size()));
        
        QElapsedTimer timer;
        timer.start();
        
        for (const QString& file : comparisonFiles) {
            singleThreadCalculator.calculateFileHash(file);
        }
        
        QSignalSpy completedSpy(&singleThreadCalculator, &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(30000);
        
        m_baselines.singleThreadTime = static_cast<double>(timer.elapsed());
        
        qDebug() << "Single thread time:" << m_baselines.singleThreadTime << "ms";
        
        return completed;
    });
}

void ThreadPoolPerformanceTest::compareDifferentThreadCounts() {
    qDebug() << "Comparing performance across different thread counts...";
    
    QVector<int> threadCounts = {1, 2, 4, 8, QThread::idealThreadCount(), 16};
    
    for (int threadCount : threadCounts) {
        QString testName = QString("ThreadCount_%1").arg(threadCount);
        
        benchmarkRunner()->registerBenchmark(testName, "scalability", [this, threadCount]() -> bool {
            setupHashCalculatorWithThreadPool(threadCount, true);
            
            QStringList scalabilityFiles = m_testFiles.mid(0, qMin(300, m_testFiles.size()));
            
            QElapsedTimer timer;
            timer.start();
            
            m_calculator->calculateFileHashesOptimized(scalabilityFiles);
            
            QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
            bool completed = completedSpy.wait(45000);
            
            HashCalculator::Statistics stats = m_calculator->getStatistics();
            double elapsedSeconds = static_cast<double>(timer.elapsed()) / 1000.0;
            double throughputMBps = (static_cast<double>(m_config.testDataSize) / 3.0 / 1024.0 / 1024.0) / elapsedSeconds;
            
            qDebug() << "Thread count" << threadCount << "- Throughput:" << throughputMBps << "MB/s";
            qDebug() << "Thread utilization:" << stats.threadUtilization << "%";
            
            return completed;
        });
    }
}

void ThreadPoolPerformanceTest::benchmarkScalabilityLimits() {
    qDebug() << "Benchmarking thread pool scalability limits...";
    
    benchmarkRunner()->registerBenchmark("Scalability_MaxThreads", "limits", [this]() -> bool {
        // Test with maximum reasonable thread count
        int maxThreads = qMin(32, QThread::idealThreadCount() * 4);
        setupHashCalculatorWithThreadPool(maxThreads, true);
        
        // Large workload
        QElapsedTimer timer;
        timer.start();
        
        m_calculator->calculateFileHashesOptimized(m_testFiles);
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(120000); // 2 minute timeout
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "Max threads test - Active threads:" << stats.activeThreads;
        qDebug() << "Peak threads:" << stats.peakThreads;
        qDebug() << "Thread utilization:" << stats.threadUtilization << "%";
        
        // Should not crash and should maintain reasonable utilization
        return completed && stats.threadUtilization > 50.0;
    });
}

void ThreadPoolPerformanceTest::stressTestHighConcurrency() {
    qDebug() << "Stress testing high concurrency scenarios...";
    
    benchmarkRunner()->registerBenchmark("StressTest_HighConcurrency", "stress", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.maxThreadCount, true);
        
        // Submit multiple batches rapidly
        QElapsedTimer timer;
        timer.start();
        
        int batchSize = 50;
        int batchCount = static_cast<int>(m_testFiles.size()) / batchSize;
        
        for (int i = 0; i < batchCount; ++i) {
            QStringList batch = m_testFiles.mid(i * batchSize, batchSize);
            m_calculator->calculateFileHashesOptimized(batch);
            // No delay - stress the system
        }
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(180000); // 3 minute timeout
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "High concurrency - Peak queue depth:" << stats.maxQueueDepth;
        qDebug() << "Work stealing events:" << stats.workStealingEvents;
        qDebug() << "Thread utilization:" << stats.threadUtilization << "%";
        
        return completed;
    });
}

void ThreadPoolPerformanceTest::stressTestVariableLoad() {
    qDebug() << "Stress testing variable load scenarios...";
    
    benchmarkRunner()->registerBenchmark("StressTest_VariableLoad", "stress", [this]() -> bool {
        setupHashCalculatorWithThreadPool(m_config.baselineThreadCount, true);
        m_calculator->setDynamicThreadsEnabled(true);
        
        QElapsedTimer timer;
        timer.start();
        
        // Simulate variable load pattern
        for (int cycle = 0; cycle < 3; ++cycle) {
            // High load phase
            QStringList highLoadBatch = m_testFiles.mid(cycle * 200, 200);
            m_calculator->calculateFileHashesOptimized(highLoadBatch);
            
            // Wait for some processing
            QThread::msleep(500);
            
            // Low load phase
            QStringList lowLoadBatch = m_testFiles.mid(cycle * 200 + 200, 50);
            m_calculator->calculateFileHashesOptimized(lowLoadBatch);
            
            QThread::msleep(200);
        }
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(120000);
        
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "Variable load - Thread efficiency:" << calculateEfficiencyScore(stats);
        qDebug() << "Dynamic scaling events:" << (stats.peakThreads > 2 ? "Yes" : "No");
        
        return completed;
    });
}

void ThreadPoolPerformanceTest::stressTestMemoryPressure() {
    qDebug() << "Stress testing under memory pressure...";
    
    benchmarkRunner()->registerBenchmark("StressTest_MemoryPressure", "stress", [this]() -> bool {
        // Create memory pressure by reducing chunk size and increasing concurrent tasks
        setupHashCalculatorWithThreadPool(m_config.maxThreadCount, true);
        
        HashCalculator::HashOptions options = m_calculator->getOptions();
        options.chunkSize = 4096; // Small chunks to increase memory allocations
        options.maxConcurrentBatches = 10; // More concurrent operations
        m_calculator->setOptions(options);
        
        // Monitor memory usage
        qint64 initialMemory = m_calculator->getCurrentMemoryUsage();
        
        QElapsedTimer timer;
        timer.start();
        
        m_calculator->calculateFileHashesOptimized(m_testFiles);
        
        QSignalSpy completedSpy(m_calculator.data(), &HashCalculator::allOperationsComplete);
        bool completed = completedSpy.wait(180000);
        
        qint64 peakMemory = m_calculator->getCurrentMemoryUsage();
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "Memory pressure - Initial memory:" << (initialMemory / 1024 / 1024) << "MB";
        qDebug() << "Peak memory:" << (peakMemory / 1024 / 1024) << "MB";
        qDebug() << "Memory efficiency maintained:" << (stats.threadUtilization > 60.0 ? "Yes" : "No");
        
        return completed && stats.threadUtilization > 60.0;
    });
}

// Helper methods

void ThreadPoolPerformanceTest::setupHashCalculatorWithThreadPool(int threadCount, bool enableAdvancedFeatures) {
    m_calculator.reset(new HashCalculator);
    
    HashCalculator::HashOptions options;
    options.threadPoolSize = threadCount;
    options.enableBatchProcessing = enableAdvancedFeatures;
    options.enableDynamicThreads = enableAdvancedFeatures;
    options.enableWorkStealing = enableAdvancedFeatures;
    options.chunkSize = 64 * 1024; // 64KB chunks
    options.enableCaching = true;
    
    m_calculator->setOptions(options);
}

void ThreadPoolPerformanceTest::measureSingleThreadBaseline() {
    qDebug() << "Measuring single-thread baseline performance...";
    
    setupHashCalculatorWithThreadPool(1, false);
    
    QStringList baselineFiles = m_testFiles.mid(0, qMin(100, m_testFiles.size()));
    
    QElapsedTimer timer;
    timer.start();
    
    for (const QString& file : baselineFiles) {
        m_calculator->calculateFileHashSync(file);
    }
    
    m_baselines.singleThreadTime = static_cast<double>(timer.elapsed());
    qDebug() << "Single-thread baseline:" << m_baselines.singleThreadTime << "ms";
}

bool ThreadPoolPerformanceTest::verifyWorkStealingOccurred(const HashCalculator::Statistics& stats) {
    // Work stealing should occur when there's load imbalance
    return stats.workStealingEvents > 0;
}

bool ThreadPoolPerformanceTest::verifyDynamicScaling(const HashCalculator::Statistics& stats) {
    // Dynamic scaling should increase thread count under load
    return stats.peakThreads > 2; // Started with 2 threads
}

double ThreadPoolPerformanceTest::calculateEfficiencyScore(const HashCalculator::Statistics& stats) {
    // Combine multiple efficiency metrics
    double threadEfficiency = stats.threadUtilization / 100.0;
    double queueEfficiency = stats.queueEfficiency;
    double throughputEfficiency = qMin(1.0, stats.batchThroughput / 10.0); // Normalize to reasonable range
    
    return (threadEfficiency + queueEfficiency + throughputEfficiency) / 3.0;
}

void ThreadPoolPerformanceTest::analyzeThreadPoolPerformance() {
    qDebug() << "\n=== Thread Pool Performance Analysis ===";
    
    auto results = benchmarkRunner()->getAllResults();
    
    // Find best performing thread count
    double bestThroughput = 0.0;
    int optimalThreadCount = 0;
    
    for (const auto& result : results) {
        if (result.testName.startsWith("ThreadCount_") && result.throughputMBps > bestThroughput) {
            bestThroughput = result.throughputMBps;
            QString threadCountStr = result.testName.mid(12); // Remove "ThreadCount_"
            optimalThreadCount = threadCountStr.toInt();
        }
    }
    
    if (optimalThreadCount > 0) {
        qDebug() << "Optimal thread count:" << optimalThreadCount;
        qDebug() << "Peak throughput:" << bestThroughput << "MB/s";
    }
    
    // Calculate speedup vs single thread
    if (m_baselines.singleThreadTime > 0) {
        for (const auto& result : results) {
            if (result.category == "threadpool") {
                double speedup = m_baselines.singleThreadTime / result.meanTime;
                qDebug() << result.testName << "speedup:" << QString::number(speedup, 'f', 2) << "x";
            }
        }
    }
    
    qDebug() << "========================================\n";
}

#include "test_thread_pool_performance.moc"