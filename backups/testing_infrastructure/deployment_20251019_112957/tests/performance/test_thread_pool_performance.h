#ifndef TEST_THREAD_POOL_PERFORMANCE_H
#define TEST_THREAD_POOL_PERFORMANCE_H

#include <QtTest>
#include "performance_test_framework.h"
#include "hash_calculator.h"

/**
 * @brief Performance tests for thread pool management optimizations
 * 
 * This test suite benchmarks the advanced thread pool features implemented
 * in HC-002a, including:
 * 
 * - Work-stealing thread pool performance
 * - Dynamic thread scaling efficiency
 * - Priority-based task scheduling
 * - Thread utilization optimization
 * - Load balancing effectiveness
 */

class ThreadPoolPerformanceTest : public PerformanceTest::PerformanceTestCase {
    Q_OBJECT

public:
    explicit ThreadPoolPerformanceTest(QObject* parent = nullptr);
    virtual ~ThreadPoolPerformanceTest();

private slots:
    void initTestCase() override;
    void cleanupTestCase() override;
    
    // Core thread pool benchmarks
    void runPerformanceTests() override;
    void benchmarkBasicThreadPool();
    void benchmarkWorkStealingEfficiency();
    void benchmarkDynamicThreadScaling();
    void benchmarkPriorityScheduling();
    void benchmarkThreadUtilization();
    void benchmarkLoadBalancing();
    
    // Comparative tests
    void compareWithStandardThreadPool();
    void compareDifferentThreadCounts();
    void benchmarkScalabilityLimits();
    
    // Stress tests
    void stressTestHighConcurrency();
    void stressTestVariableLoad();
    void stressTestMemoryPressure();

private:
    // Test setup helpers
    void setupHashCalculatorWithThreadPool(int threadCount, bool enableAdvancedFeatures);
    void generateWorkload(int fileCount, const QString& scenario);
    
    // Performance measurement helpers
    void measureThroughput(const QString& testName, std::function<void()> workload);
    void measureScalability(const QString& testName, const QVector<int>& threadCounts);
    void analyzeThreadUtilization(HashCalculator& calculator);
    
    // Verification helpers
    bool verifyWorkStealingOccurred(const HashCalculator::Statistics& stats);
    bool verifyDynamicScaling(const HashCalculator::Statistics& stats);
    double calculateEfficiencyScore(const HashCalculator::Statistics& stats);
    
    // Additional helper methods
    void measureSingleThreadBaseline();
    void analyzeThreadPoolPerformance();

private:
    QScopedPointer<HashCalculator> m_calculator;
    QStringList m_testFiles;
    PerformanceTest::TestDataGenerator::DataGenerationOptions m_workloadOptions;
    
    // Test configuration
    struct TestConfig {
        int baselineThreadCount = 4;
        int maxThreadCount = 16;
        int testFileCount = 1000;
        qint64 testDataSize = 100 * 1024 * 1024; // 100MB
        double performanceTolerancePercent = 15.0; // 15% tolerance
    } m_config;
    
    // Performance baselines for comparison
    struct PerformanceBaselines {
        double standardThreadPoolThroughput = 0.0;
        double basicHashingTime = 0.0;
        double singleThreadTime = 0.0;
    } m_baselines;
};

#endif // TEST_THREAD_POOL_PERFORMANCE_H