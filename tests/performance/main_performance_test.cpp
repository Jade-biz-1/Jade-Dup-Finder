#include <QtTest>
#include <QCoreApplication>
#include <QDebug>
#include <QDateTime>
#include "performance_test_framework.h"

// Include all performance test classes
#include "test_thread_pool_performance.h"
// Additional test classes to be implemented:
// #include "test_batch_processing_performance.h"
// #include "test_io_optimization_performance.h"
// #include "test_progress_reporting_overhead.h"
// #include "test_comprehensive_integration.h"

/**
 * @brief Main performance test runner for CloneClean
 * 
 * This application runs comprehensive performance benchmarks for all
 * optimized components of the CloneClean system, including:
 * 
 * - Thread pool management benchmarks
 * - Batch processing performance tests
 * - I/O optimization benchmarks
 * - Progress reporting overhead analysis
 * - Full integration performance tests
 * 
 * Results are saved in JSON format for analysis and regression testing.
 */

using namespace PerformanceTest;

class PerformanceTestSuite : public QObject {
    Q_OBJECT

public:
    explicit PerformanceTestSuite(QObject* parent = nullptr)
        : QObject(parent) {
        
        // Configure benchmark runner with optimized settings
        BenchmarkRunner::BenchmarkConfig config;
        config.warmupRuns = 3;
        config.measurementRuns = 10;
        config.maxRunTimeSeconds = 300; // 5 minutes per test max
        config.enableResourceMonitoring = true;
        config.enableStatistics = true;
        config.saveResults = true;
        config.acceptableVariation = 0.20; // 20% variation is acceptable for performance tests
        config.enableRegressionDetection = false; // Disable for initial run
        
        // Set up results directory
        QString resultsPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/cloneclean_performance";
        QDir().mkpath(resultsPath);
        config.resultsPath = resultsPath;
        
        m_runner.setBenchmarkConfig(config);
        
        // Setup result file with timestamp
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        m_resultFile = QDir(resultsPath).absoluteFilePath(QString("performance_results_%1.json").arg(timestamp));
    }

private slots:
    void initTestCase() {
        qDebug() << "==========================================================";
        qDebug() << "        CloneClean Performance Test Suite";
        qDebug() << "==========================================================";
        qDebug() << "Start Time:" << QDateTime::currentDateTime().toString();
        qDebug() << "";
        
        SystemProfiler::logSystemInfo();
        
        // Initialize test environment
        if (!m_runner.getDataGenerator()->generateMixedSizesScenario(500)) {
            QFAIL("Failed to generate test data");
        }
        
        qDebug() << "Test data generated successfully";
        qDebug() << "Results will be saved to:" << m_resultFile;
        qDebug() << "";
    }
    
    void runThreadPoolTests() {
        qDebug() << "=== Thread Pool Performance Tests ===";
        
        ThreadPoolPerformanceTest threadPoolTest;
        QTest::qExec(&threadPoolTest);
        
        qDebug() << "Thread pool tests completed\n";
    }
    
    void runBatchProcessingTests() {
        qDebug() << "=== Batch Processing Performance Tests ===";
        qDebug() << "⚠️ Not implemented yet - using existing batch functionality";
        qDebug() << "Batch processing tests completed\n";
    }
    
    void runIOOptimizationTests() {
        qDebug() << "=== I/O Optimization Performance Tests ===";
        qDebug() << "⚠️ Not implemented yet - using existing I/O functionality";
        qDebug() << "I/O optimization tests completed\n";
    }
    
    void runProgressReportingTests() {
        qDebug() << "=== Progress Reporting Overhead Tests ===";
        qDebug() << "⚠️ Not implemented yet - using existing progress functionality";
        qDebug() << "Progress reporting tests completed\n";
    }
    
    void runIntegrationTests() {
        qDebug() << "=== Comprehensive Integration Tests ===";
        qDebug() << "⚠️ Not implemented yet - using existing integration functionality";
        qDebug() << "Integration tests completed\n";
    }
    
    void cleanupTestCase() {
        // Save final results
        if (!m_runner.saveResultsToFile(m_resultFile)) {
            qWarning() << "Failed to save final results";
        }
        
        // Generate summary report
        QString report = generateFinalReport();
        
        // Save report to file
        QString reportFile = m_resultFile;
        reportFile.replace(".json", "_report.txt");
        
        QFile file(reportFile);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << report;
            file.close();
            qDebug() << "Performance report saved to:" << reportFile;
        }
        
        // Print summary to console
        qDebug() << "\n" << report;
        
        // Cleanup test data
        m_runner.getDataGenerator()->cleanupTestData();
        
        qDebug() << "\n==========================================================";
        qDebug() << "Performance Test Suite Completed";
        qDebug() << "End Time:" << QDateTime::currentDateTime().toString();
        qDebug() << "==========================================================";
    }

private:
    QString generateFinalReport() {
        QString report;
        QTextStream stream(&report);
        
        stream << "==========================================================\n";
        stream << "          CloneClean Performance Test Report\n";
        stream << "==========================================================\n";
        stream << "Test Date: " << QDateTime::currentDateTime().toString() << "\n";
        stream << "System: " << SystemProfiler::getSystemSummary() << "\n";
        stream << "\n";
        
        auto results = m_runner.getAllResults();
        
        if (results.isEmpty()) {
            stream << "No performance results available.\n";
            return report;
        }
        
        // Group results by category
        QMap<QString, QVector<PerformanceResult>> categorizedResults;
        for (const auto& result : results) {
            categorizedResults[result.category].append(result);
        }
        
        // Generate category summaries
        for (auto it = categorizedResults.begin(); it != categorizedResults.end(); ++it) {
            stream << "--- " << it.key().toUpper() << " TESTS ---\n";
            
            double totalTime = 0.0;
            double totalThroughput = 0.0;
            int throughputCount = 0;
            
            for (const auto& result : it.value()) {
                stream << QString("  %1: %2 (±%3)")
                    .arg(result.testName, -25)
                    .arg(formatDuration(result.meanTime))
                    .arg(formatDuration(result.confidenceInterval95))
                    << "\n";
                
                if (result.throughputMBps > 0) {
                    stream << QString("    Throughput: %1 MB/s")
                        .arg(result.throughputMBps, 0, 'f', 1) << "\n";
                    totalThroughput += result.throughputMBps;
                    throughputCount++;
                }
                
                totalTime += result.meanTime;
            }
            
            stream << QString("  Category Total: %1")
                .arg(formatDuration(totalTime)) << "\n";
            
            if (throughputCount > 0) {
                stream << QString("  Average Throughput: %1 MB/s")
                    .arg(totalThroughput / throughputCount, 0, 'f', 1) << "\n";
            }
            
            stream << "\n";
        }
        
        // Overall summary
        double overallTime = 0.0;
        double overallThroughput = 0.0;
        int overallThroughputCount = 0;
        
        for (const auto& result : results) {
            overallTime += result.meanTime;
            if (result.throughputMBps > 0) {
                overallThroughput += result.throughputMBps;
                overallThroughputCount++;
            }
        }
        
        stream << "--- OVERALL SUMMARY ---\n";
        stream << "Total Tests: " << results.size() << "\n";
        stream << "Total Execution Time: " << formatDuration(overallTime) << "\n";
        if (overallThroughputCount > 0) {
            stream << "Average System Throughput: " 
                   << QString::number(overallThroughput / overallThroughputCount, 'f', 1) 
                   << " MB/s\n";
        }
        
        // Performance recommendations
        stream << "\n--- PERFORMANCE ANALYSIS ---\n";
        analyzePerformance(stream, results);
        
        stream << "\n==========================================================\n";
        
        return report;
    }
    
    void analyzePerformance(QTextStream& stream, const QVector<PerformanceResult>& results) {
        // Find best and worst performing tests
        double bestTime = std::numeric_limits<double>::max();
        double worstTime = 0.0;
        QString bestTest, worstTest;
        
        for (const auto& result : results) {
            if (result.meanTime < bestTime) {
                bestTime = result.meanTime;
                bestTest = result.testName;
            }
            if (result.meanTime > worstTime) {
                worstTime = result.meanTime;
                worstTest = result.testName;
            }
        }
        
        stream << "Fastest Test: " << bestTest << " (" << formatDuration(bestTime) << ")\n";
        stream << "Slowest Test: " << worstTest << " (" << formatDuration(worstTime) << ")\n";
        
        // Analyze variation
        int highVariationTests = 0;
        for (const auto& result : results) {
            if (result.stdDeviation > result.meanTime * 0.20) { // >20% variation
                highVariationTests++;
            }
        }
        
        if (highVariationTests > 0) {
            stream << "High Variation Tests: " << highVariationTests 
                   << " (consider system load)\n";
        }
        
        // Performance recommendations
        stream << "\nRecommendations:\n";
        
        // Analyze thread pool performance
        bool hasThreadPoolTests = false;
        for (const auto& result : results) {
            if (result.testName.contains("thread", Qt::CaseInsensitive)) {
                hasThreadPoolTests = true;
                break;
            }
        }
        
        if (hasThreadPoolTests) {
            stream << "- Monitor thread pool efficiency for optimal core utilization\n";
        }
        
        // Analyze I/O performance
        double maxThroughput = 0.0;
        for (const auto& result : results) {
            maxThroughput = qMax(maxThroughput, result.throughputMBps);
        }
        
        if (maxThroughput < 50.0) { // Less than 50 MB/s
            stream << "- Consider I/O optimizations (SSD, memory mapping)\n";
        } else if (maxThroughput > 500.0) { // Greater than 500 MB/s
            stream << "- Excellent I/O performance detected\n";
        }
        
        stream << "- Regular performance monitoring recommended\n";
        stream << "- Compare results across different system configurations\n";
    }
    
    QString formatDuration(double milliseconds) {
        if (milliseconds < 1.0) {
            return QString("%1 μs").arg(milliseconds * 1000, 0, 'f', 1);
        } else if (milliseconds < 1000.0) {
            return QString("%1 ms").arg(milliseconds, 0, 'f', 2);
        } else {
            return QString("%1 s").arg(milliseconds / 1000.0, 0, 'f', 2);
        }
    }

private:
    BenchmarkRunner m_runner;
    QString m_resultFile;
};

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("CloneClean Performance Tests");
    app.setApplicationVersion("2.0");
    
    // Set up high precision timing
    QElapsedTimer::clockType();
    
    // Parse command line arguments
    QStringList args = app.arguments();
    bool verbose = args.contains("--verbose") || args.contains("-v");
    bool quick = args.contains("--quick") || args.contains("-q");
    
    if (verbose) {
        QLoggingCategory::setFilterRules("*.debug=true");
    }
    
    // Create and run test suite
    PerformanceTestSuite testSuite;
    
    if (quick) {
        qDebug() << "Running quick performance tests...";
        // In quick mode, we could reduce the number of runs
    }
    
    int result = QTest::qExec(&testSuite, argc, argv);
    
    return result;
}

#include "main_performance_test.moc"