#include "parallel_test_executor.h"
#include "test_maintenance_tools.h"
#include "advanced_reporting.h"
#include "test_harness.h"
#include <QCoreApplication>
#include <QDebug>
#include <QTimer>

/**
 * @brief Example demonstrating advanced testing features
 */
class AdvancedTestingExample : public QObject {
    Q_OBJECT

public:
    AdvancedTestingExample(QObject* parent = nullptr) : QObject(parent) {}

    void runExample() {
        qDebug() << "=== Advanced Testing Features Example ===\n";
        
        // 1. Demonstrate parallel test execution
        demonstrateParallelExecution();
        
        // 2. Demonstrate test maintenance tools
        demonstrateMaintenanceTools();
        
        // 3. Demonstrate advanced reporting
        demonstrateAdvancedReporting();
        
        qDebug() << "\n=== Example completed ===";
    }

private:
    void demonstrateParallelExecution() {
        qDebug() << "1. Parallel Test Execution Demo";
        qDebug() << "--------------------------------";
        
        // Create parallel test executor
        ParallelTestExecutor executor;
        executor.setMaxWorkers(4);
        executor.setLoadBalancingStrategy(TestLoadBalancer::Strategy::Adaptive);
        executor.setResourceIsolation(true);
        
        // Create sample test contexts
        QList<TestExecutionContext> testContexts;
        
        for (int i = 0; i < 10; ++i) {
            TestExecutionContext context;
            context.suiteName = QString("TestSuite%1").arg(i / 3 + 1);
            context.testName = QString("test_method_%1").arg(i);
            context.priority = static_cast<TestPriority>(i % 4);
            
            // Set resource requirements
            context.resources.requiresFileSystem = (i % 2 == 0);
            context.resources.requiresNetwork = (i % 3 == 0);
            context.resources.requiresDisplay = (i == 0); // Only first test needs display
            context.resources.estimatedDurationMs = 1000 + (i * 500);
            
            testContexts.append(context);
        }
        
        // Connect signals to monitor execution
        connect(&executor, &ParallelTestExecutor::executionStarted,
                [](int totalTests) {
                    qDebug() << "Started parallel execution of" << totalTests << "tests";
                });
        
        connect(&executor, &ParallelTestExecutor::testCompleted,
                [](const QString& suiteName, const QString& testName, bool passed, qint64 duration) {
                    qDebug() << "Completed:" << suiteName << "::" << testName 
                             << (passed ? "PASSED" : "FAILED") << "in" << duration << "ms";
                });
        
        connect(&executor, &ParallelTestExecutor::executionCompleted,
                [](const ParallelExecutionStats& stats) {
                    qDebug() << "Parallel execution completed:";
                    qDebug() << "- Total tests:" << stats.totalTests;
                    qDebug() << "- Completed:" << stats.completedTests;
                    qDebug() << "- Failed:" << stats.failedTests;
                    qDebug() << "- Peak concurrency:" << stats.peakConcurrency;
                    qDebug() << "- Parallel efficiency:" << QString::number(stats.parallelEfficiency, 'f', 1) << "%";
                });
        
        // Execute tests (this would normally be asynchronous)
        qDebug() << "Executing" << testContexts.size() << "tests in parallel...";
        // executor.executeTests(testContexts); // Commented out for example
        
        qDebug() << "Parallel execution demo completed\n";
    }
    
    void demonstrateMaintenanceTools() {
        qDebug() << "2. Test Maintenance Tools Demo";
        qDebug() << "------------------------------";
        
        // Create maintenance coordinator
        TestMaintenanceCoordinator coordinator;
        
        // Create and configure components
        auto flakinessDetector = std::make_shared<TestFlakinessDetector>();
        flakinessDetector->setFlakinessThreshold(0.05); // 5% threshold
        flakinessDetector->setMinimumRuns(10);
        
        auto executionOptimizer = std::make_shared<TestExecutionOptimizer>();
        executionOptimizer->setPerformanceRegressionThreshold(1.5); // 50% slower
        
        auto coverageAnalyzer = std::make_shared<TestCoverageAnalyzer>();
        coverageAnalyzer->setCoverageThreshold(0.85); // 85% minimum coverage
        
        auto baselineManager = std::make_shared<BaselineManager>();
        baselineManager->setAutoUpdateThreshold(0.9); // 90% confidence for auto-update
        
        // Set components
        coordinator.setFlakinessDetector(flakinessDetector);
        coordinator.setExecutionOptimizer(executionOptimizer);
        coordinator.setCoverageAnalyzer(coverageAnalyzer);
        coordinator.setBaselineManager(baselineManager);
        
        // Connect signals
        connect(&coordinator, &TestMaintenanceCoordinator::analysisCompleted,
                [](int totalRecommendations, int highPriority) {
                    qDebug() << "Maintenance analysis completed:";
                    qDebug() << "- Total recommendations:" << totalRecommendations;
                    qDebug() << "- High priority:" << highPriority;
                });
        
        // Simulate analysis
        qDebug() << "Running maintenance analysis...";
        
        // Simulate some flaky test data
        QList<bool> flakyResults = {true, false, true, true, false, true, true, false, true, true};
        QList<qint64> executionTimes = {1000, 1200, 950, 1100, 1300, 1050, 1150, 1400, 1000, 1080};
        flakinessDetector->analyzeTestHistory("TestSuite::flaky_test", flakyResults, executionTimes);
        
        // Get flaky tests
        QList<TestFlakinessInfo> flakyTests = flakinessDetector->getFlakyTests();
        qDebug() << "Found" << flakyTests.size() << "flaky tests";
        
        for (const TestFlakinessInfo& info : flakyTests) {
            qDebug() << "- Flaky test:" << info.testName 
                     << "Rate:" << QString::number(info.flakinessRate * 100, 'f', 1) << "%"
                     << "Recommendation:" << info.recommendation;
        }
        
        // Simulate performance analysis
        QList<qint64> performanceTimes = {1000, 1050, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800};
        executionOptimizer->analyzeTestPerformance("TestSuite::slow_test", performanceTimes);
        
        QList<TestPerformanceMetrics> slowTests = executionOptimizer->getSlowTests(5);
        qDebug() << "Found" << slowTests.size() << "slow tests";
        
        for (const TestPerformanceMetrics& metrics : slowTests) {
            qDebug() << "- Slow test:" << metrics.testName 
                     << "Avg time:" << metrics.averageExecutionTime << "ms"
                     << "Suggestion:" << metrics.optimizationSuggestion;
        }
        
        qDebug() << "Maintenance tools demo completed\n";
    }
    
    void demonstrateAdvancedReporting() {
        qDebug() << "3. Advanced Reporting Demo";
        qDebug() << "--------------------------";
        
        // Create trend analyzer
        TestTrendAnalyzer trendAnalyzer;
        
        // Add sample trend data
        QDateTime baseTime = QDateTime::currentDateTime().addDays(-30);
        for (int i = 0; i < 30; ++i) {
            TestTrendDataPoint point;
            point.timestamp = baseTime.addDays(i);
            point.testName = "performance_test";
            point.suiteName = "PerformanceTests";
            point.passed = (i % 10 != 0); // Fail every 10th test
            point.executionTime = 1000 + (i * 50) + (i > 20 ? 500 : 0); // Performance regression after day 20
            point.memoryUsage = 100.0 + (i * 2.5);
            point.cpuUsage = 50.0 + (i * 1.2);
            
            trendAnalyzer.addDataPoint(point);
        }
        
        // Analyze trends
        PerformanceTrendAnalysis trend = trendAnalyzer.analyzePerformanceTrend("performance_test", "executionTime", 30);
        qDebug() << "Performance trend analysis:";
        qDebug() << "- Current value:" << trend.currentValue << "ms";
        qDebug() << "- Trend slope:" << QString::number(trend.trendSlope * 100, 'f', 2) << "%";
        qDebug() << "- Direction:" << trend.trendDirection;
        qDebug() << "- Prediction:" << trend.prediction;
        
        if (!trend.alerts.isEmpty()) {
            qDebug() << "- Alerts:";
            for (const QString& alert : trend.alerts) {
                qDebug() << "  *" << alert;
            }
        }
        
        // Test effectiveness analyzer
        TestEffectivenessAnalyzer effectivenessAnalyzer;
        effectivenessAnalyzer.setCostPerHour(100.0);
        
        // Simulate some effectiveness data
        effectivenessAnalyzer.recordDefectFound("TestSuite::effective_test", "BUG-001");
        effectivenessAnalyzer.recordDefectFound("TestSuite::effective_test", "BUG-002");
        effectivenessAnalyzer.recordMaintenanceTime("TestSuite::effective_test", 2.0, "Test refactoring");
        
        effectivenessAnalyzer.analyzeTestEffectiveness("TestSuite::effective_test");
        
        TestEffectivenessMetrics effectiveness = effectivenessAnalyzer.getTestEffectiveness("TestSuite::effective_test");
        qDebug() << "Test effectiveness analysis:";
        qDebug() << "- Effectiveness score:" << QString::number(effectiveness.effectivenessScore, 'f', 2);
        qDebug() << "- Value score:" << QString::number(effectiveness.valueScore, 'f', 2);
        qDebug() << "- Defects found:" << effectiveness.defectsFound;
        qDebug() << "- Maintenance cost: $" << QString::number(effectiveness.maintenanceCost, 'f', 2);
        qDebug() << "- Recommendation:" << effectiveness.recommendation;
        
        // HTML Report Generator
        HtmlReportGenerator htmlGenerator;
        htmlGenerator.setOutputDirectory("test_reports");
        htmlGenerator.setIncludeCharts(true);
        htmlGenerator.setIncludeInteractivity(true);
        
        // Connect signals
        connect(&htmlGenerator, &HtmlReportGenerator::reportGenerationCompleted,
                [](const QString& outputPath) {
                    qDebug() << "HTML report generated:" << outputPath;
                });
        
        // Create sample test results for report
        TestResults sampleResults;
        sampleResults.totalTests = 100;
        sampleResults.passedTests = 85;
        sampleResults.failedTests = 12;
        sampleResults.skippedTests = 3;
        sampleResults.executionTimeMs = 45000;
        sampleResults.codeCoverage = 0.87;
        
        // Add sample failure
        TestFailure failure;
        failure.testName = "SampleTest::failing_test";
        failure.category = "Unit";
        failure.errorMessage = "Assertion failed: expected 42, got 24";
        failure.timestamp = QDateTime::currentMSecsSinceEpoch();
        sampleResults.failures.append(failure);
        
        qDebug() << "Generating comprehensive HTML report...";
        // htmlGenerator.generateComprehensiveReport(sampleResults, "test_reports/comprehensive_report.html");
        
        qDebug() << "Advanced reporting demo completed\n";
    }
};

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    
    AdvancedTestingExample example;
    
    // Run the example
    QTimer::singleShot(0, &example, &AdvancedTestingExample::runExample);
    
    // Exit after a short delay to allow async operations to complete
    QTimer::singleShot(2000, &app, &QCoreApplication::quit);
    
    return app.exec();
}

#include "example_advanced_testing.moc"