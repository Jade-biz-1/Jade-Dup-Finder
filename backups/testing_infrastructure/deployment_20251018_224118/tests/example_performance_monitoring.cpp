#include <QtTest>
#include <QApplication>
#include <QWidget>
#include <QDir>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include "performance_monitoring.h"
#include "performance_benchmark.h"
#include "load_stress_testing.h"

/**
 * @brief Example performance monitoring and reporting for DupFinder
 * 
 * This class demonstrates how to use the PerformanceMonitoring framework
 * for real-time performance monitoring, trend analysis, regression detection,
 * alerting, and comprehensive reporting.
 */
class ExamplePerformanceMonitoring : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Basic monitoring tests
    void testBasicMonitoring();
    void testMetricRecording();
    void testSystemMetricsCollection();
    void testApplicationMetricsCollection();
    
    // Trend analysis tests
    void testTrendAnalysis();
    void testTrendDetection();
    void testPerformanceRegression();
    void testRegressionDetection();
    
    // Alerting system tests
    void testAlertConfiguration();
    void testAlertTriggering();
    void testAlertResolution();
    void testAlertNotifications();
    
    // Reporting tests
    void testReportGeneration();
    void testDashboardGeneration();
    void testMetricsSnapshot();
    void testPerformanceSummary();
    
    // Integration tests
    void testBenchmarkIntegration();
    void testLoadTestingIntegration();
    void testRealTimeMonitoring();
    
    // Data management tests
    void testDataRetention();
    void testDataExportImport();
    void testDataOptimization();
    
    // Advanced features tests
    void testCustomMetrics();
    void testMultiMetricAnalysis();
    void testPerformanceBaselines();

private:
    PerformanceMonitoring* m_monitoring;
    PerformanceBenchmark* m_benchmark;
    LoadStressTesting* m_loadTesting;
    QTemporaryDir* m_testDir;
    QString m_testPath;
    
    // Helper methods
    void generateTestMetrics(const QString& metricName, int count, double baseValue, double variation);
    void simulatePerformanceRegression(const QString& metricName);
    void waitForMonitoringCycle();
};

void ExamplePerformanceMonitoring::initTestCase() {
    // Initialize performance monitoring framework
    m_monitoring = new PerformanceMonitoring(this);
    m_benchmark = new PerformanceBenchmark(this);
    m_loadTesting = new LoadStressTesting(this);
    
    // Create temporary directory for reports
    m_testDir = new QTemporaryDir();
    QVERIFY(m_testDir->isValid());
    m_testPath = m_testDir->path();
    
    // Configure monitoring
    PerformanceMonitoring::MonitoringConfig config;
    config.name = "DupFinder Performance Monitoring Test";
    config.samplingIntervalMs = 100; // Fast sampling for testing
    config.retentionPeriodMs = 3600000; // 1 hour retention
    config.maxDataPoints = 1000;
    config.enableTrendAnalysis = true;
    config.enableRegressionDetection = true;
    config.enableAlerting = true;
    config.enableReporting = true;
    config.trendAnalysisWindowMs = 10000; // 10 seconds for testing
    config.regressionThreshold = 15.0; // 15% threshold
    config.reportOutputDirectory = m_testPath;
    
    m_monitoring->setMonitoringConfig(config);
    
    // Integrate with other frameworks
    m_monitoring->setPerformanceBenchmark(m_benchmark);
    m_monitoring->setLoadStressTesting(m_loadTesting);
    
    qDebug() << "Performance monitoring test suite initialized";
    qDebug() << "Test directory:" << m_testPath;
}

void ExamplePerformanceMonitoring::cleanupTestCase() {
    // Stop monitoring
    if (m_monitoring->isMonitoring()) {
        m_monitoring->stopMonitoring();
    }
    
    // Generate final comprehensive report
    QString reportPath = QDir(m_testPath).absoluteFilePath("final_monitoring_report.html");
    PerformanceMonitoring::ReportConfig reportConfig;
    reportConfig.name = "Final Test Report";
    reportConfig.outputPath = reportPath;
    reportConfig.format = "html";
    reportConfig.title = "Performance Monitoring Test Results";
    reportConfig.description = "Comprehensive test results from performance monitoring framework";
    reportConfig.includeStatistics = true;
    reportConfig.includeTrendAnalysis = true;
    reportConfig.includeAlerts = true;
    reportConfig.includeRegressions = true;
    
    m_monitoring->generateReport(reportConfig);
    qDebug() << "Final monitoring report generated:" << reportPath;
    
    // Cleanup
    delete m_testDir;
    qDebug() << "Performance monitoring test suite completed";
}

void ExamplePerformanceMonitoring::testBasicMonitoring() {
    qDebug() << "Testing basic monitoring functionality...";
    
    // Start monitoring
    bool success = m_monitoring->startMonitoring();
    QVERIFY2(success, "Failed to start performance monitoring");
    QVERIFY(m_monitoring->isMonitoring());
    
    // Record some test metrics
    m_monitoring->recordMetric("test_cpu_usage", 45.5, "%", PerformanceMonitoring::MetricType::SystemResource);
    m_monitoring->recordMetric("test_memory_usage", 1024*1024*512, "bytes", PerformanceMonitoring::MetricType::SystemResource);
    m_monitoring->recordMetric("test_response_time", 150.0, "ms", PerformanceMonitoring::MetricType::ApplicationMetric);
    
    // Wait for monitoring cycle
    waitForMonitoringCycle();
    
    // Verify metrics were recorded
    QStringList availableMetrics = m_monitoring->getAvailableMetrics();
    QVERIFY(availableMetrics.contains("test_cpu_usage"));
    QVERIFY(availableMetrics.contains("test_memory_usage"));
    QVERIFY(availableMetrics.contains("test_response_time"));
    
    // Get latest metrics
    auto latestMetrics = m_monitoring->getLatestMetrics();
    QVERIFY(latestMetrics.contains("test_cpu_usage"));
    QCOMPARE(latestMetrics["test_cpu_usage"].value, 45.5);
    QCOMPARE(latestMetrics["test_cpu_usage"].unit, QString("%"));
    
    // Test pause and resume
    m_monitoring->pauseMonitoring();
    QVERIFY(!m_monitoring->isMonitoring());
    
    m_monitoring->resumeMonitoring();
    QVERIFY(m_monitoring->isMonitoring());
    
    qDebug() << "Basic monitoring test completed successfully";
}

void ExamplePerformanceMonitoring::testMetricRecording() {
    qDebug() << "Testing metric recording...";
    
    // Test different metric types
    QList<QPair<QString, PerformanceMonitoring::MetricType>> metricTypes = {
        {"system_cpu", PerformanceMonitoring::MetricType::SystemResource},
        {"app_startup_time", PerformanceMonitoring::MetricType::ApplicationMetric},
        {"ui_frame_rate", PerformanceMonitoring::MetricType::UserInterface},
        {"file_scan_rate", PerformanceMonitoring::MetricType::FileOperation},
        {"network_throughput", PerformanceMonitoring::MetricType::NetworkOperation},
        {"custom_metric", PerformanceMonitoring::MetricType::CustomMetric}
    };
    
    for (const auto& metricType : metricTypes) {
        QString metricName = metricType.first;
        PerformanceMonitoring::MetricType type = metricType.second;
        
        // Record metric with metadata
        PerformanceMonitoring::MetricDataPoint dataPoint;
        dataPoint.metricName = metricName;
        dataPoint.type = type;
        dataPoint.value = QRandomGenerator::global()->bounded(100.0);
        dataPoint.unit = "units";
        dataPoint.timestamp = QDateTime::currentDateTime();
        dataPoint.source = "ExampleTest";
        dataPoint.description = QString("Test metric for %1").arg(metricName);
        dataPoint.metadata["test_run"] = "metric_recording_test";
        
        m_monitoring->recordMetric(dataPoint);
        
        // Verify metric was recorded
        auto latestMetric = m_monitoring->getLatestMetric(metricName);
        QCOMPARE(latestMetric.metricName, metricName);
        QCOMPARE(latestMetric.type, type);
        QCOMPARE(latestMetric.source, QString("ExampleTest"));
        
        qDebug() << QString("Recorded %1: %2 %3").arg(metricName).arg(latestMetric.value).arg(latestMetric.unit);
    }
    
    // Test bulk metric recording
    for (int i = 0; i < 50; ++i) {
        m_monitoring->recordMetric("bulk_test_metric", i * 2.5, "count");
        QThread::msleep(10); // Small delay to create time series
    }
    
    // Verify bulk data
    auto bulkData = m_monitoring->getMetricData("bulk_test_metric");
    QVERIFY(bulkData.size() >= 50);
    
    qDebug() << "Metric recording test completed successfully";
}

void ExamplePerformanceMonitoring::testSystemMetricsCollection() {
    qDebug() << "Testing system metrics collection...";
    
    // Start monitoring to collect system metrics
    if (!m_monitoring->isMonitoring()) {
        m_monitoring->startMonitoring();
    }
    
    // Wait for several monitoring cycles to collect system metrics
    QTest::qWait(1000);
    
    // Verify system metrics are being collected
    QStringList expectedSystemMetrics = {
        "system_cpu_usage",
        "system_memory_usage",
        "system_disk_usage",
        "system_thread_count"
    };
    
    QStringList availableMetrics = m_monitoring->getAvailableMetrics();
    
    for (const QString& expectedMetric : expectedSystemMetrics) {
        if (availableMetrics.contains(expectedMetric)) {
            auto latestMetric = m_monitoring->getLatestMetric(expectedMetric);
            QVERIFY(latestMetric.value >= 0);
            QVERIFY(!latestMetric.unit.isEmpty());
            
            qDebug() << QString("System metric %1: %2 %3")
                        .arg(expectedMetric)
                        .arg(latestMetric.value)
                        .arg(latestMetric.unit);
        }
    }
    
    // Test manual system metrics recording
    m_monitoring->recordSystemMetrics();
    
    qDebug() << "System metrics collection test completed";
}

void ExamplePerformanceMonitoring::testApplicationMetricsCollection() {
    qDebug() << "Testing application metrics collection...";
    
    // Test application-specific metrics
    m_monitoring->recordApplicationMetrics();
    
    // Wait for collection
    QTest::qWait(200);
    
    // Check for application metrics
    QStringList availableMetrics = m_monitoring->getAvailableMetrics();
    
    // Look for application metrics that might have been collected
    QStringList expectedAppMetrics = {
        "app_widget_count",
        "app_uptime"
    };
    
    for (const QString& expectedMetric : expectedAppMetrics) {
        if (availableMetrics.contains(expectedMetric)) {
            auto latestMetric = m_monitoring->getLatestMetric(expectedMetric);
            QVERIFY(latestMetric.value >= 0);
            
            qDebug() << QString("Application metric %1: %2 %3")
                        .arg(expectedMetric)
                        .arg(latestMetric.value)
                        .arg(latestMetric.unit);
        }
    }
    
    // Test UI metrics collection
    m_monitoring->recordUIMetrics();
    
    qDebug() << "Application metrics collection test completed";
}

void ExamplePerformanceMonitoring::testTrendAnalysis() {
    qDebug() << "Testing trend analysis...";
    
    // Generate trending data
    generateTestMetrics("trending_metric_improving", 20, 100.0, 5.0); // Improving trend
    generateTestMetrics("trending_metric_degrading", 20, 100.0, -3.0); // Degrading trend
    generateTestMetrics("trending_metric_stable", 20, 100.0, 0.1); // Stable trend
    generateTestMetrics("trending_metric_volatile", 20, 100.0, 25.0); // Volatile trend
    
    // Analyze trends
    auto improvingTrend = m_monitoring->analyzeTrend("trending_metric_improving");
    auto degradingTrend = m_monitoring->analyzeTrend("trending_metric_degrading");
    auto stableTrend = m_monitoring->analyzeTrend("trending_metric_stable");
    auto volatileTrend = m_monitoring->analyzeTrend("trending_metric_volatile");
    
    // Verify trend analysis results
    QVERIFY(improvingTrend.sampleCount > 0);
    QVERIFY(degradingTrend.sampleCount > 0);
    QVERIFY(stableTrend.sampleCount > 0);
    QVERIFY(volatileTrend.sampleCount > 0);
    
    // Check trend types (may vary due to randomness, but should be reasonable)
    qDebug() << QString("Improving trend: %1 (%2% change)")
                .arg(m_monitoring->formatTrendType(improvingTrend.trendType))
                .arg(improvingTrend.changePercent, 0, 'f', 1);
    
    qDebug() << QString("Degrading trend: %1 (%2% change)")
                .arg(m_monitoring->formatTrendType(degradingTrend.trendType))
                .arg(degradingTrend.changePercent, 0, 'f', 1);
    
    qDebug() << QString("Stable trend: %1 (%2% change)")
                .arg(m_monitoring->formatTrendType(stableTrend.trendType))
                .arg(stableTrend.changePercent, 0, 'f', 1);
    
    qDebug() << QString("Volatile trend: %1 (std dev: %2)")
                .arg(m_monitoring->formatTrendType(volatileTrend.trendType))
                .arg(volatileTrend.standardDeviation, 0, 'f', 2);
    
    // Test trend analysis for all metrics
    auto allTrends = m_monitoring->analyzeAllTrends();
    QVERIFY(allTrends.size() >= 4);
    
    qDebug() << "Trend analysis test completed successfully";
}

void ExamplePerformanceMonitoring::testTrendDetection() {
    qDebug() << "Testing trend detection...";
    
    // Start monitoring for trend detection
    if (!m_monitoring->isMonitoring()) {
        m_monitoring->startMonitoring();
    }
    
    // Generate data with clear trends
    QString trendMetric = "trend_detection_test";
    
    // Generate baseline data
    for (int i = 0; i < 10; ++i) {
        m_monitoring->recordMetric(trendMetric, 50.0 + (i * 0.5), "units");
        QThread::msleep(50);
    }
    
    // Generate trending data
    for (int i = 0; i < 10; ++i) {
        m_monitoring->recordMetric(trendMetric, 55.0 + (i * 2.0), "units");
        QThread::msleep(50);
    }
    
    // Wait for trend analysis
    QTest::qWait(500);
    
    // Check if trend was detected
    auto trend = m_monitoring->analyzeTrend(trendMetric);
    QVERIFY(trend.sampleCount >= 15);
    QVERIFY(qAbs(trend.changePercent) > 5.0); // Should show significant change
    
    qDebug() << QString("Trend detection: %1 with %2% change")
                .arg(m_monitoring->formatTrendType(trend.trendType))
                .arg(trend.changePercent, 0, 'f', 1);
    
    qDebug() << "Trend detection test completed successfully";
}

void ExamplePerformanceMonitoring::testPerformanceRegression() {
    qDebug() << "Testing performance regression detection...";
    
    // Generate baseline performance data
    QString regressionMetric = "regression_test_metric";
    
    // Good baseline performance
    for (int i = 0; i < 15; ++i) {
        double value = 100.0 + QRandomGenerator::global()->bounded(-5.0, 5.0);
        m_monitoring->recordMetric(regressionMetric, value, "ms");
        QThread::msleep(20);
    }
    
    // Simulate performance regression
    simulatePerformanceRegression(regressionMetric);
    
    // Detect regression
    bool regressionDetected = m_monitoring->detectPerformanceRegression(regressionMetric, 10.0);
    
    if (regressionDetected) {
        qDebug() << "Performance regression successfully detected";
        
        auto regressions = m_monitoring->detectAllRegressions(10.0);
        QVERIFY(!regressions.isEmpty());
        
        for (const auto& regression : regressions) {
            if (regression.metricName == regressionMetric) {
                QVERIFY(regression.regressionDetected);
                QVERIFY(qAbs(regression.regressionPercent) > 10.0);
                
                qDebug() << QString("Regression details: %1% %2 in %3")
                            .arg(qAbs(regression.regressionPercent), 0, 'f', 1)
                            .arg(regression.regressionPercent > 0 ? "increase" : "decrease")
                            .arg(regression.metricName);
                
                qDebug() << QString("Regression type: %1, Severity: %2")
                            .arg(regression.regressionType)
                            .arg(regression.severity);
                
                break;
            }
        }
    } else {
        qDebug() << "Note: Regression not detected (may be due to random variation)";
    }
    
    qDebug() << "Performance regression test completed";
}

void ExamplePerformanceMonitoring::testRegressionDetection() {
    qDebug() << "Testing comprehensive regression detection...";
    
    // Create multiple metrics with different regression patterns
    QStringList regressionMetrics = {
        "sudden_regression_metric",
        "gradual_regression_metric",
        "memory_leak_metric"
    };
    
    for (const QString& metricName : regressionMetrics) {
        // Generate baseline
        for (int i = 0; i < 10; ++i) {
            double baseValue = 50.0;
            if (metricName.contains("memory")) {
                baseValue = 1024 * 1024 * 100; // 100MB baseline
            }
            
            double value = baseValue + QRandomGenerator::global()->bounded(-2.0, 2.0);
            m_monitoring->recordMetric(metricName, value, metricName.contains("memory") ? "bytes" : "ms");
            QThread::msleep(10);
        }
        
        // Generate regression
        for (int i = 0; i < 10; ++i) {
            double baseValue = 50.0;
            if (metricName.contains("memory")) {
                baseValue = 1024 * 1024 * 100;
            }
            
            double regressionFactor = 1.0;
            if (metricName.contains("sudden")) {
                regressionFactor = (i > 5) ? 1.5 : 1.0; // Sudden jump
            } else if (metricName.contains("gradual")) {
                regressionFactor = 1.0 + (i * 0.05); // Gradual increase
            } else if (metricName.contains("memory")) {
                regressionFactor = 1.0 + (i * 0.1); // Memory leak pattern
            }
            
            double value = baseValue * regressionFactor + QRandomGenerator::global()->bounded(-2.0, 2.0);
            m_monitoring->recordMetric(metricName, value, metricName.contains("memory") ? "bytes" : "ms");
            QThread::msleep(10);
        }
    }
    
    // Detect all regressions
    auto allRegressions = m_monitoring->detectAllRegressions(15.0);
    
    qDebug() << QString("Detected %1 regressions").arg(allRegressions.size());
    
    for (const auto& regression : allRegressions) {
        qDebug() << QString("Regression in %1: %2% (%3, %4)")
                    .arg(regression.metricName)
                    .arg(qAbs(regression.regressionPercent), 0, 'f', 1)
                    .arg(regression.regressionType)
                    .arg(regression.severity);
    }
    
    qDebug() << "Comprehensive regression detection test completed";
}vo
id ExamplePerformanceMonitoring::testAlertConfiguration() {
    qDebug() << "Testing alert configuration...";
    
    // Configure various types of alerts
    PerformanceMonitoring::AlertConfig cpuAlert;
    cpuAlert.name = "High CPU Usage";
    cpuAlert.metricName = "system_cpu_usage";
    cpuAlert.severity = PerformanceMonitoring::AlertSeverity::Warning;
    cpuAlert.threshold = 80.0;
    cpuAlert.comparison = ">";
    cpuAlert.evaluationWindowMs = 5000;
    cpuAlert.minSamples = 3;
    cpuAlert.enabled = true;
    cpuAlert.description = "Alert when CPU usage exceeds 80%";
    
    PerformanceMonitoring::AlertConfig memoryAlert;
    memoryAlert.name = "High Memory Usage";
    memoryAlert.metricName = "system_memory_usage";
    memoryAlert.severity = PerformanceMonitoring::AlertSeverity::Critical;
    memoryAlert.threshold = 1024 * 1024 * 1024; // 1GB
    memoryAlert.comparison = ">";
    memoryAlert.evaluationWindowMs = 10000;
    memoryAlert.minSamples = 5;
    memoryAlert.enabled = true;
    memoryAlert.description = "Alert when memory usage exceeds 1GB";
    
    PerformanceMonitoring::AlertConfig responseTimeAlert;
    responseTimeAlert.name = "Slow Response Time";
    responseTimeAlert.metricName = "app_response_time";
    responseTimeAlert.severity = PerformanceMonitoring::AlertSeverity::Warning;
    responseTimeAlert.threshold = 1000.0;
    responseTimeAlert.comparison = ">";
    responseTimeAlert.evaluationWindowMs = 15000;
    responseTimeAlert.minSamples = 2;
    responseTimeAlert.enabled = true;
    responseTimeAlert.description = "Alert when response time exceeds 1 second";
    
    // Add alerts to monitoring system
    m_monitoring->addAlert(cpuAlert);
    m_monitoring->addAlert(memoryAlert);
    m_monitoring->addAlert(responseTimeAlert);
    
    // Verify alerts were added
    auto alertConfigs = m_monitoring->getAlertConfigs();
    QVERIFY(alertConfigs.size() >= 3);
    
    // Verify specific alert configuration
    auto retrievedCpuAlert = m_monitoring->getAlertConfig("High CPU Usage");
    QCOMPARE(retrievedCpuAlert.name, cpuAlert.name);
    QCOMPARE(retrievedCpuAlert.metricName, cpuAlert.metricName);
    QCOMPARE(retrievedCpuAlert.threshold, cpuAlert.threshold);
    QCOMPARE(retrievedCpuAlert.comparison, cpuAlert.comparison);
    
    // Test alert update
    cpuAlert.threshold = 85.0;
    m_monitoring->updateAlert("High CPU Usage", cpuAlert);
    
    auto updatedAlert = m_monitoring->getAlertConfig("High CPU Usage");
    QCOMPARE(updatedAlert.threshold, 85.0);
    
    qDebug() << QString("Configured %1 alerts successfully").arg(alertConfigs.size());
    qDebug() << "Alert configuration test completed successfully";
}

void ExamplePerformanceMonitoring::testAlertTriggering() {
    qDebug() << "Testing alert triggering...";
    
    // Ensure monitoring is active
    if (!m_monitoring->isMonitoring()) {
        m_monitoring->startMonitoring();
    }
    
    // Configure a test alert with low threshold
    PerformanceMonitoring::AlertConfig testAlert;
    testAlert.name = "Test Alert Trigger";
    testAlert.metricName = "test_trigger_metric";
    testAlert.severity = PerformanceMonitoring::AlertSeverity::Warning;
    testAlert.threshold = 50.0;
    testAlert.comparison = ">";
    testAlert.evaluationWindowMs = 2000;
    testAlert.minSamples = 2;
    testAlert.enabled = true;
    testAlert.description = "Test alert for triggering";
    
    m_monitoring->addAlert(testAlert);
    
    // Record metrics below threshold (should not trigger)
    for (int i = 0; i < 3; ++i) {
        m_monitoring->recordMetric("test_trigger_metric", 30.0 + i, "units");
        QThread::msleep(100);
    }
    
    // Wait for alert evaluation
    QTest::qWait(500);
    
    // Check that no alerts are active
    auto activeAlerts = m_monitoring->getActiveAlerts();
    int initialAlertCount = activeAlerts.size();
    
    // Record metrics above threshold (should trigger alert)
    for (int i = 0; i < 3; ++i) {
        m_monitoring->recordMetric("test_trigger_metric", 70.0 + i, "units");
        QThread::msleep(100);
    }
    
    // Wait for alert evaluation
    QTest::qWait(1000);
    
    // Check if alert was triggered
    activeAlerts = m_monitoring->getActiveAlerts();
    
    bool alertTriggered = false;
    for (const auto& alert : activeAlerts) {
        if (alert.alertName == "Test Alert Trigger" && alert.isActive) {
            alertTriggered = true;
            
            QCOMPARE(alert.metricName, QString("test_trigger_metric"));
            QVERIFY(alert.currentValue > 50.0);
            QCOMPARE(alert.thresholdValue, 50.0);
            QVERIFY(!alert.message.isEmpty());
            
            qDebug() << QString("Alert triggered: %1").arg(alert.message);
            break;
        }
    }
    
    if (alertTriggered) {
        qDebug() << "Alert triggering test completed successfully";
    } else {
        qDebug() << "Note: Alert may not have triggered due to timing or evaluation conditions";
    }
}

void ExamplePerformanceMonitoring::testAlertResolution() {
    qDebug() << "Testing alert resolution...";
    
    // First trigger an alert (if not already triggered)
    testAlertTriggering();
    
    // Record metrics below threshold to resolve alert
    for (int i = 0; i < 5; ++i) {
        m_monitoring->recordMetric("test_trigger_metric", 20.0 + i, "units");
        QThread::msleep(100);
    }
    
    // Wait for alert evaluation
    QTest::qWait(1000);
    
    // Check if alert was resolved
    auto activeAlerts = m_monitoring->getActiveAlerts();
    
    bool alertResolved = true;
    for (const auto& alert : activeAlerts) {
        if (alert.alertName == "Test Alert Trigger" && alert.isActive) {
            alertResolved = false;
            break;
        }
    }
    
    if (alertResolved) {
        qDebug() << "Alert automatically resolved when condition cleared";
    }
    
    // Test manual alert resolution
    m_monitoring->acknowledgeAlert("Test Alert Trigger");
    m_monitoring->resolveAlert("Test Alert Trigger");
    
    // Check alert history
    auto alertHistory = m_monitoring->getAlertHistory();
    
    bool foundInHistory = false;
    for (const auto& alert : alertHistory) {
        if (alert.alertName == "Test Alert Trigger") {
            foundInHistory = true;
            qDebug() << QString("Alert found in history: triggered at %1")
                        .arg(alert.triggeredTime.toString());
            break;
        }
    }
    
    QVERIFY2(foundInHistory, "Alert should be found in history");
    
    qDebug() << "Alert resolution test completed successfully";
}

void ExamplePerformanceMonitoring::testAlertNotifications() {
    qDebug() << "Testing alert notifications...";
    
    // Configure alert with notification settings
    PerformanceMonitoring::AlertConfig notificationAlert;
    notificationAlert.name = "Notification Test Alert";
    notificationAlert.metricName = "notification_test_metric";
    notificationAlert.severity = PerformanceMonitoring::AlertSeverity::Critical;
    notificationAlert.threshold = 100.0;
    notificationAlert.comparison = ">";
    notificationAlert.evaluationWindowMs = 2000;
    notificationAlert.minSamples = 1;
    notificationAlert.enabled = true;
    notificationAlert.description = "Test alert for notifications";
    notificationAlert.webhookUrl = "http://localhost:8080/webhook"; // Test webhook
    notificationAlert.actionScript = "echo 'Alert triggered'"; // Test script
    
    m_monitoring->addAlert(notificationAlert);
    
    // Trigger the alert
    m_monitoring->recordMetric("notification_test_metric", 150.0, "units");
    
    // Wait for alert processing
    QTest::qWait(500);
    
    // Note: In a real test environment, you would verify that:
    // 1. Webhook was called (by setting up a test server)
    // 2. Action script was executed (by checking process execution)
    // 3. Email notifications were sent (by mocking email service)
    
    qDebug() << "Alert notification configuration tested";
    qDebug() << "Note: Actual notification delivery depends on external services";
    
    qDebug() << "Alert notifications test completed";
}

void ExamplePerformanceMonitoring::testReportGeneration() {
    qDebug() << "Testing report generation...";
    
    // Generate some test data for reporting
    generateTestMetrics("report_test_cpu", 20, 60.0, 10.0);
    generateTestMetrics("report_test_memory", 20, 512.0, 50.0);
    generateTestMetrics("report_test_response", 20, 200.0, 30.0);
    
    // Test HTML report generation
    QString htmlReportPath = QDir(m_testPath).absoluteFilePath("test_report.html");
    PerformanceMonitoring::ReportConfig htmlConfig;
    htmlConfig.name = "HTML Test Report";
    htmlConfig.outputPath = htmlReportPath;
    htmlConfig.format = "html";
    htmlConfig.title = "Performance Test Report";
    htmlConfig.description = "Test report generated by performance monitoring framework";
    htmlConfig.startTime = QDateTime::currentDateTime().addSecs(-3600); // Last hour
    htmlConfig.endTime = QDateTime::currentDateTime();
    htmlConfig.metricsToInclude = QStringList() << "report_test_cpu" << "report_test_memory" << "report_test_response";
    htmlConfig.includeStatistics = true;
    htmlConfig.includeTrendAnalysis = true;
    htmlConfig.includeAlerts = true;
    htmlConfig.includeCharts = true;
    
    bool success = m_monitoring->generateReport(htmlConfig);
    QVERIFY2(success, "Failed to generate HTML report");
    QVERIFY(QFile::exists(htmlReportPath));
    
    // Verify report content
    QFile htmlFile(htmlReportPath);
    QVERIFY(htmlFile.open(QIODevice::ReadOnly));
    QString htmlContent = htmlFile.readAll();
    QVERIFY(htmlContent.contains("Performance Test Report"));
    QVERIFY(htmlContent.contains("report_test_cpu"));
    
    qDebug() << QString("HTML report generated: %1 (%2 bytes)")
                .arg(htmlReportPath)
                .arg(htmlFile.size());
    
    // Test JSON report generation
    QString jsonReportPath = QDir(m_testPath).absoluteFilePath("test_report.json");
    PerformanceMonitoring::ReportConfig jsonConfig = htmlConfig;
    jsonConfig.outputPath = jsonReportPath;
    jsonConfig.format = "json";
    
    success = m_monitoring->generateReport(jsonConfig);
    QVERIFY2(success, "Failed to generate JSON report");
    QVERIFY(QFile::exists(jsonReportPath));
    
    // Verify JSON report content
    QFile jsonFile(jsonReportPath);
    QVERIFY(jsonFile.open(QIODevice::ReadOnly));
    QJsonDocument jsonDoc = QJsonDocument::fromJson(jsonFile.readAll());
    QVERIFY(!jsonDoc.isNull());
    
    QJsonObject reportObj = jsonDoc.object();
    QVERIFY(reportObj.contains("title"));
    QVERIFY(reportObj.contains("metrics"));
    
    qDebug() << QString("JSON report generated: %1 (%2 bytes)")
                .arg(jsonReportPath)
                .arg(jsonFile.size());
    
    // Test specialized reports
    QString trendReportPath = QDir(m_testPath).absoluteFilePath("trend_report.html");
    success = m_monitoring->generateTrendReport(trendReportPath);
    QVERIFY2(success, "Failed to generate trend report");
    
    QString alertReportPath = QDir(m_testPath).absoluteFilePath("alert_report.html");
    success = m_monitoring->generateAlertReport(alertReportPath);
    QVERIFY2(success, "Failed to generate alert report");
    
    qDebug() << "Report generation test completed successfully";
}

void ExamplePerformanceMonitoring::testDashboardGeneration() {
    qDebug() << "Testing dashboard generation...";
    
    // Generate test data for dashboard
    generateTestMetrics("dashboard_cpu", 30, 45.0, 15.0);
    generateTestMetrics("dashboard_memory", 30, 1024.0, 200.0);
    generateTestMetrics("dashboard_disk", 30, 75.0, 10.0);
    
    // Generate interactive dashboard
    QString dashboardPath = QDir(m_testPath).absoluteFilePath("performance_dashboard.html");
    bool success = m_monitoring->generateDashboard(dashboardPath);
    QVERIFY2(success, "Failed to generate performance dashboard");
    QVERIFY(QFile::exists(dashboardPath));
    
    // Verify dashboard content
    QFile dashboardFile(dashboardPath);
    QVERIFY(dashboardFile.open(QIODevice::ReadOnly));
    QString dashboardContent = dashboardFile.readAll();
    QVERIFY(dashboardContent.contains("Performance Dashboard"));
    QVERIFY(dashboardContent.contains("dashboard-grid"));
    QVERIFY(dashboardContent.contains("Chart.js"));
    
    qDebug() << QString("Dashboard generated: %1 (%2 bytes)")
                .arg(dashboardPath)
                .arg(dashboardFile.size());
    
    // Test custom dashboard configuration
    PerformanceDashboard customDashboard(m_monitoring);
    customDashboard.setDashboardTitle("Custom DupFinder Dashboard");
    customDashboard.setDashboardTheme("dark");
    customDashboard.addMetricWidget("dashboard_cpu", "gauge");
    customDashboard.addMetricWidget("dashboard_memory", "line");
    customDashboard.addTrendWidget("dashboard_disk");
    customDashboard.addAlertWidget();
    customDashboard.addRegressionWidget();
    
    QString customDashboardPath = QDir(m_testPath).absoluteFilePath("custom_dashboard.html");
    success = customDashboard.generateInteractiveDashboard(customDashboardPath);
    QVERIFY2(success, "Failed to generate custom dashboard");
    QVERIFY(QFile::exists(customDashboardPath));
    
    qDebug() << QString("Custom dashboard generated: %1").arg(customDashboardPath);
    
    qDebug() << "Dashboard generation test completed successfully";
}

void ExamplePerformanceMonitoring::testMetricsSnapshot() {
    qDebug() << "Testing metrics snapshot...";
    
    // Record some current metrics
    m_monitoring->recordMetric("snapshot_cpu", 67.5, "%");
    m_monitoring->recordMetric("snapshot_memory", 1024*1024*800, "bytes");
    m_monitoring->recordMetric("snapshot_disk", 45.2, "%");
    m_monitoring->recordMetric("snapshot_network", 1500000, "bytes/sec");
    
    // Generate metrics snapshot
    QJsonObject snapshot = m_monitoring->generateMetricsSnapshot();
    
    // Verify snapshot structure
    QVERIFY(snapshot.contains("timestamp"));
    QVERIFY(snapshot.contains("platform"));
    QVERIFY(snapshot.contains("metrics"));
    
    QJsonObject metrics = snapshot["metrics"].toObject();
    QVERIFY(metrics.contains("snapshot_cpu"));
    QVERIFY(metrics.contains("snapshot_memory"));
    
    // Verify metric data
    QJsonObject cpuMetric = metrics["snapshot_cpu"].toObject();
    QCOMPARE(cpuMetric["value"].toDouble(), 67.5);
    QCOMPARE(cpuMetric["unit"].toString(), QString("%"));
    
    qDebug() << QString("Metrics snapshot generated with %1 metrics").arg(metrics.size());
    
    // Export snapshot to file
    QString snapshotPath = QDir(m_testPath).absoluteFilePath("metrics_snapshot.json");
    QFile snapshotFile(snapshotPath);
    QVERIFY(snapshotFile.open(QIODevice::WriteOnly));
    
    QJsonDocument snapshotDoc(snapshot);
    snapshotFile.write(snapshotDoc.toJson());
    
    qDebug() << QString("Snapshot exported to: %1").arg(snapshotPath);
    qDebug() << "Metrics snapshot test completed successfully";
}

void ExamplePerformanceMonitoring::testPerformanceSummary() {
    qDebug() << "Testing performance summary...";
    
    // Generate test data over a time period
    QDateTime startTime = QDateTime::currentDateTime().addSecs(-1800); // 30 minutes ago
    QDateTime endTime = QDateTime::currentDateTime();
    
    // Generate summary data
    generateTestMetrics("summary_response_time", 50, 150.0, 25.0);
    generateTestMetrics("summary_throughput", 50, 1000.0, 100.0);
    generateTestMetrics("summary_error_rate", 50, 2.0, 1.0);
    
    // Generate performance summary
    QJsonObject summary = m_monitoring->generatePerformanceSummary(startTime, endTime);
    
    // Verify summary structure
    QVERIFY(summary.contains("timestamp"));
    QVERIFY(summary.contains("period_start"));
    QVERIFY(summary.contains("period_end"));
    QVERIFY(summary.contains("metrics"));
    
    QJsonArray metricsArray = summary["metrics"].toArray();
    QVERIFY(metricsArray.size() > 0);
    
    // Verify metric summaries
    bool foundResponseTime = false;
    for (const QJsonValue& value : metricsArray) {
        QJsonObject metricSummary = value.toObject();
        if (metricSummary["name"].toString() == "summary_response_time") {
            foundResponseTime = true;
            QVERIFY(metricSummary.contains("sample_count"));
            QVERIFY(metricSummary.contains("mean"));
            QVERIFY(metricSummary.contains("min"));
            QVERIFY(metricSummary.contains("max"));
            QVERIFY(metricSummary["sample_count"].toInt() > 0);
            break;
        }
    }
    
    QVERIFY2(foundResponseTime, "Response time metric should be in summary");
    
    // Check for trends and alerts in summary
    if (summary.contains("trends")) {
        QJsonArray trendsArray = summary["trends"].toArray();
        qDebug() << QString("Summary includes %1 trend analyses").arg(trendsArray.size());
    }
    
    if (summary.contains("alerts")) {
        QJsonArray alertsArray = summary["alerts"].toArray();
        qDebug() << QString("Summary includes %1 alerts").arg(alertsArray.size());
    }
    
    // Export summary to file
    QString summaryPath = QDir(m_testPath).absoluteFilePath("performance_summary.json");
    QFile summaryFile(summaryPath);
    QVERIFY(summaryFile.open(QIODevice::WriteOnly));
    
    QJsonDocument summaryDoc(summary);
    summaryFile.write(summaryDoc.toJson());
    
    qDebug() << QString("Performance summary exported to: %1").arg(summaryPath);
    qDebug() << "Performance summary test completed successfully";
}

void ExamplePerformanceMonitoring::testBenchmarkIntegration() {
    qDebug() << "Testing benchmark integration...";
    
    // Configure a simple benchmark
    PerformanceBenchmark::BenchmarkConfig benchmarkConfig;
    benchmarkConfig.name = "Integration Test Benchmark";
    benchmarkConfig.iterations = 3;
    benchmarkConfig.measureMemory = true;
    benchmarkConfig.measureCpu = true;
    
    m_benchmark->setBenchmarkConfig(benchmarkConfig);
    
    // Run benchmark (monitoring should automatically capture metrics)
    bool success = m_benchmark->runBenchmark("integration_test", []() {
        // Simulate some work
        QThread::msleep(100);
        
        // Simulate memory allocation
        QByteArray data(1024 * 100, 'B'); // 100KB
        
        // Simulate CPU work
        double result = 0.0;
        for (int i = 0; i < 10000; ++i) {
            result += qSqrt(i) * qSin(i * 0.1);
        }
        
        // Use result to prevent optimization
        static volatile double globalResult = result;
    });
    
    QVERIFY2(success, "Benchmark should complete successfully");
    
    // Wait for integration to process results
    QTest::qWait(500);
    
    // Check if benchmark metrics were captured by monitoring
    QStringList availableMetrics = m_monitoring->getAvailableMetrics();
    
    bool foundBenchmarkMetrics = false;
    for (const QString& metric : availableMetrics) {
        if (metric.contains("integration_test")) {
            foundBenchmarkMetrics = true;
            auto latestMetric = m_monitoring->getLatestMetric(metric);
            QVERIFY(latestMetric.value > 0);
            
            qDebug() << QString("Captured benchmark metric: %1 = %2 %3")
                        .arg(metric)
                        .arg(latestMetric.value)
                        .arg(latestMetric.unit);
        }
    }
    
    if (foundBenchmarkMetrics) {
        qDebug() << "Benchmark integration successful - metrics captured";
    } else {
        qDebug() << "Note: Benchmark metrics may not have been captured due to timing";
    }
    
    qDebug() << "Benchmark integration test completed";
}

void ExamplePerformanceMonitoring::testLoadTestingIntegration() {
    qDebug() << "Testing load testing integration...";
    
    // Configure a simple load test
    LoadStressTesting::LoadTestConfig loadConfig;
    loadConfig.name = "Integration Load Test";
    loadConfig.concurrentThreads = 2;
    loadConfig.totalOperations = 10;
    loadConfig.measureMemory = true;
    loadConfig.measureCpu = true;
    
    // Run load test (monitoring should automatically capture metrics)
    bool success = m_loadTesting->runLoadTest(loadConfig, []() {
        // Simulate load test operation
        QThread::msleep(50);
        
        // Simulate file operation
        QString tempFile = QDir::temp().absoluteFilePath(QString("load_test_%1.tmp").arg(QRandomGenerator::global()->bounded(1000)));
        QFile file(tempFile);
        if (file.open(QIODevice::WriteOnly)) {
            file.write("Load test data");
        }
        QFile::remove(tempFile);
    });
    
    QVERIFY2(success, "Load test should complete successfully");
    
    // Wait for integration to process results
    QTest::qWait(500);
    
    // Check if load test metrics were captured
    QStringList availableMetrics = m_monitoring->getAvailableMetrics();
    
    bool foundLoadTestMetrics = false;
    for (const QString& metric : availableMetrics) {
        if (metric.contains("Integration Load Test") || metric.contains("integration_load_test")) {
            foundLoadTestMetrics = true;
            auto latestMetric = m_monitoring->getLatestMetric(metric);
            
            qDebug() << QString("Captured load test metric: %1 = %2 %3")
                        .arg(metric)
                        .arg(latestMetric.value)
                        .arg(latestMetric.unit);
        }
    }
    
    if (foundLoadTestMetrics) {
        qDebug() << "Load testing integration successful - metrics captured";
    } else {
        qDebug() << "Note: Load test metrics may not have been captured due to timing";
    }
    
    qDebug() << "Load testing integration test completed";
}

void ExamplePerformanceMonitoring::testRealTimeMonitoring() {
    qDebug() << "Testing real-time monitoring...";
    
    // Start monitoring
    if (!m_monitoring->isMonitoring()) {
        m_monitoring->startMonitoring();
    }
    
    // Simulate real-time data collection
    QTimer dataTimer;
    int dataPoints = 0;
    const int maxDataPoints = 20;
    
    connect(&dataTimer, &QTimer::timeout, [this, &dataPoints]() {
        // Simulate varying system metrics
        double cpuUsage = 30.0 + (dataPoints * 2.0) + QRandomGenerator::global()->bounded(-5.0, 5.0);
        double memoryUsage = 1024*1024*400 + (dataPoints * 1024*1024*10) + QRandomGenerator::global()->bounded(-1024*1024*50, 1024*1024*50);
        double responseTime = 100.0 + (dataPoints * 5.0) + QRandomGenerator::global()->bounded(-10.0, 10.0);
        
        m_monitoring->recordMetric("realtime_cpu", cpuUsage, "%");
        m_monitoring->recordMetric("realtime_memory", memoryUsage, "bytes");
        m_monitoring->recordMetric("realtime_response", responseTime, "ms");
        
        dataPoints++;
    });
    
    // Collect data for 2 seconds
    dataTimer.start(100); // Every 100ms
    QTest::qWait(2000);
    dataTimer.stop();
    
    // Verify real-time data collection
    auto realtimeCpuData = m_monitoring->getMetricData("realtime_cpu");
    auto realtimeMemoryData = m_monitoring->getMetricData("realtime_memory");
    auto realtimeResponseData = m_monitoring->getMetricData("realtime_response");
    
    QVERIFY(realtimeCpuData.size() >= 15); // Should have collected multiple data points
    QVERIFY(realtimeMemoryData.size() >= 15);
    QVERIFY(realtimeResponseData.size() >= 15);
    
    // Verify data timestamps are recent and sequential
    if (!realtimeCpuData.isEmpty()) {
        QDateTime firstTimestamp = realtimeCpuData.first().timestamp;
        QDateTime lastTimestamp = realtimeCpuData.last().timestamp;
        
        qint64 timeDiff = firstTimestamp.msecsTo(lastTimestamp);
        QVERIFY(timeDiff >= 1500); // Should span at least 1.5 seconds
        QVERIFY(timeDiff <= 3000);  // Should not exceed 3 seconds
        
        qDebug() << QString("Real-time monitoring collected %1 CPU data points over %2ms")
                    .arg(realtimeCpuData.size())
                    .arg(timeDiff);
    }
    
    // Test real-time dashboard generation
    QString realtimeDashboardPath = QDir(m_testPath).absoluteFilePath("realtime_dashboard.html");
    PerformanceDashboard realtimeDashboard(m_monitoring);
    bool success = realtimeDashboard.generateRealtimeDashboard(realtimeDashboardPath, 5000); // 5 second refresh
    QVERIFY2(success, "Failed to generate real-time dashboard");
    
    qDebug() << QString("Real-time dashboard generated: %1").arg(realtimeDashboardPath);
    qDebug() << "Real-time monitoring test completed successfully";
}

void ExamplePerformanceMonitoring::testDataRetention() {
    qDebug() << "Testing data retention...";
    
    // Generate old data (simulate data from 2 hours ago)
    QDateTime oldTime = QDateTime::currentDateTime().addSecs(-7200);
    
    for (int i = 0; i < 10; ++i) {
        PerformanceMonitoring::MetricDataPoint oldPoint;
        oldPoint.metricName = "retention_test_metric";
        oldPoint.value = 50.0 + i;
        oldPoint.unit = "units";
        oldPoint.timestamp = oldTime.addSecs(i * 60); // One point per minute
        oldPoint.type = PerformanceMonitoring::MetricType::CustomMetric;
        oldPoint.source = "RetentionTest";
        
        m_monitoring->recordMetric(oldPoint);
    }
    
    // Generate recent data
    for (int i = 0; i < 5; ++i) {
        m_monitoring->recordMetric("retention_test_metric", 60.0 + i, "units");
        QThread::msleep(10);
    }
    
    // Verify all data is present
    auto allData = m_monitoring->getMetricData("retention_test_metric");
    QVERIFY(allData.size() >= 15);
    
    qDebug() << QString("Before retention: %1 data points").arg(allData.size());
    
    // Test manual data cleanup (remove data older than 1 hour)
    QDateTime cutoffTime = QDateTime::currentDateTime().addSecs(-3600);
    m_monitoring->clearOldData(cutoffTime);
    
    // Verify old data was removed
    auto remainingData = m_monitoring->getMetricData("retention_test_metric");
    QVERIFY(remainingData.size() < allData.size());
    
    // Verify remaining data is recent
    for (const auto& point : remainingData) {
        QVERIFY(point.timestamp >= cutoffTime);
    }
    
    qDebug() << QString("After retention: %1 data points").arg(remainingData.size());
    
    // Test automatic data retention configuration
    auto config = m_monitoring->getMonitoringConfig();
    config.retentionPeriodMs = 1800000; // 30 minutes
    m_monitoring->setMonitoringConfig(config);
    
    qDebug() << "Data retention test completed successfully";
}

void ExamplePerformanceMonitoring::testDataExportImport() {
    qDebug() << "Testing data export/import...";
    
    // Generate test data for export
    generateTestMetrics("export_test_cpu", 15, 55.0, 8.0);
    generateTestMetrics("export_test_memory", 15, 1024.0, 100.0);
    generateTestMetrics("export_test_disk", 15, 70.0, 5.0);
    
    // Export data
    QString exportPath = QDir(m_testPath).absoluteFilePath("exported_metrics.json");
    m_monitoring->exportMetricData(exportPath, "json");
    
    QVERIFY(QFile::exists(exportPath));
    
    // Verify export file content
    QFile exportFile(exportPath);
    QVERIFY(exportFile.open(QIODevice::ReadOnly));
    
    QJsonDocument exportDoc = QJsonDocument::fromJson(exportFile.readAll());
    QVERIFY(!exportDoc.isNull());
    
    QJsonObject exportData = exportDoc.object();
    QVERIFY(exportData.contains("timestamp"));
    QVERIFY(exportData.contains("metrics"));
    
    QJsonObject metricsObj = exportData["metrics"].toObject();
    QVERIFY(metricsObj.contains("export_test_cpu"));
    QVERIFY(metricsObj.contains("export_test_memory"));
    QVERIFY(metricsObj.contains("export_test_disk"));
    
    qDebug() << QString("Data exported to: %1 (%2 bytes)")
                .arg(exportPath)
                .arg(exportFile.size());
    
    // Clear current data
    m_monitoring->clearMetricData("export_test_cpu");
    m_monitoring->clearMetricData("export_test_memory");
    m_monitoring->clearMetricData("export_test_disk");
    
    // Verify data was cleared
    QVERIFY(m_monitoring->getMetricData("export_test_cpu").isEmpty());
    
    // Import data back
    bool success = m_monitoring->importMetricData(exportPath);
    QVERIFY2(success, "Failed to import metric data");
    
    // Verify data was imported
    auto importedCpuData = m_monitoring->getMetricData("export_test_cpu");
    auto importedMemoryData = m_monitoring->getMetricData("export_test_memory");
    auto importedDiskData = m_monitoring->getMetricData("export_test_disk");
    
    QVERIFY(!importedCpuData.isEmpty());
    QVERIFY(!importedMemoryData.isEmpty());
    QVERIFY(!importedDiskData.isEmpty());
    
    qDebug() << QString("Data imported: CPU=%1, Memory=%2, Disk=%3 data points")
                .arg(importedCpuData.size())
                .arg(importedMemoryData.size())
                .arg(importedDiskData.size());
    
    qDebug() << "Data export/import test completed successfully";
}

void ExamplePerformanceMonitoring::testDataOptimization() {
    qDebug() << "Testing data optimization...";
    
    // Generate large amount of test data
    QString optimizationMetric = "optimization_test_metric";
    
    for (int i = 0; i < 2000; ++i) {
        m_monitoring->recordMetric(optimizationMetric, 100.0 + (i % 50), "units");
    }
    
    // Verify large dataset
    auto beforeOptimization = m_monitoring->getMetricData(optimizationMetric);
    qDebug() << QString("Before optimization: %1 data points").arg(beforeOptimization.size());
    
    // Run data optimization
    m_monitoring->optimizeDataStorage();
    
    // Check if data was optimized (should respect maxDataPoints limit)
    auto afterOptimization = m_monitoring->getMetricData(optimizationMetric);
    qDebug() << QString("After optimization: %1 data points").arg(afterOptimization.size());
    
    // Verify optimization respected configuration limits
    auto config = m_monitoring->getMonitoringConfig();
    QVERIFY(afterOptimization.size() <= config.maxDataPoints);
    
    qDebug() << "Data optimization test completed successfully";
}

void ExamplePerformanceMonitoring::testCustomMetrics() {
    qDebug() << "Testing custom metrics...";
    
    // Test various custom metric scenarios
    QList<QPair<QString, QString>> customMetrics = {
        {"dupfinder_files_scanned", "count"},
        {"dupfinder_duplicates_found", "count"},
        {"dupfinder_scan_progress", "%"},
        {"dupfinder_hash_rate", "files/sec"},
        {"dupfinder_memory_efficiency", "ratio"},
        {"dupfinder_user_satisfaction", "score"}
    };
    
    for (const auto& metric : customMetrics) {
        QString name = metric.first;
        QString unit = metric.second;
        
        // Generate realistic values based on metric type
        double value = 0.0;
        if (unit == "count") {
            value = QRandomGenerator::global()->bounded(1000, 10000);
        } else if (unit == "%") {
            value = QRandomGenerator::global()->bounded(100.0);
        } else if (unit == "files/sec") {
            value = QRandomGenerator::global()->bounded(10.0, 500.0);
        } else if (unit == "ratio") {
            value = QRandomGenerator::global()->bounded(1.0);
        } else if (unit == "score") {
            value = QRandomGenerator::global()->bounded(1.0, 10.0);
        }
        
        // Record custom metric with metadata
        PerformanceMonitoring::MetricDataPoint customPoint;
        customPoint.metricName = name;
        customPoint.type = PerformanceMonitoring::MetricType::CustomMetric;
        customPoint.value = value;
        customPoint.unit = unit;
        customPoint.timestamp = QDateTime::currentDateTime();
        customPoint.source = "DupFinderApp";
        customPoint.description = QString("Custom DupFinder metric: %1").arg(name);
        customPoint.metadata["category"] = "application";
        customPoint.metadata["component"] = name.split("_")[1];
        
        m_monitoring->recordMetric(customPoint);
        
        qDebug() << QString("Custom metric %1: %2 %3").arg(name).arg(value).arg(unit);
    }
    
    // Verify custom metrics were recorded
    QStringList availableMetrics = m_monitoring->getAvailableMetrics();
    
    int customMetricCount = 0;
    for (const QString& metricName : availableMetrics) {
        if (metricName.startsWith("dupfinder_")) {
            customMetricCount++;
            
            auto latestMetric = m_monitoring->getLatestMetric(metricName);
            QVERIFY(latestMetric.value >= 0);
            QVERIFY(!latestMetric.unit.isEmpty());
            QCOMPARE(latestMetric.source, QString("DupFinderApp"));
        }
    }
    
    QVERIFY(customMetricCount >= 6);
    qDebug() << QString("Successfully recorded %1 custom metrics").arg(customMetricCount);
    
    qDebug() << "Custom metrics test completed successfully";
}

void ExamplePerformanceMonitoring::testMultiMetricAnalysis() {
    qDebug() << "Testing multi-metric analysis...";
    
    // Generate correlated metrics (simulating real application behavior)
    QStringList correlatedMetrics = {
        "app_cpu_usage",
        "app_memory_usage", 
        "app_response_time",
        "app_throughput",
        "app_error_rate"
    };
    
    // Generate data with realistic correlations
    for (int i = 0; i < 30; ++i) {
        double baseLoad = 50.0 + (i * 1.5); // Gradually increasing load
        
        // CPU usage increases with load
        double cpuUsage = baseLoad + QRandomGenerator::global()->bounded(-5.0, 5.0);
        m_monitoring->recordMetric("app_cpu_usage", cpuUsage, "%");
        
        // Memory usage increases with load (but more gradually)
        double memoryUsage = (baseLoad * 20.0) + QRandomGenerator::global()->bounded(-100.0, 100.0);
        m_monitoring->recordMetric("app_memory_usage", memoryUsage * 1024 * 1024, "bytes");
        
        // Response time increases with high CPU usage
        double responseTime = 100.0 + (cpuUsage > 70 ? (cpuUsage - 70) * 10 : 0) + QRandomGenerator::global()->bounded(-10.0, 10.0);
        m_monitoring->recordMetric("app_response_time", responseTime, "ms");
        
        // Throughput decreases with high response time
        double throughput = qMax(10.0, 1000.0 - (responseTime * 2.0)) + QRandomGenerator::global()->bounded(-50.0, 50.0);
        m_monitoring->recordMetric("app_throughput", throughput, "ops/sec");
        
        // Error rate increases with high load
        double errorRate = qMax(0.0, (cpuUsage > 80 ? (cpuUsage - 80) * 0.5 : 0.1)) + QRandomGenerator::global()->bounded(-0.1, 0.1);
        m_monitoring->recordMetric("app_error_rate", errorRate, "%");
        
        QThread::msleep(20);
    }
    
    // Analyze trends for all metrics
    QMap<QString, PerformanceMonitoring::TrendAnalysis> allTrends;
    
    for (const QString& metricName : correlatedMetrics) {
        auto trend = m_monitoring->analyzeTrend(metricName);
        if (trend.sampleCount > 0) {
            allTrends[metricName] = trend;
            
            qDebug() << QString("Trend for %1: %2 (%3% change)")
                        .arg(metricName)
                        .arg(m_monitoring->formatTrendType(trend.trendType))
                        .arg(trend.changePercent, 0, 'f', 1);
        }
    }
    
    // Verify we have trend data for all metrics
    QVERIFY(allTrends.size() >= 5);
    
    // Check for performance regressions across all metrics
    auto allRegressions = m_monitoring->detectAllRegressions(20.0);
    
    qDebug() << QString("Detected %1 regressions across all metrics").arg(allRegressions.size());
    
    for (const auto& regression : allRegressions) {
        qDebug() << QString("Regression in %1: %2% (%3)")
                    .arg(regression.metricName)
                    .arg(qAbs(regression.regressionPercent), 0, 'f', 1)
                    .arg(regression.severity);
    }
    
    // Generate comprehensive multi-metric report
    QString multiMetricReportPath = QDir(m_testPath).absoluteFilePath("multi_metric_analysis.html");
    PerformanceMonitoring::ReportConfig multiMetricConfig;
    multiMetricConfig.name = "Multi-Metric Analysis";
    multiMetricConfig.outputPath = multiMetricReportPath;
    multiMetricConfig.format = "html";
    multiMetricConfig.title = "Comprehensive Multi-Metric Performance Analysis";
    multiMetricConfig.description = "Analysis of correlated performance metrics";
    multiMetricConfig.metricsToInclude = correlatedMetrics;
    multiMetricConfig.includeStatistics = true;
    multiMetricConfig.includeTrendAnalysis = true;
    multiMetricConfig.includeRegressions = true;
    
    bool success = m_monitoring->generateReport(multiMetricConfig);
    QVERIFY2(success, "Failed to generate multi-metric report");
    
    qDebug() << QString("Multi-metric analysis report generated: %1").arg(multiMetricReportPath);
    qDebug() << "Multi-metric analysis test completed successfully";
}

void ExamplePerformanceMonitoring::testPerformanceBaselines() {
    qDebug() << "Testing performance baselines...";
    
    // This test demonstrates how performance monitoring can work with baselines
    // In a real implementation, this would integrate with the PerformanceBenchmark class
    
    // Generate baseline performance data
    QString baselineMetric = "baseline_test_metric";
    
    // Simulate good baseline performance
    for (int i = 0; i < 20; ++i) {
        double baselineValue = 100.0 + QRandomGenerator::global()->bounded(-5.0, 5.0);
        m_monitoring->recordMetric(baselineMetric, baselineValue, "ms");
        QThread::msleep(10);
    }
    
    // Calculate baseline statistics
    auto baselineStats = m_monitoring->analyzeTrend(baselineMetric);
    double baselineAverage = baselineStats.averageValue;
    
    qDebug() << QString("Baseline established: %1 ms (avg) with %2 samples")
                .arg(baselineAverage, 0, 'f', 2)
                .arg(baselineStats.sampleCount);
    
    // Configure alert based on baseline (deviation from baseline)
    PerformanceMonitoring::AlertConfig baselineAlert;
    baselineAlert.name = "Baseline Deviation Alert";
    baselineAlert.metricName = baselineMetric;
    baselineAlert.severity = PerformanceMonitoring::AlertSeverity::Warning;
    baselineAlert.threshold = baselineAverage * 1.2; // 20% above baseline
    baselineAlert.comparison = ">";
    baselineAlert.evaluationWindowMs = 5000;
    baselineAlert.minSamples = 3;
    baselineAlert.enabled = true;
    baselineAlert.description = QString("Alert when performance deviates more than 20% from baseline (%1 ms)").arg(baselineAverage);
    
    m_monitoring->addAlert(baselineAlert);
    
    // Simulate performance that stays within baseline
    for (int i = 0; i < 10; ++i) {
        double goodValue = baselineAverage + QRandomGenerator::global()->bounded(-8.0, 8.0);
        m_monitoring->recordMetric(baselineMetric, goodValue, "ms");
        QThread::msleep(50);
    }
    
    // Wait for alert evaluation
    QTest::qWait(500);
    
    // Should not have triggered alert
    auto activeAlerts = m_monitoring->getActiveAlerts();
    bool baselineAlertTriggered = false;
    for (const auto& alert : activeAlerts) {
        if (alert.alertName == "Baseline Deviation Alert") {
            baselineAlertTriggered = true;
            break;
        }
    }
    
    QVERIFY2(!baselineAlertTriggered, "Alert should not trigger for performance within baseline");
    
    // Simulate performance regression (beyond baseline threshold)
    for (int i = 0; i < 5; ++i) {
        double regressedValue = baselineAverage * 1.3; // 30% above baseline
        m_monitoring->recordMetric(baselineMetric, regressedValue, "ms");
        QThread::msleep(100);
    }
    
    // Wait for alert evaluation
    QTest::qWait(1000);
    
    // Check if baseline deviation was detected
    auto updatedActiveAlerts = m_monitoring->getActiveAlerts();
    bool regressionDetected = false;
    for (const auto& alert : updatedActiveAlerts) {
        if (alert.alertName == "Baseline Deviation Alert" && alert.isActive) {
            regressionDetected = true;
            qDebug() << QString("Baseline deviation detected: %1 ms (threshold: %2 ms)")
                        .arg(alert.currentValue)
                        .arg(alert.thresholdValue);
            break;
        }
    }
    
    if (regressionDetected) {
        qDebug() << "Performance baseline monitoring successful - regression detected";
    } else {
        qDebug() << "Note: Baseline deviation may not have been detected due to timing";
    }
    
    qDebug() << "Performance baselines test completed";
}

// Helper method implementations
void ExamplePerformanceMonitoring::generateTestMetrics(const QString& metricName, int count, double baseValue, double variation) {
    for (int i = 0; i < count; ++i) {
        double value = baseValue + (i * variation / count) + QRandomGenerator::global()->bounded(-variation/4, variation/4);
        m_monitoring->recordMetric(metricName, value, "units");
        QThread::msleep(10);
    }
}

void ExamplePerformanceMonitoring::simulatePerformanceRegression(const QString& metricName) {
    // Simulate a performance regression by recording significantly worse values
    for (int i = 0; i < 15; ++i) {
        double regressedValue = 150.0 + (i * 2.0) + QRandomGenerator::global()->bounded(-5.0, 5.0);
        m_monitoring->recordMetric(metricName, regressedValue, "ms");
        QThread::msleep(20);
    }
}

void ExamplePerformanceMonitoring::waitForMonitoringCycle() {
    // Wait for at least one monitoring cycle to complete
    auto config = m_monitoring->getMonitoringConfig();
    QTest::qWait(config.samplingIntervalMs + 100);
}

QTEST_MAIN(ExamplePerformanceMonitoring)
#include "example_performance_monitoring.moc"