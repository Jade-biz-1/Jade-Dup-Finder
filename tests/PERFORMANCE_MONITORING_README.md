# Performance Monitoring and Reporting Framework

## Overview

The Performance Monitoring and Reporting Framework provides comprehensive real-time performance monitoring, trend analysis, regression detection, alerting, and interactive reporting capabilities for the DupFinder application. It enables continuous performance validation, proactive issue detection, and detailed performance analytics.

## Key Features

### 🔍 Real-Time Performance Monitoring
- **Continuous Metrics Collection**: Automated collection of system and application metrics
- **Multi-Type Metrics**: Support for system resources, application metrics, UI performance, file operations, and custom metrics
- **Configurable Sampling**: Adjustable sampling intervals and data retention policies
- **Resource Monitoring**: CPU, memory, disk, network, and thread usage tracking
- **Application Integration**: Seamless integration with benchmarking and load testing frameworks

### 📈 Advanced Trend Analysis
- **Statistical Analysis**: Mean, median, standard deviation, percentiles, and correlation analysis
- **Trend Detection**: Automatic identification of improving, degrading, stable, and volatile trends
- **Regression Analysis**: Linear regression analysis for trend slope calculation
- **Change Detection**: Percentage change analysis over configurable time windows
- **Predictive Insights**: Trend-based performance predictions and recommendations

### 🚨 Intelligent Alerting System
- **Configurable Alerts**: Flexible alert configuration with multiple severity levels
- **Threshold Monitoring**: Support for various comparison operators and evaluation windows
- **Alert Lifecycle**: Automatic alert triggering, acknowledgment, and resolution
- **Notification Integration**: Webhook notifications and custom action script execution
- **Alert History**: Complete audit trail of all alert events and resolutions

### 📊 Comprehensive Reporting
- **Multiple Formats**: HTML, JSON, and PDF report generation
- **Interactive Dashboards**: Real-time performance dashboards with charts and widgets
- **Specialized Reports**: Trend analysis, alert summaries, and regression reports
- **Custom Reports**: Configurable report templates and content selection
- **Export Capabilities**: Data export and import for analysis and backup

### 🔧 Performance Regression Detection
- **Automatic Detection**: Statistical analysis to identify performance regressions
- **Baseline Comparison**: Compare current performance against historical baselines
- **Severity Classification**: Automatic classification of regression severity levels
- **Root Cause Analysis**: Detailed analysis of regression patterns and characteristics
- **Recommendation Engine**: Automated recommendations for performance improvements

## Core Classes

### PerformanceMonitoring

The main monitoring class that provides all performance monitoring capabilities.

```cpp
#include "performance_monitoring.h"

PerformanceMonitoring monitor;

// Configure monitoring
PerformanceMonitoring::MonitoringConfig config;
config.name = "DupFinder Performance Monitor";
config.samplingIntervalMs = 1000;           // 1 second sampling
config.retentionPeriodMs = 86400000;        // 24 hour retention
config.enableTrendAnalysis = true;
config.enableRegressionDetection = true;
config.enableAlerting = true;
config.regressionThreshold = 10.0;          // 10% regression threshold

monitor.setMonitoringConfig(config);
```

### PerformanceDashboard

Interactive dashboard generator for real-time performance visualization.

```cpp
PerformanceDashboard dashboard(&monitor);
dashboard.setDashboardTitle("DupFinder Performance Dashboard");
dashboard.addMetricWidget("cpu_usage", "gauge");
dashboard.addMetricWidget("memory_usage", "line");
dashboard.addTrendWidget("response_time");
dashboard.addAlertWidget();

dashboard.generateInteractiveDashboard("performance_dashboard.html");
```

## Usage Examples

### Basic Performance Monitoring

```cpp
// Start monitoring
monitor.startMonitoring();

// Record custom metrics
monitor.recordMetric("files_processed", 1500, "count", 
                    PerformanceMonitoring::MetricType::ApplicationMetric);
monitor.recordMetric("scan_rate", 250.5, "files/sec", 
                    PerformanceMonitoring::MetricType::ApplicationMetric);

// Record system metrics automatically
monitor.recordSystemMetrics();

// Get latest metrics
auto latestMetrics = monitor.getLatestMetrics();
for (auto it = latestMetrics.begin(); it != latestMetrics.end(); ++it) {
    qDebug() << it.key() << ":" << it.value().value << it.value().unit;
}
```

### Advanced Metric Recording

```cpp
// Record metric with detailed metadata
PerformanceMonitoring::MetricDataPoint dataPoint;
dataPoint.metricName = "duplicate_detection_accuracy";
dataPoint.type = PerformanceMonitoring::MetricType::ApplicationMetric;
dataPoint.value = 98.7;
dataPoint.unit = "%";
dataPoint.timestamp = QDateTime::currentDateTime();
dataPoint.source = "DuplicateDetector";
dataPoint.description = "Accuracy of duplicate detection algorithm";
dataPoint.metadata["algorithm"] = "SHA256";
dataPoint.metadata["file_count"] = 10000;
dataPoint.metadata["dataset"] = "test_suite_v2";

monitor.recordMetric(dataPoint);
```

### Trend Analysis

```cpp
// Analyze trend for specific metric
auto trend = monitor.analyzeTrend("response_time");

qDebug() << "Trend Type:" << monitor.formatTrendType(trend.trendType);
qDebug() << "Change Percentage:" << trend.changePercent << "%";
qDebug() << "Sample Count:" << trend.sampleCount;
qDebug() << "Correlation:" << trend.correlation;
qDebug() << "Description:" << trend.trendDescription;

// Analyze all trends
auto allTrends = monitor.analyzeAllTrends();
for (const auto& trend : allTrends) {
    if (trend.trendType == PerformanceMonitoring::TrendType::Degrading) {
        qWarning() << "Performance degrading in" << trend.metricName;
    }
}

// Get trend summary
auto trendSummary = monitor.getTrendSummary();
```

### Performance Regression Detection

```cpp
// Detect regression for specific metric
bool hasRegression = monitor.detectPerformanceRegression("memory_usage", 15.0);
if (hasRegression) {
    qWarning() << "Memory usage regression detected!";
}

// Detect all regressions
auto regressions = monitor.detectAllRegressions(10.0);
for (const auto& regression : regressions) {
    qDebug() << "Regression in" << regression.metricName 
             << ":" << regression.regressionPercent << "% (" 
             << regression.severity << ")";
    qDebug() << "Recommendation:" << regression.recommendation;
}
```

### Alert Configuration and Management

```cpp
// Configure performance alert
PerformanceMonitoring::AlertConfig alert;
alert.name = "High Memory Usage";
alert.metricName = "system_memory_usage";
alert.severity = PerformanceMonitoring::AlertSeverity::Critical;
alert.threshold = 1024 * 1024 * 1024; // 1GB
alert.comparison = ">";
alert.evaluationWindowMs = 30000;      // 30 seconds
alert.minSamples = 5;
alert.enabled = true;
alert.description = "Alert when memory usage exceeds 1GB";
alert.webhookUrl = "https://alerts.company.com/webhook";
alert.actionScript = "/scripts/memory_alert.sh";

monitor.addAlert(alert);

// Handle alert events
connect(&monitor, &PerformanceMonitoring::alertTriggered, 
        [](const PerformanceMonitoring::PerformanceAlert& alert) {
    qWarning() << "ALERT:" << alert.alertName << "-" << alert.message;
    // Take corrective action
});

connect(&monitor, &PerformanceMonitoring::alertResolved,
        [](const PerformanceMonitoring::PerformanceAlert& alert) {
    qInfo() << "RESOLVED:" << alert.alertName;
});
```

### Report Generation

```cpp
// Generate comprehensive HTML report
PerformanceMonitoring::ReportConfig reportConfig;
reportConfig.name = "Weekly Performance Report";
reportConfig.outputPath = "weekly_performance_report.html";
reportConfig.format = "html";
reportConfig.title = "DupFinder Weekly Performance Analysis";
reportConfig.description = "Comprehensive performance analysis for the past week";
reportConfig.startTime = QDateTime::currentDateTime().addDays(-7);
reportConfig.endTime = QDateTime::currentDateTime();
reportConfig.metricsToInclude = QStringList() 
    << "cpu_usage" << "memory_usage" << "response_time" 
    << "throughput" << "error_rate";
reportConfig.includeStatistics = true;
reportConfig.includeTrendAnalysis = true;
reportConfig.includeAlerts = true;
reportConfig.includeRegressions = true;
reportConfig.includeCharts = true;

bool success = monitor.generateReport(reportConfig);

// Generate specialized reports
monitor.generateTrendReport("trend_analysis.html");
monitor.generateAlertReport("alert_summary.html");
monitor.generateRegressionReport("regression_analysis.html");
```

### Dashboard Generation

```cpp
// Create interactive dashboard
PerformanceDashboard dashboard(&monitor);
dashboard.setDashboardTitle("DupFinder Real-Time Performance");
dashboard.setDashboardTheme("dark");

// Add metric widgets
dashboard.addMetricWidget("cpu_usage", "gauge");
dashboard.addMetricWidget("memory_usage", "line");
dashboard.addMetricWidget("disk_usage", "bar");
dashboard.addMetricWidget("network_throughput", "area");

// Add analysis widgets
dashboard.addTrendWidget("response_time");
dashboard.addTrendWidget("throughput");
dashboard.addAlertWidget();
dashboard.addRegressionWidget();

// Generate dashboard
dashboard.generateInteractiveDashboard("dashboard.html");
dashboard.generateRealtimeDashboard("realtime_dashboard.html", 5000); // 5s refresh
```

### Integration with Benchmarking

```cpp
// Integrate with performance benchmark
PerformanceBenchmark benchmark;
monitor.setPerformanceBenchmark(&benchmark);

// Monitoring will automatically capture benchmark results
benchmark.runBenchmark("file_processing", []() {
    // Benchmark code here
    processFiles();
});

// Integration with load testing
LoadStressTesting loadTester;
monitor.setLoadStressTesting(&loadTester);

// Monitoring will automatically capture load test metrics
loadTester.runLoadTest(config, []() {
    // Load test operation
});
```

### Data Management

```cpp
// Export performance data
monitor.exportMetricData("performance_data_backup.json", "json");

// Import performance data
monitor.importMetricData("performance_data_backup.json");

// Clear old data (older than 24 hours)
QDateTime cutoff = QDateTime::currentDateTime().addDays(-1);
monitor.clearOldData(cutoff);

// Optimize data storage
monitor.optimizeDataStorage();

// Get performance summary
QJsonObject summary = monitor.generatePerformanceSummary(
    QDateTime::currentDateTime().addDays(-7),  // Last week
    QDateTime::currentDateTime()
);
```

## Configuration Options

### MonitoringConfig Structure

```cpp
struct MonitoringConfig {
    QString name;                           // Configuration name
    qint64 samplingIntervalMs = 1000;      // Sampling interval (1 second)
    qint64 retentionPeriodMs = 86400000;   // Data retention (24 hours)
    int maxDataPoints = 10000;             // Maximum data points per metric
    bool enableTrendAnalysis = true;       // Enable trend analysis
    bool enableRegressionDetection = true; // Enable regression detection
    bool enableAlerting = true;            // Enable alerting system
    bool enableReporting = true;           // Enable report generation
    qint64 trendAnalysisWindowMs = 3600000; // Trend analysis window (1 hour)
    double regressionThreshold = 10.0;     // Regression threshold (10%)
    QString reportOutputDirectory;          // Report output directory
    QStringList metricsToMonitor;          // Specific metrics to monitor
    QMap<QString, QVariant> customSettings; // Custom settings
};
```

### AlertConfig Structure

```cpp
struct AlertConfig {
    QString name;                           // Alert name
    QString metricName;                     // Metric to monitor
    AlertSeverity severity;                 // Alert severity level
    double threshold = 0.0;                // Threshold value
    QString comparison = ">";               // Comparison operator
    qint64 evaluationWindowMs = 60000;     // Evaluation window (1 minute)
    int minSamples = 5;                    // Minimum samples required
    bool enabled = true;                   // Whether alert is enabled
    QString description;                    // Alert description
    QString actionScript;                   // Script to execute on trigger
    QStringList notificationEmails;        // Email notification addresses
    QString webhookUrl;                    // Webhook URL for notifications
};
```

### ReportConfig Structure

```cpp
struct ReportConfig {
    QString name;                           // Report name
    QString templatePath;                   // Report template path
    QString outputPath;                     // Output file path
    QString format = "html";               // Report format (html, pdf, json)
    QDateTime startTime;                    // Report start time
    QDateTime endTime;                      // Report end time
    QStringList metricsToInclude;          // Metrics to include
    bool includeTrendAnalysis = true;      // Include trend analysis
    bool includeAlerts = true;             // Include alerts
    bool includeRegressions = true;        // Include regression analysis
    bool includeCharts = true;             // Include performance charts
    bool includeStatistics = true;         // Include statistical analysis
    QString title;                         // Report title
    QString description;                    // Report description
    QMap<QString, QVariant> customData;    // Custom report data
};
```

## Metric Types

The framework supports various types of performance metrics:

### SystemResource
- CPU usage percentage
- Memory consumption (bytes)
- Disk usage percentage
- Network throughput (bytes/sec)
- Thread count

### ApplicationMetric
- Response times (ms)
- Throughput (operations/sec)
- Error rates (%)
- Processing times (ms)
- Queue lengths (count)

### UserInterface
- Frame rates (fps)
- UI response times (ms)
- Event queue sizes (count)
- Rendering times (ms)

### FileOperation
- File scan rates (files/sec)
- Hash calculation rates (MB/sec)
- Duplicate detection rates (files/sec)
- I/O throughput (MB/sec)

### CustomMetric
- Application-specific metrics
- Business logic metrics
- User-defined performance indicators

## Integration Patterns

### Qt Test Framework Integration

```cpp
class PerformanceTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase() {
        m_monitor = new PerformanceMonitoring(this);
        m_monitor->startMonitoring();
    }
    
    void testPerformanceRequirements() {
        // Run performance-critical operation
        MONITOR_EXECUTION_TIME("critical_operation", {
            performCriticalOperation();
        });
        
        // Verify performance requirements
        auto result = m_monitor->getLatestMetric("critical_operation");
        QVERIFY2(result.value < 1000, "Operation must complete within 1 second");
        
        // Check for regressions
        bool hasRegression = m_monitor->detectPerformanceRegression("critical_operation", 15.0);
        QVERIFY2(!hasRegression, "No performance regression should be detected");
    }

private:
    PerformanceMonitoring* m_monitor;
};
```

### Continuous Integration Integration

```cpp
// CI/CD Pipeline Performance Validation
class CIPipelinePerformanceValidator {
public:
    bool validatePerformance() {
        PerformanceMonitoring monitor;
        monitor.startMonitoring();
        
        // Run performance tests
        runPerformanceTestSuite();
        
        // Check for regressions
        auto regressions = monitor.detectAllRegressions(10.0);
        if (!regressions.isEmpty()) {
            generateRegressionReport(regressions);
            return false; // Fail CI build
        }
        
        // Generate performance report
        monitor.generateReport(createCIReportConfig());
        return true;
    }
};
```

### Real-Time Application Monitoring

```cpp
class ApplicationPerformanceMonitor : public QObject {
    Q_OBJECT

public:
    ApplicationPerformanceMonitor() {
        m_monitor = new PerformanceMonitoring(this);
        
        // Configure for production monitoring
        PerformanceMonitoring::MonitoringConfig config;
        config.samplingIntervalMs = 5000;      // 5 second sampling
        config.retentionPeriodMs = 604800000;  // 7 day retention
        config.enableAlerting = true;
        config.regressionThreshold = 20.0;     // 20% threshold for production
        
        m_monitor->setMonitoringConfig(config);
        
        // Setup production alerts
        setupProductionAlerts();
        
        // Start monitoring
        m_monitor->startMonitoring();
        
        // Generate daily reports
        QTimer* reportTimer = new QTimer(this);
        connect(reportTimer, &QTimer::timeout, this, &ApplicationPerformanceMonitor::generateDailyReport);
        reportTimer->start(86400000); // 24 hours
    }

private slots:
    void onApplicationMetric(const QString& name, double value, const QString& unit) {
        m_monitor->recordMetric(name, value, unit, PerformanceMonitoring::MetricType::ApplicationMetric);
    }
    
    void generateDailyReport() {
        QString reportPath = QString("daily_performance_%1.html")
                           .arg(QDate::currentDate().toString("yyyy-MM-dd"));
        
        PerformanceMonitoring::ReportConfig config;
        config.outputPath = reportPath;
        config.title = "Daily Performance Report";
        config.startTime = QDateTime::currentDateTime().addDays(-1);
        config.endTime = QDateTime::currentDateTime();
        
        m_monitor->generateReport(config);
    }

private:
    PerformanceMonitoring* m_monitor;
};
```

## Convenience Macros

The framework provides convenient macros for common monitoring tasks:

```cpp
// Simple metric recording
MONITOR_METRIC("files_processed", fileCount, "count");

// Execution time monitoring
MONITOR_EXECUTION_TIME("database_query", {
    database.executeQuery(sql);
});

// Memory usage monitoring
MONITOR_MEMORY_USAGE("after_file_processing");

// Monitoring session management
START_MONITORING_SESSION("duplicate_detection_session");
// ... perform operations ...
STOP_MONITORING_SESSION();

// Report generation
GENERATE_PERFORMANCE_REPORT("session_report.html");
```

## Best Practices

### 1. Monitoring Configuration
- Use appropriate sampling intervals (1-5 seconds for most applications)
- Set reasonable data retention periods based on storage capacity
- Enable only necessary monitoring features to reduce overhead
- Configure alerts with appropriate thresholds and evaluation windows

### 2. Metric Design
- Use descriptive metric names with consistent naming conventions
- Include appropriate units and metadata for all metrics
- Group related metrics using consistent prefixes
- Avoid recording too many metrics that could impact performance

### 3. Alert Management
- Set conservative thresholds to avoid alert fatigue
- Use different severity levels appropriately
- Implement proper alert acknowledgment and resolution workflows
- Test alert configurations thoroughly before production deployment

### 4. Trend Analysis
- Collect sufficient data points for meaningful trend analysis
- Consider seasonal patterns and expected variations
- Use appropriate time windows for different types of analysis
- Combine multiple metrics for comprehensive performance assessment

### 5. Regression Detection
- Establish stable baselines before enabling regression detection
- Use appropriate regression thresholds for different environments
- Investigate all detected regressions promptly
- Update baselines when making intentional performance changes

### 6. Reporting and Dashboards
- Generate reports regularly for proactive monitoring
- Customize dashboards for different audiences (developers, operations, management)
- Include actionable insights and recommendations in reports
- Archive historical reports for long-term trend analysis

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce data retention period
   - Decrease sampling frequency
   - Limit maximum data points per metric
   - Enable data optimization

2. **Missing Metrics**
   - Verify monitoring is started
   - Check metric names for typos
   - Ensure proper integration setup
   - Verify sampling intervals

3. **Alert Issues**
   - Check alert configuration
   - Verify evaluation windows and thresholds
   - Test webhook endpoints
   - Check alert enable/disable status

4. **Performance Impact**
   - Reduce sampling frequency
   - Disable unnecessary features
   - Optimize metric collection code
   - Use asynchronous processing

### Performance Tips

1. **Optimize Monitoring Overhead**
   - Use appropriate sampling intervals
   - Batch metric recordings when possible
   - Implement efficient data structures
   - Consider using separate monitoring threads

2. **Efficient Data Management**
   - Implement proper data retention policies
   - Use data compression for storage
   - Optimize database queries for historical data
   - Implement efficient data export/import

3. **Scalable Architecture**
   - Design for horizontal scaling
   - Use distributed monitoring for large systems
   - Implement proper load balancing
   - Consider using external monitoring services

## Example Implementation

See `example_performance_monitoring.cpp` for a comprehensive example that demonstrates:

- Real-time performance monitoring setup
- Trend analysis and regression detection
- Alert configuration and management
- Report and dashboard generation
- Integration with benchmarking and load testing
- Data management and optimization
- Custom metrics and multi-metric analysis

This example serves as both documentation and a working implementation that can be adapted for specific monitoring needs.

## Integration with DupFinder

The performance monitoring framework is specifically designed to monitor DupFinder's performance characteristics:

- **File Processing Performance**: Monitor file scanning, hash calculation, and duplicate detection rates
- **Memory Usage Patterns**: Track memory consumption during large file operations
- **UI Responsiveness**: Ensure user interface remains responsive during intensive operations
- **System Resource Usage**: Monitor CPU, disk, and network usage during operations
- **Error Rate Monitoring**: Track and alert on error rates and failure conditions
- **Scalability Metrics**: Monitor performance scaling with different dataset sizes

This comprehensive monitoring ensures DupFinder maintains optimal performance and provides early warning of potential issues before they impact users.