#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QElapsedTimer>
#include <QDateTime>
#include <QMap>
#include <QVariant>
#include <QJsonObject>
#include <QJsonDocument>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QQueue>
#include <QUrl>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <functional>

class PerformanceBenchmark;
class LoadStressTesting;

/**
 * @brief Real-time performance monitoring and reporting framework
 * 
 * Provides comprehensive real-time performance metrics collection, trend analysis,
 * regression detection, alerting, and interactive reporting dashboard capabilities
 * for continuous performance monitoring of DupFinder operations.
 */
class PerformanceMonitoring : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Performance metric types for monitoring
     */
    enum class MetricType {
        SystemResource,     ///< System resource metrics (CPU, memory, disk)
        ApplicationMetric,  ///< Application-specific metrics
        UserInterface,      ///< UI performance metrics
        FileOperation,      ///< File operation performance
        NetworkOperation,   ///< Network operation performance
        DatabaseOperation,  ///< Database operation performance
        CustomMetric       ///< User-defined custom metrics
    };

    /**
     * @brief Alert severity levels
     */
    enum class AlertSeverity {
        Info,       ///< Informational alert
        Warning,    ///< Warning level alert
        Critical,   ///< Critical performance issue
        Emergency   ///< Emergency - system at risk
    };

    /**
     * @brief Trend analysis types
     */
    enum class TrendType {
        Improving,      ///< Performance is improving
        Stable,         ///< Performance is stable
        Degrading,      ///< Performance is degrading
        Volatile,       ///< Performance is highly variable
        Unknown         ///< Insufficient data for analysis
    };

    /**
     * @brief Performance metric data point
     */
    struct MetricDataPoint {
        QString metricName;                     ///< Name of the metric
        MetricType type;                        ///< Type of metric
        double value = 0.0;                    ///< Measured value
        QString unit;                          ///< Unit of measurement
        QDateTime timestamp;                    ///< When measurement was taken
        QString source;                        ///< Source of the measurement
        QMap<QString, QVariant> metadata;      ///< Additional metadata
        QString description;                    ///< Description of the metric
    };

    /**
     * @brief Performance trend analysis result
     */
    struct TrendAnalysis {
        QString metricName;                     ///< Name of the metric
        TrendType trendType;                    ///< Type of trend detected
        double trendSlope = 0.0;               ///< Slope of the trend line
        double correlation = 0.0;              ///< Correlation coefficient
        double averageValue = 0.0;             ///< Average value over period
        double standardDeviation = 0.0;        ///< Standard deviation
        double changePercent = 0.0;            ///< Percentage change over period
        QDateTime analysisStartTime;            ///< Start time of analysis period
        QDateTime analysisEndTime;              ///< End time of analysis period
        int sampleCount = 0;                   ///< Number of samples analyzed
        QString trendDescription;               ///< Human-readable trend description
        QList<double> rawValues;               ///< Raw values used in analysis
        QList<QDateTime> timestamps;           ///< Timestamps for raw values
    };

    /**
     * @brief Performance alert configuration
     */
    struct AlertConfig {
        QString name;                           ///< Alert name
        QString metricName;                     ///< Metric to monitor
        AlertSeverity severity;                 ///< Alert severity level
        double threshold = 0.0;                ///< Threshold value
        QString comparison = ">";               ///< Comparison operator (>, <, ==, !=)
        qint64 evaluationWindowMs = 60000;     ///< Evaluation window in milliseconds
        int minSamples = 5;                    ///< Minimum samples required
        bool enabled = true;                   ///< Whether alert is enabled
        QString description;                    ///< Alert description
        QString actionScript;                   ///< Script to run when triggered
        QStringList notificationEmails;        ///< Email addresses for notifications
        QString webhookUrl;                    ///< Webhook URL for notifications
    };

    /**
     * @brief Performance alert instance
     */
    struct PerformanceAlert {
        QString alertName;                      ///< Name of the alert
        AlertSeverity severity;                 ///< Severity level
        QString metricName;                     ///< Metric that triggered alert
        double currentValue = 0.0;             ///< Current metric value
        double thresholdValue = 0.0;           ///< Threshold that was exceeded
        QDateTime triggeredTime;                ///< When alert was triggered
        QDateTime resolvedTime;                 ///< When alert was resolved (if applicable)
        bool isActive = true;                  ///< Whether alert is currently active
        QString message;                        ///< Alert message
        QString recommendation;                 ///< Recommended action
        QMap<QString, QVariant> context;       ///< Additional context information
    };

    /**
     * @brief Performance regression detection result
     */
    struct RegressionDetection {
        QString metricName;                     ///< Name of the metric
        bool regressionDetected = false;       ///< Whether regression was detected
        double currentValue = 0.0;             ///< Current performance value
        double baselineValue = 0.0;            ///< Baseline performance value
        double regressionPercent = 0.0;        ///< Percentage regression
        QDateTime detectionTime;                ///< When regression was detected
        QString regressionType;                 ///< Type of regression (sudden, gradual)
        QString severity;                       ///< Severity of regression
        QString description;                    ///< Description of regression
        QString recommendation;                 ///< Recommended action
        QList<MetricDataPoint> evidencePoints; ///< Data points showing regression
    };

    /**
     * @brief Performance monitoring configuration
     */
    struct MonitoringConfig {
        QString name;                           ///< Configuration name
        qint64 samplingIntervalMs = 1000;      ///< Sampling interval in milliseconds
        qint64 retentionPeriodMs = 86400000;   ///< Data retention period (24 hours)
        int maxDataPoints = 10000;             ///< Maximum data points to retain
        bool enableTrendAnalysis = true;       ///< Enable trend analysis
        bool enableRegressionDetection = true; ///< Enable regression detection
        bool enableAlerting = true;            ///< Enable alerting system
        bool enableReporting = true;           ///< Enable report generation
        qint64 trendAnalysisWindowMs = 3600000; ///< Trend analysis window (1 hour)
        double regressionThreshold = 10.0;     ///< Regression detection threshold (%)
        QString reportOutputDirectory;          ///< Directory for report output
        QStringList metricsToMonitor;          ///< List of metrics to monitor
        QMap<QString, QVariant> customSettings; ///< Custom configuration settings
    };

    /**
     * @brief Performance report configuration
     */
    struct ReportConfig {
        QString name;                           ///< Report name
        QString templatePath;                   ///< Path to report template
        QString outputPath;                     ///< Output file path
        QString format = "html";               ///< Report format (html, pdf, json)
        QDateTime startTime;                    ///< Report start time
        QDateTime endTime;                      ///< Report end time
        QStringList metricsToInclude;          ///< Metrics to include in report
        bool includeTrendAnalysis = true;      ///< Include trend analysis
        bool includeAlerts = true;             ///< Include alerts
        bool includeRegressions = true;        ///< Include regression analysis
        bool includeCharts = true;             ///< Include performance charts
        bool includeStatistics = true;         ///< Include statistical analysis
        QString title;                         ///< Report title
        QString description;                    ///< Report description
        QMap<QString, QVariant> customData;    ///< Custom data for report
    };

    explicit PerformanceMonitoring(QObject* parent = nullptr);
    ~PerformanceMonitoring();

    // Configuration methods
    void setMonitoringConfig(const MonitoringConfig& config);
    MonitoringConfig getMonitoringConfig() const;
    void setPerformanceBenchmark(PerformanceBenchmark* benchmark);
    void setLoadStressTesting(LoadStressTesting* loadTesting);

    // Monitoring control
    bool startMonitoring();
    bool stopMonitoring();
    bool isMonitoring() const;
    void pauseMonitoring();
    void resumeMonitoring();

    // Metric collection
    void recordMetric(const MetricDataPoint& dataPoint);
    void recordMetric(const QString& name, double value, const QString& unit, MetricType type = MetricType::CustomMetric);
    void recordSystemMetrics();
    void recordApplicationMetrics();
    void recordUIMetrics();
    void recordFileOperationMetrics();

    // Data retrieval
    QList<MetricDataPoint> getMetricData(const QString& metricName, const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime()) const;
    QList<MetricDataPoint> getAllMetricData(const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime()) const;
    QStringList getAvailableMetrics() const;
    MetricDataPoint getLatestMetric(const QString& metricName) const;
    QMap<QString, MetricDataPoint> getLatestMetrics() const;

    // Trend analysis
    TrendAnalysis analyzeTrend(const QString& metricName, const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime()) const;
    QList<TrendAnalysis> analyzeAllTrends(const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime()) const;
    QMap<QString, TrendAnalysis> getTrendSummary() const;
    bool detectPerformanceRegression(const QString& metricName, double thresholdPercent = 10.0) const;
    QList<RegressionDetection> detectAllRegressions(double thresholdPercent = 10.0) const;

    // Alerting system
    void addAlert(const AlertConfig& config);
    void removeAlert(const QString& alertName);
    void updateAlert(const QString& alertName, const AlertConfig& config);
    QList<AlertConfig> getAlertConfigs() const;
    AlertConfig getAlertConfig(const QString& alertName) const;
    QList<PerformanceAlert> getActiveAlerts() const;
    QList<PerformanceAlert> getAlertHistory(const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime()) const;
    void acknowledgeAlert(const QString& alertName);
    void resolveAlert(const QString& alertName);

    // Reporting system
    bool generateReport(const ReportConfig& config);
    bool generateDashboard(const QString& outputPath);
    bool generateTrendReport(const QString& outputPath, const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime());
    bool generateAlertReport(const QString& outputPath, const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime());
    bool generateRegressionReport(const QString& outputPath);
    QJsonObject generateMetricsSnapshot() const;
    QJsonObject generatePerformanceSummary(const QDateTime& startTime = QDateTime(), const QDateTime& endTime = QDateTime()) const;

    // Data management
    void clearMetricData(const QString& metricName = "");
    void clearOldData(const QDateTime& cutoffTime);
    void exportMetricData(const QString& filePath, const QString& format = "json") const;
    bool importMetricData(const QString& filePath);
    void optimizeDataStorage();

    // Integration methods
    void integrateWithBenchmark(PerformanceBenchmark* benchmark);
    void integrateWithLoadTesting(LoadStressTesting* loadTesting);
    void startBenchmarkMonitoring(const QString& benchmarkName);
    void stopBenchmarkMonitoring();
    void startLoadTestMonitoring(const QString& testName);
    void stopLoadTestMonitoring();

    // Utility methods
    QString formatMetricValue(double value, const QString& unit) const;
    QString formatTrendType(TrendType type) const;
    QString formatAlertSeverity(AlertSeverity severity) const;
    QString formatDuration(qint64 milliseconds) const;
    QColor getMetricColor(const QString& metricName) const;
    QString getMetricIcon(MetricType type) const;

signals:
    void monitoringStarted();
    void monitoringStopped();
    void metricRecorded(const MetricDataPoint& dataPoint);
    void trendDetected(const TrendAnalysis& trend);
    void regressionDetected(const RegressionDetection& regression);
    void alertTriggered(const PerformanceAlert& alert);
    void alertResolved(const PerformanceAlert& alert);
    void reportGenerated(const QString& reportPath);
    void dataRetentionPerformed(int removedDataPoints);

private slots:
    void onMonitoringTimer();
    void onTrendAnalysisTimer();
    void onAlertEvaluationTimer();
    void onDataRetentionTimer();

private:
    // Configuration
    MonitoringConfig m_config;
    PerformanceBenchmark* m_performanceBenchmark;
    LoadStressTesting* m_loadStressTesting;

    // Monitoring state
    bool m_isMonitoring;
    bool m_isPaused;
    QTimer* m_monitoringTimer;
    QTimer* m_trendAnalysisTimer;
    QTimer* m_alertEvaluationTimer;
    QTimer* m_dataRetentionTimer;

    // Data storage
    QMap<QString, QQueue<MetricDataPoint>> m_metricData;
    QList<AlertConfig> m_alertConfigs;
    QList<PerformanceAlert> m_activeAlerts;
    QList<PerformanceAlert> m_alertHistory;
    QMap<QString, TrendAnalysis> m_trendCache;

    // Thread safety
    mutable QMutex m_dataMutex;
    mutable QMutex m_alertMutex;
    mutable QMutex m_trendMutex;

    // Network for notifications
    QNetworkAccessManager* m_networkManager;

    // Internal helper methods
    void collectSystemMetrics();
    void collectApplicationMetrics();
    void collectUIMetrics();
    void collectFileOperationMetrics();
    void collectCustomMetrics();

    // Trend analysis helpers
    TrendAnalysis calculateTrend(const QString& metricName, const QList<MetricDataPoint>& dataPoints) const;
    double calculateTrendSlope(const QList<double>& values, const QList<qint64>& timestamps) const;
    double calculateCorrelation(const QList<double>& values, const QList<qint64>& timestamps) const;
    TrendType determineTrendType(double slope, double correlation, double changePercent) const;
    QString generateTrendDescription(const TrendAnalysis& trend) const;

    // Regression detection helpers
    RegressionDetection detectRegression(const QString& metricName, const QList<MetricDataPoint>& dataPoints, double threshold) const;
    bool isSignificantRegression(double currentValue, double baselineValue, double threshold) const;
    QString determineRegressionType(const QList<MetricDataPoint>& dataPoints) const;
    QString determineRegressionSeverity(double regressionPercent) const;

    // Alert evaluation helpers
    void evaluateAlerts();
    bool evaluateAlertCondition(const AlertConfig& config, const QList<MetricDataPoint>& recentData) const;
    void triggerAlert(const AlertConfig& config, double currentValue);
    void resolveAlert(const QString& alertName, const QString& reason);
    void sendAlertNotification(const PerformanceAlert& alert);
    void sendWebhookNotification(const PerformanceAlert& alert, const QString& webhookUrl);

    // Report generation helpers
    QString generateHtmlReport(const ReportConfig& config) const;
    QString generateJsonReport(const ReportConfig& config) const;
    QString generatePdfReport(const ReportConfig& config) const;
    QString generateMetricChart(const QString& metricName, const QList<MetricDataPoint>& data) const;
    QString generateTrendChart(const TrendAnalysis& trend) const;
    QString generateAlertSummary(const QList<PerformanceAlert>& alerts) const;

    // Data management helpers
    void enforceDataRetention();
    void removeOldDataPoints(const QString& metricName, const QDateTime& cutoffTime);
    void optimizeMetricStorage(const QString& metricName);

    // System monitoring helpers
    double getCurrentCpuUsage() const;
    qint64 getCurrentMemoryUsage() const;
    double getCurrentDiskUsage() const;
    double getCurrentNetworkUsage() const;
    int getCurrentThreadCount() const;
    double getCurrentFrameRate() const;

    // Platform-specific helpers
    QString detectPlatform() const;
    QMap<QString, QVariant> getSystemInfo() const;
    QMap<QString, QVariant> getEnvironmentInfo() const;
};

/**
 * @brief Performance dashboard generator
 */
class PerformanceDashboard : public QObject {
    Q_OBJECT

public:
    explicit PerformanceDashboard(PerformanceMonitoring* monitoring, QObject* parent = nullptr);

    // Dashboard generation
    bool generateInteractiveDashboard(const QString& outputPath);
    bool generateStaticDashboard(const QString& outputPath);
    bool generateRealtimeDashboard(const QString& outputPath, int refreshIntervalMs = 5000);

    // Dashboard customization
    void setDashboardTitle(const QString& title);
    void setDashboardTheme(const QString& theme);
    void addMetricWidget(const QString& metricName, const QString& widgetType);
    void addTrendWidget(const QString& metricName);
    void addAlertWidget();
    void addRegressionWidget();

    // Chart generation
    QString generateLineChart(const QString& metricName, const QList<PerformanceMonitoring::MetricDataPoint>& data) const;
    QString generateBarChart(const QMap<QString, double>& data) const;
    QString generatePieChart(const QMap<QString, double>& data) const;
    QString generateGaugeChart(const QString& metricName, double currentValue, double minValue, double maxValue) const;

private:
    PerformanceMonitoring* m_monitoring;
    QString m_dashboardTitle;
    QString m_dashboardTheme;
    QStringList m_metricWidgets;
    QStringList m_trendWidgets;
    bool m_includeAlerts;
    bool m_includeRegressions;

    // HTML generation helpers
    QString generateDashboardHtml() const;
    QString generateMetricWidgetHtml(const QString& metricName) const;
    QString generateTrendWidgetHtml(const QString& metricName) const;
    QString generateAlertWidgetHtml() const;
    QString generateRegressionWidgetHtml() const;
    QString generateChartScript(const QString& chartId, const QString& chartType, const QJsonObject& data) const;
};

/**
 * @brief Convenience macros for performance monitoring
 */
#define MONITOR_METRIC(name, value, unit) \
    performanceMonitoring.recordMetric(name, value, unit)

#define MONITOR_EXECUTION_TIME(name, code) \
    do { \
        QElapsedTimer timer; \
        timer.start(); \
        code; \
        performanceMonitoring.recordMetric(name, timer.elapsed(), "ms", PerformanceMonitoring::MetricType::ApplicationMetric); \
    } while(0)

#define MONITOR_MEMORY_USAGE(name) \
    performanceMonitoring.recordMetric(name, performanceMonitoring.getCurrentMemoryUsage(), "bytes", PerformanceMonitoring::MetricType::SystemResource)

#define START_MONITORING_SESSION(name) \
    performanceMonitoring.startBenchmarkMonitoring(name)

#define STOP_MONITORING_SESSION() \
    performanceMonitoring.stopBenchmarkMonitoring()

#define GENERATE_PERFORMANCE_REPORT(path) \
    do { \
        PerformanceMonitoring::ReportConfig config; \
        config.outputPath = path; \
        config.format = "html"; \
        performanceMonitoring.generateReport(config); \
    } while(0)