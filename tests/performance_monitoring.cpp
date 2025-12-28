#include "performance_monitoring.h"
#include "performance_benchmark.h"
#include "load_stress_testing.h"
#include <QApplication>
#include <QWidget>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>
#include <QProcess>
#include <QSysInfo>
#include <QDebug>
#include <QtMath>
#include <QRandomGenerator>
#include <QNetworkRequest>
#include <QUrlQuery>
#include <QHttpMultiPart>
#include <algorithm>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#elif defined(Q_OS_LINUX)
#include <unistd.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#elif defined(Q_OS_MAC)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/mount.h>
#endif

PerformanceMonitoring::PerformanceMonitoring(QObject* parent)
    : QObject(parent)
    , m_performanceBenchmark(nullptr)
    , m_loadStressTesting(nullptr)
    , m_isMonitoring(false)
    , m_isPaused(false)
    , m_monitoringTimer(new QTimer(this))
    , m_trendAnalysisTimer(new QTimer(this))
    , m_alertEvaluationTimer(new QTimer(this))
    , m_dataRetentionTimer(new QTimer(this))
    , m_networkManager(new QNetworkAccessManager(this))
{
    // Set default configuration
    m_config.name = "CloneClean Performance Monitoring";
    m_config.samplingIntervalMs = 1000;
    m_config.retentionPeriodMs = 86400000; // 24 hours
    m_config.maxDataPoints = 10000;
    m_config.enableTrendAnalysis = true;
    m_config.enableRegressionDetection = true;
    m_config.enableAlerting = true;
    m_config.enableReporting = true;
    m_config.trendAnalysisWindowMs = 3600000; // 1 hour
    m_config.regressionThreshold = 10.0;
    m_config.reportOutputDirectory = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/CloneClean/Reports";
    
    // Setup timers
    connect(m_monitoringTimer, &QTimer::timeout, this, &PerformanceMonitoring::onMonitoringTimer);
    connect(m_trendAnalysisTimer, &QTimer::timeout, this, &PerformanceMonitoring::onTrendAnalysisTimer);
    connect(m_alertEvaluationTimer, &QTimer::timeout, this, &PerformanceMonitoring::onAlertEvaluationTimer);
    connect(m_dataRetentionTimer, &QTimer::timeout, this, &PerformanceMonitoring::onDataRetentionTimer);
    
    // Create output directory
    QDir().mkpath(m_config.reportOutputDirectory);
    
    qDebug() << "PerformanceMonitoring initialized";
}

PerformanceMonitoring::~PerformanceMonitoring() {
    if (m_isMonitoring) {
        stopMonitoring();
    }
}

void PerformanceMonitoring::setMonitoringConfig(const MonitoringConfig& config) {
    QMutexLocker locker(&m_dataMutex);
    m_config = config;
    
    // Update timer intervals if monitoring is active
    if (m_isMonitoring) {
        m_monitoringTimer->setInterval(m_config.samplingIntervalMs);
        m_trendAnalysisTimer->setInterval(m_config.trendAnalysisWindowMs / 10); // Analyze trends 10 times per window
        m_alertEvaluationTimer->setInterval(m_config.samplingIntervalMs * 5); // Evaluate alerts every 5 samples
    }
}

PerformanceMonitoring::MonitoringConfig PerformanceMonitoring::getMonitoringConfig() const {
    QMutexLocker locker(&m_dataMutex);
    return m_config;
}

void PerformanceMonitoring::setPerformanceBenchmark(PerformanceBenchmark* benchmark) {
    m_performanceBenchmark = benchmark;
    if (benchmark) {
        integrateWithBenchmark(benchmark);
    }
}

void PerformanceMonitoring::setLoadStressTesting(LoadStressTesting* loadTesting) {
    m_loadStressTesting = loadTesting;
    if (loadTesting) {
        integrateWithLoadTesting(loadTesting);
    }
}

bool PerformanceMonitoring::startMonitoring() {
    if (m_isMonitoring) {
        qWarning() << "Performance monitoring is already active";
        return false;
    }
    
    m_isMonitoring = true;
    m_isPaused = false;
    
    // Start monitoring timer
    m_monitoringTimer->start(m_config.samplingIntervalMs);
    
    // Start analysis timers if enabled
    if (m_config.enableTrendAnalysis) {
        m_trendAnalysisTimer->start(m_config.trendAnalysisWindowMs / 10);
    }
    
    if (m_config.enableAlerting) {
        m_alertEvaluationTimer->start(m_config.samplingIntervalMs * 5);
    }
    
    // Start data retention timer (run every hour)
    m_dataRetentionTimer->start(3600000);
    
    emit monitoringStarted();
    qDebug() << "Performance monitoring started";
    return true;
}

bool PerformanceMonitoring::stopMonitoring() {
    if (!m_isMonitoring) {
        return false;
    }
    
    m_isMonitoring = false;
    m_isPaused = false;
    
    // Stop all timers
    m_monitoringTimer->stop();
    m_trendAnalysisTimer->stop();
    m_alertEvaluationTimer->stop();
    m_dataRetentionTimer->stop();
    
    emit monitoringStopped();
    qDebug() << "Performance monitoring stopped";
    return true;
}

bool PerformanceMonitoring::isMonitoring() const {
    return m_isMonitoring && !m_isPaused;
}

void PerformanceMonitoring::pauseMonitoring() {
    if (m_isMonitoring && !m_isPaused) {
        m_isPaused = true;
        m_monitoringTimer->stop();
        qDebug() << "Performance monitoring paused";
    }
}

void PerformanceMonitoring::resumeMonitoring() {
    if (m_isMonitoring && m_isPaused) {
        m_isPaused = false;
        m_monitoringTimer->start(m_config.samplingIntervalMs);
        qDebug() << "Performance monitoring resumed";
    }
}

void PerformanceMonitoring::recordMetric(const MetricDataPoint& dataPoint) {
    QMutexLocker locker(&m_dataMutex);
    
    // Add to metric data queue
    if (!m_metricData.contains(dataPoint.metricName)) {
        m_metricData[dataPoint.metricName] = QQueue<MetricDataPoint>();
    }
    
    m_metricData[dataPoint.metricName].enqueue(dataPoint);
    
    // Enforce maximum data points per metric
    while (m_metricData[dataPoint.metricName].size() > m_config.maxDataPoints) {
        m_metricData[dataPoint.metricName].dequeue();
    }
    
    emit metricRecorded(dataPoint);
}

void PerformanceMonitoring::recordMetric(const QString& name, double value, const QString& unit, MetricType type) {
    MetricDataPoint dataPoint;
    dataPoint.metricName = name;
    dataPoint.type = type;
    dataPoint.value = value;
    dataPoint.unit = unit;
    dataPoint.timestamp = QDateTime::currentDateTime();
    dataPoint.source = "PerformanceMonitoring";
    
    recordMetric(dataPoint);
}

void PerformanceMonitoring::recordSystemMetrics() {
    collectSystemMetrics();
}

void PerformanceMonitoring::recordApplicationMetrics() {
    collectApplicationMetrics();
}

void PerformanceMonitoring::recordUIMetrics() {
    collectUIMetrics();
}

void PerformanceMonitoring::recordFileOperationMetrics() {
    collectFileOperationMetrics();
}

QList<PerformanceMonitoring::MetricDataPoint> PerformanceMonitoring::getMetricData(const QString& metricName, const QDateTime& startTime, const QDateTime& endTime) const {
    QMutexLocker locker(&m_dataMutex);
    
    QList<MetricDataPoint> result;
    
    if (!m_metricData.contains(metricName)) {
        return result;
    }
    
    const QQueue<MetricDataPoint>& queue = m_metricData[metricName];
    
    QDateTime effectiveStartTime = startTime.isValid() ? startTime : QDateTime::currentDateTime().addDays(-1);
    QDateTime effectiveEndTime = endTime.isValid() ? endTime : QDateTime::currentDateTime();
    
    for (const MetricDataPoint& point : queue) {
        if (point.timestamp >= effectiveStartTime && point.timestamp <= effectiveEndTime) {
            result.append(point);
        }
    }
    
    return result;
}

QList<PerformanceMonitoring::MetricDataPoint> PerformanceMonitoring::getAllMetricData(const QDateTime& startTime, const QDateTime& endTime) const {
    QMutexLocker locker(&m_dataMutex);
    
    QList<MetricDataPoint> result;
    
    QDateTime effectiveStartTime = startTime.isValid() ? startTime : QDateTime::currentDateTime().addDays(-1);
    QDateTime effectiveEndTime = endTime.isValid() ? endTime : QDateTime::currentDateTime();
    
    for (auto it = m_metricData.begin(); it != m_metricData.end(); ++it) {
        const QQueue<MetricDataPoint>& queue = it.value();
        
        for (const MetricDataPoint& point : queue) {
            if (point.timestamp >= effectiveStartTime && point.timestamp <= effectiveEndTime) {
                result.append(point);
            }
        }
    }
    
    // Sort by timestamp
    std::sort(result.begin(), result.end(), [](const MetricDataPoint& a, const MetricDataPoint& b) {
        return a.timestamp < b.timestamp;
    });
    
    return result;
}

QStringList PerformanceMonitoring::getAvailableMetrics() const {
    QMutexLocker locker(&m_dataMutex);
    return m_metricData.keys();
}

PerformanceMonitoring::MetricDataPoint PerformanceMonitoring::getLatestMetric(const QString& metricName) const {
    QMutexLocker locker(&m_dataMutex);
    
    if (!m_metricData.contains(metricName) || m_metricData[metricName].isEmpty()) {
        return MetricDataPoint();
    }
    
    return m_metricData[metricName].last();
}

QMap<QString, PerformanceMonitoring::MetricDataPoint> PerformanceMonitoring::getLatestMetrics() const {
    QMutexLocker locker(&m_dataMutex);
    
    QMap<QString, MetricDataPoint> result;
    
    for (auto it = m_metricData.begin(); it != m_metricData.end(); ++it) {
        if (!it.value().isEmpty()) {
            result[it.key()] = it.value().last();
        }
    }
    
    return result;
}

PerformanceMonitoring::TrendAnalysis PerformanceMonitoring::analyzeTrend(const QString& metricName, const QDateTime& startTime, const QDateTime& endTime) const {
    QList<MetricDataPoint> dataPoints = getMetricData(metricName, startTime, endTime);
    return calculateTrend(metricName, dataPoints);
}

QList<PerformanceMonitoring::TrendAnalysis> PerformanceMonitoring::analyzeAllTrends(const QDateTime& startTime, const QDateTime& endTime) const {
    QList<TrendAnalysis> results;
    
    QStringList metrics = getAvailableMetrics();
    for (const QString& metricName : metrics) {
        TrendAnalysis trend = analyzeTrend(metricName, startTime, endTime);
        if (trend.sampleCount > 0) {
            results.append(trend);
        }
    }
    
    return results;
}

QMap<QString, PerformanceMonitoring::TrendAnalysis> PerformanceMonitoring::getTrendSummary() const {
    QMutexLocker locker(&m_trendMutex);
    return m_trendCache;
}

bool PerformanceMonitoring::detectPerformanceRegression(const QString& metricName, double thresholdPercent) const {
    QList<MetricDataPoint> dataPoints = getMetricData(metricName);
    RegressionDetection regression = detectRegression(metricName, dataPoints, thresholdPercent);
    return regression.regressionDetected;
}

QList<PerformanceMonitoring::RegressionDetection> PerformanceMonitoring::detectAllRegressions(double thresholdPercent) const {
    QList<RegressionDetection> results;
    
    QStringList metrics = getAvailableMetrics();
    for (const QString& metricName : metrics) {
        QList<MetricDataPoint> dataPoints = getMetricData(metricName);
        RegressionDetection regression = detectRegression(metricName, dataPoints, thresholdPercent);
        if (regression.regressionDetected) {
            results.append(regression);
        }
    }
    
    return results;
}

void PerformanceMonitoring::addAlert(const AlertConfig& config) {
    QMutexLocker locker(&m_alertMutex);
    
    // Remove existing alert with same name
    for (int i = 0; i < m_alertConfigs.size(); ++i) {
        if (m_alertConfigs[i].name == config.name) {
            m_alertConfigs.removeAt(i);
            break;
        }
    }
    
    m_alertConfigs.append(config);
    qDebug() << "Added performance alert:" << config.name;
}

void PerformanceMonitoring::removeAlert(const QString& alertName) {
    QMutexLocker locker(&m_alertMutex);
    
    for (int i = 0; i < m_alertConfigs.size(); ++i) {
        if (m_alertConfigs[i].name == alertName) {
            m_alertConfigs.removeAt(i);
            qDebug() << "Removed performance alert:" << alertName;
            break;
        }
    }
}

void PerformanceMonitoring::updateAlert(const QString& alertName, const AlertConfig& config) {
    QMutexLocker locker(&m_alertMutex);
    
    for (AlertConfig& existingConfig : m_alertConfigs) {
        if (existingConfig.name == alertName) {
            existingConfig = config;
            qDebug() << "Updated performance alert:" << alertName;
            break;
        }
    }
}

QList<PerformanceMonitoring::AlertConfig> PerformanceMonitoring::getAlertConfigs() const {
    QMutexLocker locker(&m_alertMutex);
    return m_alertConfigs;
}

PerformanceMonitoring::AlertConfig PerformanceMonitoring::getAlertConfig(const QString& alertName) const {
    QMutexLocker locker(&m_alertMutex);
    
    for (const AlertConfig& config : m_alertConfigs) {
        if (config.name == alertName) {
            return config;
        }
    }
    
    return AlertConfig();
}

QList<PerformanceMonitoring::PerformanceAlert> PerformanceMonitoring::getActiveAlerts() const {
    QMutexLocker locker(&m_alertMutex);
    
    QList<PerformanceAlert> activeAlerts;
    for (const PerformanceAlert& alert : m_activeAlerts) {
        if (alert.isActive) {
            activeAlerts.append(alert);
        }
    }
    
    return activeAlerts;
}

QList<PerformanceMonitoring::PerformanceAlert> PerformanceMonitoring::getAlertHistory(const QDateTime& startTime, const QDateTime& endTime) const {
    QMutexLocker locker(&m_alertMutex);
    
    QList<PerformanceAlert> result;
    
    QDateTime effectiveStartTime = startTime.isValid() ? startTime : QDateTime::currentDateTime().addDays(-7);
    QDateTime effectiveEndTime = endTime.isValid() ? endTime : QDateTime::currentDateTime();
    
    for (const PerformanceAlert& alert : m_alertHistory) {
        if (alert.triggeredTime >= effectiveStartTime && alert.triggeredTime <= effectiveEndTime) {
            result.append(alert);
        }
    }
    
    return result;
}

void PerformanceMonitoring::acknowledgeAlert(const QString& alertName) {
    QMutexLocker locker(&m_alertMutex);
    
    for (PerformanceAlert& alert : m_activeAlerts) {
        if (alert.alertName == alertName && alert.isActive) {
            alert.context["acknowledged"] = true;
            alert.context["acknowledged_time"] = QDateTime::currentDateTime();
            qDebug() << "Acknowledged alert:" << alertName;
            break;
        }
    }
}

void PerformanceMonitoring::resolveAlert(const QString& alertName) {
    resolveAlert(alertName, "Manually resolved");
}boo
l PerformanceMonitoring::generateReport(const ReportConfig& config) {
    qDebug() << "Generating performance report:" << config.name;
    
    QString reportContent;
    
    if (config.format.toLower() == "html") {
        reportContent = generateHtmlReport(config);
    } else if (config.format.toLower() == "json") {
        reportContent = generateJsonReport(config);
    } else if (config.format.toLower() == "pdf") {
        reportContent = generatePdfReport(config);
    } else {
        qWarning() << "Unsupported report format:" << config.format;
        return false;
    }
    
    // Write report to file
    QFile reportFile(config.outputPath);
    if (!reportFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to open report file for writing:" << config.outputPath;
        return false;
    }
    
    QTextStream stream(&reportFile);
    stream << reportContent;
    
    emit reportGenerated(config.outputPath);
    qDebug() << "Performance report generated:" << config.outputPath;
    return true;
}

bool PerformanceMonitoring::generateDashboard(const QString& outputPath) {
    PerformanceDashboard dashboard(this);
    return dashboard.generateInteractiveDashboard(outputPath);
}

bool PerformanceMonitoring::generateTrendReport(const QString& outputPath, const QDateTime& startTime, const QDateTime& endTime) {
    ReportConfig config;
    config.name = "Trend Analysis Report";
    config.outputPath = outputPath;
    config.format = "html";
    config.startTime = startTime.isValid() ? startTime : QDateTime::currentDateTime().addDays(-7);
    config.endTime = endTime.isValid() ? endTime : QDateTime::currentDateTime();
    config.includeTrendAnalysis = true;
    config.includeAlerts = false;
    config.includeRegressions = false;
    config.title = "Performance Trend Analysis";
    config.description = "Analysis of performance trends over time";
    
    return generateReport(config);
}

bool PerformanceMonitoring::generateAlertReport(const QString& outputPath, const QDateTime& startTime, const QDateTime& endTime) {
    ReportConfig config;
    config.name = "Alert Report";
    config.outputPath = outputPath;
    config.format = "html";
    config.startTime = startTime.isValid() ? startTime : QDateTime::currentDateTime().addDays(-7);
    config.endTime = endTime.isValid() ? endTime : QDateTime::currentDateTime();
    config.includeTrendAnalysis = false;
    config.includeAlerts = true;
    config.includeRegressions = false;
    config.title = "Performance Alert Report";
    config.description = "Summary of performance alerts and incidents";
    
    return generateReport(config);
}

bool PerformanceMonitoring::generateRegressionReport(const QString& outputPath) {
    ReportConfig config;
    config.name = "Regression Report";
    config.outputPath = outputPath;
    config.format = "html";
    config.startTime = QDateTime::currentDateTime().addDays(-7);
    config.endTime = QDateTime::currentDateTime();
    config.includeTrendAnalysis = false;
    config.includeAlerts = false;
    config.includeRegressions = true;
    config.title = "Performance Regression Analysis";
    config.description = "Analysis of detected performance regressions";
    
    return generateReport(config);
}

QJsonObject PerformanceMonitoring::generateMetricsSnapshot() const {
    QJsonObject snapshot;
    snapshot["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    snapshot["platform"] = detectPlatform();
    
    QJsonObject metrics;
    QMap<QString, MetricDataPoint> latestMetrics = getLatestMetrics();
    
    for (auto it = latestMetrics.begin(); it != latestMetrics.end(); ++it) {
        QJsonObject metricObj;
        metricObj["value"] = it.value().value;
        metricObj["unit"] = it.value().unit;
        metricObj["timestamp"] = it.value().timestamp.toString(Qt::ISODate);
        metricObj["type"] = static_cast<int>(it.value().type);
        metrics[it.key()] = metricObj;
    }
    
    snapshot["metrics"] = metrics;
    return snapshot;
}

QJsonObject PerformanceMonitoring::generatePerformanceSummary(const QDateTime& startTime, const QDateTime& endTime) const {
    QJsonObject summary;
    summary["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    summary["period_start"] = (startTime.isValid() ? startTime : QDateTime::currentDateTime().addDays(-1)).toString(Qt::ISODate);
    summary["period_end"] = (endTime.isValid() ? endTime : QDateTime::currentDateTime()).toString(Qt::ISODate);
    
    // Metrics summary
    QJsonArray metricsArray;
    QStringList metrics = getAvailableMetrics();
    
    for (const QString& metricName : metrics) {
        QList<MetricDataPoint> data = getMetricData(metricName, startTime, endTime);
        if (!data.isEmpty()) {
            QJsonObject metricSummary;
            metricSummary["name"] = metricName;
            metricSummary["sample_count"] = data.size();
            
            // Calculate statistics
            QList<double> values;
            for (const MetricDataPoint& point : data) {
                values.append(point.value);
            }
            
            if (!values.isEmpty()) {
                std::sort(values.begin(), values.end());
                
                double sum = std::accumulate(values.begin(), values.end(), 0.0);
                double mean = sum / values.size();
                
                metricSummary["mean"] = mean;
                metricSummary["min"] = values.first();
                metricSummary["max"] = values.last();
                metricSummary["median"] = values[values.size() / 2];
                metricSummary["unit"] = data.first().unit;
            }
            
            metricsArray.append(metricSummary);
        }
    }
    
    summary["metrics"] = metricsArray;
    
    // Trends summary
    if (m_config.enableTrendAnalysis) {
        QJsonArray trendsArray;
        QList<TrendAnalysis> trends = analyzeAllTrends(startTime, endTime);
        
        for (const TrendAnalysis& trend : trends) {
            QJsonObject trendObj;
            trendObj["metric"] = trend.metricName;
            trendObj["type"] = formatTrendType(trend.trendType);
            trendObj["slope"] = trend.trendSlope;
            trendObj["change_percent"] = trend.changePercent;
            trendObj["description"] = trend.trendDescription;
            trendsArray.append(trendObj);
        }
        
        summary["trends"] = trendsArray;
    }
    
    // Alerts summary
    if (m_config.enableAlerting) {
        QJsonArray alertsArray;
        QList<PerformanceAlert> alerts = getAlertHistory(startTime, endTime);
        
        for (const PerformanceAlert& alert : alerts) {
            QJsonObject alertObj;
            alertObj["name"] = alert.alertName;
            alertObj["severity"] = formatAlertSeverity(alert.severity);
            alertObj["metric"] = alert.metricName;
            alertObj["triggered_time"] = alert.triggeredTime.toString(Qt::ISODate);
            alertObj["is_active"] = alert.isActive;
            alertObj["message"] = alert.message;
            alertsArray.append(alertObj);
        }
        
        summary["alerts"] = alertsArray;
    }
    
    return summary;
}

void PerformanceMonitoring::clearMetricData(const QString& metricName) {
    QMutexLocker locker(&m_dataMutex);
    
    if (metricName.isEmpty()) {
        m_metricData.clear();
        qDebug() << "Cleared all metric data";
    } else if (m_metricData.contains(metricName)) {
        m_metricData[metricName].clear();
        qDebug() << "Cleared metric data for:" << metricName;
    }
}

void PerformanceMonitoring::clearOldData(const QDateTime& cutoffTime) {
    QMutexLocker locker(&m_dataMutex);
    
    int totalRemoved = 0;
    
    for (auto it = m_metricData.begin(); it != m_metricData.end(); ++it) {
        QQueue<MetricDataPoint>& queue = it.value();
        
        while (!queue.isEmpty() && queue.first().timestamp < cutoffTime) {
            queue.dequeue();
            totalRemoved++;
        }
    }
    
    if (totalRemoved > 0) {
        emit dataRetentionPerformed(totalRemoved);
        qDebug() << "Removed" << totalRemoved << "old data points";
    }
}

void PerformanceMonitoring::exportMetricData(const QString& filePath, const QString& format) const {
    QJsonObject exportData;
    exportData["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    exportData["format_version"] = "1.0";
    exportData["platform"] = detectPlatform();
    
    QJsonObject metricsObj;
    
    QMutexLocker locker(&m_dataMutex);
    for (auto it = m_metricData.begin(); it != m_metricData.end(); ++it) {
        QJsonArray dataArray;
        
        for (const MetricDataPoint& point : it.value()) {
            QJsonObject pointObj;
            pointObj["value"] = point.value;
            pointObj["unit"] = point.unit;
            pointObj["timestamp"] = point.timestamp.toString(Qt::ISODate);
            pointObj["type"] = static_cast<int>(point.type);
            pointObj["source"] = point.source;
            pointObj["description"] = point.description;
            dataArray.append(pointObj);
        }
        
        metricsObj[it.key()] = dataArray;
    }
    
    exportData["metrics"] = metricsObj;
    
    QJsonDocument doc(exportData);
    
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        qDebug() << "Exported metric data to:" << filePath;
    } else {
        qWarning() << "Failed to export metric data to:" << filePath;
    }
}

bool PerformanceMonitoring::importMetricData(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open metric data file:" << filePath;
        return false;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject importData = doc.object();
    
    if (!importData.contains("metrics")) {
        qWarning() << "Invalid metric data format in file:" << filePath;
        return false;
    }
    
    QJsonObject metricsObj = importData["metrics"].toObject();
    
    QMutexLocker locker(&m_dataMutex);
    
    for (auto it = metricsObj.begin(); it != metricsObj.end(); ++it) {
        QString metricName = it.key();
        QJsonArray dataArray = it.value().toArray();
        
        QQueue<MetricDataPoint> dataQueue;
        
        for (const QJsonValue& value : dataArray) {
            QJsonObject pointObj = value.toObject();
            
            MetricDataPoint point;
            point.metricName = metricName;
            point.value = pointObj["value"].toDouble();
            point.unit = pointObj["unit"].toString();
            point.timestamp = QDateTime::fromString(pointObj["timestamp"].toString(), Qt::ISODate);
            point.type = static_cast<MetricType>(pointObj["type"].toInt());
            point.source = pointObj["source"].toString();
            point.description = pointObj["description"].toString();
            
            dataQueue.enqueue(point);
        }
        
        m_metricData[metricName] = dataQueue;
    }
    
    qDebug() << "Imported metric data from:" << filePath;
    return true;
}

void PerformanceMonitoring::optimizeDataStorage() {
    QMutexLocker locker(&m_dataMutex);
    
    for (auto it = m_metricData.begin(); it != m_metricData.end(); ++it) {
        optimizeMetricStorage(it.key());
    }
}

void PerformanceMonitoring::integrateWithBenchmark(PerformanceBenchmark* benchmark) {
    if (!benchmark) return;
    
    // Connect to benchmark signals
    connect(benchmark, &PerformanceBenchmark::benchmarkStarted, this, [this](const QString& name) {
        startBenchmarkMonitoring(name);
    });
    
    connect(benchmark, &PerformanceBenchmark::benchmarkCompleted, this, [this](const QString& name, const QList<PerformanceBenchmark::PerformanceResult>& results) {
        stopBenchmarkMonitoring();
        
        // Record benchmark results as metrics
        for (const auto& result : results) {
            recordMetric(result.benchmarkName + "_execution_time", result.value, result.unit, MetricType::ApplicationMetric);
        }
    });
    
    connect(benchmark, &PerformanceBenchmark::measurementRecorded, this, [this](const PerformanceBenchmark::PerformanceResult& result) {
        recordMetric(result.benchmarkName + "_" + result.metricName, result.value, result.unit, MetricType::ApplicationMetric);
    });
}

void PerformanceMonitoring::integrateWithLoadTesting(LoadStressTesting* loadTesting) {
    if (!loadTesting) return;
    
    // Connect to load testing signals
    connect(loadTesting, &LoadStressTesting::loadTestStarted, this, [this](const QString& name) {
        startLoadTestMonitoring(name);
    });
    
    connect(loadTesting, &LoadStressTesting::loadTestCompleted, this, [this](const QString& name, const LoadStressTesting::LoadTestResult& result) {
        stopLoadTestMonitoring();
        
        // Record load test results as metrics
        recordMetric(name + "_operations_per_second", result.operationsPerSecond, "ops/sec", MetricType::ApplicationMetric);
        recordMetric(name + "_average_response_time", result.averageResponseTime, "ms", MetricType::ApplicationMetric);
        recordMetric(name + "_peak_memory_usage", result.peakMemoryUsage, "bytes", MetricType::SystemResource);
    });
    
    connect(loadTesting, &LoadStressTesting::stressTestCompleted, this, [this](const QString& name, const LoadStressTesting::StressTestResult& result) {
        // Record stress test results as metrics
        recordMetric(name + "_peak_memory_mb", result.peakMemoryUsageMB, "MB", MetricType::SystemResource);
        recordMetric(name + "_peak_cpu_percent", result.peakCpuUsagePercent, "%", MetricType::SystemResource);
        recordMetric(name + "_max_concurrent_ops", result.maxConcurrentOperationsReached, "count", MetricType::ApplicationMetric);
    });
}

void PerformanceMonitoring::startBenchmarkMonitoring(const QString& benchmarkName) {
    qDebug() << "Started monitoring for benchmark:" << benchmarkName;
    
    // Record start event
    recordMetric(benchmarkName + "_started", 1.0, "event", MetricType::ApplicationMetric);
    
    // Increase monitoring frequency during benchmarks
    if (m_isMonitoring) {
        m_monitoringTimer->setInterval(m_config.samplingIntervalMs / 2);
    }
}

void PerformanceMonitoring::stopBenchmarkMonitoring() {
    qDebug() << "Stopped benchmark monitoring";
    
    // Restore normal monitoring frequency
    if (m_isMonitoring) {
        m_monitoringTimer->setInterval(m_config.samplingIntervalMs);
    }
}

void PerformanceMonitoring::startLoadTestMonitoring(const QString& testName) {
    qDebug() << "Started monitoring for load test:" << testName;
    
    // Record start event
    recordMetric(testName + "_started", 1.0, "event", MetricType::ApplicationMetric);
    
    // Increase monitoring frequency during load tests
    if (m_isMonitoring) {
        m_monitoringTimer->setInterval(m_config.samplingIntervalMs / 4);
    }
}

void PerformanceMonitoring::stopLoadTestMonitoring() {
    qDebug() << "Stopped load test monitoring";
    
    // Restore normal monitoring frequency
    if (m_isMonitoring) {
        m_monitoringTimer->setInterval(m_config.samplingIntervalMs);
    }
}

// Timer slot implementations
void PerformanceMonitoring::onMonitoringTimer() {
    if (!m_isMonitoring || m_isPaused) {
        return;
    }
    
    // Collect all enabled metrics
    collectSystemMetrics();
    collectApplicationMetrics();
    
    if (m_config.metricsToMonitor.contains("ui") || m_config.metricsToMonitor.isEmpty()) {
        collectUIMetrics();
    }
    
    if (m_config.metricsToMonitor.contains("file_operations") || m_config.metricsToMonitor.isEmpty()) {
        collectFileOperationMetrics();
    }
    
    collectCustomMetrics();
}

void PerformanceMonitoring::onTrendAnalysisTimer() {
    if (!m_config.enableTrendAnalysis) {
        return;
    }
    
    QMutexLocker trendLocker(&m_trendMutex);
    
    // Analyze trends for all metrics
    QStringList metrics = getAvailableMetrics();
    for (const QString& metricName : metrics) {
        TrendAnalysis trend = analyzeTrend(metricName);
        if (trend.sampleCount > 0) {
            m_trendCache[metricName] = trend;
            emit trendDetected(trend);
        }
    }
}

void PerformanceMonitoring::onAlertEvaluationTimer() {
    if (!m_config.enableAlerting) {
        return;
    }
    
    evaluateAlerts();
}

void PerformanceMonitoring::onDataRetentionTimer() {
    enforceDataRetention();
}// H
elper method implementations
void PerformanceMonitoring::collectSystemMetrics() {
    // CPU usage
    double cpuUsage = getCurrentCpuUsage();
    recordMetric("system_cpu_usage", cpuUsage, "%", MetricType::SystemResource);
    
    // Memory usage
    qint64 memoryUsage = getCurrentMemoryUsage();
    recordMetric("system_memory_usage", memoryUsage, "bytes", MetricType::SystemResource);
    
    // Disk usage
    double diskUsage = getCurrentDiskUsage();
    recordMetric("system_disk_usage", diskUsage, "%", MetricType::SystemResource);
    
    // Network usage
    double networkUsage = getCurrentNetworkUsage();
    recordMetric("system_network_usage", networkUsage, "bytes/sec", MetricType::SystemResource);
    
    // Thread count
    int threadCount = getCurrentThreadCount();
    recordMetric("system_thread_count", threadCount, "count", MetricType::SystemResource);
}

void PerformanceMonitoring::collectApplicationMetrics() {
    // Application-specific metrics would be collected here
    // For now, we'll collect some basic Qt application metrics
    
    if (QApplication::instance()) {
        // Number of top-level widgets
        QWidgetList widgets = QApplication::topLevelWidgets();
        recordMetric("app_widget_count", widgets.size(), "count", MetricType::ApplicationMetric);
        
        // Application uptime
        static QElapsedTimer appTimer;
        if (!appTimer.isValid()) {
            appTimer.start();
        }
        recordMetric("app_uptime", appTimer.elapsed(), "ms", MetricType::ApplicationMetric);
    }
}

void PerformanceMonitoring::collectUIMetrics() {
    if (QApplication::instance()) {
        // Frame rate (simplified)
        double frameRate = getCurrentFrameRate();
        recordMetric("ui_frame_rate", frameRate, "fps", MetricType::UserInterface);
        
        // Event queue size (approximation)
        recordMetric("ui_event_queue_size", QApplication::instance()->hasPendingEvents() ? 1.0 : 0.0, "count", MetricType::UserInterface);
    }
}

void PerformanceMonitoring::collectFileOperationMetrics() {
    // File operation metrics would be collected from actual file operations
    // This is a placeholder implementation
    static QElapsedTimer lastFileOp;
    if (lastFileOp.isValid()) {
        recordMetric("file_operation_interval", lastFileOp.elapsed(), "ms", MetricType::FileOperation);
    }
    lastFileOp.restart();
}

void PerformanceMonitoring::collectCustomMetrics() {
    // Custom metrics collection - can be extended by users
    // This is where integration with specific application metrics would go
}

PerformanceMonitoring::TrendAnalysis PerformanceMonitoring::calculateTrend(const QString& metricName, const QList<MetricDataPoint>& dataPoints) const {
    TrendAnalysis trend;
    trend.metricName = metricName;
    trend.sampleCount = dataPoints.size();
    
    if (dataPoints.size() < 2) {
        trend.trendType = TrendType::Unknown;
        return trend;
    }
    
    // Extract values and timestamps
    QList<double> values;
    QList<qint64> timestamps;
    
    for (const MetricDataPoint& point : dataPoints) {
        values.append(point.value);
        timestamps.append(point.timestamp.toMSecsSinceEpoch());
    }
    
    trend.rawValues = values;
    for (qint64 timestamp : timestamps) {
        trend.timestamps.append(QDateTime::fromMSecsSinceEpoch(timestamp));
    }
    
    // Calculate statistics
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    trend.averageValue = sum / values.size();
    
    double sumSquaredDiffs = 0.0;
    for (double value : values) {
        double diff = value - trend.averageValue;
        sumSquaredDiffs += diff * diff;
    }
    trend.standardDeviation = qSqrt(sumSquaredDiffs / (values.size() - 1));
    
    // Calculate trend slope and correlation
    trend.trendSlope = calculateTrendSlope(values, timestamps);
    trend.correlation = calculateCorrelation(values, timestamps);
    
    // Calculate percentage change
    if (values.first() != 0) {
        trend.changePercent = ((values.last() - values.first()) / values.first()) * 100.0;
    }
    
    // Determine trend type
    trend.trendType = determineTrendType(trend.trendSlope, trend.correlation, trend.changePercent);
    
    // Set analysis period
    trend.analysisStartTime = dataPoints.first().timestamp;
    trend.analysisEndTime = dataPoints.last().timestamp;
    
    // Generate description
    trend.trendDescription = generateTrendDescription(trend);
    
    return trend;
}

double PerformanceMonitoring::calculateTrendSlope(const QList<double>& values, const QList<qint64>& timestamps) const {
    if (values.size() != timestamps.size() || values.size() < 2) {
        return 0.0;
    }
    
    int n = values.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (int i = 0; i < n; ++i) {
        double x = timestamps[i];
        double y = values[i];
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
    }
    
    double denominator = n * sumXX - sumX * sumX;
    if (qAbs(denominator) < 1e-10) {
        return 0.0;
    }
    
    return (n * sumXY - sumX * sumY) / denominator;
}

double PerformanceMonitoring::calculateCorrelation(const QList<double>& values, const QList<qint64>& timestamps) const {
    if (values.size() != timestamps.size() || values.size() < 2) {
        return 0.0;
    }
    
    int n = values.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;
    
    for (int i = 0; i < n; ++i) {
        double x = timestamps[i];
        double y = values[i];
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
        sumYY += y * y;
    }
    
    double numerator = n * sumXY - sumX * sumY;
    double denominatorX = n * sumXX - sumX * sumX;
    double denominatorY = n * sumYY - sumY * sumY;
    
    if (denominatorX <= 0 || denominatorY <= 0) {
        return 0.0;
    }
    
    return numerator / qSqrt(denominatorX * denominatorY);
}

PerformanceMonitoring::TrendType PerformanceMonitoring::determineTrendType(double slope, double correlation, double changePercent) const {
    double absCorrelation = qAbs(correlation);
    
    if (absCorrelation < 0.3) {
        return TrendType::Volatile;
    }
    
    if (qAbs(changePercent) < 2.0) {
        return TrendType::Stable;
    }
    
    if (slope > 0 && changePercent > 5.0) {
        return TrendType::Improving;
    }
    
    if (slope < 0 && changePercent < -5.0) {
        return TrendType::Degrading;
    }
    
    return TrendType::Stable;
}

QString PerformanceMonitoring::generateTrendDescription(const TrendAnalysis& trend) const {
    QString description;
    
    switch (trend.trendType) {
        case TrendType::Improving:
            description = QString("Performance is improving with a %1% increase over the analysis period")
                         .arg(trend.changePercent, 0, 'f', 1);
            break;
        case TrendType::Degrading:
            description = QString("Performance is degrading with a %1% decrease over the analysis period")
                         .arg(-trend.changePercent, 0, 'f', 1);
            break;
        case TrendType::Stable:
            description = QString("Performance is stable with minimal variation (%1% change)")
                         .arg(trend.changePercent, 0, 'f', 1);
            break;
        case TrendType::Volatile:
            description = QString("Performance is highly variable with high standard deviation (%1)")
                         .arg(trend.standardDeviation, 0, 'f', 2);
            break;
        default:
            description = "Insufficient data for trend analysis";
            break;
    }
    
    return description;
}

PerformanceMonitoring::RegressionDetection PerformanceMonitoring::detectRegression(const QString& metricName, const QList<MetricDataPoint>& dataPoints, double threshold) const {
    RegressionDetection regression;
    regression.metricName = metricName;
    regression.detectionTime = QDateTime::currentDateTime();
    
    if (dataPoints.size() < 10) {
        return regression; // Need at least 10 data points
    }
    
    // Split data into baseline (first half) and current (second half)
    int splitPoint = dataPoints.size() / 2;
    
    QList<double> baselineValues;
    QList<double> currentValues;
    
    for (int i = 0; i < splitPoint; ++i) {
        baselineValues.append(dataPoints[i].value);
    }
    
    for (int i = splitPoint; i < dataPoints.size(); ++i) {
        currentValues.append(dataPoints[i].value);
        regression.evidencePoints.append(dataPoints[i]);
    }
    
    // Calculate averages
    double baselineSum = std::accumulate(baselineValues.begin(), baselineValues.end(), 0.0);
    double currentSum = std::accumulate(currentValues.begin(), currentValues.end(), 0.0);
    
    regression.baselineValue = baselineSum / baselineValues.size();
    regression.currentValue = currentSum / currentValues.size();
    
    // Check for regression
    regression.regressionDetected = isSignificantRegression(regression.currentValue, regression.baselineValue, threshold);
    
    if (regression.regressionDetected) {
        regression.regressionPercent = ((regression.currentValue - regression.baselineValue) / regression.baselineValue) * 100.0;
        regression.regressionType = determineRegressionType(dataPoints);
        regression.severity = determineRegressionSeverity(qAbs(regression.regressionPercent));
        
        regression.description = QString("Performance regression detected: %1% %2 in %3")
                                .arg(qAbs(regression.regressionPercent), 0, 'f', 1)
                                .arg(regression.regressionPercent > 0 ? "increase" : "decrease")
                                .arg(metricName);
        
        regression.recommendation = QString("Investigate recent changes that may have caused the %1 regression in %2")
                                   .arg(regression.severity.toLower())
                                   .arg(metricName);
    }
    
    return regression;
}

bool PerformanceMonitoring::isSignificantRegression(double currentValue, double baselineValue, double threshold) const {
    if (baselineValue == 0) {
        return false;
    }
    
    double changePercent = qAbs((currentValue - baselineValue) / baselineValue) * 100.0;
    return changePercent > threshold;
}

QString PerformanceMonitoring::determineRegressionType(const QList<MetricDataPoint>& dataPoints) const {
    if (dataPoints.size() < 5) {
        return "Unknown";
    }
    
    // Check if regression is sudden (last few points) or gradual (trend over time)
    QList<double> recentValues;
    for (int i = qMax(0, dataPoints.size() - 5); i < dataPoints.size(); ++i) {
        recentValues.append(dataPoints[i].value);
    }
    
    double recentVariance = 0.0;
    if (recentValues.size() > 1) {
        double recentSum = std::accumulate(recentValues.begin(), recentValues.end(), 0.0);
        double recentMean = recentSum / recentValues.size();
        
        for (double value : recentValues) {
            recentVariance += (value - recentMean) * (value - recentMean);
        }
        recentVariance /= (recentValues.size() - 1);
    }
    
    return recentVariance > 100.0 ? "Sudden" : "Gradual";
}

QString PerformanceMonitoring::determineRegressionSeverity(double regressionPercent) const {
    if (regressionPercent > 50.0) {
        return "Critical";
    } else if (regressionPercent > 25.0) {
        return "High";
    } else if (regressionPercent > 10.0) {
        return "Medium";
    } else {
        return "Low";
    }
}

void PerformanceMonitoring::evaluateAlerts() {
    QMutexLocker locker(&m_alertMutex);
    
    for (const AlertConfig& config : m_alertConfigs) {
        if (!config.enabled) {
            continue;
        }
        
        // Get recent data for evaluation
        QDateTime windowStart = QDateTime::currentDateTime().addMSecs(-config.evaluationWindowMs);
        QList<MetricDataPoint> recentData = getMetricData(config.metricName, windowStart);
        
        if (recentData.size() < config.minSamples) {
            continue;
        }
        
        bool conditionMet = evaluateAlertCondition(config, recentData);
        
        // Check if alert is already active
        bool alertActive = false;
        for (const PerformanceAlert& alert : m_activeAlerts) {
            if (alert.alertName == config.name && alert.isActive) {
                alertActive = true;
                break;
            }
        }
        
        if (conditionMet && !alertActive) {
            // Trigger new alert
            double currentValue = recentData.last().value;
            triggerAlert(config, currentValue);
        } else if (!conditionMet && alertActive) {
            // Resolve existing alert
            resolveAlert(config.name, "Condition no longer met");
        }
    }
}

bool PerformanceMonitoring::evaluateAlertCondition(const AlertConfig& config, const QList<MetricDataPoint>& recentData) const {
    if (recentData.isEmpty()) {
        return false;
    }
    
    double currentValue = recentData.last().value;
    
    if (config.comparison == ">") {
        return currentValue > config.threshold;
    } else if (config.comparison == "<") {
        return currentValue < config.threshold;
    } else if (config.comparison == "==") {
        return qAbs(currentValue - config.threshold) < 0.001;
    } else if (config.comparison == "!=") {
        return qAbs(currentValue - config.threshold) >= 0.001;
    } else if (config.comparison == ">=") {
        return currentValue >= config.threshold;
    } else if (config.comparison == "<=") {
        return currentValue <= config.threshold;
    }
    
    return false;
}

void PerformanceMonitoring::triggerAlert(const AlertConfig& config, double currentValue) {
    PerformanceAlert alert;
    alert.alertName = config.name;
    alert.severity = config.severity;
    alert.metricName = config.metricName;
    alert.currentValue = currentValue;
    alert.thresholdValue = config.threshold;
    alert.triggeredTime = QDateTime::currentDateTime();
    alert.isActive = true;
    alert.message = QString("Alert '%1' triggered: %2 %3 %4 (current: %5)")
                   .arg(config.name)
                   .arg(config.metricName)
                   .arg(config.comparison)
                   .arg(config.threshold)
                   .arg(currentValue);
    
    // Generate recommendation based on severity
    switch (config.severity) {
        case AlertSeverity::Critical:
            alert.recommendation = "Immediate action required - system performance is critically degraded";
            break;
        case AlertSeverity::Warning:
            alert.recommendation = "Monitor closely - performance is approaching critical levels";
            break;
        case AlertSeverity::Info:
            alert.recommendation = "Performance threshold exceeded - consider investigation";
            break;
        default:
            alert.recommendation = "Review performance metrics and take appropriate action";
            break;
    }
    
    m_activeAlerts.append(alert);
    m_alertHistory.append(alert);
    
    // Send notifications
    sendAlertNotification(alert);
    
    emit alertTriggered(alert);
    qDebug() << "Performance alert triggered:" << alert.alertName;
}

void PerformanceMonitoring::resolveAlert(const QString& alertName, const QString& reason) {
    QMutexLocker locker(&m_alertMutex);
    
    for (PerformanceAlert& alert : m_activeAlerts) {
        if (alert.alertName == alertName && alert.isActive) {
            alert.isActive = false;
            alert.resolvedTime = QDateTime::currentDateTime();
            alert.context["resolution_reason"] = reason;
            
            emit alertResolved(alert);
            qDebug() << "Performance alert resolved:" << alertName << "Reason:" << reason;
            break;
        }
    }
}

void PerformanceMonitoring::sendAlertNotification(const PerformanceAlert& alert) {
    // Find alert configuration
    AlertConfig config;
    for (const AlertConfig& cfg : m_alertConfigs) {
        if (cfg.name == alert.alertName) {
            config = cfg;
            break;
        }
    }
    
    // Send webhook notification if configured
    if (!config.webhookUrl.isEmpty()) {
        sendWebhookNotification(alert, config.webhookUrl);
    }
    
    // Execute action script if configured
    if (!config.actionScript.isEmpty()) {
        QProcess::startDetached(config.actionScript, QStringList() << alert.alertName << QString::number(alert.currentValue));
    }
}

void PerformanceMonitoring::sendWebhookNotification(const PerformanceAlert& alert, const QString& webhookUrl) {
    QJsonObject payload;
    payload["alert_name"] = alert.alertName;
    payload["severity"] = formatAlertSeverity(alert.severity);
    payload["metric_name"] = alert.metricName;
    payload["current_value"] = alert.currentValue;
    payload["threshold_value"] = alert.thresholdValue;
    payload["triggered_time"] = alert.triggeredTime.toString(Qt::ISODate);
    payload["message"] = alert.message;
    payload["recommendation"] = alert.recommendation;
    
    QJsonDocument doc(payload);
    QByteArray data = doc.toJson();
    
    QNetworkRequest request(QUrl(webhookUrl));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    
    QNetworkReply* reply = m_networkManager->post(request, data);
    connect(reply, &QNetworkReply::finished, reply, &QNetworkReply::deleteLater);
}

// System monitoring helper implementations
double PerformanceMonitoring::getCurrentCpuUsage() const {
    // Simplified CPU usage - real implementation would use platform-specific APIs
    return QRandomGenerator::global()->bounded(100.0);
}

qint64 PerformanceMonitoring::getCurrentMemoryUsage() const {
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
                    return parts[1].toLongLong() * 1024;
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

double PerformanceMonitoring::getCurrentDiskUsage() const {
    // Simplified disk usage calculation
    QString path = QDir::currentPath();
    
#ifdef Q_OS_WIN
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes;
    if (GetDiskFreeSpaceExA(path.toLocal8Bit().constData(), &freeBytesAvailable, &totalNumberOfBytes, &totalNumberOfFreeBytes)) {
        double usedBytes = totalNumberOfBytes.QuadPart - totalNumberOfFreeBytes.QuadPart;
        return (usedBytes / totalNumberOfBytes.QuadPart) * 100.0;
    }
#elif defined(Q_OS_LINUX) || defined(Q_OS_MAC)
    struct statvfs stat;
    if (statvfs(path.toLocal8Bit().constData(), &stat) == 0) {
        unsigned long totalBytes = stat.f_blocks * stat.f_frsize;
        unsigned long freeBytes = stat.f_bavail * stat.f_frsize;
        unsigned long usedBytes = totalBytes - freeBytes;
        return (static_cast<double>(usedBytes) / totalBytes) * 100.0;
    }
#endif
    return 0.0;
}

double PerformanceMonitoring::getCurrentNetworkUsage() const {
    // Simplified network usage - would need platform-specific implementation
    return QRandomGenerator::global()->bounded(1000000.0); // Random bytes/sec
}

int PerformanceMonitoring::getCurrentThreadCount() const {
    return QThread::idealThreadCount();
}

double PerformanceMonitoring::getCurrentFrameRate() const {
    // Simplified frame rate calculation
    return 60.0; // Assume 60 FPS for now
}

QString PerformanceMonitoring::detectPlatform() const {
    return QString("%1_%2_%3")
        .arg(QSysInfo::kernelType())
        .arg(QSysInfo::currentCpuArchitecture())
        .arg(QSysInfo::productVersion());
}

QMap<QString, QVariant> PerformanceMonitoring::getSystemInfo() const {
    QMap<QString, QVariant> info;
    info["os_type"] = QSysInfo::productType();
    info["os_version"] = QSysInfo::productVersion();
    info["kernel_type"] = QSysInfo::kernelType();
    info["kernel_version"] = QSysInfo::kernelVersion();
    info["cpu_architecture"] = QSysInfo::currentCpuArchitecture();
    info["machine_hostname"] = QSysInfo::machineHostName();
    return info;
}

QMap<QString, QVariant> PerformanceMonitoring::getEnvironmentInfo() const {
    QMap<QString, QVariant> env;
    env["qt_version"] = QT_VERSION_STR;
    env["application_name"] = QApplication::applicationName();
    env["application_version"] = QApplication::applicationVersion();
    env["working_directory"] = QDir::currentPath();
    return env;
}

// Utility method implementations
QString PerformanceMonitoring::formatMetricValue(double value, const QString& unit) const {
    if (unit == "bytes") {
        const QStringList units = {"B", "KB", "MB", "GB", "TB"};
        int unitIndex = 0;
        double size = value;
        
        while (size >= 1024.0 && unitIndex < units.size() - 1) {
            size /= 1024.0;
            unitIndex++;
        }
        
        return QString("%1 %2").arg(size, 0, 'f', 2).arg(units[unitIndex]);
    } else if (unit == "ms") {
        if (value < 1000) {
            return QString("%1 ms").arg(value, 0, 'f', 1);
        } else {
            return QString("%1 s").arg(value / 1000.0, 0, 'f', 2);
        }
    } else if (unit == "%") {
        return QString("%1%").arg(value, 0, 'f', 1);
    } else {
        return QString("%1 %2").arg(value, 0, 'f', 2).arg(unit);
    }
}

QString PerformanceMonitoring::formatTrendType(TrendType type) const {
    switch (type) {
        case TrendType::Improving: return "Improving";
        case TrendType::Stable: return "Stable";
        case TrendType::Degrading: return "Degrading";
        case TrendType::Volatile: return "Volatile";
        case TrendType::Unknown: return "Unknown";
        default: return "Unknown";
    }
}

QString PerformanceMonitoring::formatAlertSeverity(AlertSeverity severity) const {
    switch (severity) {
        case AlertSeverity::Info: return "Info";
        case AlertSeverity::Warning: return "Warning";
        case AlertSeverity::Critical: return "Critical";
        case AlertSeverity::Emergency: return "Emergency";
        default: return "Unknown";
    }
}

QString PerformanceMonitoring::formatDuration(qint64 milliseconds) const {
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

QColor PerformanceMonitoring::getMetricColor(const QString& metricName) const {
    // Simple color assignment based on metric name hash
    uint hash = qHash(metricName);
    QList<QColor> colors = {
        QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
        QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
        QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE")
    };
    
    return colors[hash % colors.size()];
}

QString PerformanceMonitoring::getMetricIcon(MetricType type) const {
    switch (type) {
        case MetricType::SystemResource: return "🖥️";
        case MetricType::ApplicationMetric: return "📱";
        case MetricType::UserInterface: return "🖼️";
        case MetricType::FileOperation: return "📁";
        case MetricType::NetworkOperation: return "🌐";
        case MetricType::DatabaseOperation: return "🗄️";
        case MetricType::CustomMetric: return "⚙️";
        default: return "📊";
    }
}

// Data management helper implementations
void PerformanceMonitoring::enforceDataRetention() {
    QDateTime cutoffTime = QDateTime::currentDateTime().addMSecs(-m_config.retentionPeriodMs);
    clearOldData(cutoffTime);
}

void PerformanceMonitoring::removeOldDataPoints(const QString& metricName, const QDateTime& cutoffTime) {
    QMutexLocker locker(&m_dataMutex);
    
    if (!m_metricData.contains(metricName)) {
        return;
    }
    
    QQueue<MetricDataPoint>& queue = m_metricData[metricName];
    
    while (!queue.isEmpty() && queue.first().timestamp < cutoffTime) {
        queue.dequeue();
    }
}

void PerformanceMonitoring::optimizeMetricStorage(const QString& metricName) {
    QMutexLocker locker(&m_dataMutex);
    
    if (!m_metricData.contains(metricName)) {
        return;
    }
    
    QQueue<MetricDataPoint>& queue = m_metricData[metricName];
    
    // If we have too many data points, remove older ones
    while (queue.size() > m_config.maxDataPoints) {
        queue.dequeue();
    }
}

// Report generation helper implementations
QString PerformanceMonitoring::generateHtmlReport(const ReportConfig& config) const {
    QString html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>)" + config.title + R"(</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .metric { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }
        .trend { background-color: #e8f5e8; }
        .alert { background-color: #ffe8e8; }
        .chart { width: 100%; height: 300px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>)" + config.title + R"(</h1>
        <p>)" + config.description + R"(</p>
        <p>Generated: )" + QDateTime::currentDateTime().toString() + R"(</p>
        <p>Period: )" + config.startTime.toString() + " to " + config.endTime.toString() + R"(</p>
    </div>
)";
    
    // Add metrics section
    if (config.includeStatistics) {
        html += "<h2>Performance Metrics</h2>\n";
        
        for (const QString& metricName : config.metricsToInclude) {
            QList<MetricDataPoint> data = getMetricData(metricName, config.startTime, config.endTime);
            if (!data.isEmpty()) {
                html += QString("<div class=\"metric\"><h3>%1</h3>").arg(metricName);
                
                // Calculate basic statistics
                QList<double> values;
                for (const MetricDataPoint& point : data) {
                    values.append(point.value);
                }
                
                if (!values.isEmpty()) {
                    std::sort(values.begin(), values.end());
                    double sum = std::accumulate(values.begin(), values.end(), 0.0);
                    double mean = sum / values.size();
                    
                    html += QString("<p>Samples: %1</p>").arg(values.size());
                    html += QString("<p>Average: %1</p>").arg(formatMetricValue(mean, data.first().unit));
                    html += QString("<p>Min: %1</p>").arg(formatMetricValue(values.first(), data.first().unit));
                    html += QString("<p>Max: %1</p>").arg(formatMetricValue(values.last(), data.first().unit));
                    html += QString("<p>Median: %1</p>").arg(formatMetricValue(values[values.size()/2], data.first().unit));
                }
                
                html += "</div>\n";
            }
        }
    }
    
    // Add trends section
    if (config.includeTrendAnalysis) {
        html += "<h2>Trend Analysis</h2>\n";
        
        QList<TrendAnalysis> trends = analyzeAllTrends(config.startTime, config.endTime);
        for (const TrendAnalysis& trend : trends) {
            if (config.metricsToInclude.isEmpty() || config.metricsToInclude.contains(trend.metricName)) {
                html += QString("<div class=\"metric trend\"><h3>%1</h3>").arg(trend.metricName);
                html += QString("<p>Trend: %1</p>").arg(formatTrendType(trend.trendType));
                html += QString("<p>Change: %1%</p>").arg(trend.changePercent, 0, 'f', 1);
                html += QString("<p>%1</p>").arg(trend.trendDescription);
                html += "</div>\n";
            }
        }
    }
    
    // Add alerts section
    if (config.includeAlerts) {
        html += "<h2>Performance Alerts</h2>\n";
        
        QList<PerformanceAlert> alerts = getAlertHistory(config.startTime, config.endTime);
        for (const PerformanceAlert& alert : alerts) {
            html += QString("<div class=\"metric alert\"><h3>%1</h3>").arg(alert.alertName);
            html += QString("<p>Severity: %1</p>").arg(formatAlertSeverity(alert.severity));
            html += QString("<p>Metric: %1</p>").arg(alert.metricName);
            html += QString("<p>Triggered: %1</p>").arg(alert.triggeredTime.toString());
            html += QString("<p>Status: %1</p>").arg(alert.isActive ? "Active" : "Resolved");
            html += QString("<p>%1</p>").arg(alert.message);
            html += "</div>\n";
        }
    }
    
    html += "</body></html>";
    return html;
}

QString PerformanceMonitoring::generateJsonReport(const ReportConfig& config) const {
    QJsonObject report;
    report["title"] = config.title;
    report["description"] = config.description;
    report["generated"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["period_start"] = config.startTime.toString(Qt::ISODate);
    report["period_end"] = config.endTime.toString(Qt::ISODate);
    
    // Add metrics
    if (config.includeStatistics) {
        QJsonArray metricsArray;
        
        for (const QString& metricName : config.metricsToInclude) {
            QList<MetricDataPoint> data = getMetricData(metricName, config.startTime, config.endTime);
            if (!data.isEmpty()) {
                QJsonObject metricObj;
                metricObj["name"] = metricName;
                metricObj["sample_count"] = data.size();
                
                QJsonArray dataArray;
                for (const MetricDataPoint& point : data) {
                    QJsonObject pointObj;
                    pointObj["timestamp"] = point.timestamp.toString(Qt::ISODate);
                    pointObj["value"] = point.value;
                    pointObj["unit"] = point.unit;
                    dataArray.append(pointObj);
                }
                metricObj["data"] = dataArray;
                
                metricsArray.append(metricObj);
            }
        }
        
        report["metrics"] = metricsArray;
    }
    
    // Add trends
    if (config.includeTrendAnalysis) {
        QJsonArray trendsArray;
        
        QList<TrendAnalysis> trends = analyzeAllTrends(config.startTime, config.endTime);
        for (const TrendAnalysis& trend : trends) {
            if (config.metricsToInclude.isEmpty() || config.metricsToInclude.contains(trend.metricName)) {
                QJsonObject trendObj;
                trendObj["metric"] = trend.metricName;
                trendObj["type"] = formatTrendType(trend.trendType);
                trendObj["slope"] = trend.trendSlope;
                trendObj["correlation"] = trend.correlation;
                trendObj["change_percent"] = trend.changePercent;
                trendObj["description"] = trend.trendDescription;
                trendsArray.append(trendObj);
            }
        }
        
        report["trends"] = trendsArray;
    }
    
    // Add alerts
    if (config.includeAlerts) {
        QJsonArray alertsArray;
        
        QList<PerformanceAlert> alerts = getAlertHistory(config.startTime, config.endTime);
        for (const PerformanceAlert& alert : alerts) {
            QJsonObject alertObj;
            alertObj["name"] = alert.alertName;
            alertObj["severity"] = formatAlertSeverity(alert.severity);
            alertObj["metric"] = alert.metricName;
            alertObj["triggered_time"] = alert.triggeredTime.toString(Qt::ISODate);
            alertObj["is_active"] = alert.isActive;
            alertObj["message"] = alert.message;
            alertObj["recommendation"] = alert.recommendation;
            alertsArray.append(alertObj);
        }
        
        report["alerts"] = alertsArray;
    }
    
    QJsonDocument doc(report);
    return doc.toJson();
}

QString PerformanceMonitoring::generatePdfReport(const ReportConfig& config) const {
    // PDF generation would require additional libraries like QPrinter
    // For now, return HTML that can be converted to PDF
    return generateHtmlReport(config);
}

#include "performance_monitoring.moc"// P
erformanceDashboard implementation
PerformanceDashboard::PerformanceDashboard(PerformanceMonitoring* monitoring, QObject* parent)
    : QObject(parent)
    , m_monitoring(monitoring)
    , m_dashboardTitle("Performance Dashboard")
    , m_dashboardTheme("default")
    , m_includeAlerts(true)
    , m_includeRegressions(true)
{
}

bool PerformanceDashboard::generateInteractiveDashboard(const QString& outputPath) {
    QString html = generateDashboardHtml();
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create dashboard file:" << outputPath;
        return false;
    }
    
    QTextStream stream(&file);
    stream << html;
    
    qDebug() << "Interactive dashboard generated:" << outputPath;
    return true;
}

bool PerformanceDashboard::generateStaticDashboard(const QString& outputPath) {
    // Static dashboard is similar to interactive but without real-time updates
    return generateInteractiveDashboard(outputPath);
}

bool PerformanceDashboard::generateRealtimeDashboard(const QString& outputPath, int refreshIntervalMs) {
    QString html = generateDashboardHtml();
    
    // Add auto-refresh meta tag
    QString refreshTag = QString("<meta http-equiv=\"refresh\" content=\"%1\">").arg(refreshIntervalMs / 1000);
    html.replace("<head>", "<head>\n" + refreshTag);
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create real-time dashboard file:" << outputPath;
        return false;
    }
    
    QTextStream stream(&file);
    stream << html;
    
    qDebug() << "Real-time dashboard generated:" << outputPath;
    return true;
}

void PerformanceDashboard::setDashboardTitle(const QString& title) {
    m_dashboardTitle = title;
}

void PerformanceDashboard::setDashboardTheme(const QString& theme) {
    m_dashboardTheme = theme;
}

void PerformanceDashboard::addMetricWidget(const QString& metricName, const QString& widgetType) {
    m_metricWidgets.append(QString("%1:%2").arg(metricName, widgetType));
}

void PerformanceDashboard::addTrendWidget(const QString& metricName) {
    m_trendWidgets.append(metricName);
}

void PerformanceDashboard::addAlertWidget() {
    m_includeAlerts = true;
}

void PerformanceDashboard::addRegressionWidget() {
    m_includeRegressions = true;
}

QString PerformanceDashboard::generateDashboardHtml() const {
    QString html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>)" + m_dashboardTitle + R"(</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .widget {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .widget:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .widget-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-unit {
            font-size: 14px;
            color: #666;
            margin-left: 5px;
        }
        .trend-indicator {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .trend-improving { background-color: #d4edda; color: #155724; }
        .trend-stable { background-color: #d1ecf1; color: #0c5460; }
        .trend-degrading { background-color: #f8d7da; color: #721c24; }
        .trend-volatile { background-color: #fff3cd; color: #856404; }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-critical { border-color: #dc3545; background-color: #f8d7da; }
        .alert-warning { border-color: #ffc107; background-color: #fff3cd; }
        .alert-info { border-color: #17a2b8; background-color: #d1ecf1; }
        .chart-container {
            width: 100%;
            height: 200px;
            margin-top: 15px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-good { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-critical { background-color: #dc3545; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="dashboard-header">
        <h1>)" + m_dashboardTitle + R"(</h1>
        <p>Real-time Performance Monitoring Dashboard</p>
        <p>Last Updated: <span id="lastUpdate">)" + QDateTime::currentDateTime().toString() + R"(</span></p>
    </div>
    
    <div class="dashboard-grid">
)";
    
    // Add metric widgets
    for (const QString& widgetSpec : m_metricWidgets) {
        QStringList parts = widgetSpec.split(":");
        if (parts.size() == 2) {
            html += generateMetricWidgetHtml(parts[0]);
        }
    }
    
    // Add trend widgets
    for (const QString& metricName : m_trendWidgets) {
        html += generateTrendWidgetHtml(metricName);
    }
    
    // Add alert widget
    if (m_includeAlerts) {
        html += generateAlertWidgetHtml();
    }
    
    // Add regression widget
    if (m_includeRegressions) {
        html += generateRegressionWidgetHtml();
    }
    
    html += R"(
    </div>
    
    <script>
        // Auto-refresh functionality
        function updateDashboard() {
            document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
        }
        
        // Update every 5 seconds
        setInterval(updateDashboard, 5000);
        
        // Initialize charts
        function initializeCharts() {
            // Chart initialization code would go here
        }
        
        window.onload = initializeCharts;
    </script>
</body>
</html>
)";
    
    return html;
}

QString PerformanceDashboard::generateMetricWidgetHtml(const QString& metricName) const {
    if (!m_monitoring) {
        return "";
    }
    
    PerformanceMonitoring::MetricDataPoint latestMetric = m_monitoring->getLatestMetric(metricName);
    
    if (latestMetric.metricName.isEmpty()) {
        return "";
    }
    
    QString formattedValue = m_monitoring->formatMetricValue(latestMetric.value, latestMetric.unit);
    QString icon = m_monitoring->getMetricIcon(latestMetric.type);
    
    // Determine status based on metric value (simplified logic)
    QString statusClass = "status-good";
    if (metricName.contains("cpu") && latestMetric.value > 80) {
        statusClass = "status-critical";
    } else if (metricName.contains("memory") && latestMetric.value > 1024*1024*1024) { // 1GB
        statusClass = "status-warning";
    }
    
    return QString(R"(
        <div class="widget">
            <div class="widget-title">
                %1 %2
                <span class="status-indicator %3"></span>
            </div>
            <div class="metric-value">%4</div>
            <div style="margin-top: 10px; font-size: 12px; color: #666;">
                Last updated: %5
            </div>
            <div class="chart-container">
                <canvas id="chart_%6"></canvas>
            </div>
        </div>
    )").arg(icon, metricName, statusClass, formattedValue, 
            latestMetric.timestamp.toString(), 
            QString(metricName).replace(" ", "_"));
}

QString PerformanceDashboard::generateTrendWidgetHtml(const QString& metricName) const {
    if (!m_monitoring) {
        return "";
    }
    
    PerformanceMonitoring::TrendAnalysis trend = m_monitoring->analyzeTrend(metricName);
    
    if (trend.metricName.isEmpty()) {
        return "";
    }
    
    QString trendClass;
    switch (trend.trendType) {
        case PerformanceMonitoring::TrendType::Improving:
            trendClass = "trend-improving";
            break;
        case PerformanceMonitoring::TrendType::Degrading:
            trendClass = "trend-degrading";
            break;
        case PerformanceMonitoring::TrendType::Volatile:
            trendClass = "trend-volatile";
            break;
        default:
            trendClass = "trend-stable";
            break;
    }
    
    return QString(R"(
        <div class="widget">
            <div class="widget-title">
                📈 %1 Trend
                <span class="trend-indicator %2">%3</span>
            </div>
            <div style="margin: 10px 0;">
                <strong>Change:</strong> %4%
            </div>
            <div style="margin: 10px 0;">
                <strong>Samples:</strong> %5
            </div>
            <div style="font-size: 14px; color: #666;">
                %6
            </div>
        </div>
    )").arg(metricName, trendClass, 
            m_monitoring->formatTrendType(trend.trendType),
            QString::number(trend.changePercent, 'f', 1),
            QString::number(trend.sampleCount),
            trend.trendDescription);
}

QString PerformanceDashboard::generateAlertWidgetHtml() const {
    if (!m_monitoring) {
        return "";
    }
    
    QList<PerformanceMonitoring::PerformanceAlert> activeAlerts = m_monitoring->getActiveAlerts();
    
    QString alertsHtml = R"(
        <div class="widget">
            <div class="widget-title">🚨 Active Alerts</div>
    )";
    
    if (activeAlerts.isEmpty()) {
        alertsHtml += "<div style=\"color: #28a745; font-weight: bold;\">✅ No active alerts</div>";
    } else {
        for (const PerformanceMonitoring::PerformanceAlert& alert : activeAlerts) {
            QString alertClass;
            switch (alert.severity) {
                case PerformanceMonitoring::AlertSeverity::Critical:
                case PerformanceMonitoring::AlertSeverity::Emergency:
                    alertClass = "alert-critical";
                    break;
                case PerformanceMonitoring::AlertSeverity::Warning:
                    alertClass = "alert-warning";
                    break;
                default:
                    alertClass = "alert-info";
                    break;
            }
            
            alertsHtml += QString(R"(
                <div class="alert-item %1">
                    <strong>%2</strong><br>
                    <small>%3 - %4</small><br>
                    <small>%5</small>
                </div>
            )").arg(alertClass, alert.alertName, 
                    m_monitoring->formatAlertSeverity(alert.severity),
                    alert.metricName,
                    alert.triggeredTime.toString());
        }
    }
    
    alertsHtml += "</div>";
    return alertsHtml;
}

QString PerformanceDashboard::generateRegressionWidgetHtml() const {
    if (!m_monitoring) {
        return "";
    }
    
    QList<PerformanceMonitoring::RegressionDetection> regressions = m_monitoring->detectAllRegressions();
    
    QString regressionsHtml = R"(
        <div class="widget">
            <div class="widget-title">📉 Performance Regressions</div>
    )";
    
    if (regressions.isEmpty()) {
        regressionsHtml += "<div style=\"color: #28a745; font-weight: bold;\">✅ No regressions detected</div>";
    } else {
        for (const PerformanceMonitoring::RegressionDetection& regression : regressions) {
            QString severityClass;
            if (regression.severity == "Critical") {
                severityClass = "alert-critical";
            } else if (regression.severity == "High" || regression.severity == "Medium") {
                severityClass = "alert-warning";
            } else {
                severityClass = "alert-info";
            }
            
            regressionsHtml += QString(R"(
                <div class="alert-item %1">
                    <strong>%2</strong><br>
                    <small>%3% regression (%4)</small><br>
                    <small>%5</small>
                </div>
            )").arg(severityClass, regression.metricName,
                    QString::number(qAbs(regression.regressionPercent), 'f', 1),
                    regression.severity,
                    regression.description);
        }
    }
    
    regressionsHtml += "</div>";
    return regressionsHtml;
}

QString PerformanceDashboard::generateLineChart(const QString& metricName, const QList<PerformanceMonitoring::MetricDataPoint>& data) const {
    // Generate Chart.js configuration for line chart
    QJsonObject chartConfig;
    chartConfig["type"] = "line";
    
    QJsonObject chartData;
    QJsonArray labels;
    QJsonArray values;
    
    for (const PerformanceMonitoring::MetricDataPoint& point : data) {
        labels.append(point.timestamp.toString("hh:mm:ss"));
        values.append(point.value);
    }
    
    QJsonObject dataset;
    dataset["label"] = metricName;
    dataset["data"] = values;
    dataset["borderColor"] = "rgb(75, 192, 192)";
    dataset["tension"] = 0.1;
    
    chartData["labels"] = labels;
    chartData["datasets"] = QJsonArray({dataset});
    
    chartConfig["data"] = chartData;
    
    QJsonDocument doc(chartConfig);
    return doc.toJson(QJsonDocument::Compact);
}

QString PerformanceDashboard::generateBarChart(const QMap<QString, double>& data) const {
    QJsonObject chartConfig;
    chartConfig["type"] = "bar";
    
    QJsonObject chartData;
    QJsonArray labels;
    QJsonArray values;
    
    for (auto it = data.begin(); it != data.end(); ++it) {
        labels.append(it.key());
        values.append(it.value());
    }
    
    QJsonObject dataset;
    dataset["label"] = "Values";
    dataset["data"] = values;
    dataset["backgroundColor"] = "rgba(54, 162, 235, 0.2)";
    dataset["borderColor"] = "rgba(54, 162, 235, 1)";
    dataset["borderWidth"] = 1;
    
    chartData["labels"] = labels;
    chartData["datasets"] = QJsonArray({dataset});
    
    chartConfig["data"] = chartData;
    
    QJsonDocument doc(chartConfig);
    return doc.toJson(QJsonDocument::Compact);
}

QString PerformanceDashboard::generatePieChart(const QMap<QString, double>& data) const {
    QJsonObject chartConfig;
    chartConfig["type"] = "pie";
    
    QJsonObject chartData;
    QJsonArray labels;
    QJsonArray values;
    
    for (auto it = data.begin(); it != data.end(); ++it) {
        labels.append(it.key());
        values.append(it.value());
    }
    
    QJsonObject dataset;
    dataset["data"] = values;
    dataset["backgroundColor"] = QJsonArray({
        "rgba(255, 99, 132, 0.2)",
        "rgba(54, 162, 235, 0.2)",
        "rgba(255, 205, 86, 0.2)",
        "rgba(75, 192, 192, 0.2)",
        "rgba(153, 102, 255, 0.2)"
    });
    
    chartData["labels"] = labels;
    chartData["datasets"] = QJsonArray({dataset});
    
    chartConfig["data"] = chartData;
    
    QJsonDocument doc(chartConfig);
    return doc.toJson(QJsonDocument::Compact);
}

QString PerformanceDashboard::generateGaugeChart(const QString& metricName, double currentValue, double minValue, double maxValue) const {
    // Gauge chart implementation would require a specialized chart library
    // For now, return a simple representation
    double percentage = ((currentValue - minValue) / (maxValue - minValue)) * 100.0;
    
    return QString("Gauge: %1 = %2% (%3 of %4-%5)")
           .arg(metricName)
           .arg(percentage, 0, 'f', 1)
           .arg(currentValue)
           .arg(minValue)
           .arg(maxValue);
}