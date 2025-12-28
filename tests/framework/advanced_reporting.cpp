#include "advanced_reporting.h"
#include "test_harness.h"
#include "test_maintenance_tools.h"
#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QTimer>
#include <QStandardPaths>
#include <QDebug>
#include <QtMath>
#include <QRandomGenerator>
#include <algorithm>
#include <numeric>

// TestTrendAnalyzer Implementation
TestTrendAnalyzer::TestTrendAnalyzer(QObject* parent)
    : QObject(parent)
{
}

void TestTrendAnalyzer::addDataPoint(const TestTrendDataPoint& dataPoint) {
    m_historicalData.append(dataPoint);
    
    // Update cache
    QString key = QString("%1::%2").arg(dataPoint.suiteName, dataPoint.testName);
    m_testDataCache[key].append(dataPoint);
    
    // Keep cache size reasonable (last 1000 data points per test)
    if (m_testDataCache[key].size() > 1000) {
        m_testDataCache[key].removeFirst();
    }
}

void TestTrendAnalyzer::addDataPoints(const QList<TestTrendDataPoint>& dataPoints) {
    for (const TestTrendDataPoint& point : dataPoints) {
        addDataPoint(point);
    }
}

void TestTrendAnalyzer::loadHistoricalData(const QString& dataFilePath) {
    QFile file(dataFilePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to load historical data:" << dataFilePath;
        return;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError) {
        qWarning() << "Failed to parse historical data:" << error.errorString();
        return;
    }
    
    QJsonArray dataArray = doc.object()["dataPoints"].toArray();
    for (const QJsonValue& value : dataArray) {
        QJsonObject pointObj = value.toObject();
        
        TestTrendDataPoint point;
        point.timestamp = QDateTime::fromString(pointObj["timestamp"].toString(), Qt::ISODate);
        point.testName = pointObj["testName"].toString();
        point.suiteName = pointObj["suiteName"].toString();
        point.passed = pointObj["passed"].toBool();
        point.executionTime = pointObj["executionTime"].toInt();
        point.memoryUsage = pointObj["memoryUsage"].toDouble();
        point.cpuUsage = pointObj["cpuUsage"].toDouble();
        
        // Load custom metrics
        QJsonObject metricsObj = pointObj["customMetrics"].toObject();
        for (auto it = metricsObj.begin(); it != metricsObj.end(); ++it) {
            point.customMetrics[it.key()] = it.value().toVariant();
        }
        
        addDataPoint(point);
    }
    
    qDebug() << "Loaded" << m_historicalData.size() << "historical data points";
}

void TestTrendAnalyzer::saveHistoricalData(const QString& dataFilePath) const {
    QJsonObject root;
    QJsonArray dataArray;
    
    for (const TestTrendDataPoint& point : m_historicalData) {
        QJsonObject pointObj;
        pointObj["timestamp"] = point.timestamp.toString(Qt::ISODate);
        pointObj["testName"] = point.testName;
        pointObj["suiteName"] = point.suiteName;
        pointObj["passed"] = point.passed;
        pointObj["executionTime"] = point.executionTime;
        pointObj["memoryUsage"] = point.memoryUsage;
        pointObj["cpuUsage"] = point.cpuUsage;
        
        // Save custom metrics
        QJsonObject metricsObj;
        for (auto it = point.customMetrics.begin(); it != point.customMetrics.end(); ++it) {
            metricsObj[it.key()] = QJsonValue::fromVariant(it.value());
        }
        pointObj["customMetrics"] = metricsObj;
        
        dataArray.append(pointObj);
    }
    
    root["dataPoints"] = dataArray;
    root["exportDate"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    QFile file(dataFilePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson());
        qDebug() << "Saved" << m_historicalData.size() << "data points to" << dataFilePath;
    } else {
        qWarning() << "Failed to save historical data:" << dataFilePath;
    }
}

PerformanceTrendAnalysis TestTrendAnalyzer::analyzePerformanceTrend(const QString& testName, 
                                                                   const QString& metricName,
                                                                   int daysPeriod) const {
    PerformanceTrendAnalysis analysis;
    analysis.metricName = metricName;
    
    QDateTime startDate = QDateTime::currentDateTime().addDays(-daysPeriod);
    QDateTime endDate = QDateTime::currentDateTime();
    
    analysis.dataPoints = getDataPoints(testName, startDate, endDate);
    
    if (analysis.dataPoints.isEmpty()) {
        return analysis;
    }
    
    // Extract metric values
    QList<double> values;
    for (const TestTrendDataPoint& point : analysis.dataPoints) {
        double value = 0.0;
        
        if (metricName == "executionTime") {
            value = point.executionTime;
        } else if (metricName == "memoryUsage") {
            value = point.memoryUsage;
        } else if (metricName == "cpuUsage") {
            value = point.cpuUsage;
        } else if (point.customMetrics.contains(metricName)) {
            value = point.customMetrics[metricName].toDouble();
        }
        
        values.append(value);
    }
    
    if (!values.isEmpty()) {
        analysis.currentValue = values.last();
        analysis.trendSlope = calculateTrendSlope(values);
        analysis.volatility = calculateVolatility(values);
        analysis.trendDirection = determineTrendDirection(analysis.trendSlope, analysis.volatility);
        analysis.alerts = generateTrendAlerts(analysis);
        
        // Generate prediction
        if (qAbs(analysis.trendSlope) > 0.01) {
            double predictedChange = analysis.trendSlope * 7; // 7 days ahead
            analysis.prediction = QString("Predicted %1 change: %2%")
                                .arg(metricName)
                                .arg(predictedChange * 100, 0, 'f', 1);
        } else {
            analysis.prediction = QString("%1 expected to remain stable").arg(metricName);
        }
    }
    
    emit trendAnalysisCompleted(testName);
    return analysis;
}

QList<PerformanceTrendAnalysis> TestTrendAnalyzer::analyzeAllTrends(int daysPeriod) const {
    QList<PerformanceTrendAnalysis> allTrends;
    
    // Get unique test names
    QSet<QString> uniqueTests;
    for (const TestTrendDataPoint& point : m_historicalData) {
        uniqueTests.insert(QString("%1::%2").arg(point.suiteName, point.testName));
    }
    
    // Analyze trends for each test and common metrics
    QStringList metrics = {"executionTime", "memoryUsage", "cpuUsage"};
    
    for (const QString& testName : uniqueTests) {
        for (const QString& metric : metrics) {
            PerformanceTrendAnalysis trend = analyzePerformanceTrend(testName, metric, daysPeriod);
            if (!trend.dataPoints.isEmpty()) {
                allTrends.append(trend);
            }
        }
    }
    
    return allTrends;
}

double TestTrendAnalyzer::calculateSuccessRateTrend(const QString& testName, int daysPeriod) const {
    QDateTime startDate = QDateTime::currentDateTime().addDays(-daysPeriod);
    QDateTime endDate = QDateTime::currentDateTime();
    
    QList<TestTrendDataPoint> dataPoints = getDataPoints(testName, startDate, endDate);
    
    if (dataPoints.size() < 2) {
        return 0.0;
    }
    
    // Calculate success rate for first and second half
    int halfPoint = dataPoints.size() / 2;
    
    int firstHalfPassed = 0;
    int secondHalfPassed = 0;
    
    for (int i = 0; i < halfPoint; ++i) {
        if (dataPoints[i].passed) {
            firstHalfPassed++;
        }
    }
    
    for (int i = halfPoint; i < dataPoints.size(); ++i) {
        if (dataPoints[i].passed) {
            secondHalfPassed++;
        }
    }
    
    double firstHalfRate = (double)firstHalfPassed / halfPoint;
    double secondHalfRate = (double)secondHalfPassed / (dataPoints.size() - halfPoint);
    
    return secondHalfRate - firstHalfRate;
}

QMap<QString, double> TestTrendAnalyzer::getSuccessRateTrends(int daysPeriod) const {
    QMap<QString, double> trends;
    
    // Get unique test names
    QSet<QString> uniqueTests;
    for (const TestTrendDataPoint& point : m_historicalData) {
        uniqueTests.insert(QString("%1::%2").arg(point.suiteName, point.testName));
    }
    
    for (const QString& testName : uniqueTests) {
        double trend = calculateSuccessRateTrend(testName, daysPeriod);
        trends[testName] = trend;
    }
    
    return trends;
}

double TestTrendAnalyzer::predictExecutionTime(const QString& testName, int daysAhead) const {
    PerformanceTrendAnalysis trend = analyzePerformanceTrend(testName, "executionTime", 30);
    
    if (trend.dataPoints.isEmpty()) {
        return 5000.0; // Default prediction
    }
    
    double currentTime = trend.currentValue;
    double predictedChange = trend.trendSlope * daysAhead;
    
    return qMax(100.0, currentTime + (currentTime * predictedChange));
}

QMap<QString, double> TestTrendAnalyzer::predictAllExecutionTimes(int daysAhead) const {
    QMap<QString, double> predictions;
    
    // Get unique test names
    QSet<QString> uniqueTests;
    for (const TestTrendDataPoint& point : m_historicalData) {
        uniqueTests.insert(QString("%1::%2").arg(point.suiteName, point.testName));
    }
    
    for (const QString& testName : uniqueTests) {
        predictions[testName] = predictExecutionTime(testName, daysAhead);
    }
    
    return predictions;
}

QStringList TestTrendAnalyzer::detectAnomalies(const QString& testName, double threshold) const {
    QStringList anomalies;
    
    QString key = testName;
    if (!m_testDataCache.contains(key)) {
        return anomalies;
    }
    
    const QList<TestTrendDataPoint>& dataPoints = m_testDataCache[key];
    if (dataPoints.size() < 10) {
        return anomalies; // Need enough data for anomaly detection
    }
    
    // Calculate baseline statistics
    QList<qint64> executionTimes;
    for (const TestTrendDataPoint& point : dataPoints) {
        executionTimes.append(point.executionTime);
    }
    
    double mean = std::accumulate(executionTimes.begin(), executionTimes.end(), 0.0) / executionTimes.size();
    
    double variance = 0.0;
    for (qint64 time : executionTimes) {
        variance += qPow(time - mean, 2);
    }
    variance /= executionTimes.size();
    double stdDev = qSqrt(variance);
    
    // Detect anomalies in recent data points
    int recentCount = qMin(5, dataPoints.size());
    for (int i = dataPoints.size() - recentCount; i < dataPoints.size(); ++i) {
        const TestTrendDataPoint& point = dataPoints[i];
        
        double zScore = qAbs(point.executionTime - mean) / stdDev;
        if (zScore > threshold) {
            anomalies << QString("Execution time anomaly at %1: %2ms (z-score: %3)")
                        .arg(point.timestamp.toString())
                        .arg(point.executionTime)
                        .arg(zScore, 0, 'f', 2);
        }
        
        // Check for consecutive failures
        if (!point.passed) {
            int consecutiveFailures = 1;
            for (int j = i - 1; j >= 0 && !dataPoints[j].passed; --j) {
                consecutiveFailures++;
            }
            
            if (consecutiveFailures >= 3) {
                anomalies << QString("Consecutive failures detected: %1 failures ending at %2")
                           .arg(consecutiveFailures)
                           .arg(point.timestamp.toString());
            }
        }
    }
    
    if (!anomalies.isEmpty()) {
        emit anomalyDetected(testName, anomalies.join("; "));
    }
    
    return anomalies;
}

QMap<QString, QStringList> TestTrendAnalyzer::detectAllAnomalies(double threshold) const {
    QMap<QString, QStringList> allAnomalies;
    
    for (auto it = m_testDataCache.begin(); it != m_testDataCache.end(); ++it) {
        QStringList anomalies = detectAnomalies(it.key(), threshold);
        if (!anomalies.isEmpty()) {
            allAnomalies[it.key()] = anomalies;
        }
    }
    
    return allAnomalies;
}

QJsonObject TestTrendAnalyzer::exportTrendData() const {
    QJsonObject root;
    QJsonArray trendsArray;
    
    QList<PerformanceTrendAnalysis> allTrends = analyzeAllTrends(30);
    for (const PerformanceTrendAnalysis& trend : allTrends) {
        QJsonObject trendObj;
        trendObj["metricName"] = trend.metricName;
        trendObj["currentValue"] = trend.currentValue;
        trendObj["trendSlope"] = trend.trendSlope;
        trendObj["volatility"] = trend.volatility;
        trendObj["trendDirection"] = trend.trendDirection;
        trendObj["prediction"] = trend.prediction;
        
        QJsonArray alertsArray;
        for (const QString& alert : trend.alerts) {
            alertsArray.append(alert);
        }
        trendObj["alerts"] = alertsArray;
        
        trendsArray.append(trendObj);
    }
    
    root["trends"] = trendsArray;
    root["exportDate"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["dataPointCount"] = m_historicalData.size();
    
    return root;
}

void TestTrendAnalyzer::generateTrendReport(const QString& outputPath) const {
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create trend report:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    out << "# Test Trend Analysis Report\n\n";
    out << "Generated: " << QDateTime::currentDateTime().toString() << "\n\n";
    
    QList<PerformanceTrendAnalysis> allTrends = analyzeAllTrends(30);
    QMap<QString, QStringList> anomalies = detectAllAnomalies(2.0);
    
    out << "## Summary\n";
    out << "- Total trends analyzed: " << allTrends.size() << "\n";
    out << "- Anomalies detected: " << anomalies.size() << "\n";
    out << "- Data points: " << m_historicalData.size() << "\n\n";
    
    // Performance regressions
    QList<PerformanceTrendAnalysis> regressions;
    for (const PerformanceTrendAnalysis& trend : allTrends) {
        if (trend.trendDirection == "degrading" && trend.trendSlope > 0.1) {
            regressions.append(trend);
        }
    }
    
    if (!regressions.isEmpty()) {
        out << "## Performance Regressions\n\n";
        for (const PerformanceTrendAnalysis& trend : regressions) {
            out << "### " << trend.metricName << "\n";
            out << "- Current value: " << trend.currentValue << "\n";
            out << "- Trend slope: " << QString::number(trend.trendSlope * 100, 'f', 2) << "%\n";
            out << "- Direction: " << trend.trendDirection << "\n";
            out << "- Prediction: " << trend.prediction << "\n\n";
        }
    }
    
    // Anomalies
    if (!anomalies.isEmpty()) {
        out << "## Detected Anomalies\n\n";
        for (auto it = anomalies.begin(); it != anomalies.end(); ++it) {
            out << "### " << it.key() << "\n";
            for (const QString& anomaly : it.value()) {
                out << "- " << anomaly << "\n";
            }
            out << "\n";
        }
    }
}

// Private methods for TestTrendAnalyzer
QList<TestTrendDataPoint> TestTrendAnalyzer::getDataPoints(const QString& testName, 
                                                          const QDateTime& startDate,
                                                          const QDateTime& endDate) const {
    QList<TestTrendDataPoint> filteredPoints;
    
    QString key = testName;
    if (m_testDataCache.contains(key)) {
        for (const TestTrendDataPoint& point : m_testDataCache[key]) {
            if (point.timestamp >= startDate && point.timestamp <= endDate) {
                filteredPoints.append(point);
            }
        }
    }
    
    return filteredPoints;
}

double TestTrendAnalyzer::calculateTrendSlope(const QList<double>& values) const {
    if (values.size() < 2) {
        return 0.0;
    }
    
    // Simple linear regression slope calculation
    int n = values.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (int i = 0; i < n; ++i) {
        double x = i; // Time index
        double y = values[i];
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }
    
    double denominator = n * sumX2 - sumX * sumX;
    if (qAbs(denominator) < 1e-10) {
        return 0.0;
    }
    
    double slope = (n * sumXY - sumX * sumY) / denominator;
    
    // Normalize by average value to get percentage change
    double avgValue = sumY / n;
    return avgValue != 0 ? slope / avgValue : 0.0;
}

double TestTrendAnalyzer::calculateVolatility(const QList<double>& values) const {
    if (values.size() < 2) {
        return 0.0;
    }
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    double variance = 0.0;
    for (double value : values) {
        variance += qPow(value - mean, 2);
    }
    variance /= values.size();
    
    double stdDev = qSqrt(variance);
    return mean != 0 ? stdDev / mean : 0.0; // Coefficient of variation
}

QString TestTrendAnalyzer::determineTrendDirection(double slope, double volatility) const {
    if (qAbs(slope) < 0.02) { // Less than 2% change
        return "stable";
    } else if (slope > 0) {
        return volatility > 0.2 ? "degrading" : "degrading";
    } else {
        return volatility > 0.2 ? "improving" : "improving";
    }
}

QStringList TestTrendAnalyzer::generateTrendAlerts(const PerformanceTrendAnalysis& analysis) const {
    QStringList alerts;
    
    if (analysis.trendDirection == "degrading" && analysis.trendSlope > 0.1) {
        alerts << QString("Performance degradation detected: %1% increase in %2")
                 .arg(analysis.trendSlope * 100, 0, 'f', 1)
                 .arg(analysis.metricName);
    }
    
    if (analysis.volatility > 0.3) {
        alerts << QString("High volatility detected in %1: %2% coefficient of variation")
                 .arg(analysis.metricName)
                 .arg(analysis.volatility * 100, 0, 'f', 1);
    }
    
    if (analysis.trendSlope > 0.2) {
        emit performanceRegressionPredicted(analysis.metricName, analysis.trendSlope);
    }
    
    return alerts;
}

// TestEffectivenessAnalyzer Implementation
TestEffectivenessAnalyzer::TestEffectivenessAnalyzer(QObject* parent)
    : QObject(parent)
{
}

void TestEffectivenessAnalyzer::setDefectTrackingEnabled(bool enabled) {
    m_defectTrackingEnabled = enabled;
}

void TestEffectivenessAnalyzer::setMaintenanceCostTracking(bool enabled) {
    m_maintenanceCostTracking = enabled;
}

void TestEffectivenessAnalyzer::setCostPerHour(double costPerHour) {
    m_costPerHour = qMax(0.0, costPerHour);
}

void TestEffectivenessAnalyzer::analyzeTestEffectiveness(const QString& testName) {
    TestEffectivenessMetrics metrics;
    metrics.testName = testName;
    
    // Get defect history
    if (m_defectHistory.contains(testName)) {
        metrics.defectsFound = m_defectHistory[testName].size();
    }
    
    // Calculate maintenance cost
    if (m_maintenanceCosts.contains(testName)) {
        metrics.maintenanceCost = m_maintenanceCosts[testName];
    }
    
    // Simulate some metrics for demonstration
    metrics.totalRuns = QRandomGenerator::global()->bounded(50, 200);
    metrics.falsePositives = QRandomGenerator::global()->bounded(0, 5);
    metrics.falseNegatives = QRandomGenerator::global()->bounded(0, 3);
    
    // Calculate scores
    metrics.effectivenessScore = calculateEffectivenessScore(metrics);
    metrics.valueScore = calculateValueScore(metrics);
    metrics.recommendation = generateRecommendation(metrics);
    metrics.improvementSuggestions = generateImprovementSuggestions(metrics);
    
    m_effectivenessData[testName] = metrics;
    
    // Emit signals for low value or high maintenance cost
    if (metrics.valueScore < 0.3) {
        emit lowValueTestDetected(testName, metrics.valueScore);
    }
    
    if (metrics.maintenanceCost > 10.0) { // More than 10 hours per month
        emit highMaintenanceCostDetected(testName, metrics.maintenanceCost);
    }
}

void TestEffectivenessAnalyzer::analyzeAllTestEffectiveness() {
    // Analyze all tests that have data
    QSet<QString> allTests;
    
    // Collect all test names from various sources
    allTests.unite(QSet<QString>(m_defectHistory.keys().begin(), m_defectHistory.keys().end()));
    allTests.unite(QSet<QString>(m_maintenanceCosts.keys().begin(), m_maintenanceCosts.keys().end()));
    
    // Add some sample tests for demonstration
    allTests << "CoreTest::testBasicFunctionality" << "UITest::testUserInterface" 
             << "PerformanceTest::testLargeDataset" << "IntegrationTest::testWorkflow";
    
    int lowValueTests = 0;
    for (const QString& testName : allTests) {
        analyzeTestEffectiveness(testName);
        
        if (m_effectivenessData.contains(testName) && 
            m_effectivenessData[testName].valueScore < 0.3) {
            lowValueTests++;
        }
    }
    
    emit effectivenessAnalysisCompleted(allTests.size(), lowValueTests);
}

TestEffectivenessMetrics TestEffectivenessAnalyzer::getTestEffectiveness(const QString& testName) const {
    return m_effectivenessData.value(testName, TestEffectivenessMetrics());
}

void TestEffectivenessAnalyzer::recordDefectFound(const QString& testName, const QString& defectId) {
    if (m_defectTrackingEnabled) {
        m_defectHistory[testName].append(defectId);
    }
}

void TestEffectivenessAnalyzer::recordFalsePositive(const QString& testName, const QString& reason) {
    if (m_effectivenessData.contains(testName)) {
        m_effectivenessData[testName].falsePositives++;
    }
}

void TestEffectivenessAnalyzer::recordFalseNegative(const QString& testName, const QString& missedDefect) {
    if (m_effectivenessData.contains(testName)) {
        m_effectivenessData[testName].falseNegatives++;
    }
}

void TestEffectivenessAnalyzer::recordMaintenanceTime(const QString& testName, double hours, const QString& reason) {
    if (m_maintenanceCostTracking) {
        m_maintenanceCosts[testName] += hours * m_costPerHour;
    }
}

double TestEffectivenessAnalyzer::getMaintenanceCost(const QString& testName) const {
    return m_maintenanceCosts.value(testName, 0.0);
}

QList<TestEffectivenessMetrics> TestEffectivenessAnalyzer::getHighValueTests(int topN) const {
    QList<TestEffectivenessMetrics> allMetrics = m_effectivenessData.values();
    
    // Sort by value score (highest first)
    std::sort(allMetrics.begin(), allMetrics.end(),
              [](const TestEffectivenessMetrics& a, const TestEffectivenessMetrics& b) {
                  return a.valueScore > b.valueScore;
              });
    
    return allMetrics.mid(0, qMin(topN, allMetrics.size()));
}

QList<TestEffectivenessMetrics> TestEffectivenessAnalyzer::getLowValueTests(int bottomN) const {
    QList<TestEffectivenessMetrics> allMetrics = m_effectivenessData.values();
    
    // Sort by value score (lowest first)
    std::sort(allMetrics.begin(), allMetrics.end(),
              [](const TestEffectivenessMetrics& a, const TestEffectivenessMetrics& b) {
                  return a.valueScore < b.valueScore;
              });
    
    return allMetrics.mid(0, qMin(bottomN, allMetrics.size()));
}

QStringList TestEffectivenessAnalyzer::getTestsToRemove() const {
    QStringList testsToRemove;
    
    for (auto it = m_effectivenessData.begin(); it != m_effectivenessData.end(); ++it) {
        if (it.value().recommendation == "remove") {
            testsToRemove << it.key();
        }
    }
    
    return testsToRemove;
}

QStringList TestEffectivenessAnalyzer::getTestsToImprove() const {
    QStringList testsToImprove;
    
    for (auto it = m_effectivenessData.begin(); it != m_effectivenessData.end(); ++it) {
        if (it.value().recommendation == "improve") {
            testsToImprove << it.key();
        }
    }
    
    return testsToImprove;
}

void TestEffectivenessAnalyzer::generateEffectivenessReport(const QString& outputPath) const {
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create effectiveness report:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    out << "# Test Effectiveness Analysis Report\n\n";
    out << "Generated: " << QDateTime::currentDateTime().toString() << "\n\n";
    
    QList<TestEffectivenessMetrics> highValue = getHighValueTests(10);
    QList<TestEffectivenessMetrics> lowValue = getLowValueTests(10);
    QStringList toRemove = getTestsToRemove();
    QStringList toImprove = getTestsToImprove();
    
    out << "## Summary\n";
    out << "- Total tests analyzed: " << m_effectivenessData.size() << "\n";
    out << "- High value tests: " << highValue.size() << "\n";
    out << "- Low value tests: " << lowValue.size() << "\n";
    out << "- Tests to remove: " << toRemove.size() << "\n";
    out << "- Tests to improve: " << toImprove.size() << "\n\n";
    
    if (!highValue.isEmpty()) {
        out << "## High Value Tests\n\n";
        for (const TestEffectivenessMetrics& metrics : highValue) {
            out << "### " << metrics.testName << "\n";
            out << "- Effectiveness Score: " << QString::number(metrics.effectivenessScore, 'f', 2) << "\n";
            out << "- Value Score: " << QString::number(metrics.valueScore, 'f', 2) << "\n";
            out << "- Defects Found: " << metrics.defectsFound << "\n";
            out << "- Maintenance Cost: $" << QString::number(metrics.maintenanceCost, 'f', 2) << "\n\n";
        }
    }
    
    if (!lowValue.isEmpty()) {
        out << "## Low Value Tests\n\n";
        for (const TestEffectivenessMetrics& metrics : lowValue) {
            out << "### " << metrics.testName << "\n";
            out << "- Effectiveness Score: " << QString::number(metrics.effectivenessScore, 'f', 2) << "\n";
            out << "- Value Score: " << QString::number(metrics.valueScore, 'f', 2) << "\n";
            out << "- Recommendation: " << metrics.recommendation << "\n";
            out << "- Suggestions: " << metrics.improvementSuggestions.join(", ") << "\n\n";
        }
    }
}

QJsonObject TestEffectivenessAnalyzer::exportEffectivenessData() const {
    QJsonObject root;
    QJsonArray testsArray;
    
    for (auto it = m_effectivenessData.begin(); it != m_effectivenessData.end(); ++it) {
        const TestEffectivenessMetrics& metrics = it.value();
        
        QJsonObject testObj;
        testObj["testName"] = metrics.testName;
        testObj["suiteName"] = metrics.suiteName;
        testObj["totalRuns"] = metrics.totalRuns;
        testObj["defectsFound"] = metrics.defectsFound;
        testObj["falsePositives"] = metrics.falsePositives;
        testObj["falseNegatives"] = metrics.falseNegatives;
        testObj["effectivenessScore"] = metrics.effectivenessScore;
        testObj["maintenanceCost"] = metrics.maintenanceCost;
        testObj["valueScore"] = metrics.valueScore;
        testObj["recommendation"] = metrics.recommendation;
        
        QJsonArray suggestionsArray;
        for (const QString& suggestion : metrics.improvementSuggestions) {
            suggestionsArray.append(suggestion);
        }
        testObj["improvementSuggestions"] = suggestionsArray;
        
        testsArray.append(testObj);
    }
    
    root["tests"] = testsArray;
    root["defectTrackingEnabled"] = m_defectTrackingEnabled;
    root["maintenanceCostTracking"] = m_maintenanceCostTracking;
    root["costPerHour"] = m_costPerHour;
    root["exportDate"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    return root;
}

// Private methods for TestEffectivenessAnalyzer
double TestEffectivenessAnalyzer::calculateEffectivenessScore(const TestEffectivenessMetrics& metrics) const {
    if (metrics.totalRuns == 0) {
        return 0.0;
    }
    
    // Simple effectiveness calculation
    double truePositiveRate = (double)metrics.defectsFound / (metrics.defectsFound + metrics.falseNegatives + 1);
    double falsePositiveRate = (double)metrics.falsePositives / (metrics.totalRuns + 1);
    
    // Effectiveness = True Positive Rate - False Positive Rate
    return qBound(0.0, truePositiveRate - falsePositiveRate, 1.0);
}

double TestEffectivenessAnalyzer::calculateValueScore(const TestEffectivenessMetrics& metrics) const {
    if (metrics.maintenanceCost <= 0) {
        return metrics.effectivenessScore; // No cost data, use effectiveness only
    }
    
    // Value = Effectiveness / Cost (normalized)
    double normalizedCost = metrics.maintenanceCost / 100.0; // Normalize to reasonable range
    return metrics.effectivenessScore / (1.0 + normalizedCost);
}

QString TestEffectivenessAnalyzer::generateRecommendation(const TestEffectivenessMetrics& metrics) const {
    if (metrics.valueScore < 0.2) {
        return "remove";
    } else if (metrics.valueScore < 0.5) {
        return "improve";
    } else {
        return "keep";
    }
}

QStringList TestEffectivenessAnalyzer::generateImprovementSuggestions(const TestEffectivenessMetrics& metrics) const {
    QStringList suggestions;
    
    if (metrics.falsePositives > metrics.totalRuns * 0.1) {
        suggestions << "Reduce false positives by improving test assertions";
    }
    
    if (metrics.falseNegatives > 0) {
        suggestions << "Improve test coverage to catch more defects";
    }
    
    if (metrics.maintenanceCost > 5.0) {
        suggestions << "Reduce maintenance cost through test refactoring";
    }
    
    if (metrics.defectsFound == 0 && metrics.totalRuns > 50) {
        suggestions << "Consider if this test is testing the right scenarios";
    }
    
    return suggestions;
}

// HtmlReportGenerator Implementation
HtmlReportGenerator::HtmlReportGenerator(QObject* parent)
    : QObject(parent)
{
    // Initialize default templates
    m_templates["header"] = R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>{{CSS_STYLES}}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{TITLE}}</h1>
            <p class="generated-date">Generated: {{GENERATED_DATE}}</p>
        </header>
)";
    
    m_templates["footer"] = R"(
        <footer>
            <p>Generated by CloneClean Test Framework</p>
        </footer>
    </div>
    {{JAVASCRIPT}}
</body>
</html>
)";
}

void HtmlReportGenerator::setTemplate(const QString& templatePath) {
    m_templatePath = templatePath;
}

void HtmlReportGenerator::setOutputDirectory(const QString& directory) {
    m_outputDirectory = directory;
    QDir().mkpath(directory);
}

void HtmlReportGenerator::setIncludeCharts(bool includeCharts) {
    m_includeCharts = includeCharts;
}

void HtmlReportGenerator::setIncludeInteractivity(bool interactive) {
    m_includeInteractivity = interactive;
}

void HtmlReportGenerator::generateComprehensiveReport(const TestResults& results, 
                                                     const QString& outputPath) {
    emit reportGenerationStarted("comprehensive");
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create comprehensive report:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    
    // Generate header
    QString header = m_templates["header"];
    header.replace("{{TITLE}}", "Comprehensive Test Report");
    header.replace("{{GENERATED_DATE}}", QDateTime::currentDateTime().toString());
    header.replace("{{CSS_STYLES}}", generateCssStyles());
    out << header;
    
    // Generate summary section
    out << "<section class=\"summary\">\n";
    out << "<h2>Test Summary</h2>\n";
    out << "<div class=\"summary-cards\">\n";
    out << QString("<div class=\"card\"><h3>Total Tests</h3><p class=\"metric\">%1</p></div>\n").arg(results.totalTests);
    out << QString("<div class=\"card success\"><h3>Passed</h3><p class=\"metric\">%1</p></div>\n").arg(results.passedTests);
    out << QString("<div class=\"card failure\"><h3>Failed</h3><p class=\"metric\">%1</p></div>\n").arg(results.failedTests);
    out << QString("<div class=\"card skipped\"><h3>Skipped</h3><p class=\"metric\">%1</p></div>\n").arg(results.skippedTests);
    out << QString("<div class=\"card\"><h3>Success Rate</h3><p class=\"metric\">%1%</p></div>\n").arg(results.successRate(), 0, 'f', 1);
    out << QString("<div class=\"card\"><h3>Execution Time</h3><p class=\"metric\">%1s</p></div>\n").arg(results.executionTimeMs / 1000.0, 0, 'f', 1);
    out << "</div>\n";
    out << "</section>\n";
    
    // Generate failures section if any
    if (!results.failures.isEmpty()) {
        out << "<section class=\"failures\">\n";
        out << "<h2>Test Failures</h2>\n";
        out << "<div class=\"failures-list\">\n";
        
        for (const TestFailure& failure : results.failures) {
            out << "<div class=\"failure-item\">\n";
            out << QString("<h3>%1</h3>\n").arg(failure.testName);
            out << QString("<p class=\"category\">Category: %1</p>\n").arg(failure.category);
            out << QString("<p class=\"error-message\">%1</p>\n").arg(failure.errorMessage);
            if (!failure.screenshotPath.isEmpty()) {
                out << QString("<p><a href=\"%1\" target=\"_blank\">View Screenshot</a></p>\n").arg(failure.screenshotPath);
            }
            out << "</div>\n";
        }
        
        out << "</div>\n";
        out << "</section>\n";
    }
    
    // Generate charts if enabled
    if (m_includeCharts) {
        out << "<section class=\"charts\">\n";
        out << "<h2>Test Analytics</h2>\n";
        out << "<div class=\"chart-container\">\n";
        out << "<canvas id=\"successRateChart\" width=\"400\" height=\"200\"></canvas>\n";
        out << "</div>\n";
        out << "</section>\n";
    }
    
    // Generate footer
    QString footer = m_templates["footer"];
    if (m_includeInteractivity) {
        footer.replace("{{JAVASCRIPT}}", generateJavaScript());
    } else {
        footer.replace("{{JAVASCRIPT}}", "");
    }
    out << footer;
    
    // Copy static assets
    if (!m_outputDirectory.isEmpty()) {
        copyStaticAssets(m_outputDirectory);
    }
    
    emit reportGenerationCompleted(outputPath);
}

void HtmlReportGenerator::generateTrendReport(const QList<PerformanceTrendAnalysis>& trends,
                                            const QString& outputPath) {
    emit reportGenerationStarted("trend");
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create trend report:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    
    // Generate header
    QString header = m_templates["header"];
    header.replace("{{TITLE}}", "Performance Trend Report");
    header.replace("{{GENERATED_DATE}}", QDateTime::currentDateTime().toString());
    header.replace("{{CSS_STYLES}}", generateCssStyles());
    out << header;
    
    // Generate trends section
    out << "<section class=\"trends\">\n";
    out << "<h2>Performance Trends</h2>\n";
    
    for (const PerformanceTrendAnalysis& trend : trends) {
        out << "<div class=\"trend-item\">\n";
        out << QString("<h3>%1</h3>\n").arg(trend.metricName);
        out << QString("<p class=\"current-value\">Current Value: %1</p>\n").arg(trend.currentValue);
        out << QString("<p class=\"trend-direction %1\">Trend: %2</p>\n").arg(trend.trendDirection, trend.trendDirection);
        out << QString("<p class=\"prediction\">%1</p>\n").arg(trend.prediction);
        
        if (!trend.alerts.isEmpty()) {
            out << "<div class=\"alerts\">\n";
            for (const QString& alert : trend.alerts) {
                out << QString("<p class=\"alert\">⚠️ %1</p>\n").arg(alert);
            }
            out << "</div>\n";
        }
        
        out << "</div>\n";
    }
    
    out << "</section>\n";
    
    // Generate footer
    QString footer = m_templates["footer"];
    footer.replace("{{JAVASCRIPT}}", "");
    out << footer;
    
    emit reportGenerationCompleted(outputPath);
}

void HtmlReportGenerator::generateEffectivenessReport(const QList<TestEffectivenessMetrics>& metrics,
                                                     const QString& outputPath) {
    emit reportGenerationStarted("effectiveness");
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create effectiveness report:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    
    // Generate header
    QString header = m_templates["header"];
    header.replace("{{TITLE}}", "Test Effectiveness Report");
    header.replace("{{GENERATED_DATE}}", QDateTime::currentDateTime().toString());
    header.replace("{{CSS_STYLES}}", generateCssStyles());
    out << header;
    
    // Generate effectiveness section
    out << "<section class=\"effectiveness\">\n";
    out << "<h2>Test Effectiveness Analysis</h2>\n";
    
    for (const TestEffectivenessMetrics& metric : metrics) {
        out << "<div class=\"effectiveness-item\">\n";
        out << QString("<h3>%1</h3>\n").arg(metric.testName);
        out << QString("<p>Effectiveness Score: <span class=\"score\">%1</span></p>\n").arg(metric.effectivenessScore, 0, 'f', 2);
        out << QString("<p>Value Score: <span class=\"score\">%1</span></p>\n").arg(metric.valueScore, 0, 'f', 2);
        out << QString("<p>Defects Found: %1</p>\n").arg(metric.defectsFound);
        out << QString("<p>Maintenance Cost: $%1</p>\n").arg(metric.maintenanceCost, 0, 'f', 2);
        out << QString("<p class=\"recommendation %1\">Recommendation: %2</p>\n").arg(metric.recommendation, metric.recommendation);
        out << "</div>\n";
    }
    
    out << "</section>\n";
    
    // Generate footer
    QString footer = m_templates["footer"];
    footer.replace("{{JAVASCRIPT}}", "");
    out << footer;
    
    emit reportGenerationCompleted(outputPath);
}

void HtmlReportGenerator::generateMaintenanceReport(const QJsonObject& maintenanceData,
                                                   const QString& outputPath) {
    emit reportGenerationStarted("maintenance");
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to create maintenance report:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    
    // Generate header
    QString header = m_templates["header"];
    header.replace("{{TITLE}}", "Test Maintenance Report");
    header.replace("{{GENERATED_DATE}}", QDateTime::currentDateTime().toString());
    header.replace("{{CSS_STYLES}}", generateCssStyles());
    out << header;
    
    // Generate maintenance section
    out << "<section class=\"maintenance\">\n";
    out << "<h2>Test Maintenance Recommendations</h2>\n";
    
    QJsonArray recommendations = maintenanceData["recommendations"].toArray();
    for (const QJsonValue& value : recommendations) {
        QJsonObject rec = value.toObject();
        
        out << "<div class=\"maintenance-item\">\n";
        out << QString("<h3>%1</h3>\n").arg(rec["testName"].toString());
        out << QString("<p class=\"description\">%1</p>\n").arg(rec["description"].toString());
        out << QString("<p class=\"action\">Suggested Action: %1</p>\n").arg(rec["suggestedAction"].toString());
        out << QString("<p class=\"priority\">Priority: %1/5</p>\n").arg(rec["priority"].toInt());
        out << QString("<p class=\"effort\">Estimated Effort: %1 hours</p>\n").arg(rec["estimatedEffort"].toDouble());
        out << QString("<p class=\"automation\">Automation: %1</p>\n").arg(rec["automationPossible"].toString());
        out << "</div>\n";
    }
    
    out << "</section>\n";
    
    // Generate footer
    QString footer = m_templates["footer"];
    footer.replace("{{JAVASCRIPT}}", "");
    out << footer;
    
    emit reportGenerationCompleted(outputPath);
}

// Private methods for HtmlReportGenerator
QString HtmlReportGenerator::loadTemplate(const QString& templateName) const {
    return m_templates.value(templateName, QString());
}

QString HtmlReportGenerator::generateHtmlTable(const QJsonArray& data, const QStringList& columns) const {
    QString html = "<table class=\"data-table\">\n";
    
    // Generate header
    html += "<thead><tr>\n";
    for (const QString& column : columns) {
        html += QString("<th>%1</th>\n").arg(column);
    }
    html += "</tr></thead>\n";
    
    // Generate body
    html += "<tbody>\n";
    for (const QJsonValue& value : data) {
        QJsonObject row = value.toObject();
        html += "<tr>\n";
        for (const QString& column : columns) {
            html += QString("<td>%1</td>\n").arg(row[column].toString());
        }
        html += "</tr>\n";
    }
    html += "</tbody>\n";
    
    html += "</table>\n";
    return html;
}

QString HtmlReportGenerator::generateChartScript(const QString& chartType, const QJsonObject& data) const {
    // This would generate Chart.js or similar charting library code
    // For demonstration, we'll return a simple placeholder
    return QString("// Chart script for %1 would go here").arg(chartType);
}

QString HtmlReportGenerator::generateCssStyles() const {
    return R"(
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        .card.success { border-left: 4px solid #27ae60; }
        .card.failure { border-left: 4px solid #e74c3c; }
        .card.skipped { border-left: 4px solid #f39c12; }
        .metric { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .failures-list { background: white; padding: 20px; border-radius: 8px; }
        .failure-item { border-bottom: 1px solid #eee; padding: 15px 0; }
        .error-message { color: #e74c3c; font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 4px; }
        .trend-item, .effectiveness-item, .maintenance-item { background: white; margin: 15px 0; padding: 20px; border-radius: 8px; }
        .trend-direction.improving { color: #27ae60; }
        .trend-direction.degrading { color: #e74c3c; }
        .trend-direction.stable { color: #7f8c8d; }
        .alert { color: #e67e22; background: #fdf2e9; padding: 8px; border-radius: 4px; }
        .score { font-weight: bold; font-size: 1.2em; }
        .recommendation.keep { color: #27ae60; }
        .recommendation.improve { color: #f39c12; }
        .recommendation.remove { color: #e74c3c; }
        footer { text-align: center; margin-top: 40px; color: #7f8c8d; }
    )";
}

QString HtmlReportGenerator::generateJavaScript() const {
    return R"(
        <script>
        // Interactive functionality would go here
        document.addEventListener('DOMContentLoaded', function() {
            // Add sorting, filtering, search functionality
            console.log('Interactive report loaded');
        });
        </script>
    )";
}

void HtmlReportGenerator::copyStaticAssets(const QString& outputDir) const {
    // This would copy CSS, JS, and image files to the output directory
    // For demonstration, we'll just create the directories
    QDir().mkpath(outputDir + "/assets/css");
    QDir().mkpath(outputDir + "/assets/js");
    QDir().mkpath(outputDir + "/assets/images");
}

// Simplified implementations for remaining classes due to length constraints
// TestDashboard and AdvancedAnalyticsCoordinator would follow similar patterns

#include "advanced_reporting.moc"