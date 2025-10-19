#pragma once

#include "test_harness.h"
#include <QObject>
#include <QString>
#include <QDateTime>
#include <QMap>
#include <QTextStream>
#include <QJsonObject>
#include <QJsonDocument>
#include <memory>

/**
 * @brief Report format types
 */
enum class ReportFormat {
    HTML,
    JSON,
    XML,
    JUnit,
    Console,
    CSV
};

/**
 * @brief Report configuration options
 */
struct ReportConfig {
    QList<ReportFormat> formats = {ReportFormat::HTML, ReportFormat::JUnit};
    QString outputDirectory = "test_reports";
    QString reportTitle = "DupFinder Test Results";
    bool includeScreenshots = true;
    bool includeMetrics = true;
    bool includeCodeCoverage = true;
    bool generateTrendAnalysis = false;
    QString baselineReportPath; // For comparison
    QString customCssPath; // For HTML reports
    QString customTemplatePath; // For custom report templates
};

/**
 * @brief Performance trend data point
 */
struct TrendDataPoint {
    QDateTime timestamp;
    QString testName;
    qint64 executionTimeMs;
    qint64 memoryUsageMB;
    double cpuUsagePercent;
    QMap<QString, QVariant> customMetrics;
};

/**
 * @brief Comprehensive test reporting system
 * 
 * Generates detailed test reports in multiple formats including:
 * - HTML reports with interactive features
 * - JUnit XML for CI/CD integration
 * - JSON for programmatic analysis
 * - Console output for immediate feedback
 */
class TestReporting : public QObject {
    Q_OBJECT

public:
    explicit TestReporting(QObject* parent = nullptr);
    ~TestReporting() = default;

    // Configuration
    void setReportConfig(const ReportConfig& config);
    ReportConfig getReportConfig() const { return m_config; }

    // Report generation
    bool generateReport(const TestResults& overallResults, 
                       const QMap<QString, TestResults>& suiteResults,
                       const QString& outputPath = QString());
    
    bool generateHTMLReport(const TestResults& overallResults,
                           const QMap<QString, TestResults>& suiteResults,
                           const QString& filePath);
    
    bool generateJUnitReport(const TestResults& overallResults,
                            const QMap<QString, TestResults>& suiteResults,
                            const QString& filePath);
    
    bool generateJSONReport(const TestResults& overallResults,
                           const QMap<QString, TestResults>& suiteResults,
                           const QString& filePath);
    
    bool generateConsoleReport(const TestResults& overallResults,
                              const QMap<QString, TestResults>& suiteResults);

    // Trend analysis
    void addTrendData(const QString& testName, const TrendDataPoint& dataPoint);
    void loadTrendData(const QString& filePath);
    void saveTrendData(const QString& filePath);
    QList<TrendDataPoint> getTrendData(const QString& testName) const;
    
    // Report comparison
    bool compareWithBaseline(const TestResults& currentResults, const QString& baselineReportPath);
    QJsonObject generateComparisonReport(const TestResults& current, const TestResults& baseline);

    // Utilities
    static QString formatDuration(qint64 milliseconds);
    static QString formatBytes(qint64 bytes);
    static QString formatPercentage(double percentage);
    static QString escapeHtml(const QString& text);

signals:
    void reportGenerated(const QString& filePath, ReportFormat format);
    void reportGenerationFailed(const QString& error);

private:
    // HTML report generation helpers
    QString generateHTMLHeader(const QString& title);
    QString generateHTMLSummary(const TestResults& results);
    QString generateHTMLSuiteDetails(const QMap<QString, TestResults>& suiteResults);
    QString generateHTMLFailureDetails(const QList<TestFailure>& failures);
    QString generateHTMLMetrics(const QMap<QString, QVariant>& metrics);
    QString generateHTMLTrendCharts();
    QString generateHTMLFooter();

    // JUnit XML helpers
    QString generateJUnitTestSuite(const QString& suiteName, const TestResults& results);
    QString generateJUnitTestCase(const QString& testName, bool passed, const QString& error = QString());

    // JSON helpers
    QJsonObject testResultsToJson(const TestResults& results);
    QJsonObject testFailureToJson(const TestFailure& failure);
    QJsonObject testWarningToJson(const TestWarning& warning);

    // Trend analysis helpers
    void generateTrendChart(const QString& testName, QTextStream& html);
    QString generatePerformanceTrendData(const QString& testName);

    // File operations
    bool writeToFile(const QString& filePath, const QString& content);
    QString readTemplate(const QString& templatePath);
    QString getOutputFilePath(const QString& baseName, ReportFormat format);

    // Configuration and state
    ReportConfig m_config;
    QMap<QString, QList<TrendDataPoint>> m_trendData;
    QDateTime m_reportGenerationTime;
};

/**
 * @brief Console report formatter for immediate feedback
 */
class ConsoleReporter {
public:
    static void printSummary(const TestResults& results);
    static void printSuiteResults(const QString& suiteName, const TestResults& results);
    static void printFailures(const QList<TestFailure>& failures);
    static void printWarnings(const QList<TestWarning>& warnings);
    static void printMetrics(const QMap<QString, QVariant>& metrics);
    static void printProgressBar(int current, int total, const QString& label = QString());

private:
    static QString colorize(const QString& text, const QString& color);
    static QString formatTestResult(bool passed);
};

/**
 * @brief Report template system for customizable reports
 */
class ReportTemplate {
public:
    explicit ReportTemplate(const QString& templatePath);
    
    void setVariable(const QString& name, const QString& value);
    void setVariable(const QString& name, const QJsonObject& value);
    QString render();
    
    static QString getDefaultHTMLTemplate();
    static QString getDefaultJUnitTemplate();

private:
    QString m_template;
    QMap<QString, QString> m_variables;
    QMap<QString, QJsonObject> m_jsonVariables;
    
    QString processTemplate(const QString& content);
    QString processVariable(const QString& varName);
};

/**
 * @brief Performance metrics collector for trend analysis
 */
class PerformanceMetrics {
public:
    static void startCollection();
    static void stopCollection();
    static TrendDataPoint getCurrentMetrics(const QString& testName);
    
    static qint64 getCurrentMemoryUsage();
    static double getCurrentCpuUsage();
    static QMap<QString, QVariant> getSystemMetrics();

private:
    static bool s_collecting;
    static QDateTime s_startTime;
    static qint64 s_startMemory;
};

/**
 * @brief Utility macros for reporting
 */
#define REPORT_METRIC(reporter, name, value) \
    do { \
        if (reporter) { \
            reporter->addMetric(name, value); \
        } \
    } while(0)

#define REPORT_PERFORMANCE_START() \
    PerformanceMetrics::startCollection()

#define REPORT_PERFORMANCE_END(reporter, testName) \
    do { \
        auto metrics = PerformanceMetrics::getCurrentMetrics(testName); \
        if (reporter) { \
            reporter->addTrendData(testName, metrics); \
        } \
        PerformanceMetrics::stopCollection(); \
    } while(0)