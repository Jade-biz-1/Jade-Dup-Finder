#include "test_reporting.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>
#include <QDebug>
#include <QProcess>
#include <QStandardPaths>
#include <QCoreApplication>

TestReporting::TestReporting(QObject* parent)
    : QObject(parent)
{
    m_reportGenerationTime = QDateTime::currentDateTime();
}

void TestReporting::setReportConfig(const ReportConfig& config) {
    m_config = config;
    
    // Ensure output directory exists
    QDir().mkpath(m_config.outputDirectory);
}

bool TestReporting::generateReport(const TestResults& overallResults, 
                                  const QMap<QString, TestResults>& suiteResults,
                                  const QString& outputPath) {
    
    QString reportDir = outputPath.isEmpty() ? m_config.outputDirectory : outputPath;
    QDir().mkpath(reportDir);
    
    bool allSuccess = true;
    
    // Generate reports in all configured formats
    for (ReportFormat format : m_config.formats) {
        QString filePath = getOutputFilePath(reportDir + "/test_report", format);
        bool success = false;
        
        switch (format) {
            case ReportFormat::HTML:
                success = generateHTMLReport(overallResults, suiteResults, filePath);
                break;
            case ReportFormat::JUnit:
                success = generateJUnitReport(overallResults, suiteResults, filePath);
                break;
            case ReportFormat::JSON:
                success = generateJSONReport(overallResults, suiteResults, filePath);
                break;
            case ReportFormat::Console:
                success = generateConsoleReport(overallResults, suiteResults);
                break;
            default:
                qWarning() << "Unsupported report format:" << static_cast<int>(format);
                continue;
        }
        
        if (success) {
            emit reportGenerated(filePath, format);
            qDebug() << "Generated report:" << filePath;
        } else {
            allSuccess = false;
            emit reportGenerationFailed(QString("Failed to generate %1 report").arg(static_cast<int>(format)));
        }
    }
    
    return allSuccess;
}

bool TestReporting::generateHTMLReport(const TestResults& overallResults,
                                      const QMap<QString, TestResults>& suiteResults,
                                      const QString& filePath) {
    
    QString html;
    QTextStream stream(&html);
    
    // Generate HTML content
    stream << generateHTMLHeader(m_config.reportTitle);
    stream << generateHTMLSummary(overallResults);
    stream << generateHTMLSuiteDetails(suiteResults);
    
    if (!overallResults.failures.isEmpty()) {
        stream << generateHTMLFailureDetails(overallResults.failures);
    }
    
    if (m_config.includeMetrics && !overallResults.metrics.isEmpty()) {
        stream << generateHTMLMetrics(overallResults.metrics);
    }
    
    if (m_config.generateTrendAnalysis) {
        stream << generateHTMLTrendCharts();
    }
    
    stream << generateHTMLFooter();
    
    return writeToFile(filePath, html);
}

bool TestReporting::generateJUnitReport(const TestResults& overallResults,
                                       const QMap<QString, TestResults>& suiteResults,
                                       const QString& filePath) {
    
    QString xml;
    QTextStream stream(&xml);
    
    stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    stream << "<testsuites name=\"DupFinder Tests\" ";
    stream << "tests=\"" << overallResults.totalTests << "\" ";
    stream << "failures=\"" << overallResults.failedTests << "\" ";
    stream << "errors=\"0\" ";
    stream << "time=\"" << (overallResults.executionTimeMs / 1000.0) << "\" ";
    stream << "timestamp=\"" << m_reportGenerationTime.toString(Qt::ISODate) << "\">\n";
    
    // Generate test suites
    for (auto it = suiteResults.begin(); it != suiteResults.end(); ++it) {
        stream << generateJUnitTestSuite(it.key(), it.value());
    }
    
    stream << "</testsuites>\n";
    
    return writeToFile(filePath, xml);
}

bool TestReporting::generateJSONReport(const TestResults& overallResults,
                                      const QMap<QString, TestResults>& suiteResults,
                                      const QString& filePath) {
    
    QJsonObject report;
    report["timestamp"] = m_reportGenerationTime.toString(Qt::ISODate);
    report["overallResults"] = testResultsToJson(overallResults);
    
    QJsonObject suites;
    for (auto it = suiteResults.begin(); it != suiteResults.end(); ++it) {
        suites[it.key()] = testResultsToJson(it.value());
    }
    report["suiteResults"] = suites;
    
    if (m_config.includeMetrics) {
        QJsonObject metrics;
        for (auto it = overallResults.metrics.begin(); it != overallResults.metrics.end(); ++it) {
            metrics[it.key()] = QJsonValue::fromVariant(it.value());
        }
        report["metrics"] = metrics;
    }
    
    QJsonDocument doc(report);
    return writeToFile(filePath, doc.toJson());
}

bool TestReporting::generateConsoleReport(const TestResults& overallResults,
                                         const QMap<QString, TestResults>& suiteResults) {
    
    ConsoleReporter::printSummary(overallResults);
    
    for (auto it = suiteResults.begin(); it != suiteResults.end(); ++it) {
        ConsoleReporter::printSuiteResults(it.key(), it.value());
    }
    
    if (!overallResults.failures.isEmpty()) {
        ConsoleReporter::printFailures(overallResults.failures);
    }
    
    if (!overallResults.warnings.isEmpty()) {
        ConsoleReporter::printWarnings(overallResults.warnings);
    }
    
    if (m_config.includeMetrics && !overallResults.metrics.isEmpty()) {
        ConsoleReporter::printMetrics(overallResults.metrics);
    }
    
    return true;
}

QString TestReporting::generateHTMLHeader(const QString& title) {
    QString header = R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%1</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 0.9em; }
        .success { color: #27ae60; }
        .failure { color: #e74c3c; }
        .warning { color: #f39c12; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #34495e; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .failure-details { background: #fdf2f2; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; }
        .stack-trace { background: #2c3e50; color: #ecf0f1; padding: 10px; font-family: monospace; font-size: 0.9em; overflow-x: auto; }
        .chart-container { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>%1</h1>
        <p>Generated on: %2</p>
)";
    
    return header.arg(title).arg(m_reportGenerationTime.toString());
}

QString TestReporting::generateHTMLSummary(const TestResults& results) {
    QString summary = R"(
        <h2>Test Summary</h2>
        <div class="summary">
            <div class="metric">
                <div class="metric-value">%1</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">%2</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value failure">%3</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value warning">%4</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric">
                <div class="metric-value">%5%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">%6</div>
                <div class="metric-label">Execution Time</div>
            </div>
        </div>
)";
    
    return summary.arg(results.totalTests)
                 .arg(results.passedTests)
                 .arg(results.failedTests)
                 .arg(results.skippedTests)
                 .arg(results.successRate(), 0, 'f', 1)
                 .arg(formatDuration(results.executionTimeMs));
}

QString TestReporting::generateHTMLSuiteDetails(const QMap<QString, TestResults>& suiteResults) {
    QString details = "<h2>Test Suite Details</h2>\n<table>\n";
    details += "<tr><th>Suite Name</th><th>Total</th><th>Passed</th><th>Failed</th><th>Skipped</th><th>Success Rate</th><th>Duration</th></tr>\n";
    
    for (auto it = suiteResults.begin(); it != suiteResults.end(); ++it) {
        const TestResults& result = it.value();
        QString rowClass = result.hasFailures() ? "failure" : "success";
        
        details += QString("<tr class=\"%1\">").arg(rowClass);
        details += QString("<td>%1</td>").arg(escapeHtml(it.key()));
        details += QString("<td>%1</td>").arg(result.totalTests);
        details += QString("<td>%1</td>").arg(result.passedTests);
        details += QString("<td>%1</td>").arg(result.failedTests);
        details += QString("<td>%1</td>").arg(result.skippedTests);
        details += QString("<td>%1%</td>").arg(result.successRate(), 0, 'f', 1);
        details += QString("<td>%1</td>").arg(formatDuration(result.executionTimeMs));
        details += "</tr>\n";
    }
    
    details += "</table>\n";
    return details;
}

QString TestReporting::generateHTMLFailureDetails(const QList<TestFailure>& failures) {
    if (failures.isEmpty()) {
        return QString();
    }
    
    QString details = "<h2>Failure Details</h2>\n";
    
    for (const TestFailure& failure : failures) {
        details += "<div class=\"failure-details\">\n";
        details += QString("<h3>%1 (%2)</h3>\n").arg(escapeHtml(failure.testName)).arg(escapeHtml(failure.category));
        details += QString("<p><strong>Error:</strong> %1</p>\n").arg(escapeHtml(failure.errorMessage));
        
        if (!failure.stackTrace.isEmpty()) {
            details += "<div class=\"stack-trace\">\n";
            details += escapeHtml(failure.stackTrace);
            details += "</div>\n";
        }
        
        if (!failure.screenshotPath.isEmpty() && QFile::exists(failure.screenshotPath)) {
            details += QString("<p><strong>Screenshot:</strong> <a href=\"%1\">View</a></p>\n").arg(failure.screenshotPath);
        }
        
        details += "</div>\n";
    }
    
    return details;
}

QString TestReporting::generateHTMLFooter() {
    return R"(
    </div>
</body>
</html>
)";
}

QString TestReporting::formatDuration(qint64 milliseconds) {
    if (milliseconds < 1000) {
        return QString("%1ms").arg(milliseconds);
    } else if (milliseconds < 60000) {
        return QString("%1s").arg(milliseconds / 1000.0, 0, 'f', 1);
    } else {
        int minutes = milliseconds / 60000;
        int seconds = (milliseconds % 60000) / 1000;
        return QString("%1m %2s").arg(minutes).arg(seconds);
    }
}

QString TestReporting::escapeHtml(const QString& text) {
    QString escaped = text;
    escaped.replace("&", "&amp;");
    escaped.replace("<", "&lt;");
    escaped.replace(">", "&gt;");
    escaped.replace("\"", "&quot;");
    escaped.replace("'", "&#39;");
    return escaped;
}

bool TestReporting::writeToFile(const QString& filePath, const QString& content) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to open file for writing:" << filePath << file.errorString();
        return false;
    }
    
    QTextStream stream(&file);
    stream << content;
    return true;
}

QString TestReporting::getOutputFilePath(const QString& baseName, ReportFormat format) {
    QString extension;
    switch (format) {
        case ReportFormat::HTML: extension = ".html"; break;
        case ReportFormat::JSON: extension = ".json"; break;
        case ReportFormat::XML:
        case ReportFormat::JUnit: extension = ".xml"; break;
        case ReportFormat::CSV: extension = ".csv"; break;
        case ReportFormat::Console: return QString(); // No file output
    }
    
    return baseName + extension;
}

QJsonObject TestReporting::testResultsToJson(const TestResults& results) {
    QJsonObject obj;
    obj["totalTests"] = results.totalTests;
    obj["passedTests"] = results.passedTests;
    obj["failedTests"] = results.failedTests;
    obj["skippedTests"] = results.skippedTests;
    obj["executionTimeMs"] = results.executionTimeMs;
    obj["codeCoverage"] = results.codeCoverage;
    obj["successRate"] = results.successRate();
    
    QJsonArray failures;
    for (const TestFailure& failure : results.failures) {
        failures.append(testFailureToJson(failure));
    }
    obj["failures"] = failures;
    
    QJsonArray warnings;
    for (const TestWarning& warning : results.warnings) {
        warnings.append(testWarningToJson(warning));
    }
    obj["warnings"] = warnings;
    
    return obj;
}

// ConsoleReporter implementation
void ConsoleReporter::printSummary(const TestResults& results) {
    qDebug() << "========================================";
    qDebug() << "           TEST SUMMARY";
    qDebug() << "========================================";
    qDebug() << "Total Tests:" << results.totalTests;
    qDebug() << colorize(QString("Passed: %1").arg(results.passedTests), "green");
    qDebug() << colorize(QString("Failed: %1").arg(results.failedTests), "red");
    qDebug() << colorize(QString("Skipped: %1").arg(results.skippedTests), "yellow");
    qDebug() << QString("Success Rate: %1%").arg(results.successRate(), 0, 'f', 1);
    qDebug() << QString("Execution Time: %1").arg(TestReporting::formatDuration(results.executionTimeMs));
    qDebug() << "========================================";
}

void ConsoleReporter::printSuiteResults(const QString& suiteName, const TestResults& results) {
    QString status = results.hasFailures() ? colorize("FAILED", "red") : colorize("PASSED", "green");
    qDebug() << QString("[%1] %2: %3/%4 tests passed (%5)")
                .arg(status)
                .arg(suiteName)
                .arg(results.passedTests)
                .arg(results.totalTests)
                .arg(TestReporting::formatDuration(results.executionTimeMs));
}

QString ConsoleReporter::colorize(const QString& text, const QString& color) {
    // ANSI color codes for console output
    QMap<QString, QString> colors = {
        {"red", "\033[31m"},
        {"green", "\033[32m"},
        {"yellow", "\033[33m"},
        {"blue", "\033[34m"},
        {"reset", "\033[0m"}
    };
    
    return colors.value(color, "") + text + colors.value("reset", "");
}

#include "test_reporting.moc"