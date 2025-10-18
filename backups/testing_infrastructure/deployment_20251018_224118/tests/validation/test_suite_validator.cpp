#include "test_suite_validator.h"
#include <QCoreApplication>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <QDebug>
#include <QTimer>
#include <QElapsedTimer>
#include <QStandardPaths>

TestSuiteValidator::TestSuiteValidator(QObject *parent)
    : QObject(parent)
    , m_totalTests(0)
    , m_passedTests(0)
    , m_failedTests(0)
    , m_skippedTests(0)
    , m_executionTimeMs(0)
    , m_codeCoverage(0.0)
    , m_flakyTestRate(0.0)
{
}

bool TestSuiteValidator::validateComprehensiveTestSuite()
{
    qDebug() << "Starting comprehensive test suite validation...";
    
    ValidationResults results;
    results.startTime = QDateTime::currentDateTime();
    
    // Execute all test categories
    bool success = true;
    success &= executeTestCategory("unit", results);
    success &= executeTestCategory("integration", results);
    success &= executeTestCategory("performance", results);
    success &= executeTestCategory("ui", results);
    success &= executeTestCategory("end-to-end", results);
    success &= executeTestCategory("accessibility", results);
    success &= executeTestCategory("cross-platform", results);
    success &= executeTestCategory("security", results);
    
    results.endTime = QDateTime::currentDateTime();
    results.totalExecutionTimeMs = results.startTime.msecsTo(results.endTime);
    
    // Validate requirements
    success &= validateCoverageRequirement(results);
    success &= validateExecutionTimeRequirement(results);
    success &= validateReliabilityRequirement(results);
    success &= validatePlatformCoverage(results);
    
    // Generate comprehensive report
    generateValidationReport(results);
    
    return success;
}

bool TestSuiteValidator::executeTestCategory(const QString &category, ValidationResults &results)
{
    qDebug() << "Executing test category:" << category;
    
    QElapsedTimer timer;
    timer.start();
    
    CategoryResults categoryResult;
    categoryResult.category = category;
    categoryResult.startTime = QDateTime::currentDateTime();
    
    // Get test executables for this category
    QStringList testExecutables = getTestExecutablesForCategory(category);
    
    bool categorySuccess = true;
    for (const QString &executable : testExecutables) {
        TestExecutionResult execResult = executeTestExecutable(executable);
        categoryResult.testResults.append(execResult);
        
        if (!execResult.success) {
            categorySuccess = false;
        }
        
        // Update totals
        m_totalTests += execResult.totalTests;
        m_passedTests += execResult.passedTests;
        m_failedTests += execResult.failedTests;
        m_skippedTests += execResult.skippedTests;
    }
    
    categoryResult.endTime = QDateTime::currentDateTime();
    categoryResult.executionTimeMs = timer.elapsed();
    categoryResult.success = categorySuccess;
    
    results.categoryResults.append(categoryResult);
    
    qDebug() << "Category" << category << "completed in" << categoryResult.executionTimeMs << "ms";
    
    return categorySuccess;
}

QStringList TestSuiteValidator::getTestExecutablesForCategory(const QString &category)
{
    QStringList executables;
    
    if (category == "unit") {
        executables << "unit_tests"
                   << "test_file_scanner_coverage"
                   << "test_thumbnail_cache"
                   << "test_thumbnail_delegate"
                   << "test_exclude_pattern_widget"
                   << "test_preset_manager"
                   << "test_scan_configuration_validation"
                   << "test_scan_scope_preview_widget"
                   << "test_scan_progress_tracking"
                   << "test_scan_progress_dialog";
    }
    else if (category == "integration") {
        executables << "integration_tests"
                   << "test_integration_workflow"
                   << "test_filescanner_hashcalculator"
                   << "test_filescanner_duplicatedetector"
                   << "test_end_to_end_workflow"
                   << "test_scan_to_delete_workflow"
                   << "test_restore_functionality"
                   << "test_error_scenarios";
    }
    else if (category == "performance") {
        executables << "performance_tests"
                   << "test_file_scanner_performance"
                   << "test_hc002b_batch_processing"
                   << "test_hc002c_io_optimization"
                   << "example_performance_testing";
    }
    else if (category == "ui") {
        executables << "example_ui_automation"
                   << "example_visual_theme_testing";
    }
    else if (category == "end-to-end") {
        executables << "example_end_to_end_testing";
    }
    else if (category == "accessibility") {
        executables << "example_visual_theme_testing"; // Contains accessibility tests
    }
    else if (category == "cross-platform") {
        executables << "test_cross_platform_testing";
    }
    else if (category == "security") {
        executables << "example_security_safety_testing";
    }
    
    return executables;
}

TestExecutionResult TestSuiteValidator::executeTestExecutable(const QString &executable)
{
    TestExecutionResult result;
    result.executableName = executable;
    result.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer timer;
    timer.start();
    
    QProcess process;
    QString executablePath = findTestExecutable(executable);
    
    if (executablePath.isEmpty()) {
        result.success = false;
        result.errorMessage = QString("Executable not found: %1").arg(executable);
        result.executionTimeMs = timer.elapsed();
        return result;
    }
    
    // Set environment for headless execution
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("QT_QPA_PLATFORM", "offscreen");
    env.insert("DISPLAY", ":99"); // For Linux CI environments
    process.setProcessEnvironment(env);
    
    // Execute test with timeout
    process.start(executablePath, QStringList());
    
    if (!process.waitForStarted(5000)) {
        result.success = false;
        result.errorMessage = QString("Failed to start: %1").arg(executable);
        result.executionTimeMs = timer.elapsed();
        return result;
    }
    
    // Wait for completion with timeout (5 minutes per test)
    if (!process.waitForFinished(300000)) {
        process.kill();
        result.success = false;
        result.errorMessage = QString("Test timed out: %1").arg(executable);
        result.executionTimeMs = timer.elapsed();
        return result;
    }
    
    result.executionTimeMs = timer.elapsed();
    result.endTime = QDateTime::currentDateTime();
    result.exitCode = process.exitCode();
    result.success = (process.exitCode() == 0);
    
    // Parse output for test statistics
    QString output = process.readAllStandardOutput();
    QString errorOutput = process.readAllStandardError();
    
    parseTestOutput(output, result);
    
    if (!result.success) {
        result.errorMessage = errorOutput;
    }
    
    qDebug() << "Executed" << executable << "- Success:" << result.success 
             << "Time:" << result.executionTimeMs << "ms";
    
    return result;
}

QString TestSuiteValidator::findTestExecutable(const QString &executableName)
{
    // Check common build directories
    QStringList searchPaths = {
        QDir::currentPath() + "/build",
        QDir::currentPath() + "/build/tests",
        QDir::currentPath() + "/tests",
        QDir::currentPath()
    };
    
    for (const QString &path : searchPaths) {
        QString fullPath = path + "/" + executableName;
        
#ifdef Q_OS_WIN
        fullPath += ".exe";
#endif
        
        if (QFileInfo::exists(fullPath)) {
            return fullPath;
        }
    }
    
    return QString();
}

void TestSuiteValidator::parseTestOutput(const QString &output, TestExecutionResult &result)
{
    // Parse Qt Test Framework output
    QStringList lines = output.split('\n');
    
    for (const QString &line : lines) {
        if (line.contains("Totals:")) {
            // Parse totals line: "Totals: 5 passed, 0 failed, 0 skipped, 0 blacklisted"
            QRegularExpression re(R"((\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+skipped)");
            QRegularExpressionMatch match = re.match(line);
            
            if (match.hasMatch()) {
                result.passedTests = match.captured(1).toInt();
                result.failedTests = match.captured(2).toInt();
                result.skippedTests = match.captured(3).toInt();
                result.totalTests = result.passedTests + result.failedTests + result.skippedTests;
            }
        }
        else if (line.contains("PASS") || line.contains("FAIL") || line.contains("SKIP")) {
            // Individual test result
            TestCaseResult testCase;
            testCase.name = extractTestCaseName(line);
            testCase.success = line.contains("PASS");
            testCase.skipped = line.contains("SKIP");
            result.testCases.append(testCase);
        }
    }
    
    // If no totals found, count individual results
    if (result.totalTests == 0) {
        result.totalTests = result.testCases.size();
        result.passedTests = 0;
        result.failedTests = 0;
        result.skippedTests = 0;
        
        for (const TestCaseResult &testCase : result.testCases) {
            if (testCase.skipped) {
                result.skippedTests++;
            } else if (testCase.success) {
                result.passedTests++;
            } else {
                result.failedTests++;
            }
        }
    }
}

QString TestSuiteValidator::extractTestCaseName(const QString &line)
{
    // Extract test case name from Qt Test output
    QRegularExpression re(R"((PASS|FAIL|SKIP)\s+::\s*(\w+))");
    QRegularExpressionMatch match = re.match(line);
    
    if (match.hasMatch()) {
        return match.captured(2);
    }
    
    return "Unknown";
}

bool TestSuiteValidator::validateCoverageRequirement(ValidationResults &results)
{
    qDebug() << "Validating code coverage requirement (85%)...";
    
    // Run coverage analysis if available
    double coverage = calculateCodeCoverage();
    results.codeCoverage = coverage;
    m_codeCoverage = coverage;
    
    bool meetsRequirement = coverage >= 85.0;
    
    ValidationRequirement req;
    req.name = "Code Coverage";
    req.requirement = "Minimum 85% line coverage";
    req.actualValue = QString("%1%").arg(coverage, 0, 'f', 1);
    req.success = meetsRequirement;
    req.details = meetsRequirement ? "Coverage requirement met" : "Coverage below minimum threshold";
    
    results.requirements.append(req);
    
    qDebug() << "Code coverage:" << coverage << "% - Requirement met:" << meetsRequirement;
    
    return meetsRequirement;
}

bool TestSuiteValidator::validateExecutionTimeRequirement(ValidationResults &results)
{
    qDebug() << "Validating execution time requirement (<30 minutes)...";
    
    qint64 totalTimeMs = results.totalExecutionTimeMs;
    qint64 maxTimeMs = 30 * 60 * 1000; // 30 minutes in milliseconds
    
    bool meetsRequirement = totalTimeMs < maxTimeMs;
    
    ValidationRequirement req;
    req.name = "Execution Time";
    req.requirement = "Complete test suite under 30 minutes";
    req.actualValue = QString("%1 minutes").arg(totalTimeMs / 60000.0, 0, 'f', 1);
    req.success = meetsRequirement;
    req.details = meetsRequirement ? "Execution time requirement met" : "Execution time exceeds limit";
    
    results.requirements.append(req);
    
    qDebug() << "Total execution time:" << (totalTimeMs / 60000.0) << "minutes - Requirement met:" << meetsRequirement;
    
    return meetsRequirement;
}

bool TestSuiteValidator::validateReliabilityRequirement(ValidationResults &results)
{
    qDebug() << "Validating test reliability requirement (<2% flaky tests)...";
    
    // Calculate flaky test rate
    double flakyRate = calculateFlakyTestRate(results);
    results.flakyTestRate = flakyRate;
    m_flakyTestRate = flakyRate;
    
    bool meetsRequirement = flakyRate < 2.0;
    
    ValidationRequirement req;
    req.name = "Test Reliability";
    req.requirement = "Flaky test rate below 2%";
    req.actualValue = QString("%1%").arg(flakyRate, 0, 'f', 1);
    req.success = meetsRequirement;
    req.details = meetsRequirement ? "Reliability requirement met" : "Too many flaky tests detected";
    
    results.requirements.append(req);
    
    qDebug() << "Flaky test rate:" << flakyRate << "% - Requirement met:" << meetsRequirement;
    
    return meetsRequirement;
}

bool TestSuiteValidator::validatePlatformCoverage(ValidationResults &results)
{
    qDebug() << "Validating platform coverage requirement...";
    
    // Check if cross-platform tests were executed
    bool hasCrossPlatformTests = false;
    for (const CategoryResults &category : results.categoryResults) {
        if (category.category == "cross-platform" && category.success) {
            hasCrossPlatformTests = true;
            break;
        }
    }
    
    ValidationRequirement req;
    req.name = "Platform Coverage";
    req.requirement = "Support Windows, macOS, and Linux testing";
    req.actualValue = hasCrossPlatformTests ? "Cross-platform tests executed" : "No cross-platform tests found";
    req.success = hasCrossPlatformTests;
    req.details = hasCrossPlatformTests ? "Platform coverage validated" : "Cross-platform testing not available";
    
    results.requirements.append(req);
    
    qDebug() << "Platform coverage validated:" << hasCrossPlatformTests;
    
    return hasCrossPlatformTests;
}

double TestSuiteValidator::calculateCodeCoverage()
{
    // Try to run gcov or similar coverage tool
    QProcess process;
    
    // Check if lcov is available for coverage reporting
    process.start("lcov", QStringList() << "--version");
    if (process.waitForFinished(5000) && process.exitCode() == 0) {
        return calculateLcovCoverage();
    }
    
    // Check if gcov is available
    process.start("gcov", QStringList() << "--version");
    if (process.waitForFinished(5000) && process.exitCode() == 0) {
        return calculateGcovCoverage();
    }
    
    // Fallback: estimate coverage based on test execution
    return estimateCoverageFromTests();
}

double TestSuiteValidator::calculateLcovCoverage()
{
    // Run lcov to generate coverage report
    QProcess process;
    
    // Generate coverage data
    process.start("lcov", QStringList() 
                  << "--capture" 
                  << "--directory" << "."
                  << "--output-file" << "coverage.info");
    
    if (!process.waitForFinished(30000) || process.exitCode() != 0) {
        return 0.0;
    }
    
    // Generate HTML report and extract coverage percentage
    process.start("genhtml", QStringList()
                  << "coverage.info"
                  << "--output-directory" << "coverage_html");
    
    if (!process.waitForFinished(30000) || process.exitCode() != 0) {
        return 0.0;
    }
    
    // Parse coverage.info for line coverage percentage
    QFile coverageFile("coverage.info");
    if (coverageFile.open(QIODevice::ReadOnly)) {
        QTextStream stream(&coverageFile);
        QString content = stream.readAll();
        
        // Look for line coverage summary
        QRegularExpression re(R"(LF:(\d+).*?LH:(\d+))");
        QRegularExpressionMatch match = re.match(content);
        
        if (match.hasMatch()) {
            int totalLines = match.captured(1).toInt();
            int coveredLines = match.captured(2).toInt();
            
            if (totalLines > 0) {
                return (double(coveredLines) / double(totalLines)) * 100.0;
            }
        }
    }
    
    return 0.0;
}

double TestSuiteValidator::calculateGcovCoverage()
{
    // Simple gcov-based coverage calculation
    QProcess process;
    
    // Find .gcda files and run gcov
    QDir buildDir("build");
    QStringList gcovFiles = buildDir.entryList(QStringList() << "*.gcda", QDir::Files, QDir::Name);
    
    if (gcovFiles.isEmpty()) {
        return 0.0;
    }
    
    int totalLines = 0;
    int coveredLines = 0;
    
    for (const QString &gcovFile : gcovFiles) {
        process.start("gcov", QStringList() << buildDir.absoluteFilePath(gcovFile));
        
        if (process.waitForFinished(10000) && process.exitCode() == 0) {
            QString output = process.readAllStandardOutput();
            
            // Parse gcov output for coverage statistics
            QRegularExpression re(R"((\d+\.\d+)% of (\d+) lines)");
            QRegularExpressionMatch match = re.match(output);
            
            if (match.hasMatch()) {
                double percentage = match.captured(1).toDouble();
                int lines = match.captured(2).toInt();
                
                totalLines += lines;
                coveredLines += int(lines * percentage / 100.0);
            }
        }
    }
    
    if (totalLines > 0) {
        return (double(coveredLines) / double(totalLines)) * 100.0;
    }
    
    return 0.0;
}

double TestSuiteValidator::estimateCoverageFromTests()
{
    // Estimate coverage based on number of tests and their success rate
    if (m_totalTests == 0) {
        return 0.0;
    }
    
    double successRate = double(m_passedTests) / double(m_totalTests);
    
    // Rough estimation: assume good test coverage correlates with test success
    // This is a fallback when no coverage tools are available
    return successRate * 80.0; // Conservative estimate
}

double TestSuiteValidator::calculateFlakyTestRate(const ValidationResults &results)
{
    // For now, assume tests are not flaky if they pass consistently
    // In a real implementation, this would track test results over multiple runs
    
    if (m_totalTests == 0) {
        return 0.0;
    }
    
    // Simple heuristic: failed tests might indicate flakiness
    // In practice, you'd need historical data to detect true flakiness
    return (double(m_failedTests) / double(m_totalTests)) * 100.0;
}

void TestSuiteValidator::generateValidationReport(const ValidationResults &results)
{
    qDebug() << "Generating comprehensive validation report...";
    
    // Generate JSON report
    generateJsonReport(results);
    
    // Generate HTML report
    generateHtmlReport(results);
    
    // Generate console summary
    generateConsoleSummary(results);
}

void TestSuiteValidator::generateJsonReport(const ValidationResults &results)
{
    QJsonObject report;
    
    // Summary
    QJsonObject summary;
    summary["total_tests"] = m_totalTests;
    summary["passed_tests"] = m_passedTests;
    summary["failed_tests"] = m_failedTests;
    summary["skipped_tests"] = m_skippedTests;
    summary["execution_time_ms"] = results.totalExecutionTimeMs;
    summary["code_coverage"] = results.codeCoverage;
    summary["flaky_test_rate"] = results.flakyTestRate;
    summary["start_time"] = results.startTime.toString(Qt::ISODate);
    summary["end_time"] = results.endTime.toString(Qt::ISODate);
    
    report["summary"] = summary;
    
    // Requirements validation
    QJsonArray requirements;
    for (const ValidationRequirement &req : results.requirements) {
        QJsonObject reqObj;
        reqObj["name"] = req.name;
        reqObj["requirement"] = req.requirement;
        reqObj["actual_value"] = req.actualValue;
        reqObj["success"] = req.success;
        reqObj["details"] = req.details;
        requirements.append(reqObj);
    }
    report["requirements"] = requirements;
    
    // Category results
    QJsonArray categories;
    for (const CategoryResults &category : results.categoryResults) {
        QJsonObject catObj;
        catObj["category"] = category.category;
        catObj["success"] = category.success;
        catObj["execution_time_ms"] = category.executionTimeMs;
        catObj["start_time"] = category.startTime.toString(Qt::ISODate);
        catObj["end_time"] = category.endTime.toString(Qt::ISODate);
        
        QJsonArray testResults;
        for (const TestExecutionResult &testResult : category.testResults) {
            QJsonObject testObj;
            testObj["executable"] = testResult.executableName;
            testObj["success"] = testResult.success;
            testObj["total_tests"] = testResult.totalTests;
            testObj["passed_tests"] = testResult.passedTests;
            testObj["failed_tests"] = testResult.failedTests;
            testObj["skipped_tests"] = testResult.skippedTests;
            testObj["execution_time_ms"] = testResult.executionTimeMs;
            testObj["exit_code"] = testResult.exitCode;
            
            if (!testResult.errorMessage.isEmpty()) {
                testObj["error_message"] = testResult.errorMessage;
            }
            
            testResults.append(testObj);
        }
        catObj["test_results"] = testResults;
        
        categories.append(catObj);
    }
    report["categories"] = categories;
    
    // Write JSON report
    QJsonDocument doc(report);
    QFile jsonFile("test_validation_report.json");
    if (jsonFile.open(QIODevice::WriteOnly)) {
        jsonFile.write(doc.toJson());
        jsonFile.close();
        qDebug() << "JSON report written to test_validation_report.json";
    }
}

void TestSuiteValidator::generateHtmlReport(const ValidationResults &results)
{
    QString html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Test Suite Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .requirement { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .requirement.success { border-left-color: #4CAF50; background-color: #f1f8e9; }
        .requirement.failure { border-left-color: #f44336; background-color: #ffebee; }
        .category { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .test-result { margin: 5px 0; padding: 5px; background-color: #f9f9f9; }
        .success { color: #4CAF50; }
        .failure { color: #f44336; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Suite Validation Report</h1>
        <p>Generated: %1</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>%2</td></tr>
            <tr><td>Passed Tests</td><td class="success">%3</td></tr>
            <tr><td>Failed Tests</td><td class="failure">%4</td></tr>
            <tr><td>Skipped Tests</td><td>%5</td></tr>
            <tr><td>Execution Time</td><td>%6 minutes</td></tr>
            <tr><td>Code Coverage</td><td>%7%</td></tr>
            <tr><td>Flaky Test Rate</td><td>%8%</td></tr>
        </table>
    </div>
    
    <div class="requirements">
        <h2>Requirements Validation</h2>
        %9
    </div>
    
    <div class="categories">
        <h2>Test Categories</h2>
        %10
    </div>
</body>
</html>
)";
    
    // Fill in summary data
    html = html.arg(QDateTime::currentDateTime().toString())
              .arg(m_totalTests)
              .arg(m_passedTests)
              .arg(m_failedTests)
              .arg(m_skippedTests)
              .arg(results.totalExecutionTimeMs / 60000.0, 0, 'f', 1)
              .arg(results.codeCoverage, 0, 'f', 1)
              .arg(results.flakyTestRate, 0, 'f', 1);
    
    // Generate requirements section
    QString requirementsHtml;
    for (const ValidationRequirement &req : results.requirements) {
        QString reqClass = req.success ? "success" : "failure";
        requirementsHtml += QString(R"(
        <div class="requirement %1">
            <h3>%2</h3>
            <p><strong>Requirement:</strong> %3</p>
            <p><strong>Actual:</strong> %4</p>
            <p><strong>Status:</strong> %5</p>
            <p>%6</p>
        </div>
        )").arg(reqClass, req.name, req.requirement, req.actualValue,
                req.success ? "PASS" : "FAIL", req.details);
    }
    
    // Generate categories section
    QString categoriesHtml;
    for (const CategoryResults &category : results.categoryResults) {
        QString categoryClass = category.success ? "success" : "failure";
        categoriesHtml += QString(R"(
        <div class="category">
            <h3>%1 <span class="%2">(%3)</span></h3>
            <p>Execution Time: %4 ms</p>
        )").arg(category.category, categoryClass,
                category.success ? "PASS" : "FAIL",
                QString::number(category.executionTimeMs));
        
        for (const TestExecutionResult &testResult : category.testResults) {
            QString testClass = testResult.success ? "success" : "failure";
            categoriesHtml += QString(R"(
            <div class="test-result">
                <strong>%1</strong> <span class="%2">(%3)</span><br>
                Tests: %4 passed, %5 failed, %6 skipped<br>
                Time: %7 ms
            </div>
            )").arg(testResult.executableName, testClass,
                    testResult.success ? "PASS" : "FAIL")
                   .arg(testResult.passedTests)
                   .arg(testResult.failedTests)
                   .arg(testResult.skippedTests)
                   .arg(testResult.executionTimeMs);
        }
        
        categoriesHtml += "</div>";
    }
    
    html = html.arg(requirementsHtml).arg(categoriesHtml);
    
    // Write HTML report
    QFile htmlFile("test_validation_report.html");
    if (htmlFile.open(QIODevice::WriteOnly)) {
        htmlFile.write(html.toUtf8());
        htmlFile.close();
        qDebug() << "HTML report written to test_validation_report.html";
    }
}

void TestSuiteValidator::generateConsoleSummary(const ValidationResults &results)
{
    qDebug() << "\n" << QString(60, '=');
    qDebug() << "TEST SUITE VALIDATION SUMMARY";
    qDebug() << QString(60, '=');
    
    qDebug() << QString("Total Tests: %1").arg(m_totalTests);
    qDebug() << QString("Passed: %1").arg(m_passedTests);
    qDebug() << QString("Failed: %1").arg(m_failedTests);
    qDebug() << QString("Skipped: %1").arg(m_skippedTests);
    qDebug() << QString("Execution Time: %1 minutes").arg(results.totalExecutionTimeMs / 60000.0, 0, 'f', 1);
    qDebug() << QString("Code Coverage: %1%").arg(results.codeCoverage, 0, 'f', 1);
    qDebug() << QString("Flaky Test Rate: %1%").arg(results.flakyTestRate, 0, 'f', 1);
    
    qDebug() << "\nREQUIREMENTS VALIDATION:";
    for (const ValidationRequirement &req : results.requirements) {
        QString status = req.success ? "PASS" : "FAIL";
        qDebug() << QString("  %1: %2 (%3)").arg(req.name, status, req.actualValue);
    }
    
    qDebug() << "\nCATEGORY RESULTS:";
    for (const CategoryResults &category : results.categoryResults) {
        QString status = category.success ? "PASS" : "FAIL";
        qDebug() << QString("  %1: %2 (%3 ms)").arg(category.category, status).arg(category.executionTimeMs);
    }
    
    bool overallSuccess = true;
    for (const ValidationRequirement &req : results.requirements) {
        if (!req.success) {
            overallSuccess = false;
            break;
        }
    }
    
    qDebug() << QString(60, '=');
    qDebug() << QString("OVERALL RESULT: %1").arg(overallSuccess ? "PASS" : "FAIL");
    qDebug() << QString(60, '=') << "\n";
}