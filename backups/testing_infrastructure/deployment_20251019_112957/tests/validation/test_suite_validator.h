#ifndef TEST_SUITE_VALIDATOR_H
#define TEST_SUITE_VALIDATOR_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QRegularExpression>

struct TestCaseResult {
    QString name;
    bool success = false;
    bool skipped = false;
    QString errorMessage;
};

struct TestExecutionResult {
    QString executableName;
    bool success = false;
    int totalTests = 0;
    int passedTests = 0;
    int failedTests = 0;
    int skippedTests = 0;
    qint64 executionTimeMs = 0;
    int exitCode = 0;
    QString errorMessage;
    QDateTime startTime;
    QDateTime endTime;
    QList<TestCaseResult> testCases;
};

struct CategoryResults {
    QString category;
    bool success = false;
    qint64 executionTimeMs = 0;
    QDateTime startTime;
    QDateTime endTime;
    QList<TestExecutionResult> testResults;
};

struct ValidationRequirement {
    QString name;
    QString requirement;
    QString actualValue;
    bool success = false;
    QString details;
};

struct ValidationResults {
    QDateTime startTime;
    QDateTime endTime;
    qint64 totalExecutionTimeMs = 0;
    double codeCoverage = 0.0;
    double flakyTestRate = 0.0;
    QList<CategoryResults> categoryResults;
    QList<ValidationRequirement> requirements;
};

class TestSuiteValidator : public QObject
{
    Q_OBJECT

public:
    explicit TestSuiteValidator(QObject *parent = nullptr);

    // Main validation methods
    bool validateComprehensiveTestSuite();
    
    // Individual validation methods
    bool executeTestCategory(const QString &category, ValidationResults &results);
    bool validateCoverageRequirement(ValidationResults &results);
    bool validateExecutionTimeRequirement(ValidationResults &results);
    bool validateReliabilityRequirement(ValidationResults &results);
    bool validatePlatformCoverage(ValidationResults &results);
    
    // Test execution
    QStringList getTestExecutablesForCategory(const QString &category);
    TestExecutionResult executeTestExecutable(const QString &executable);
    QString findTestExecutable(const QString &executableName);
    
    // Output parsing
    void parseTestOutput(const QString &output, TestExecutionResult &result);
    QString extractTestCaseName(const QString &line);
    
    // Coverage analysis
    double calculateCodeCoverage();
    double calculateLcovCoverage();
    double calculateGcovCoverage();
    double estimateCoverageFromTests();
    
    // Reliability analysis
    double calculateFlakyTestRate(const ValidationResults &results);
    
    // Reporting
    void generateValidationReport(const ValidationResults &results);
    void generateJsonReport(const ValidationResults &results);
    void generateHtmlReport(const ValidationResults &results);
    void generateConsoleSummary(const ValidationResults &results);
    
    // Getters for test statistics
    int totalTests() const { return m_totalTests; }
    int passedTests() const { return m_passedTests; }
    int failedTests() const { return m_failedTests; }
    int skippedTests() const { return m_skippedTests; }
    qint64 executionTimeMs() const { return m_executionTimeMs; }
    double codeCoverage() const { return m_codeCoverage; }
    double flakyTestRate() const { return m_flakyTestRate; }

private:
    int m_totalTests;
    int m_passedTests;
    int m_failedTests;
    int m_skippedTests;
    qint64 m_executionTimeMs;
    double m_codeCoverage;
    double m_flakyTestRate;
};

#endif // TEST_SUITE_VALIDATOR_H