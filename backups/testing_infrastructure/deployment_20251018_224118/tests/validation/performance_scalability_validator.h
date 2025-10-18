#ifndef PERFORMANCE_SCALABILITY_VALIDATOR_H
#define PERFORMANCE_SCALABILITY_VALIDATOR_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QRegularExpression>
#include <QRandomGenerator>

struct SystemInfo {
    QString cpuArchitecture;
    QString kernelType;
    QString kernelVersion;
    QString productType;
    QString productVersion;
    int cpuCores = 0;
    int idealThreadCount = 0;
    qint64 totalMemoryMB = 0;
    qint64 availableMemoryMB = 0;
};

struct PerformanceMetrics {
    qint64 executionTimeMs = 0;
    qint64 memoryUsageMB = 0;
    qint64 peakMemoryMB = 0;
    double cpuUsagePercent = 0.0;
    int filesProcessed = 0;
    double throughputFilesPerSec = 0.0;
};

struct LargeCodebaseTest {
    QString testName;
    bool success = false;
    bool meetsPerformanceRequirements = false;
    qint64 executionTimeMs = 0;
    int exitCode = 0;
    QString errorMessage;
    QDateTime startTime;
    QDateTime endTime;
    PerformanceMetrics performanceMetrics;
};

struct ParallelExecutionResult {
    int parallelLevel = 1;
    bool success = false;
    qint64 executionTimeMs = 0;
    int exitCode = 0;
    QString errorMessage;
    QDateTime startTime;
    QDateTime endTime;
    PerformanceMetrics performanceMetrics;
};

struct ParallelExecutionTest {
    QString testName;
    bool success = false;
    bool meetsEfficiencyRequirements = false;
    qint64 executionTimeMs = 0;
    QDateTime startTime;
    QDateTime endTime;
    QList<ParallelExecutionResult> parallelResults;
    double efficiency = 0.0;
    int optimalParallelLevel = 1;
    double resourceUtilization = 0.0;
};

struct CIPipelineStage {
    QString stageName;
    bool success = false;
    qint64 executionTimeMs = 0;
    QDateTime startTime;
    QDateTime endTime;
    qint64 artifactSize = 0;
    QString testResults;
};

struct CIPipelineTest {
    QString testName;
    bool success = false;
    bool meetsCIRequirements = false;
    qint64 executionTimeMs = 0;
    QDateTime startTime;
    QDateTime endTime;
    QList<CIPipelineStage> pipelineStages;
};

struct TestResultScalabilityResult {
    int dataSize = 0;
    bool success = false;
    bool storageSuccess = false;
    bool retrievalSuccess = false;
    qint64 storageTimeMs = 0;
    qint64 retrievalTimeMs = 0;
    qint64 totalTimeMs = 0;
    double storedSizeMB = 0.0;
    QDateTime startTime;
    QDateTime endTime;
};

struct TestResultScalabilityTest {
    QString testName;
    bool success = false;
    bool meetsScalabilityRequirements = false;
    qint64 executionTimeMs = 0;
    QDateTime startTime;
    QDateTime endTime;
    QList<TestResultScalabilityResult> scalabilityResults;
    double storageScalability = 0.0;
    double retrievalScalability = 0.0;
    double memoryEfficiency = 0.0;
};

struct ScalabilityResults {
    QDateTime startTime;
    QDateTime endTime;
    qint64 totalExecutionTimeMs = 0;
    SystemInfo systemInfo;
    QList<LargeCodebaseTest> largeCodebaseTests;
    QList<ParallelExecutionTest> parallelExecutionTests;
    QList<CIPipelineTest> ciPipelineTests;
    QList<TestResultScalabilityTest> testResultScalabilityTests;
};

class PerformanceScalabilityValidator : public QObject
{
    Q_OBJECT

public:
    explicit PerformanceScalabilityValidator(QObject *parent = nullptr);

    // Main validation method
    bool validatePerformanceAndScalability();
    
    // Individual validation methods
    bool validateLargeCodebaseExecution(ScalabilityResults &results);
    bool validateParallelExecutionEfficiency(ScalabilityResults &results);
    bool validateCIPipelinePerformance(ScalabilityResults &results);
    bool validateTestResultScalability(ScalabilityResults &results);
    
    // System information
    SystemInfo collectSystemInfo();
    
    // Test data management
    bool createLargeTestDataset();
    QString findTestExecutable(const QString &executableName);
    
    // Performance analysis
    void parsePerformanceMetrics(const QString &output, PerformanceMetrics &metrics);
    bool validateLargeDatasetPerformance(const PerformanceMetrics &metrics);
    
    // Parallel execution analysis
    ParallelExecutionResult executeParallelTests(int parallelLevel);
    double calculateParallelEfficiency(const QList<ParallelExecutionResult> &results);
    int findOptimalParallelLevel(const QList<ParallelExecutionResult> &results);
    double calculateResourceUtilization(const QList<ParallelExecutionResult> &results);
    
    // CI pipeline simulation
    bool simulateCIPipelineExecution(CIPipelineTest &test);
    bool simulatePipelineStage(const QString &stageName, CIPipelineStage &stage);
    bool validateCIPerformanceRequirements(const CIPipelineTest &test);
    
    // Scalability testing
    TestResultScalabilityResult testResultStorageScalability(int dataSize);
    double calculateStorageScalability(const QList<TestResultScalabilityResult> &results);
    double calculateRetrievalScalability(const QList<TestResultScalabilityResult> &results);
    double calculateMemoryEfficiency(const QList<TestResultScalabilityResult> &results);
    
    // Reporting
    void generateScalabilityReport(const ScalabilityResults &results);
    void generateScalabilityJsonReport(const ScalabilityResults &results);
    void generateScalabilityHtmlReport(const ScalabilityResults &results);
    void generateScalabilityConsoleSummary(const ScalabilityResults &results);

private:
    int m_maxParallelTests;
};

#endif // PERFORMANCE_SCALABILITY_VALIDATOR_H