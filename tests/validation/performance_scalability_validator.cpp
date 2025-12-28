#include "performance_scalability_validator.h"
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
#include <QThread>
#include <QThreadPool>
#include <QStandardPaths>
#include <QStorageInfo>
#include <QSysInfo>

PerformanceScalabilityValidator::PerformanceScalabilityValidator(QObject *parent)
    : QObject(parent)
    , m_maxParallelTests(QThread::idealThreadCount())
{
}

bool PerformanceScalabilityValidator::validatePerformanceAndScalability()
{
    qDebug() << "Starting performance and scalability validation...";
    
    ScalabilityResults results;
    results.startTime = QDateTime::currentDateTime();
    results.systemInfo = collectSystemInfo();
    
    bool success = true;
    
    // Test suite execution with large codebases
    success &= validateLargeCodebaseExecution(results);
    
    // Validate parallel execution efficiency
    success &= validateParallelExecutionEfficiency(results);
    
    // Confirm CI/CD pipeline integration performance
    success &= validateCIPipelinePerformance(results);
    
    // Verify test result storage and retrieval scalability
    success &= validateTestResultScalability(results);
    
    results.endTime = QDateTime::currentDateTime();
    results.totalExecutionTimeMs = results.startTime.msecsTo(results.endTime);
    
    // Generate comprehensive report
    generateScalabilityReport(results);
    
    return success;
}

bool PerformanceScalabilityValidator::validateLargeCodebaseExecution(ScalabilityResults &results)
{
    qDebug() << "Validating test suite execution with large codebases...";
    
    LargeCodebaseTest test;
    test.testName = "Large Codebase Execution";
    test.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer timer;
    timer.start();
    
    // Create a large test dataset
    bool datasetCreated = createLargeTestDataset();
    if (!datasetCreated) {
        test.success = false;
        test.errorMessage = "Failed to create large test dataset";
        test.executionTimeMs = timer.elapsed();
        results.largeCodebaseTests.append(test);
        return false;
    }
    
    // Execute test suite with large dataset
    QProcess process;
    QString testExecutable = findTestExecutable("performance_tests");
    
    if (testExecutable.isEmpty()) {
        test.success = false;
        test.errorMessage = "Performance test executable not found";
        test.executionTimeMs = timer.elapsed();
        results.largeCodebaseTests.append(test);
        return false;
    }
    
    // Set environment for large dataset testing
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("QT_QPA_PLATFORM", "offscreen");
    env.insert("CLONECLEAN_TEST_LARGE_DATASET", "1");
    env.insert("CLONECLEAN_TEST_DATASET_SIZE", "10000"); // 10k files
    process.setProcessEnvironment(env);
    
    // Execute with extended timeout for large dataset
    process.start(testExecutable, QStringList());
    
    if (!process.waitForStarted(10000)) {
        test.success = false;
        test.errorMessage = "Failed to start large codebase test";
        test.executionTimeMs = timer.elapsed();
        results.largeCodebaseTests.append(test);
        return false;
    }
    
    // Wait for completion with extended timeout (30 minutes)
    if (!process.waitForFinished(1800000)) {
        process.kill();
        test.success = false;
        test.errorMessage = "Large codebase test timed out";
        test.executionTimeMs = timer.elapsed();
        results.largeCodebaseTests.append(test);
        return false;
    }
    
    test.executionTimeMs = timer.elapsed();
    test.endTime = QDateTime::currentDateTime();
    test.exitCode = process.exitCode();
    test.success = (process.exitCode() == 0);
    
    // Parse performance metrics
    QString output = process.readAllStandardOutput();
    parsePerformanceMetrics(output, test.performanceMetrics);
    
    // Validate performance requirements
    test.meetsPerformanceRequirements = validateLargeDatasetPerformance(test.performanceMetrics);
    
    if (!test.success) {
        test.errorMessage = process.readAllStandardError();
    }
    
    results.largeCodebaseTests.append(test);
    
    qDebug() << "Large codebase test completed - Success:" << test.success 
             << "Time:" << test.executionTimeMs << "ms";
    
    return test.success && test.meetsPerformanceRequirements;
}

bool PerformanceScalabilityValidator::validateParallelExecutionEfficiency(ScalabilityResults &results)
{
    qDebug() << "Validating parallel execution efficiency...";
    
    ParallelExecutionTest test;
    test.testName = "Parallel Execution Efficiency";
    test.startTime = QDateTime::currentDateTime();
    
    // Test different levels of parallelism
    QList<int> parallelLevels = {1, 2, 4, 8, m_maxParallelTests};
    
    for (int parallelLevel : parallelLevels) {
        ParallelExecutionResult result = executeParallelTests(parallelLevel);
        test.parallelResults.append(result);
    }
    
    test.endTime = QDateTime::currentDateTime();
    test.executionTimeMs = test.startTime.msecsTo(test.endTime);
    
    // Analyze efficiency
    test.efficiency = calculateParallelEfficiency(test.parallelResults);
    test.optimalParallelLevel = findOptimalParallelLevel(test.parallelResults);
    test.resourceUtilization = calculateResourceUtilization(test.parallelResults);
    
    // Validate efficiency requirements
    test.success = (test.efficiency >= 0.7); // 70% efficiency threshold
    test.meetsEfficiencyRequirements = test.success;
    
    results.parallelExecutionTests.append(test);
    
    qDebug() << "Parallel execution test completed - Efficiency:" << test.efficiency 
             << "Optimal level:" << test.optimalParallelLevel;
    
    return test.success;
}

bool PerformanceScalabilityValidator::validateCIPipelinePerformance(ScalabilityResults &results)
{
    qDebug() << "Validating CI/CD pipeline integration performance...";
    
    CIPipelineTest test;
    test.testName = "CI Pipeline Performance";
    test.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer timer;
    timer.start();
    
    // Simulate CI pipeline execution
    bool pipelineSuccess = simulateCIPipelineExecution(test);
    
    test.executionTimeMs = timer.elapsed();
    test.endTime = QDateTime::currentDateTime();
    test.success = pipelineSuccess;
    
    // Validate CI performance requirements
    test.meetsCIRequirements = validateCIPerformanceRequirements(test);
    
    results.ciPipelineTests.append(test);
    
    qDebug() << "CI pipeline test completed - Success:" << test.success 
             << "Time:" << test.executionTimeMs << "ms";
    
    return test.success && test.meetsCIRequirements;
}

bool PerformanceScalabilityValidator::validateTestResultScalability(ScalabilityResults &results)
{
    qDebug() << "Validating test result storage and retrieval scalability...";
    
    TestResultScalabilityTest test;
    test.testName = "Test Result Scalability";
    test.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer timer;
    timer.start();
    
    // Test result storage with increasing data sizes
    QList<int> dataSizes = {100, 1000, 10000, 50000}; // Number of test results
    
    for (int dataSize : dataSizes) {
        TestResultScalabilityResult result = testResultStorageScalability(dataSize);
        test.scalabilityResults.append(result);
    }
    
    test.executionTimeMs = timer.elapsed();
    test.endTime = QDateTime::currentDateTime();
    
    // Analyze scalability
    test.storageScalability = calculateStorageScalability(test.scalabilityResults);
    test.retrievalScalability = calculateRetrievalScalability(test.scalabilityResults);
    test.memoryEfficiency = calculateMemoryEfficiency(test.scalabilityResults);
    
    // Validate scalability requirements
    test.success = (test.storageScalability >= 0.8 && test.retrievalScalability >= 0.8);
    test.meetsScalabilityRequirements = test.success;
    
    results.testResultScalabilityTests.append(test);
    
    qDebug() << "Test result scalability completed - Storage:" << test.storageScalability 
             << "Retrieval:" << test.retrievalScalability;
    
    return test.success;
}

SystemInfo PerformanceScalabilityValidator::collectSystemInfo()
{
    SystemInfo info;
    
    info.cpuArchitecture = QSysInfo::currentCpuArchitecture();
    info.kernelType = QSysInfo::kernelType();
    info.kernelVersion = QSysInfo::kernelVersion();
    info.productType = QSysInfo::productType();
    info.productVersion = QSysInfo::productVersion();
    info.idealThreadCount = QThread::idealThreadCount();
    
    // Get memory information
    QStorageInfo storage = QStorageInfo::root();
    info.totalMemoryMB = storage.bytesTotal() / (1024 * 1024);
    info.availableMemoryMB = storage.bytesAvailable() / (1024 * 1024);
    
    // Get CPU information (simplified)
    info.cpuCores = QThread::idealThreadCount();
    
    return info;
}

bool PerformanceScalabilityValidator::createLargeTestDataset()
{
    qDebug() << "Creating large test dataset...";
    
    // Create a temporary directory with many test files
    QString testDataDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/cloneclean_large_test";
    QDir dir;
    
    if (!dir.mkpath(testDataDir)) {
        qDebug() << "Failed to create test data directory:" << testDataDir;
        return false;
    }
    
    // Create 10,000 test files with various sizes and content
    for (int i = 0; i < 10000; ++i) {
        QString fileName = QString("%1/test_file_%2.txt").arg(testDataDir).arg(i, 5, 10, QChar('0'));
        QFile file(fileName);
        
        if (file.open(QIODevice::WriteOnly)) {
            // Create files with different sizes and some duplicates
            QString content;
            if (i % 100 == 0) {
                // Create duplicate content every 100 files
                content = QString("Duplicate content for testing - %1").arg(i / 100);
            } else {
                content = QString("Unique content for file %1 - %2").arg(i).arg(QDateTime::currentMSecsSinceEpoch());
            }
            
            // Vary file sizes
            int repetitions = (i % 10) + 1;
            for (int j = 0; j < repetitions; ++j) {
                file.write(content.toUtf8());
                file.write("\n");
            }
            
            file.close();
        }
        
        // Progress indication
        if (i % 1000 == 0) {
            qDebug() << "Created" << i << "test files...";
        }
    }
    
    qDebug() << "Large test dataset created successfully";
    return true;
}

QString PerformanceScalabilityValidator::findTestExecutable(const QString &executableName)
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

void PerformanceScalabilityValidator::parsePerformanceMetrics(const QString &output, PerformanceMetrics &metrics)
{
    QStringList lines = output.split('\n');
    
    for (const QString &line : lines) {
        if (line.contains("Execution time:")) {
            QRegularExpression re(R"(Execution time:\s*(\d+)\s*ms)");
            QRegularExpressionMatch match = re.match(line);
            if (match.hasMatch()) {
                metrics.executionTimeMs = match.captured(1).toLongLong();
            }
        }
        else if (line.contains("Memory usage:")) {
            QRegularExpression re(R"(Memory usage:\s*(\d+)\s*MB)");
            QRegularExpressionMatch match = re.match(line);
            if (match.hasMatch()) {
                metrics.memoryUsageMB = match.captured(1).toLongLong();
            }
        }
        else if (line.contains("Peak memory:")) {
            QRegularExpression re(R"(Peak memory:\s*(\d+)\s*MB)");
            QRegularExpressionMatch match = re.match(line);
            if (match.hasMatch()) {
                metrics.peakMemoryMB = match.captured(1).toLongLong();
            }
        }
        else if (line.contains("CPU usage:")) {
            QRegularExpression re(R"(CPU usage:\s*(\d+\.\d+)%)");
            QRegularExpressionMatch match = re.match(line);
            if (match.hasMatch()) {
                metrics.cpuUsagePercent = match.captured(1).toDouble();
            }
        }
        else if (line.contains("Files processed:")) {
            QRegularExpression re(R"(Files processed:\s*(\d+))");
            QRegularExpressionMatch match = re.match(line);
            if (match.hasMatch()) {
                metrics.filesProcessed = match.captured(1).toInt();
            }
        }
        else if (line.contains("Throughput:")) {
            QRegularExpression re(R"(Throughput:\s*(\d+\.\d+)\s*files/sec)");
            QRegularExpressionMatch match = re.match(line);
            if (match.hasMatch()) {
                metrics.throughputFilesPerSec = match.captured(1).toDouble();
            }
        }
    }
}

bool PerformanceScalabilityValidator::validateLargeDatasetPerformance(const PerformanceMetrics &metrics)
{
    // Define performance requirements for large datasets
    const qint64 maxExecutionTimeMs = 30 * 60 * 1000; // 30 minutes
    const qint64 maxMemoryUsageMB = 2048; // 2GB
    const double minThroughput = 10.0; // 10 files/sec minimum
    
    bool meetsRequirements = true;
    
    if (metrics.executionTimeMs > maxExecutionTimeMs) {
        qDebug() << "Performance requirement failed: Execution time" << metrics.executionTimeMs 
                 << "ms exceeds limit of" << maxExecutionTimeMs << "ms";
        meetsRequirements = false;
    }
    
    if (metrics.memoryUsageMB > maxMemoryUsageMB) {
        qDebug() << "Performance requirement failed: Memory usage" << metrics.memoryUsageMB 
                 << "MB exceeds limit of" << maxMemoryUsageMB << "MB";
        meetsRequirements = false;
    }
    
    if (metrics.throughputFilesPerSec < minThroughput) {
        qDebug() << "Performance requirement failed: Throughput" << metrics.throughputFilesPerSec 
                 << "files/sec below minimum of" << minThroughput << "files/sec";
        meetsRequirements = false;
    }
    
    return meetsRequirements;
}

ParallelExecutionResult PerformanceScalabilityValidator::executeParallelTests(int parallelLevel)
{
    qDebug() << "Executing parallel tests with level:" << parallelLevel;
    
    ParallelExecutionResult result;
    result.parallelLevel = parallelLevel;
    result.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer timer;
    timer.start();
    
    // Set thread pool size
    QThreadPool::globalInstance()->setMaxThreadCount(parallelLevel);
    
    // Execute test suite with parallel configuration
    QProcess process;
    QString testExecutable = findTestExecutable("performance_tests");
    
    if (testExecutable.isEmpty()) {
        result.success = false;
        result.errorMessage = "Performance test executable not found";
        result.executionTimeMs = timer.elapsed();
        return result;
    }
    
    // Set environment for parallel testing
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("QT_QPA_PLATFORM", "offscreen");
    env.insert("CLONECLEAN_TEST_PARALLEL_LEVEL", QString::number(parallelLevel));
    process.setProcessEnvironment(env);
    
    process.start(testExecutable, QStringList());
    
    if (!process.waitForStarted(10000)) {
        result.success = false;
        result.errorMessage = "Failed to start parallel test";
        result.executionTimeMs = timer.elapsed();
        return result;
    }
    
    if (!process.waitForFinished(600000)) { // 10 minutes timeout
        process.kill();
        result.success = false;
        result.errorMessage = "Parallel test timed out";
        result.executionTimeMs = timer.elapsed();
        return result;
    }
    
    result.executionTimeMs = timer.elapsed();
    result.endTime = QDateTime::currentDateTime();
    result.exitCode = process.exitCode();
    result.success = (process.exitCode() == 0);
    
    // Parse performance metrics
    QString output = process.readAllStandardOutput();
    parsePerformanceMetrics(output, result.performanceMetrics);
    
    if (!result.success) {
        result.errorMessage = process.readAllStandardError();
    }
    
    qDebug() << "Parallel test level" << parallelLevel << "completed - Success:" << result.success 
             << "Time:" << result.executionTimeMs << "ms";
    
    return result;
}

double PerformanceScalabilityValidator::calculateParallelEfficiency(const QList<ParallelExecutionResult> &results)
{
    if (results.size() < 2) {
        return 0.0;
    }
    
    // Find baseline (single-threaded) result
    ParallelExecutionResult baseline;
    for (const ParallelExecutionResult &result : results) {
        if (result.parallelLevel == 1) {
            baseline = result;
            break;
        }
    }
    
    if (baseline.parallelLevel != 1 || baseline.executionTimeMs == 0) {
        return 0.0;
    }
    
    // Calculate efficiency for highest parallel level
    ParallelExecutionResult maxParallel;
    for (const ParallelExecutionResult &result : results) {
        if (result.parallelLevel > maxParallel.parallelLevel) {
            maxParallel = result;
        }
    }
    
    if (maxParallel.executionTimeMs == 0) {
        return 0.0;
    }
    
    // Efficiency = (baseline_time / parallel_time) / parallel_level
    double speedup = double(baseline.executionTimeMs) / double(maxParallel.executionTimeMs);
    double efficiency = speedup / double(maxParallel.parallelLevel);
    
    return efficiency;
}

int PerformanceScalabilityValidator::findOptimalParallelLevel(const QList<ParallelExecutionResult> &results)
{
    int optimalLevel = 1;
    qint64 bestTime = LLONG_MAX;
    
    for (const ParallelExecutionResult &result : results) {
        if (result.success && result.executionTimeMs < bestTime) {
            bestTime = result.executionTimeMs;
            optimalLevel = result.parallelLevel;
        }
    }
    
    return optimalLevel;
}

double PerformanceScalabilityValidator::calculateResourceUtilization(const QList<ParallelExecutionResult> &results)
{
    if (results.isEmpty()) {
        return 0.0;
    }
    
    double totalUtilization = 0.0;
    int validResults = 0;
    
    for (const ParallelExecutionResult &result : results) {
        if (result.success && result.performanceMetrics.cpuUsagePercent > 0) {
            totalUtilization += result.performanceMetrics.cpuUsagePercent;
            validResults++;
        }
    }
    
    return validResults > 0 ? totalUtilization / validResults : 0.0;
}

bool PerformanceScalabilityValidator::simulateCIPipelineExecution(CIPipelineTest &test)
{
    qDebug() << "Simulating CI pipeline execution...";
    
    // Simulate typical CI pipeline stages
    QStringList pipelineStages = {
        "checkout",
        "build",
        "unit_tests",
        "integration_tests",
        "performance_tests",
        "package",
        "deploy"
    };
    
    bool success = true;
    
    for (const QString &stage : pipelineStages) {
        CIPipelineStage stageResult;
        stageResult.stageName = stage;
        stageResult.startTime = QDateTime::currentDateTime();
        
        QElapsedTimer stageTimer;
        stageTimer.start();
        
        // Simulate stage execution
        bool stageSuccess = simulatePipelineStage(stage, stageResult);
        
        stageResult.executionTimeMs = stageTimer.elapsed();
        stageResult.endTime = QDateTime::currentDateTime();
        stageResult.success = stageSuccess;
        
        test.pipelineStages.append(stageResult);
        
        if (!stageSuccess) {
            success = false;
            break; // Stop on first failure
        }
        
        qDebug() << "Pipeline stage" << stage << "completed - Success:" << stageSuccess 
                 << "Time:" << stageResult.executionTimeMs << "ms";
    }
    
    return success;
}

bool PerformanceScalabilityValidator::simulatePipelineStage(const QString &stageName, CIPipelineStage &stage)
{
    if (stageName == "checkout") {
        // Simulate code checkout
        QThread::msleep(1000); // 1 second
        stage.artifactSize = 50 * 1024 * 1024; // 50MB
        return true;
    }
    else if (stageName == "build") {
        // Simulate build process
        QProcess process;
        process.start("cmake", QStringList() << "--version");
        if (process.waitForFinished(5000)) {
            stage.artifactSize = 100 * 1024 * 1024; // 100MB
            return process.exitCode() == 0;
        }
        return false;
    }
    else if (stageName == "unit_tests") {
        // Execute actual unit tests
        QString testExecutable = findTestExecutable("unit_tests");
        if (!testExecutable.isEmpty()) {
            QProcess process;
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            env.insert("QT_QPA_PLATFORM", "offscreen");
            process.setProcessEnvironment(env);
            
            process.start(testExecutable, QStringList());
            if (process.waitForFinished(120000)) { // 2 minutes timeout
                stage.testResults = process.readAllStandardOutput();
                return process.exitCode() == 0;
            }
        }
        return false;
    }
    else if (stageName == "integration_tests") {
        // Execute actual integration tests
        QString testExecutable = findTestExecutable("integration_tests");
        if (!testExecutable.isEmpty()) {
            QProcess process;
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            env.insert("QT_QPA_PLATFORM", "offscreen");
            process.setProcessEnvironment(env);
            
            process.start(testExecutable, QStringList());
            if (process.waitForFinished(300000)) { // 5 minutes timeout
                stage.testResults = process.readAllStandardOutput();
                return process.exitCode() == 0;
            }
        }
        return false;
    }
    else if (stageName == "performance_tests") {
        // Execute performance tests
        QString testExecutable = findTestExecutable("performance_tests");
        if (!testExecutable.isEmpty()) {
            QProcess process;
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            env.insert("QT_QPA_PLATFORM", "offscreen");
            process.setProcessEnvironment(env);
            
            process.start(testExecutable, QStringList());
            if (process.waitForFinished(600000)) { // 10 minutes timeout
                stage.testResults = process.readAllStandardOutput();
                return process.exitCode() == 0;
            }
        }
        return false;
    }
    else if (stageName == "package") {
        // Simulate packaging
        QThread::msleep(5000); // 5 seconds
        stage.artifactSize = 200 * 1024 * 1024; // 200MB
        return true;
    }
    else if (stageName == "deploy") {
        // Simulate deployment
        QThread::msleep(2000); // 2 seconds
        return true;
    }
    
    return false;
}

bool PerformanceScalabilityValidator::validateCIPerformanceRequirements(const CIPipelineTest &test)
{
    // Define CI performance requirements
    const qint64 maxTotalTimeMs = 45 * 60 * 1000; // 45 minutes total
    const qint64 maxStageTimeMs = 15 * 60 * 1000; // 15 minutes per stage
    
    bool meetsRequirements = true;
    
    qint64 totalTime = 0;
    for (const CIPipelineStage &stage : test.pipelineStages) {
        totalTime += stage.executionTimeMs;
        
        if (stage.executionTimeMs > maxStageTimeMs) {
            qDebug() << "CI requirement failed: Stage" << stage.stageName 
                     << "took" << stage.executionTimeMs << "ms, exceeds limit of" << maxStageTimeMs << "ms";
            meetsRequirements = false;
        }
    }
    
    if (totalTime > maxTotalTimeMs) {
        qDebug() << "CI requirement failed: Total pipeline time" << totalTime 
                 << "ms exceeds limit of" << maxTotalTimeMs << "ms";
        meetsRequirements = false;
    }
    
    return meetsRequirements;
}

TestResultScalabilityResult PerformanceScalabilityValidator::testResultStorageScalability(int dataSize)
{
    qDebug() << "Testing result storage scalability with" << dataSize << "results...";
    
    TestResultScalabilityResult result;
    result.dataSize = dataSize;
    result.startTime = QDateTime::currentDateTime();
    
    QElapsedTimer timer;
    timer.start();
    
    // Create test result data
    QJsonArray testResults;
    for (int i = 0; i < dataSize; ++i) {
        QJsonObject testResult;
        testResult["test_name"] = QString("test_%1").arg(i);
        testResult["success"] = (i % 10 != 0); // 90% success rate
        testResult["execution_time"] = QRandomGenerator::global()->bounded(1000, 10000);
        testResult["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        testResult["details"] = QString("Test details for test %1").arg(i);
        testResults.append(testResult);
    }
    
    // Measure storage time
    QElapsedTimer storageTimer;
    storageTimer.start();
    
    QString fileName = QString("test_results_%1.json").arg(dataSize);
    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly)) {
        QJsonDocument doc(testResults);
        file.write(doc.toJson());
        file.close();
        result.storageTimeMs = storageTimer.elapsed();
        result.storageSuccess = true;
        result.storedSizeMB = file.size() / (1024.0 * 1024.0);
    } else {
        result.storageSuccess = false;
        result.storageTimeMs = storageTimer.elapsed();
    }
    
    // Measure retrieval time
    QElapsedTimer retrievalTimer;
    retrievalTimer.start();
    
    if (file.open(QIODevice::ReadOnly)) {
        QByteArray data = file.readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        QJsonArray retrievedResults = doc.array();
        
        result.retrievalTimeMs = retrievalTimer.elapsed();
        result.retrievalSuccess = (retrievedResults.size() == dataSize);
        file.close();
    } else {
        result.retrievalSuccess = false;
        result.retrievalTimeMs = retrievalTimer.elapsed();
    }
    
    // Clean up
    file.remove();
    
    result.totalTimeMs = timer.elapsed();
    result.endTime = QDateTime::currentDateTime();
    result.success = result.storageSuccess && result.retrievalSuccess;
    
    qDebug() << "Storage scalability test for" << dataSize << "results completed - Success:" << result.success 
             << "Storage:" << result.storageTimeMs << "ms Retrieval:" << result.retrievalTimeMs << "ms";
    
    return result;
}

double PerformanceScalabilityValidator::calculateStorageScalability(const QList<TestResultScalabilityResult> &results)
{
    if (results.size() < 2) {
        return 0.0;
    }
    
    // Calculate scalability as the inverse of time growth rate
    // Good scalability means time grows linearly (or better) with data size
    
    double totalScalability = 0.0;
    int validComparisons = 0;
    
    for (int i = 1; i < results.size(); ++i) {
        const TestResultScalabilityResult &prev = results[i-1];
        const TestResultScalabilityResult &curr = results[i];
        
        if (prev.storageSuccess && curr.storageSuccess && prev.storageTimeMs > 0) {
            double dataRatio = double(curr.dataSize) / double(prev.dataSize);
            double timeRatio = double(curr.storageTimeMs) / double(prev.storageTimeMs);
            
            // Scalability = dataRatio / timeRatio (1.0 = linear, >1.0 = better than linear)
            double scalability = dataRatio / timeRatio;
            totalScalability += scalability;
            validComparisons++;
        }
    }
    
    return validComparisons > 0 ? totalScalability / validComparisons : 0.0;
}

double PerformanceScalabilityValidator::calculateRetrievalScalability(const QList<TestResultScalabilityResult> &results)
{
    if (results.size() < 2) {
        return 0.0;
    }
    
    double totalScalability = 0.0;
    int validComparisons = 0;
    
    for (int i = 1; i < results.size(); ++i) {
        const TestResultScalabilityResult &prev = results[i-1];
        const TestResultScalabilityResult &curr = results[i];
        
        if (prev.retrievalSuccess && curr.retrievalSuccess && prev.retrievalTimeMs > 0) {
            double dataRatio = double(curr.dataSize) / double(prev.dataSize);
            double timeRatio = double(curr.retrievalTimeMs) / double(prev.retrievalTimeMs);
            
            double scalability = dataRatio / timeRatio;
            totalScalability += scalability;
            validComparisons++;
        }
    }
    
    return validComparisons > 0 ? totalScalability / validComparisons : 0.0;
}

double PerformanceScalabilityValidator::calculateMemoryEfficiency(const QList<TestResultScalabilityResult> &results)
{
    if (results.isEmpty()) {
        return 0.0;
    }
    
    // Calculate memory efficiency as data size vs storage size ratio
    double totalEfficiency = 0.0;
    int validResults = 0;
    
    for (const TestResultScalabilityResult &result : results) {
        if (result.success && result.storedSizeMB > 0) {
            // Estimate expected data size (rough calculation)
            double expectedSizeMB = result.dataSize * 0.001; // ~1KB per test result
            double efficiency = expectedSizeMB / result.storedSizeMB;
            totalEfficiency += efficiency;
            validResults++;
        }
    }
    
    return validResults > 0 ? totalEfficiency / validResults : 0.0;
}

void PerformanceScalabilityValidator::generateScalabilityReport(const ScalabilityResults &results)
{
    qDebug() << "Generating performance and scalability report...";
    
    // Generate JSON report
    generateScalabilityJsonReport(results);
    
    // Generate HTML report
    generateScalabilityHtmlReport(results);
    
    // Generate console summary
    generateScalabilityConsoleSummary(results);
}

void PerformanceScalabilityValidator::generateScalabilityJsonReport(const ScalabilityResults &results)
{
    QJsonObject report;
    
    // System information
    QJsonObject systemInfo;
    systemInfo["cpu_architecture"] = results.systemInfo.cpuArchitecture;
    systemInfo["kernel_type"] = results.systemInfo.kernelType;
    systemInfo["kernel_version"] = results.systemInfo.kernelVersion;
    systemInfo["product_type"] = results.systemInfo.productType;
    systemInfo["product_version"] = results.systemInfo.productVersion;
    systemInfo["cpu_cores"] = results.systemInfo.cpuCores;
    systemInfo["ideal_thread_count"] = results.systemInfo.idealThreadCount;
    systemInfo["total_memory_mb"] = results.systemInfo.totalMemoryMB;
    systemInfo["available_memory_mb"] = results.systemInfo.availableMemoryMB;
    report["system_info"] = systemInfo;
    
    // Summary
    QJsonObject summary;
    summary["start_time"] = results.startTime.toString(Qt::ISODate);
    summary["end_time"] = results.endTime.toString(Qt::ISODate);
    summary["total_execution_time_ms"] = results.totalExecutionTimeMs;
    report["summary"] = summary;
    
    // Large codebase tests
    QJsonArray largeCodebaseTests;
    for (const LargeCodebaseTest &test : results.largeCodebaseTests) {
        QJsonObject testObj;
        testObj["test_name"] = test.testName;
        testObj["success"] = test.success;
        testObj["meets_performance_requirements"] = test.meetsPerformanceRequirements;
        testObj["execution_time_ms"] = test.executionTimeMs;
        testObj["exit_code"] = test.exitCode;
        
        if (!test.errorMessage.isEmpty()) {
            testObj["error_message"] = test.errorMessage;
        }
        
        // Performance metrics
        QJsonObject metrics;
        metrics["execution_time_ms"] = test.performanceMetrics.executionTimeMs;
        metrics["memory_usage_mb"] = test.performanceMetrics.memoryUsageMB;
        metrics["peak_memory_mb"] = test.performanceMetrics.peakMemoryMB;
        metrics["cpu_usage_percent"] = test.performanceMetrics.cpuUsagePercent;
        metrics["files_processed"] = test.performanceMetrics.filesProcessed;
        metrics["throughput_files_per_sec"] = test.performanceMetrics.throughputFilesPerSec;
        testObj["performance_metrics"] = metrics;
        
        largeCodebaseTests.append(testObj);
    }
    report["large_codebase_tests"] = largeCodebaseTests;
    
    // Write JSON report
    QJsonDocument doc(report);
    QFile jsonFile("performance_scalability_report.json");
    if (jsonFile.open(QIODevice::WriteOnly)) {
        jsonFile.write(doc.toJson());
        jsonFile.close();
        qDebug() << "Performance scalability JSON report written to performance_scalability_report.json";
    }
}

void PerformanceScalabilityValidator::generateScalabilityHtmlReport(const ScalabilityResults &results)
{
    QString html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Performance and Scalability Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: #4CAF50; }
        .failure { color: #f44336; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance and Scalability Validation Report</h1>
        <p>Generated: %1</p>
        <p>System: %2 %3 (%4 cores)</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total Execution Time: %5 minutes</div>
        <div class="metric">System Memory: %6 MB total, %7 MB available</div>
    </div>
    
    <div class="section">
        <h2>Large Codebase Tests</h2>
        %8
    </div>
</body>
</html>
)";
    
    // Fill in basic information
    html = html.arg(QDateTime::currentDateTime().toString())
              .arg(results.systemInfo.productType)
              .arg(results.systemInfo.productVersion)
              .arg(results.systemInfo.cpuCores)
              .arg(results.totalExecutionTimeMs / 60000.0, 0, 'f', 1)
              .arg(results.systemInfo.totalMemoryMB)
              .arg(results.systemInfo.availableMemoryMB);
    
    // Generate large codebase tests section
    QString largeCodebaseHtml;
    for (const LargeCodebaseTest &test : results.largeCodebaseTests) {
        QString testClass = test.success ? "success" : "failure";
        largeCodebaseHtml += QString(R"(
        <div class="test-result">
            <h3>%1 <span class="%2">(%3)</span></h3>
            <div class="metric">Execution Time: %4 ms</div>
            <div class="metric">Memory Usage: %5 MB</div>
            <div class="metric">Peak Memory: %6 MB</div>
            <div class="metric">CPU Usage: %7%</div>
            <div class="metric">Files Processed: %8</div>
            <div class="metric">Throughput: %9 files/sec</div>
            <div class="metric">Meets Requirements: %10</div>
        </div>
        )").arg(test.testName, testClass, test.success ? "PASS" : "FAIL")
           .arg(test.executionTimeMs)
           .arg(test.performanceMetrics.memoryUsageMB)
           .arg(test.performanceMetrics.peakMemoryMB)
           .arg(test.performanceMetrics.cpuUsagePercent, 0, 'f', 1)
           .arg(test.performanceMetrics.filesProcessed)
           .arg(test.performanceMetrics.throughputFilesPerSec, 0, 'f', 1)
           .arg(test.meetsPerformanceRequirements ? "Yes" : "No");
    }
    
    html = html.arg(largeCodebaseHtml);
    
    // Write HTML report
    QFile htmlFile("performance_scalability_report.html");
    if (htmlFile.open(QIODevice::WriteOnly)) {
        htmlFile.write(html.toUtf8());
        htmlFile.close();
        qDebug() << "Performance scalability HTML report written to performance_scalability_report.html";
    }
}

void PerformanceScalabilityValidator::generateScalabilityConsoleSummary(const ScalabilityResults &results)
{
    qDebug() << "\n" << QString(60, '=');
    qDebug() << "PERFORMANCE AND SCALABILITY VALIDATION SUMMARY";
    qDebug() << QString(60, '=');
    
    qDebug() << QString("System: %1 %2 (%3 cores)")
                .arg(results.systemInfo.productType)
                .arg(results.systemInfo.productVersion)
                .arg(results.systemInfo.cpuCores);
    
    qDebug() << QString("Total Execution Time: %1 minutes").arg(results.totalExecutionTimeMs / 60000.0, 0, 'f', 1);
    
    qDebug() << "\nLARGE CODEBASE TESTS:";
    for (const LargeCodebaseTest &test : results.largeCodebaseTests) {
        QString status = test.success ? "PASS" : "FAIL";
        QString requirements = test.meetsPerformanceRequirements ? "PASS" : "FAIL";
        qDebug() << QString("  %1: %2 (Requirements: %3)")
                    .arg(test.testName, status, requirements);
        qDebug() << QString("    Time: %1 ms, Memory: %2 MB, Throughput: %3 files/sec")
                    .arg(test.executionTimeMs)
                    .arg(test.performanceMetrics.memoryUsageMB)
                    .arg(test.performanceMetrics.throughputFilesPerSec, 0, 'f', 1);
    }
    
    qDebug() << "\nPARALLEL EXECUTION TESTS:";
    for (const ParallelExecutionTest &test : results.parallelExecutionTests) {
        QString status = test.success ? "PASS" : "FAIL";
        qDebug() << QString("  %1: %2 (Efficiency: %3)")
                    .arg(test.testName, status)
                    .arg(test.efficiency, 0, 'f', 2);
        qDebug() << QString("    Optimal Level: %1, Resource Utilization: %2%")
                    .arg(test.optimalParallelLevel)
                    .arg(test.resourceUtilization, 0, 'f', 1);
    }
    
    bool overallSuccess = true;
    for (const LargeCodebaseTest &test : results.largeCodebaseTests) {
        if (!test.success || !test.meetsPerformanceRequirements) {
            overallSuccess = false;
            break;
        }
    }
    
    for (const ParallelExecutionTest &test : results.parallelExecutionTests) {
        if (!test.success) {
            overallSuccess = false;
            break;
        }
    }
    
    qDebug() << QString(60, '=');
    qDebug() << QString("OVERALL RESULT: %1").arg(overallSuccess ? "PASS" : "FAIL");
    qDebug() << QString(60, '=') << "\n";
}