#include "performance_test_framework.h"
#include <QFile>
#include <QTextStream>
#include <QTimer>
#include <QProcessEnvironment>
#include <QSysInfo>
#include <QtConcurrent>
#include <QStorageInfo>
#include <algorithm>
#include <cmath>

namespace PerformanceTest {

// PerformanceResult Implementation
QJsonObject PerformanceResult::toJson() const {
    QJsonObject obj;
    obj["testName"] = testName;
    obj["category"] = category;
    obj["timestamp"] = timestamp.toString(Qt::ISODate);
    
    // Timing
    obj["executionTime"] = executionTime;
    obj["cpuTime"] = cpuTime;
    obj["wallTime"] = wallTime;
    
    // Throughput
    obj["throughputMBps"] = throughputMBps;
    obj["filesPerSecond"] = filesPerSecond;
    obj["operationsPerSecond"] = operationsPerSecond;
    
    // Resources
    obj["peakMemoryUsage"] = static_cast<qint64>(peakMemoryUsage);
    obj["averageMemoryUsage"] = static_cast<qint64>(averageMemoryUsage);
    obj["cpuUtilization"] = cpuUtilization;
    obj["ioUtilization"] = ioUtilization;
    
    // Statistics
    obj["meanTime"] = meanTime;
    obj["medianTime"] = medianTime;
    obj["stdDeviation"] = stdDeviation;
    obj["minTime"] = minTime;
    obj["maxTime"] = maxTime;
    obj["confidenceInterval95"] = confidenceInterval95;
    
    // System
    obj["systemInfo"] = systemInfo;
    obj["qtVersion"] = qtVersion;
    obj["buildConfig"] = buildConfig;
    obj["threadCount"] = threadCount;
    
    // Validation
    obj["validationPassed"] = validationPassed;
    obj["errorMessage"] = errorMessage;
    
    return obj;
}

void PerformanceResult::fromJson(const QJsonObject& json) {
    testName = json["testName"].toString();
    category = json["category"].toString();
    timestamp = QDateTime::fromString(json["timestamp"].toString(), Qt::ISODate);
    
    executionTime = json["executionTime"].toDouble();
    cpuTime = json["cpuTime"].toDouble();
    wallTime = json["wallTime"].toDouble();
    
    throughputMBps = json["throughputMBps"].toDouble();
    filesPerSecond = json["filesPerSecond"].toDouble();
    operationsPerSecond = json["operationsPerSecond"].toDouble();
    
    peakMemoryUsage = json["peakMemoryUsage"].toVariant().toLongLong();
    averageMemoryUsage = json["averageMemoryUsage"].toVariant().toLongLong();
    cpuUtilization = json["cpuUtilization"].toDouble();
    ioUtilization = json["ioUtilization"].toDouble();
    
    meanTime = json["meanTime"].toDouble();
    medianTime = json["medianTime"].toDouble();
    stdDeviation = json["stdDeviation"].toDouble();
    minTime = json["minTime"].toDouble();
    maxTime = json["maxTime"].toDouble();
    confidenceInterval95 = json["confidenceInterval95"].toDouble();
    
    systemInfo = json["systemInfo"].toString();
    qtVersion = json["qtVersion"].toString();
    buildConfig = json["buildConfig"].toString();
    threadCount = json["threadCount"].toInt();
    
    validationPassed = json["validationPassed"].toBool();
    errorMessage = json["errorMessage"].toString();
}

// SystemProfiler Implementation
QJsonObject SystemProfiler::SystemInfo::toJson() const {
    QJsonObject obj;
    obj["cpuModel"] = cpuModel;
    obj["cpuCores"] = cpuCores;
    obj["logicalProcessors"] = logicalProcessors;
    obj["cpuArchitecture"] = cpuArchitecture;
    obj["cpuFrequencyGHz"] = cpuFrequencyGHz;
    obj["totalMemoryMB"] = static_cast<qint64>(totalMemoryMB);
    obj["availableMemoryMB"] = static_cast<qint64>(availableMemoryMB);
    obj["osName"] = osName;
    obj["osVersion"] = osVersion;
    obj["qtVersion"] = qtVersion;
    obj["buildType"] = buildType;
    obj["primaryStorageType"] = primaryStorageType;
    
    return obj;
}

SystemProfiler::SystemInfo SystemProfiler::getSystemInfo() {
    SystemInfo info;
    
    // CPU Information
    info.cpuArchitecture = QSysInfo::currentCpuArchitecture();
    info.logicalProcessors = QThread::idealThreadCount();
    info.cpuCores = info.logicalProcessors; // Approximation
    info.cpuModel = QProcessEnvironment::systemEnvironment().value("PROCESSOR_IDENTIFIER", "Unknown");
    info.cpuFrequencyGHz = 0.0; // Would need platform-specific code
    
    // Memory Information (approximation)
    info.totalMemoryMB = 8192; // Default assumption
    info.availableMemoryMB = 4096;
    
    // OS Information
    info.osName = QSysInfo::prettyProductName();
    info.osVersion = QSysInfo::productVersion();
    info.qtVersion = QT_VERSION_STR;
    
#ifdef QT_DEBUG
    info.buildType = "Debug";
#else
    info.buildType = "Release";
#endif
    
    // Storage Information
    QStorageInfo storage = QStorageInfo::root();
    if (storage.isValid()) {
        QString device = storage.device();
        if (device.contains("nvme") || device.contains("ssd")) {
            info.primaryStorageType = "SSD";
        } else {
            info.primaryStorageType = "HDD";
        }
    } else {
        info.primaryStorageType = "Unknown";
    }
    
    return info;
}

QString SystemProfiler::getSystemSummary() {
    SystemInfo info = getSystemInfo();
    
    return QString("CPU: %1 (%2 cores), RAM: %3MB, OS: %4, Storage: %5, Qt: %6 (%7)")
        .arg(info.cpuModel.isEmpty() ? info.cpuArchitecture : info.cpuModel)
        .arg(info.logicalProcessors)
        .arg(info.totalMemoryMB)
        .arg(info.osName)
        .arg(info.primaryStorageType)
        .arg(info.qtVersion)
        .arg(info.buildType);
}

void SystemProfiler::logSystemInfo() {
    qDebug() << "=== System Information ===";
    qDebug() << getSystemSummary();
    qDebug() << "============================";
}

// TestDataGenerator Implementation
TestDataGenerator::TestDataGenerator(QObject* parent)
    : m_tempDir(new QTemporaryDir()) {
    Q_UNUSED(parent)
    if (m_tempDir->isValid()) {
        m_testDataPath = m_tempDir->path();
        qDebug() << "TestDataGenerator: Created temp directory:" << m_testDataPath;
    } else {
        qWarning() << "TestDataGenerator: Failed to create temporary directory";
    }
}

TestDataGenerator::~TestDataGenerator() {
    cleanupTestData();
}

bool TestDataGenerator::generateTestFiles(const DataGenerationOptions& options) {
    if (!m_tempDir->isValid()) {
        qWarning() << "TestDataGenerator: Invalid temporary directory";
        return false;
    }
    
    QString basePath = options.basePath.isEmpty() ? m_testDataPath : options.basePath;
    
    // Create directory structure if needed
    if (options.directoryDepth > 0) {
        basePath = createDirectoryStructure(basePath, options.directoryDepth, options.subdirectoryCount);
    }
    
    // Generate files with specified sizes or distribute total size
    QVector<qint64> fileSizes;
    if (!options.fileSizes.isEmpty()) {
        fileSizes = options.fileSizes;
    } else {
        // Distribute total size across files
        qint64 avgSize = options.totalSize / options.fileCount;
        for (int i = 0; i < options.fileCount; ++i) {
            fileSizes.append(avgSize);
        }
    }
    
    // Generate unique files
    QStringList uniqueFiles;
    int uniqueCount = options.createDuplicates ? 
        static_cast<int>(static_cast<double>(fileSizes.size()) * (1.0 - options.duplicateRatio)) : static_cast<int>(fileSizes.size());
    
    for (int i = 0; i < uniqueCount; ++i) {
        QString extension = options.fileExtensions[i % options.fileExtensions.size()];
        QString fileName = QString("test_file_%1.%2").arg(i, 4, 10, QLatin1Char('0')).arg(extension);
        QString filePath = QDir(basePath).absoluteFilePath(fileName);
        
        if (generateTestFile(filePath, fileSizes[i], options.contentPattern)) {
            uniqueFiles.append(filePath);
            m_generatedFiles.append(filePath);
        }
    }
    
    // Generate duplicates
    if (options.createDuplicates && !uniqueFiles.isEmpty()) {
        int duplicatesToCreate = static_cast<int>(fileSizes.size()) - uniqueCount;
        for (int i = 0; i < duplicatesToCreate; ++i) {
            QString sourceFile = uniqueFiles[i % uniqueFiles.size()];
            QString extension = QFileInfo(sourceFile).suffix();
            QString fileName = QString("dup_file_%1.%2").arg(i, 4, 10, QLatin1Char('0')).arg(extension);
            QString duplicatePath = QDir(basePath).absoluteFilePath(fileName);
            
            if (QFile::copy(sourceFile, duplicatePath)) {
                m_generatedFiles.append(duplicatePath);
            }
        }
    }
    
    qDebug() << "TestDataGenerator: Generated" << m_generatedFiles.size() << "test files";
    return !m_generatedFiles.isEmpty();
}

bool TestDataGenerator::generateTestFile(const QString& filePath, qint64 size, const QString& pattern) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "TestDataGenerator: Cannot create file:" << filePath;
        return false;
    }
    
    QByteArray content;
    if (!pattern.isEmpty()) {
        // Use specified pattern
        QByteArray patternBytes = pattern.toUtf8();
        while (content.size() < size) {
            content.append(patternBytes);
        }
        content = content.left(size);
    } else {
        // Generate random content
        content = generateRandomContent(size);
    }
    
    qint64 bytesWritten = file.write(content);
    file.close();
    
    return bytesWritten == size;
}

QByteArray TestDataGenerator::generateRandomContent(qint64 size, bool compressible) {
    QByteArray content;
    content.reserve(size);
    
    if (compressible) {
        // Generate compressible content with patterns
        const QByteArray pattern = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        while (content.size() < size) {
            content.append(pattern);
        }
    } else {
        // Generate truly random content
        QRandomGenerator* rng = QRandomGenerator::global();
        for (qint64 i = 0; i < size; ++i) {
            content.append(static_cast<char>(rng->bounded(256)));
        }
    }
    
    return content.left(size);
}

QString TestDataGenerator::createDirectoryStructure(const QString& basePath, int depth, int subdirCount) {
    QString currentPath = basePath;
    
    for (int d = 0; d < depth; ++d) {
        for (int s = 0; s < subdirCount; ++s) {
            QString subdirName = QString("subdir_%1_%2").arg(d).arg(s);
            QString subdirPath = QDir(currentPath).absoluteFilePath(subdirName);
            QDir().mkpath(subdirPath);
        }
        // Navigate into first subdirectory for next level
        if (subdirCount > 0) {
            currentPath = QDir(currentPath).absoluteFilePath(QString("subdir_%1_0").arg(d));
        }
    }
    
    return currentPath;
}

bool TestDataGenerator::generateSmallFilesScenario(int fileCount, qint64 avgSize) {
    DataGenerationOptions options;
    options.fileCount = fileCount;
    options.totalSize = fileCount * avgSize;
    options.fileExtensions = {"txt", "log", "cfg"};
    return generateTestFiles(options);
}

bool TestDataGenerator::generateLargeFilesScenario(int fileCount, qint64 avgSize) {
    DataGenerationOptions options;
    options.fileCount = fileCount;
    options.totalSize = fileCount * avgSize;
    options.fileExtensions = {"bin", "dat", "img"};
    return generateTestFiles(options);
}

bool TestDataGenerator::generateMixedSizesScenario(int totalFiles) {
    DataGenerationOptions options;
    options.fileCount = totalFiles;
    
    // Create a mix of file sizes
    for (int i = 0; i < totalFiles; ++i) {
        qint64 size;
        if (i < totalFiles * 0.7) {
            size = 1024 + QRandomGenerator::global()->bounded(10240); // 1-10KB
        } else if (i < totalFiles * 0.9) {
            size = 100 * 1024 + QRandomGenerator::global()->bounded(1024 * 1024); // 100KB-1MB
        } else {
            size = 10 * 1024 * 1024 + QRandomGenerator::global()->bounded(50 * 1024 * 1024); // 10-60MB
        }
        options.fileSizes.append(size);
    }
    
    return generateTestFiles(options);
}

bool TestDataGenerator::generateDuplicateScenario(int uniqueFiles, int duplicatesPerFile) {
    DataGenerationOptions options;
    options.fileCount = uniqueFiles;
    options.createDuplicates = true;
    options.duplicateRatio = static_cast<double>(duplicatesPerFile) / (uniqueFiles + duplicatesPerFile);
    options.totalSize = uniqueFiles * 50 * 1024; // 50KB average
    
    return generateTestFiles(options);
}

bool TestDataGenerator::generateDirectoryStructureScenario(int depth, int filesPerDir) {
    DataGenerationOptions options;
    options.directoryDepth = depth;
    options.subdirectoryCount = 3;
    options.fileCount = filesPerDir * depth;
    options.totalSize = options.fileCount * 10 * 1024; // 10KB per file
    
    return generateTestFiles(options);
}

void TestDataGenerator::cleanupTestData() {
    m_generatedFiles.clear();
    // QTemporaryDir will clean up automatically
}

// PerformanceTimer Implementation
PerformanceTimer::PerformanceTimer()
    : m_startMemory(0), m_peakMemory(0), m_resourceMonitoring(false) {
}

PerformanceTimer::~PerformanceTimer() {
}

void PerformanceTimer::start() {
    m_timer.start();
    m_startTime = QDateTime::currentDateTime();
    if (m_resourceMonitoring) {
        m_startMemory = getCurrentMemoryUsage();
        m_peakMemory = m_startMemory;
    }
}

void PerformanceTimer::stop() {
    m_endTime = QDateTime::currentDateTime();
    if (m_resourceMonitoring) {
        updateMemoryUsage();
    }
}

void PerformanceTimer::reset() {
    m_timer.invalidate();
    m_runTimes.clear();
    m_startMemory = 0;
    m_peakMemory = 0;
}

void PerformanceTimer::startRun() {
    start();
}

void PerformanceTimer::endRun() {
    stop();
    if (m_timer.isValid()) {
        m_runTimes.append(static_cast<double>(m_timer.elapsed()));
    }
}

void PerformanceTimer::beginSeries(const QString& name, int expectedRuns) {
    m_seriesName = name;
    m_runTimes.clear();
    m_runTimes.reserve(expectedRuns);
}

void PerformanceTimer::endSeries() {
    m_seriesName.clear();
}

double PerformanceTimer::elapsedTime() const {
    return m_timer.isValid() ? static_cast<double>(m_timer.elapsed()) : 0.0;
}

double PerformanceTimer::averageTime() const {
    if (m_runTimes.isEmpty()) return 0.0;
    
    double sum = 0.0;
    for (double time : m_runTimes) {
        sum += time;
    }
    return sum / static_cast<double>(m_runTimes.size());
}

double PerformanceTimer::medianTime() const {
    return calculateMedian();
}

double PerformanceTimer::standardDeviation() const {
    return calculateStandardDeviation();
}

double PerformanceTimer::confidenceInterval95() const {
    return calculateConfidenceInterval();
}

void PerformanceTimer::enableResourceMonitoring(bool enabled) {
    m_resourceMonitoring = enabled;
}

qint64 PerformanceTimer::getCurrentMemoryUsage() const {
    // Simplified memory usage estimation
    // In a real implementation, this would use platform-specific APIs
    return QCoreApplication::instance() ? 10 * 1024 * 1024 : 0; // 10MB default
}

qint64 PerformanceTimer::getPeakMemoryUsage() const {
    return m_peakMemory;
}

double PerformanceTimer::getCPUUsage() const {
    // Simplified CPU usage estimation
    return 0.0;
}

double PerformanceTimer::calculateStandardDeviation() const {
    if (m_runTimes.size() < 2) return 0.0;
    
    double mean = averageTime();
    double sumSquaredDiffs = 0.0;
    
    for (double time : m_runTimes) {
        double diff = time - mean;
        sumSquaredDiffs += diff * diff;
    }
    
    double variance = sumSquaredDiffs / (static_cast<double>(m_runTimes.size()) - 1.0);
    return qSqrt(variance);
}

double PerformanceTimer::calculateMedian() const {
    if (m_runTimes.isEmpty()) return 0.0;
    
    QVector<double> sortedTimes = m_runTimes;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    
    int size = static_cast<int>(sortedTimes.size());
    if (size % 2 == 0) {
        return (sortedTimes[size/2 - 1] + sortedTimes[size/2]) / 2.0;
    } else {
        return sortedTimes[size/2];
    }
}

double PerformanceTimer::calculateConfidenceInterval() const {
    if (m_runTimes.size() < 3) return 0.0;
    
    // 95% confidence interval using t-distribution approximation
    double stdDev = calculateStandardDeviation();
    double tValue = 2.0; // Approximation for 95% confidence
    
    return tValue * (stdDev / qSqrt(static_cast<double>(m_runTimes.size())));
}

void PerformanceTimer::updateMemoryUsage() {
    if (m_resourceMonitoring) {
        qint64 currentMemory = getCurrentMemoryUsage();
        m_peakMemory = qMax(m_peakMemory, currentMemory);
    }
}

PerformanceResult PerformanceTimer::getResult() const {
    PerformanceResult result;
    result.testName = m_testName;
    result.category = m_category;
    result.timestamp = m_startTime;
    
    result.executionTime = elapsedTime();
    result.meanTime = averageTime();
    result.medianTime = medianTime();
    result.stdDeviation = standardDeviation();
    result.confidenceInterval95 = confidenceInterval95();
    
    if (!m_runTimes.isEmpty()) {
        result.minTime = *std::min_element(m_runTimes.begin(), m_runTimes.end());
        result.maxTime = *std::max_element(m_runTimes.begin(), m_runTimes.end());
    }
    
    result.peakMemoryUsage = m_peakMemory;
    result.executionTimes = m_runTimes;
    
    return result;
}

// BenchmarkRunner Implementation
BenchmarkRunner::BenchmarkRunner(QObject* parent)
    : QObject(parent), m_timeoutTimer(new QTimer(this)) {
    
    m_systemInfo = SystemProfiler::getSystemInfo();
    
    // Default configuration
    BenchmarkConfig config;
    config.resultsPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/performance_results";
    setBenchmarkConfig(config);
    
    connect(m_timeoutTimer, &QTimer::timeout, this, &BenchmarkRunner::onBenchmarkTimeout);
}

BenchmarkRunner::~BenchmarkRunner() {
}

void BenchmarkRunner::setBenchmarkConfig(const BenchmarkConfig& config) {
    m_config = config;
    
    // Ensure results directory exists
    if (!m_config.resultsPath.isEmpty()) {
        QDir().mkpath(m_config.resultsPath);
    }
}

void BenchmarkRunner::registerBenchmark(const QString& name, const QString& category, 
                                      std::function<bool()> testFunction) {
    BenchmarkInfo info;
    info.name = name;
    info.category = category;
    info.testFunction = testFunction;
    
    m_benchmarks.append(info);
}

void BenchmarkRunner::registerSetup(std::function<bool()> setupFunction) {
    m_setupFunction = setupFunction;
}

void BenchmarkRunner::registerTeardown(std::function<void()> teardownFunction) {
    m_teardownFunction = teardownFunction;
}

bool BenchmarkRunner::runAllBenchmarks() {
    qDebug() << "=== Starting Performance Benchmarks ===";
    SystemProfiler::logSystemInfo();
    
    m_results.clear();
    
    // Run setup if provided
    if (m_setupFunction && !m_setupFunction()) {
        qWarning() << "Benchmark setup failed";
        return false;
    }
    
    int completed = 0;
    for (const BenchmarkInfo& benchmark : m_benchmarks) {
        emit progressUpdate(completed, static_cast<int>(m_benchmarks.size()));
        
        if (runSingleBenchmark(benchmark)) {
            completed++;
        }
        
        QCoreApplication::processEvents(); // Keep UI responsive
    }
    
    // Run teardown if provided
    if (m_teardownFunction) {
        m_teardownFunction();
    }
    
    emit progressUpdate(static_cast<int>(m_benchmarks.size()), static_cast<int>(m_benchmarks.size()));
    emit allBenchmarksCompleted();
    
    // Save results
    if (m_config.saveResults) {
        saveResultsToFile();
    }
    
    logResults();
    
    qDebug() << "=== Benchmarks Complete ===";
    return completed == m_benchmarks.size();
}

bool BenchmarkRunner::runSingleBenchmark(const BenchmarkInfo& benchmark) {
    qDebug() << "Running benchmark:" << benchmark.name;
    
    emit benchmarkStarted(benchmark.name);
    
    try {
        // Warmup runs
        if (m_config.warmupRuns > 0) {
            performWarmupRuns(benchmark);
        }
        
        // Actual measurement
        PerformanceResult result = measureBenchmark(benchmark);
        
        // Validate result
        if (validateResult(result)) {
            m_results.append(result);
            emit benchmarkCompleted(benchmark.name, result);
            
            qDebug() << "✅" << benchmark.name << "completed:"
                     << formatDuration(result.meanTime)
                     << "(" << formatThroughput(result.throughputMBps) << ")";
            return true;
        }
    } catch (const std::exception& e) {
        qWarning() << "❌ Benchmark" << benchmark.name << "failed with exception:" << e.what();
    } catch (...) {
        qWarning() << "❌ Benchmark" << benchmark.name << "failed with unknown exception";
    }
    
    return false;
}

void BenchmarkRunner::performWarmupRuns(const BenchmarkInfo& benchmark) {
    for (int i = 0; i < m_config.warmupRuns; ++i) {
        benchmark.testFunction();
        QCoreApplication::processEvents();
    }
}

PerformanceResult BenchmarkRunner::measureBenchmark(const BenchmarkInfo& benchmark) {
    PerformanceTimer timer;
    timer.setTestName(benchmark.name);
    timer.setCategory(benchmark.category);
    timer.enableResourceMonitoring(m_config.enableResourceMonitoring);
    
    // Set timeout
    m_timeoutTimer->start(m_config.maxRunTimeSeconds * 1000);
    
    timer.beginSeries(benchmark.name, m_config.measurementRuns);
    
    for (int i = 0; i < m_config.measurementRuns; ++i) {
        timer.startRun();
        bool success = benchmark.testFunction();
        timer.endRun();
        
        if (!success) {
            qWarning() << "Benchmark" << benchmark.name << "run" << i << "failed";
        }
        
        QCoreApplication::processEvents();
        
        if (!m_timeoutTimer->isActive()) {
            qWarning() << "Benchmark" << benchmark.name << "timed out";
            break;
        }
    }
    
    timer.endSeries();
    m_timeoutTimer->stop();
    
    PerformanceResult result = timer.getResult();
    result.systemInfo = SystemProfiler::getSystemSummary();
    result.qtVersion = QT_VERSION_STR;
    result.buildConfig = m_systemInfo.buildType;
    result.threadCount = m_systemInfo.logicalProcessors;
    
    return result;
}

bool BenchmarkRunner::validateResult(const PerformanceResult& result) const {
    // Check for reasonable execution times
    if (result.meanTime <= 0 || result.meanTime > m_config.maxRunTimeSeconds * 1000) {
        return false;
    }
    
    // Check for excessive variation if statistics are enabled
    if (m_config.enableStatistics && result.executionTimes.size() > 1) {
        double cv = result.stdDeviation / result.meanTime; // Coefficient of variation
        if (cv > m_config.acceptableVariation * 2) { // Allow double the acceptable variation
            qWarning() << "High variation detected in" << result.testName << "CV:" << cv;
        }
    }
    
    return result.validationPassed;
}

void BenchmarkRunner::onBenchmarkTimeout() {
    qWarning() << "Benchmark timeout occurred";
    m_timeoutTimer->stop();
}

QString BenchmarkRunner::formatDuration(double milliseconds) const {
    if (milliseconds < 1.0) {
        return QString("%1 μs").arg(milliseconds * 1000, 0, 'f', 1);
    } else if (milliseconds < 1000.0) {
        return QString("%1 ms").arg(milliseconds, 0, 'f', 2);
    } else {
        return QString("%1 s").arg(milliseconds / 1000.0, 0, 'f', 2);
    }
}

QString BenchmarkRunner::formatThroughput(double mbps) const {
    if (mbps < 1.0) {
        return QString("%1 KB/s").arg(mbps * 1024, 0, 'f', 1);
    } else if (mbps < 1024.0) {
        return QString("%1 MB/s").arg(mbps, 0, 'f', 1);
    } else {
        return QString("%1 GB/s").arg(mbps / 1024.0, 0, 'f', 2);
    }
}

QString BenchmarkRunner::formatMemory(qint64 bytes) const {
    if (bytes < 1024) {
        return QString("%1 B").arg(bytes);
    } else if (bytes < 1024 * 1024) {
        return QString("%1 KB").arg(static_cast<double>(bytes) / 1024.0, 0, 'f', 1);
    } else if (bytes < 1024 * 1024 * 1024) {
        return QString("%1 MB").arg(static_cast<double>(bytes) / (1024.0 * 1024.0), 0, 'f', 1);
    } else {
        return QString("%1 GB").arg(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2);
    }
}

bool BenchmarkRunner::saveResultsToFile(const QString& filePath) const {
    QString outputPath = filePath.isEmpty() ? 
        QDir(m_config.resultsPath).absoluteFilePath("benchmark_results.json") : filePath;
    
    QJsonObject rootObj;
    rootObj["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    rootObj["systemInfo"] = m_systemInfo.toJson();
    
    QJsonArray resultsArray;
    for (const PerformanceResult& result : m_results) {
        resultsArray.append(result.toJson());
    }
    rootObj["results"] = resultsArray;
    
    QFile file(outputPath);
    if (file.open(QIODevice::WriteOnly)) {
        QJsonDocument doc(rootObj);
        file.write(doc.toJson());
        file.close();
        
        qDebug() << "Performance results saved to:" << outputPath;
        return true;
    }
    
    qWarning() << "Failed to save results to:" << outputPath;
    return false;
}

void BenchmarkRunner::logResults() const {
    qDebug() << "\n=== Benchmark Results Summary ===";
    
    for (const PerformanceResult& result : m_results) {
        qDebug() << QString("%1: %2 (±%3)")
            .arg(result.testName, -30)
            .arg(formatDuration(result.meanTime))
            .arg(formatDuration(result.confidenceInterval95));
    }
    
    qDebug() << "================================\n";
}

// PerformanceTestCase Implementation
PerformanceTestCase::PerformanceTestCase(QObject* parent)
    : QObject(parent) {
    m_systemInfo = SystemProfiler::getSystemInfo();
}

PerformanceTestCase::~PerformanceTestCase() {
}

void PerformanceTestCase::initTestCase() {
    SystemProfiler::logSystemInfo();
    setupTestEnvironment();
}

void PerformanceTestCase::cleanupTestCase() {
    cleanupTestEnvironment();
}

bool PerformanceTestCase::setupTestEnvironment() {
    return dataGenerator()->generateMixedSizesScenario(100);
}

void PerformanceTestCase::cleanupTestEnvironment() {
    dataGenerator()->cleanupTestData();
}

void PerformanceTestCase::measureFunction(const QString& testName, std::function<void()> func) {
    benchmarkRunner()->registerBenchmark(testName, "function", [func]() -> bool {
        func();
        return true;
    });
}

void PerformanceTestCase::measureThroughput(const QString& testName, qint64 dataSize, std::function<void()> func) {
    benchmarkRunner()->registerBenchmark(testName, "throughput", [func, dataSize, testName]() -> bool {
        PerformanceTimer timer;
        timer.start();
        func();
        timer.stop();
        
        double throughput = (static_cast<double>(dataSize) / 1024.0 / 1024.0) / (timer.elapsedTime() / 1000.0);
        qDebug() << testName << "throughput:" << throughput << "MB/s";
        
        return true;
    });
}

void PerformanceTestCase::QVERIFY_PERFORMANCE(bool condition, const QString& message) {
    if (!condition) {
        QString errorMsg = message.isEmpty() ? "Performance condition failed" : message;
        qWarning() << "PERFORMANCE FAILURE:" << errorMsg;
        QVERIFY(condition);
    }
}

void PerformanceTestCase::QCOMPARE_PERFORMANCE(double actual, double expected, double tolerance, const QString& metric) {
    double diff = qAbs(actual - expected);
    double relativeDiff = diff / expected;
    
    if (relativeDiff > tolerance) {
        QString metricName = metric.isEmpty() ? "metric" : metric;
        qWarning() << QString("PERFORMANCE COMPARISON FAILED: %1 = %2, expected %3 (±%4%)")
            .arg(metricName).arg(actual).arg(expected).arg(tolerance * 100);
        QVERIFY(relativeDiff <= tolerance);
    }
}

} // namespace PerformanceTest

#include "performance_test_framework.moc"