#include "performance_benchmark.h"
#include "test_data_generator.h"
#include <QApplication>
#include <QWidget>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QJsonArray>
#include <QJsonDocument>
#include <QCryptographicHash>
#include <QThread>
#include <QProcess>
#include <QSysInfo>
#include <QDebug>
#include <QtMath>
#include <QRandomGenerator>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#elif defined(Q_OS_LINUX)
#include <unistd.h>
#include <sys/resource.h>
#elif defined(Q_OS_MAC)
#include <mach/mach.h>
#endif

PerformanceBenchmark::PerformanceBenchmark(QObject* parent)
    : QObject(parent)
    , m_resourceTimer(new QTimer(this))
    , m_resourceMonitoringActive(false)
{
    // Set default configuration
    m_config.name = "DefaultBenchmark";
    m_config.iterations = 1;
    m_config.warmupIterations = 0;
    m_config.timeoutMs = 60000;
    m_config.measureMemory = true;
    m_config.measureCpu = true;
    m_config.samplingIntervalMs = 100;
    
    // Setup resource monitoring timer
    connect(m_resourceTimer, &QTimer::timeout, this, &PerformanceBenchmark::onResourceMonitoringTimer);
    
    // Load existing baselines
    loadBaselines();
}

PerformanceBenchmark::~PerformanceBenchmark() {
    stopResourceMonitoring();
    saveBaselines();
}

void PerformanceBenchmark::setBenchmarkConfig(const BenchmarkConfig& config) {
    QMutexLocker locker(&m_resultsMutex);
    m_config = config;
}

PerformanceBenchmark::BenchmarkConfig PerformanceBenchmark::getBenchmarkConfig() const {
    QMutexLocker locker(&m_resultsMutex);
    return m_config;
}

bool PerformanceBenchmark::runBenchmark(const QString& name, std::function<void()> benchmarkFunction) {
    if (!benchmarkFunction) {
        qWarning() << "Invalid benchmark function for:" << name;
        return false;
    }
    
    emit benchmarkStarted(name);
    
    QList<PerformanceResult> benchmarkResults;
    
    // Warmup iterations
    for (int i = 0; i < m_config.warmupIterations; ++i) {
        try {
            benchmarkFunction();
        } catch (const std::exception& e) {
            qWarning() << "Exception during warmup iteration" << i << ":" << e.what();
        }
        QApplication::processEvents();
    }
    
    // Actual benchmark iterations
    for (int iteration = 0; iteration < m_config.iterations; ++iteration) {
        // Start resource monitoring for this iteration
        if (m_config.measureMemory || m_config.measureCpu) {
            startResourceMonitoring();
        }
        
        // Measure execution time
        QElapsedTimer timer;
        timer.start();
        
        qint64 memoryBefore = m_config.measureMemory ? getCurrentMemoryUsage() : 0;
        
        try {
            benchmarkFunction();
        } catch (const std::exception& e) {
            qWarning() << "Exception during benchmark iteration" << iteration << ":" << e.what();
            stopResourceMonitoring();
            return false;
        }
        
        qint64 executionTime = timer.elapsed();
        qint64 memoryAfter = m_config.measureMemory ? getCurrentMemoryUsage() : 0;
        
        // Stop resource monitoring
        if (m_config.measureMemory || m_config.measureCpu) {
            stopResourceMonitoring();
        }
        
        // Record execution time
        PerformanceResult timeResult = createResult(name, MetricType::ExecutionTime, 
                                                  "execution_time", executionTime, "ms");
        timeResult.iteration = iteration;
        recordResult(timeResult);
        benchmarkResults.append(timeResult);
        
        // Record memory usage
        if (m_config.measureMemory && memoryAfter > memoryBefore) {
            qint64 memoryUsed = memoryAfter - memoryBefore;
            PerformanceResult memoryResult = createResult(name, MetricType::MemoryUsage, 
                                                        "memory_usage", memoryUsed, "bytes");
            memoryResult.iteration = iteration;
            recordResult(memoryResult);
            benchmarkResults.append(memoryResult);
        }
        
        // Check timeout
        if (executionTime > m_config.timeoutMs) {
            qWarning() << "Benchmark" << name << "exceeded timeout on iteration" << iteration;
            break;
        }
        
        QApplication::processEvents();
    }
    
    emit benchmarkCompleted(name, benchmarkResults);
    return true;
}

bool PerformanceBenchmark::runBenchmark(const QString& name, std::function<void(int)> benchmarkFunction) {
    return runBenchmark(name, [benchmarkFunction]() {
        benchmarkFunction(0); // Call with iteration 0 for compatibility
    });
}

void PerformanceBenchmark::startMeasurement(const QString& measurementName) {
    QMutexLocker locker(&m_resultsMutex);
    m_activeTimers[measurementName].start();
}

void PerformanceBenchmark::stopMeasurement(const QString& measurementName) {
    QMutexLocker locker(&m_resultsMutex);
    
    if (!m_activeTimers.contains(measurementName)) {
        qWarning() << "No active measurement found for:" << measurementName;
        return;
    }
    
    qint64 elapsed = m_activeTimers[measurementName].elapsed();
    m_activeTimers.remove(measurementName);
    
    // Record the measurement
    recordExecutionTime(measurementName, elapsed);
}

void PerformanceBenchmark::recordMetric(const QString& metricName, double value, const QString& unit, MetricType type) {
    PerformanceResult result = createResult(m_config.name, type, metricName, value, unit);
    recordResult(result);
    emit measurementRecorded(result);
}

void PerformanceBenchmark::recordExecutionTime(const QString& operationName, qint64 timeMs) {
    recordMetric(operationName, timeMs, "ms", MetricType::ExecutionTime);
}

void PerformanceBenchmark::recordMemoryUsage(const QString& operationName, qint64 memoryBytes) {
    recordMetric(operationName, memoryBytes, "bytes", MetricType::MemoryUsage);
}

void PerformanceBenchmark::recordThroughput(const QString& operationName, double operationsPerSecond) {
    recordMetric(operationName, operationsPerSecond, "ops/sec", MetricType::ThroughputRate);
}void Perf
ormanceBenchmark::startResourceMonitoring() {
    if (m_resourceMonitoringActive) {
        return;
    }
    
    m_resourceMonitoringActive = true;
    m_resourceResults.clear();
    m_resourceTimer->start(m_config.samplingIntervalMs);
}

void PerformanceBenchmark::stopResourceMonitoring() {
    if (!m_resourceMonitoringActive) {
        return;
    }
    
    m_resourceMonitoringActive = false;
    m_resourceTimer->stop();
}

QMap<QString, QVariant> PerformanceBenchmark::getCurrentResourceUsage() const {
    QMap<QString, QVariant> resources;
    
    if (m_config.measureMemory) {
        resources["memory_usage"] = getCurrentMemoryUsage();
    }
    
    if (m_config.measureCpu) {
        resources["cpu_usage"] = getCurrentCpuUsage();
    }
    
    if (m_config.measureDiskIO) {
        QMap<QString, qint64> diskStats = getDiskIOStats();
        for (auto it = diskStats.begin(); it != diskStats.end(); ++it) {
            resources["disk_" + it.key()] = it.value();
        }
    }
    
    return resources;
}

QList<PerformanceBenchmark::PerformanceResult> PerformanceBenchmark::getResourceMonitoringResults() const {
    QMutexLocker locker(&m_resultsMutex);
    return m_resourceResults;
}

bool PerformanceBenchmark::benchmarkFileOperations(const QString& testDirectory, int fileCount, qint64 fileSize) {
    QString benchmarkName = QString("file_operations_%1_files_%2_bytes").arg(fileCount).arg(fileSize);
    
    return runBenchmark(benchmarkName, [this, testDirectory, fileCount, fileSize]() {
        // Create test directory
        QDir().mkpath(testDirectory);
        
        // Measure file creation
        qint64 creationTime = measureFileCreation(testDirectory, fileCount, fileSize);
        recordExecutionTime("file_creation", creationTime);
        
        // Get list of created files
        QDir dir(testDirectory);
        QStringList files = dir.entryList(QDir::Files);
        QStringList filePaths;
        for (const QString& file : files) {
            filePaths.append(dir.absoluteFilePath(file));
        }
        
        // Measure file reading
        qint64 readingTime = measureFileReading(filePaths);
        recordExecutionTime("file_reading", readingTime);
        
        // Calculate throughput
        qint64 totalBytes = fileCount * fileSize;
        double readThroughput = (totalBytes / 1024.0 / 1024.0) / (readingTime / 1000.0); // MB/s
        recordThroughput("read_throughput", readThroughput);
        
        // Cleanup
        for (const QString& filePath : filePaths) {
            QFile::remove(filePath);
        }
        QDir().rmdir(testDirectory);
    });
}

bool PerformanceBenchmark::benchmarkDirectoryScanning(const QString& directory, bool recursive) {
    QString benchmarkName = QString("directory_scan_%1").arg(recursive ? "recursive" : "flat");
    
    return runBenchmark(benchmarkName, [this, directory, recursive]() {
        qint64 scanTime = measureDirectoryTraversal(directory, recursive);
        recordExecutionTime("directory_scan", scanTime);
        
        // Count files and directories
        QDir dir(directory);
        QDirIterator::IteratorFlags flags = recursive ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags;
        QDirIterator iterator(directory, QDir::AllEntries | QDir::NoDotAndDotDot, flags);
        
        int fileCount = 0;
        while (iterator.hasNext()) {
            iterator.next();
            fileCount++;
        }
        
        double scanRate = fileCount / (scanTime / 1000.0); // files per second
        recordThroughput("scan_rate", scanRate);
    });
}

bool PerformanceBenchmark::benchmarkHashCalculation(const QStringList& filePaths, const QString& algorithm) {
    QString benchmarkName = QString("hash_calculation_%1").arg(algorithm.toLower());
    
    return runBenchmark(benchmarkName, [this, filePaths, algorithm]() {
        QElapsedTimer timer;
        timer.start();
        
        qint64 totalBytes = 0;
        QCryptographicHash::Algorithm hashAlgorithm = QCryptographicHash::Md5;
        
        if (algorithm.toUpper() == "SHA1") {
            hashAlgorithm = QCryptographicHash::Sha1;
        } else if (algorithm.toUpper() == "SHA256") {
            hashAlgorithm = QCryptographicHash::Sha256;
        }
        
        for (const QString& filePath : filePaths) {
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(hashAlgorithm);
                hash.addData(&file);
                hash.result(); // Force calculation
                totalBytes += file.size();
            }
        }
        
        qint64 hashTime = timer.elapsed();
        recordExecutionTime("hash_calculation", hashTime);
        
        if (hashTime > 0) {
            double hashThroughput = (totalBytes / 1024.0 / 1024.0) / (hashTime / 1000.0); // MB/s
            recordThroughput("hash_throughput", hashThroughput);
        }
    });
}

bool PerformanceBenchmark::benchmarkDuplicateDetection(const QString& testDirectory) {
    QString benchmarkName = "duplicate_detection";
    
    return runBenchmark(benchmarkName, [this, testDirectory]() {
        QElapsedTimer timer;
        timer.start();
        
        // Simulate duplicate detection process
        QDir dir(testDirectory);
        QStringList files = dir.entryList(QDir::Files, QDir::Name);
        
        // Hash all files
        QMap<QString, QStringList> hashGroups;
        for (const QString& fileName : files) {
            QString filePath = dir.absoluteFilePath(fileName);
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(QCryptographicHash::Md5);
                hash.addData(&file);
                QString hashString = hash.result().toHex();
                hashGroups[hashString].append(filePath);
            }
        }
        
        // Count duplicates
        int duplicateGroups = 0;
        int totalDuplicates = 0;
        for (auto it = hashGroups.begin(); it != hashGroups.end(); ++it) {
            if (it.value().size() > 1) {
                duplicateGroups++;
                totalDuplicates += it.value().size();
            }
        }
        
        qint64 detectionTime = timer.elapsed();
        recordExecutionTime("duplicate_detection", detectionTime);
        recordMetric("duplicate_groups_found", duplicateGroups, "count", MetricType::Custom);
        recordMetric("total_duplicates_found", totalDuplicates, "count", MetricType::Custom);
        
        if (detectionTime > 0) {
            double detectionRate = files.size() / (detectionTime / 1000.0); // files per second
            recordThroughput("detection_rate", detectionRate);
        }
    });
}

bool PerformanceBenchmark::benchmarkUIResponsiveness(QWidget* widget, int operationCount) {
    if (!widget) {
        return false;
    }
    
    QString benchmarkName = QString("ui_responsiveness_%1_ops").arg(operationCount);
    
    return runBenchmark(benchmarkName, [this, widget, operationCount]() {
        qint64 totalTime = measureWidgetUpdate(widget, operationCount);
        recordExecutionTime("ui_updates", totalTime);
        
        double updateRate = operationCount / (totalTime / 1000.0); // updates per second
        recordThroughput("ui_update_rate", updateRate);
        
        // Measure frame rate
        double frameRate = measureFrameRate(widget, 1000); // 1 second measurement
        recordMetric("frame_rate", frameRate, "fps", MetricType::UIResponsiveness);
    });
}

bool PerformanceBenchmark::benchmarkWidgetRendering(QWidget* widget, int frameCount) {
    if (!widget) {
        return false;
    }
    
    QString benchmarkName = QString("widget_rendering_%1_frames").arg(frameCount);
    
    return runBenchmark(benchmarkName, [this, widget, frameCount]() {
        QElapsedTimer timer;
        timer.start();
        
        for (int i = 0; i < frameCount; ++i) {
            widget->update();
            QApplication::processEvents();
        }
        
        qint64 renderTime = timer.elapsed();
        recordExecutionTime("widget_rendering", renderTime);
        
        if (renderTime > 0) {
            double frameRate = (frameCount * 1000.0) / renderTime; // frames per second
            recordMetric("rendering_fps", frameRate, "fps", MetricType::UIResponsiveness);
        }
    });
}

bool PerformanceBenchmark::benchmarkThemeSwitching(QWidget* widget, const QStringList& themes) {
    if (!widget || themes.isEmpty()) {
        return false;
    }
    
    QString benchmarkName = QString("theme_switching_%1_themes").arg(themes.size());
    
    return runBenchmark(benchmarkName, [this, widget, themes]() {
        QElapsedTimer timer;
        timer.start();
        
        for (const QString& theme : themes) {
            // Simulate theme switching (would need actual theme manager integration)
            widget->setStyleSheet(QString("/* Theme: %1 */").arg(theme));
            widget->update();
            QApplication::processEvents();
        }
        
        qint64 switchTime = timer.elapsed();
        recordExecutionTime("theme_switching", switchTime);
        
        if (switchTime > 0) {
            double switchRate = themes.size() / (switchTime / 1000.0); // switches per second
            recordThroughput("theme_switch_rate", switchRate);
        }
    });
}Perfor
manceBenchmark::BenchmarkStatistics PerformanceBenchmark::calculateStatistics(const QString& benchmarkName, const QString& metricName) const {
    QMutexLocker locker(&m_resultsMutex);
    
    BenchmarkStatistics stats;
    stats.benchmarkName = benchmarkName;
    stats.metricName = metricName;
    
    QList<double> values;
    for (const PerformanceResult& result : m_results) {
        if ((benchmarkName.isEmpty() || result.benchmarkName == benchmarkName) &&
            (metricName.isEmpty() || result.metricName == metricName)) {
            values.append(result.value);
            if (stats.unit.isEmpty()) {
                stats.unit = result.unit;
            }
        }
    }
    
    if (values.isEmpty()) {
        return stats;
    }
    
    stats.sampleCount = values.size();
    stats.rawValues = values;
    
    // Calculate basic statistics
    stats.mean = calculateMean(values);
    stats.median = calculateMedian(values);
    stats.standardDeviation = calculateStandardDeviation(values, stats.mean);
    
    std::sort(values.begin(), values.end());
    stats.minimum = values.first();
    stats.maximum = values.last();
    
    stats.percentile95 = calculatePercentile(values, 95.0);
    stats.percentile99 = calculatePercentile(values, 99.0);
    
    return stats;
}

QList<PerformanceBenchmark::BenchmarkStatistics> PerformanceBenchmark::calculateAllStatistics() const {
    QMutexLocker locker(&m_resultsMutex);
    
    QList<BenchmarkStatistics> allStats;
    QSet<QPair<QString, QString>> uniquePairs;
    
    // Collect unique benchmark/metric pairs
    for (const PerformanceResult& result : m_results) {
        QPair<QString, QString> pair(result.benchmarkName, result.metricName);
        uniquePairs.insert(pair);
    }
    
    // Calculate statistics for each pair
    for (const auto& pair : uniquePairs) {
        BenchmarkStatistics stats = calculateStatistics(pair.first, pair.second);
        if (stats.sampleCount > 0) {
            allStats.append(stats);
        }
    }
    
    return allStats;
}

QMap<QString, PerformanceBenchmark::BenchmarkStatistics> PerformanceBenchmark::groupStatisticsByBenchmark() const {
    QMap<QString, BenchmarkStatistics> groupedStats;
    QList<BenchmarkStatistics> allStats = calculateAllStatistics();
    
    for (const BenchmarkStatistics& stats : allStats) {
        QString key = QString("%1_%2").arg(stats.benchmarkName, stats.metricName);
        groupedStats[key] = stats;
    }
    
    return groupedStats;
}

bool PerformanceBenchmark::createBaseline(const QString& name, const QString& benchmarkName, const QString& metricName) {
    QMutexLocker locker(&m_baselinesMutex);
    
    // Calculate current statistics for the metric
    BenchmarkStatistics stats = calculateStatistics(benchmarkName, metricName);
    if (stats.sampleCount == 0) {
        qWarning() << "No data available for baseline creation:" << benchmarkName << metricName;
        return false;
    }
    
    PerformanceBaseline baseline;
    baseline.name = name;
    baseline.benchmarkName = benchmarkName;
    baseline.metricName = metricName;
    baseline.expectedValue = stats.mean;
    baseline.tolerancePercent = 10.0; // Default 10% tolerance
    baseline.warningThreshold = 5.0;  // Default 5% warning threshold
    baseline.created = QDateTime::currentDateTime();
    baseline.platform = getPlatformIdentifier();
    baseline.version = "1.0"; // Would be replaced with actual version
    baseline.environment = getEnvironmentInfo();
    
    // Remove existing baseline with same name
    for (int i = 0; i < m_baselines.size(); ++i) {
        if (m_baselines[i].name == name) {
            m_baselines.removeAt(i);
            break;
        }
    }
    
    m_baselines.append(baseline);
    emit baselineCreated(name);
    return true;
}

bool PerformanceBenchmark::updateBaseline(const QString& name, double newValue) {
    QMutexLocker locker(&m_baselinesMutex);
    
    for (PerformanceBaseline& baseline : m_baselines) {
        if (baseline.name == name) {
            baseline.expectedValue = newValue;
            emit baselineUpdated(name);
            return true;
        }
    }
    
    qWarning() << "Baseline not found:" << name;
    return false;
}

bool PerformanceBenchmark::deleteBaseline(const QString& name) {
    QMutexLocker locker(&m_baselinesMutex);
    
    for (int i = 0; i < m_baselines.size(); ++i) {
        if (m_baselines[i].name == name) {
            m_baselines.removeAt(i);
            return true;
        }
    }
    
    return false;
}

QList<PerformanceBenchmark::PerformanceBaseline> PerformanceBenchmark::getBaselines() const {
    QMutexLocker locker(&m_baselinesMutex);
    return m_baselines;
}

PerformanceBenchmark::PerformanceBaseline PerformanceBenchmark::getBaseline(const QString& name) const {
    QMutexLocker locker(&m_baselinesMutex);
    
    for (const PerformanceBaseline& baseline : m_baselines) {
        if (baseline.name == name) {
            return baseline;
        }
    }
    
    return PerformanceBaseline(); // Return empty baseline if not found
}

PerformanceBenchmark::ComparisonResult PerformanceBenchmark::compareWithBaseline(const QString& baselineName, const QString& benchmarkName, const QString& metricName) const {
    ComparisonResult result;
    result.benchmarkName = benchmarkName;
    result.metricName = metricName;
    
    // Get baseline
    PerformanceBaseline baseline = getBaseline(baselineName);
    if (baseline.name.isEmpty()) {
        result.status = "Baseline not found";
        return result;
    }
    
    // Get current statistics
    BenchmarkStatistics stats = calculateStatistics(benchmarkName, metricName);
    if (stats.sampleCount == 0) {
        result.status = "No current data available";
        return result;
    }
    
    result.currentValue = stats.mean;
    result.baselineValue = baseline.expectedValue;
    
    if (baseline.expectedValue != 0) {
        result.deviationPercent = ((stats.mean - baseline.expectedValue) / baseline.expectedValue) * 100.0;
    }
    
    result.withinTolerance = qAbs(result.deviationPercent) <= baseline.tolerancePercent;
    result.isRegression = result.deviationPercent > baseline.tolerancePercent;
    result.isImprovement = result.deviationPercent < -baseline.warningThreshold;
    
    if (result.isRegression) {
        result.status = "Performance Regression";
        result.recommendation = QString("Performance degraded by %1%. Investigate recent changes.").arg(result.deviationPercent, 0, 'f', 2);
        emit performanceRegression(result);
    } else if (result.isImprovement) {
        result.status = "Performance Improvement";
        result.recommendation = QString("Performance improved by %1%. Consider updating baseline.").arg(-result.deviationPercent, 0, 'f', 2);
        emit performanceImprovement(result);
    } else if (result.withinTolerance) {
        result.status = "Within Tolerance";
        result.recommendation = "Performance is within acceptable range.";
    } else {
        result.status = "Warning";
        result.recommendation = QString("Performance deviation of %1% detected. Monitor closely.").arg(result.deviationPercent, 0, 'f', 2);
    }
    
    return result;
}

QList<PerformanceBenchmark::ComparisonResult> PerformanceBenchmark::compareAllWithBaselines() const {
    QList<ComparisonResult> results;
    
    QMutexLocker locker(&m_baselinesMutex);
    for (const PerformanceBaseline& baseline : m_baselines) {
        ComparisonResult result = compareWithBaseline(baseline.name, baseline.benchmarkName, baseline.metricName);
        if (!result.status.isEmpty()) {
            results.append(result);
        }
    }
    
    return results;
}

bool PerformanceBenchmark::detectPerformanceRegressions(double regressionThreshold) const {
    QList<ComparisonResult> results = compareAllWithBaselines();
    
    for (const ComparisonResult& result : results) {
        if (result.deviationPercent > regressionThreshold) {
            return true;
        }
    }
    
    return false;
}QJ
sonObject PerformanceBenchmark::generateReport() const {
    QMutexLocker locker(&m_resultsMutex);
    
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["platform"] = getPlatformIdentifier();
    report["environment"] = QJsonObject::fromVariantMap(getEnvironmentInfo());
    
    // Add configuration
    QJsonObject config;
    config["name"] = m_config.name;
    config["iterations"] = m_config.iterations;
    config["warmup_iterations"] = m_config.warmupIterations;
    config["timeout_ms"] = m_config.timeoutMs;
    report["configuration"] = config;
    
    // Add results
    QJsonArray resultsArray;
    for (const PerformanceResult& result : m_results) {
        QJsonObject resultObj;
        resultObj["benchmark_name"] = result.benchmarkName;
        resultObj["metric_name"] = result.metricName;
        resultObj["value"] = result.value;
        resultObj["unit"] = result.unit;
        resultObj["timestamp"] = result.timestamp.toString(Qt::ISODate);
        resultObj["iteration"] = result.iteration;
        resultObj["description"] = result.description;
        resultsArray.append(resultObj);
    }
    report["results"] = resultsArray;
    
    // Add statistics
    QJsonArray statsArray;
    QList<BenchmarkStatistics> allStats = calculateAllStatistics();
    for (const BenchmarkStatistics& stats : allStats) {
        QJsonObject statsObj;
        statsObj["benchmark_name"] = stats.benchmarkName;
        statsObj["metric_name"] = stats.metricName;
        statsObj["sample_count"] = stats.sampleCount;
        statsObj["mean"] = stats.mean;
        statsObj["median"] = stats.median;
        statsObj["std_deviation"] = stats.standardDeviation;
        statsObj["minimum"] = stats.minimum;
        statsObj["maximum"] = stats.maximum;
        statsObj["percentile_95"] = stats.percentile95;
        statsObj["percentile_99"] = stats.percentile99;
        statsObj["unit"] = stats.unit;
        statsArray.append(statsObj);
    }
    report["statistics"] = statsArray;
    
    return report;
}

QJsonObject PerformanceBenchmark::generateComparisonReport(const QList<ComparisonResult>& comparisons) const {
    QJsonObject report;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["comparison_count"] = comparisons.size();
    
    QJsonArray comparisonsArray;
    for (const ComparisonResult& comparison : comparisons) {
        QJsonObject compObj;
        compObj["benchmark_name"] = comparison.benchmarkName;
        compObj["metric_name"] = comparison.metricName;
        compObj["current_value"] = comparison.currentValue;
        compObj["baseline_value"] = comparison.baselineValue;
        compObj["deviation_percent"] = comparison.deviationPercent;
        compObj["within_tolerance"] = comparison.withinTolerance;
        compObj["is_regression"] = comparison.isRegression;
        compObj["is_improvement"] = comparison.isImprovement;
        compObj["status"] = comparison.status;
        compObj["recommendation"] = comparison.recommendation;
        comparisonsArray.append(compObj);
    }
    report["comparisons"] = comparisonsArray;
    
    return report;
}

bool PerformanceBenchmark::exportResults(const QString& filePath, const QString& format) const {
    QJsonObject report = generateReport();
    
    if (format.toLower() == "json") {
        QJsonDocument doc(report);
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(doc.toJson());
            return true;
        }
    }
    
    return false;
}

bool PerformanceBenchmark::exportBaselines(const QString& filePath) const {
    QMutexLocker locker(&m_baselinesMutex);
    
    QJsonArray baselinesArray;
    for (const PerformanceBaseline& baseline : m_baselines) {
        QJsonObject baselineObj;
        baselineObj["name"] = baseline.name;
        baselineObj["benchmark_name"] = baseline.benchmarkName;
        baselineObj["metric_name"] = baseline.metricName;
        baselineObj["expected_value"] = baseline.expectedValue;
        baselineObj["tolerance_percent"] = baseline.tolerancePercent;
        baselineObj["warning_threshold"] = baseline.warningThreshold;
        baselineObj["created"] = baseline.created.toString(Qt::ISODate);
        baselineObj["platform"] = baseline.platform;
        baselineObj["version"] = baseline.version;
        baselineObj["environment"] = QJsonObject::fromVariantMap(baseline.environment);
        baselinesArray.append(baselineObj);
    }
    
    QJsonObject root;
    root["baselines"] = baselinesArray;
    root["exported"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    QJsonDocument doc(root);
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        return true;
    }
    
    return false;
}

bool PerformanceBenchmark::importBaselines(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();
    QJsonArray baselinesArray = root["baselines"].toArray();
    
    QMutexLocker locker(&m_baselinesMutex);
    
    for (const QJsonValue& value : baselinesArray) {
        QJsonObject baselineObj = value.toObject();
        
        PerformanceBaseline baseline;
        baseline.name = baselineObj["name"].toString();
        baseline.benchmarkName = baselineObj["benchmark_name"].toString();
        baseline.metricName = baselineObj["metric_name"].toString();
        baseline.expectedValue = baselineObj["expected_value"].toDouble();
        baseline.tolerancePercent = baselineObj["tolerance_percent"].toDouble();
        baseline.warningThreshold = baselineObj["warning_threshold"].toDouble();
        baseline.created = QDateTime::fromString(baselineObj["created"].toString(), Qt::ISODate);
        baseline.platform = baselineObj["platform"].toString();
        baseline.version = baselineObj["version"].toString();
        baseline.environment = baselineObj["environment"].toObject().toVariantMap();
        
        // Remove existing baseline with same name
        for (int i = 0; i < m_baselines.size(); ++i) {
            if (m_baselines[i].name == baseline.name) {
                m_baselines.removeAt(i);
                break;
            }
        }
        
        m_baselines.append(baseline);
    }
    
    return true;
}

QList<PerformanceBenchmark::PerformanceResult> PerformanceBenchmark::getResults(const QString& benchmarkName) const {
    QMutexLocker locker(&m_resultsMutex);
    
    if (benchmarkName.isEmpty()) {
        return m_results;
    }
    
    QList<PerformanceResult> filteredResults;
    for (const PerformanceResult& result : m_results) {
        if (result.benchmarkName == benchmarkName) {
            filteredResults.append(result);
        }
    }
    
    return filteredResults;
}

void PerformanceBenchmark::clearResults() {
    QMutexLocker locker(&m_resultsMutex);
    m_results.clear();
    m_resourceResults.clear();
}

void PerformanceBenchmark::clearBaselines() {
    QMutexLocker locker(&m_baselinesMutex);
    m_baselines.clear();
}

QString PerformanceBenchmark::formatDuration(qint64 milliseconds) const {
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

QString PerformanceBenchmark::formatBytes(qint64 bytes) const {
    const QStringList units = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = bytes;
    
    while (size >= 1024.0 && unitIndex < units.size() - 1) {
        size /= 1024.0;
        unitIndex++;
    }
    
    return QString("%1 %2").arg(size, 0, 'f', 2).arg(units[unitIndex]);
}

QString PerformanceBenchmark::formatRate(double rate, const QString& unit) const {
    if (rate < 1000) {
        return QString("%1 %2").arg(rate, 0, 'f', 2).arg(unit);
    } else if (rate < 1000000) {
        return QString("%1K %2").arg(rate / 1000.0, 0, 'f', 2).arg(unit);
    } else {
        return QString("%1M %2").arg(rate / 1000000.0, 0, 'f', 2).arg(unit);
    }
}

QMap<QString, QVariant> PerformanceBenchmark::getSystemInfo() const {
    QMap<QString, QVariant> info;
    
    info["os_type"] = QSysInfo::productType();
    info["os_version"] = QSysInfo::productVersion();
    info["kernel_type"] = QSysInfo::kernelType();
    info["kernel_version"] = QSysInfo::kernelVersion();
    info["cpu_architecture"] = QSysInfo::currentCpuArchitecture();
    info["machine_hostname"] = QSysInfo::machineHostName();
    info["cpu_info"] = detectCpuInfo();
    info["total_memory"] = detectTotalMemory();
    
    return info;
}

QMap<QString, QVariant> PerformanceBenchmark::getEnvironmentInfo() const {
    QMap<QString, QVariant> env;
    
    env["qt_version"] = QT_VERSION_STR;
    env["application_name"] = QApplication::applicationName();
    env["application_version"] = QApplication::applicationVersion();
    env["working_directory"] = QDir::currentPath();
    env["temp_directory"] = QDir::tempPath();
    
    return env;
}

QString PerformanceBenchmark::getPlatformIdentifier() const {
    return QString("%1_%2_%3")
        .arg(QSysInfo::kernelType())
        .arg(QSysInfo::currentCpuArchitecture())
        .arg(QSysInfo::productVersion());
}voi
d PerformanceBenchmark::onResourceMonitoringTimer() {
    if (!m_resourceMonitoringActive) {
        return;
    }
    
    QMap<QString, QVariant> resources = getCurrentResourceUsage();
    
    // Record resource measurements
    for (auto it = resources.begin(); it != resources.end(); ++it) {
        PerformanceResult result = createResult(m_config.name, MetricType::Custom, 
                                              it.key(), it.value().toDouble(), "");
        m_resourceResults.append(result);
    }
    
    emit resourceMonitoringUpdate(resources);
}

void PerformanceBenchmark::recordResult(const PerformanceResult& result) {
    QMutexLocker locker(&m_resultsMutex);
    m_results.append(result);
}

PerformanceBenchmark::PerformanceResult PerformanceBenchmark::createResult(const QString& benchmarkName, MetricType type, 
                                                                         const QString& metricName, double value, const QString& unit) const {
    PerformanceResult result;
    result.benchmarkName = benchmarkName;
    result.metricType = type;
    result.metricName = metricName;
    result.value = value;
    result.unit = unit;
    result.timestamp = QDateTime::currentDateTime();
    return result;
}

qint64 PerformanceBenchmark::getCurrentMemoryUsage() const {
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
                    return parts[1].toLongLong() * 1024; // Convert KB to bytes
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

double PerformanceBenchmark::getCurrentCpuUsage() const {
    // Simplified CPU usage measurement
    // In a real implementation, this would measure actual CPU usage
    static QElapsedTimer lastMeasurement;
    static qint64 lastCpuTime = 0;
    
    if (!lastMeasurement.isValid()) {
        lastMeasurement.start();
        return 0.0;
    }
    
    qint64 elapsed = lastMeasurement.restart();
    if (elapsed > 0) {
        // This is a placeholder - real CPU measurement would require platform-specific code
        return QRandomGenerator::global()->bounded(100.0); // Random value for demonstration
    }
    
    return 0.0;
}

QMap<QString, qint64> PerformanceBenchmark::getDiskIOStats() const {
    QMap<QString, qint64> stats;
    
    // Placeholder implementation - real disk I/O measurement would require platform-specific code
    stats["bytes_read"] = 0;
    stats["bytes_written"] = 0;
    stats["read_operations"] = 0;
    stats["write_operations"] = 0;
    
    return stats;
}

QMap<QString, qint64> PerformanceBenchmark::getNetworkIOStats() const {
    QMap<QString, qint64> stats;
    
    // Placeholder implementation - real network I/O measurement would require platform-specific code
    stats["bytes_received"] = 0;
    stats["bytes_sent"] = 0;
    stats["packets_received"] = 0;
    stats["packets_sent"] = 0;
    
    return stats;
}

qint64 PerformanceBenchmark::measureFileCreation(const QString& directory, int fileCount, qint64 fileSize) {
    QElapsedTimer timer;
    timer.start();
    
    QByteArray data(fileSize, 'A');
    
    for (int i = 0; i < fileCount; ++i) {
        QString fileName = QString("test_file_%1.dat").arg(i);
        QString filePath = QDir(directory).absoluteFilePath(fileName);
        
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(data);
        }
    }
    
    return timer.elapsed();
}

qint64 PerformanceBenchmark::measureFileReading(const QStringList& filePaths) {
    QElapsedTimer timer;
    timer.start();
    
    for (const QString& filePath : filePaths) {
        QFile file(filePath);
        if (file.open(QIODevice::ReadOnly)) {
            file.readAll();
        }
    }
    
    return timer.elapsed();
}

qint64 PerformanceBenchmark::measureFileWriting(const QStringList& filePaths, const QByteArray& data) {
    QElapsedTimer timer;
    timer.start();
    
    for (const QString& filePath : filePaths) {
        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(data);
        }
    }
    
    return timer.elapsed();
}

qint64 PerformanceBenchmark::measureDirectoryTraversal(const QString& directory, bool recursive) {
    QElapsedTimer timer;
    timer.start();
    
    QDirIterator::IteratorFlags flags = recursive ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags;
    QDirIterator iterator(directory, QDir::AllEntries | QDir::NoDotAndDotDot, flags);
    
    int count = 0;
    while (iterator.hasNext()) {
        iterator.next();
        count++;
    }
    
    return timer.elapsed();
}

qint64 PerformanceBenchmark::measureWidgetUpdate(QWidget* widget, int updateCount) {
    QElapsedTimer timer;
    timer.start();
    
    for (int i = 0; i < updateCount; ++i) {
        widget->update();
        QApplication::processEvents();
    }
    
    return timer.elapsed();
}

qint64 PerformanceBenchmark::measureWidgetResize(QWidget* widget, int resizeCount) {
    QElapsedTimer timer;
    timer.start();
    
    QSize originalSize = widget->size();
    
    for (int i = 0; i < resizeCount; ++i) {
        QSize newSize(originalSize.width() + (i % 100), originalSize.height() + (i % 100));
        widget->resize(newSize);
        QApplication::processEvents();
    }
    
    widget->resize(originalSize); // Restore original size
    
    return timer.elapsed();
}

double PerformanceBenchmark::measureFrameRate(QWidget* widget, int durationMs) {
    QElapsedTimer timer;
    timer.start();
    
    int frameCount = 0;
    while (timer.elapsed() < durationMs) {
        widget->update();
        QApplication::processEvents();
        frameCount++;
    }
    
    qint64 actualDuration = timer.elapsed();
    if (actualDuration > 0) {
        return (frameCount * 1000.0) / actualDuration; // frames per second
    }
    
    return 0.0;
}

double PerformanceBenchmark::calculateMean(const QList<double>& values) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    
    return sum / values.size();
}

double PerformanceBenchmark::calculateMedian(QList<double> values) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    std::sort(values.begin(), values.end());
    
    int size = values.size();
    if (size % 2 == 0) {
        return (values[size / 2 - 1] + values[size / 2]) / 2.0;
    } else {
        return values[size / 2];
    }
}

double PerformanceBenchmark::calculateStandardDeviation(const QList<double>& values, double mean) const {
    if (values.size() <= 1) {
        return 0.0;
    }
    
    double sumSquaredDiffs = 0.0;
    for (double value : values) {
        double diff = value - mean;
        sumSquaredDiffs += diff * diff;
    }
    
    return qSqrt(sumSquaredDiffs / (values.size() - 1));
}

double PerformanceBenchmark::calculatePercentile(QList<double> values, double percentile) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    std::sort(values.begin(), values.end());
    
    double index = (percentile / 100.0) * (values.size() - 1);
    int lowerIndex = qFloor(index);
    int upperIndex = qCeil(index);
    
    if (lowerIndex == upperIndex) {
        return values[lowerIndex];
    }
    
    double weight = index - lowerIndex;
    return values[lowerIndex] * (1.0 - weight) + values[upperIndex] * weight;
}

void PerformanceBenchmark::saveBaselines() const {
    QString filePath = getBaselinesFilePath();
    exportBaselines(filePath);
}

void PerformanceBenchmark::loadBaselines() {
    QString filePath = getBaselinesFilePath();
    if (QFile::exists(filePath)) {
        importBaselines(filePath);
    }
}

QString PerformanceBenchmark::getBaselinesFilePath() const {
    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dataDir);
    return QDir(dataDir).absoluteFilePath("performance_baselines.json");
}

QString PerformanceBenchmark::detectPlatform() const {
    return QSysInfo::prettyProductName();
}

QString PerformanceBenchmark::detectCpuInfo() const {
    // Simplified CPU detection - real implementation would use platform-specific APIs
    return QString("%1 (%2)").arg(QSysInfo::currentCpuArchitecture()).arg("Unknown CPU");
}

qint64 PerformanceBenchmark::detectTotalMemory() const {
    // Simplified memory detection - real implementation would use platform-specific APIs
#ifdef Q_OS_LINUX
    QFile file("/proc/meminfo");
    if (file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        while (stream.readLineInto(&line)) {
            if (line.startsWith("MemTotal:")) {
                QStringList parts = line.split(QRegExp("\\s+"));
                if (parts.size() >= 2) {
                    return parts[1].toLongLong() * 1024; // Convert KB to bytes
                }
            }
        }
    }
#endif
    return 0;
}

// BenchmarkRunner implementation
BenchmarkRunner::BenchmarkRunner(QObject* parent)
    : QObject(parent)
    , m_benchmark(new PerformanceBenchmark(this))
{
}

void BenchmarkRunner::addBenchmark(const QString& name, std::function<void()> benchmarkFunction, 
                                 const PerformanceBenchmark::BenchmarkConfig& config) {
    BenchmarkInfo info;
    info.name = name;
    info.function = benchmarkFunction;
    info.config = config;
    
    // Remove existing benchmark with same name
    for (int i = 0; i < m_benchmarks.size(); ++i) {
        if (m_benchmarks[i].name == name) {
            m_benchmarks.removeAt(i);
            break;
        }
    }
    
    m_benchmarks.append(info);
}

void BenchmarkRunner::removeBenchmark(const QString& name) {
    for (int i = 0; i < m_benchmarks.size(); ++i) {
        if (m_benchmarks[i].name == name) {
            m_benchmarks.removeAt(i);
            break;
        }
    }
}

QStringList BenchmarkRunner::getBenchmarkNames() const {
    QStringList names;
    for (const BenchmarkInfo& info : m_benchmarks) {
        names.append(info.name);
    }
    return names;
}

bool BenchmarkRunner::runAllBenchmarks() {
    return runBenchmarkSuite(getBenchmarkNames());
}

bool BenchmarkRunner::runBenchmark(const QString& name) {
    for (const BenchmarkInfo& info : m_benchmarks) {
        if (info.name == name) {
            m_benchmark->setBenchmarkConfig(info.config);
            return m_benchmark->runBenchmark(name, info.function);
        }
    }
    return false;
}

bool BenchmarkRunner::runBenchmarkSuite(const QStringList& names) {
    emit suiteStarted(names.size());
    
    int successfulBenchmarks = 0;
    
    for (int i = 0; i < names.size(); ++i) {
        const QString& name = names[i];
        emit benchmarkProgress(name, i + 1, names.size());
        
        if (runBenchmark(name)) {
            successfulBenchmarks =++;
        }
    }
    
    emit suiteCompleted(names.size(), successfulBenchmarks);
    return successfulBenchmarks == names.size();
}

QList<PerformanceBenchmark::PerformanceResult> BenchmarkRunner::getAllResults() const {
    return m_benchmark->getResults();
}

QMap<QString, PerformanceBenchmark::BenchmarkStatistics> BenchmarkRunner::getAllStatistics() const {
    return m_benchmark->groupStatisticsByBenchmark();
}

bool BenchmarkRunner::generateSuiteReport(const QString& outputPath) const {
    return m_benchmark->exportResults(outputPath);
}

#include "performance_benchmark.moc"