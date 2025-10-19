#include "performance_benchmarks.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QTextStream>
#include <QDebug>
#include <QSysInfo>
#include <QStandardPaths>
#include <QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QRandomGenerator>
#include <QtMath>
#include <QProcess>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#elif defined(Q_OS_LINUX)
#include <unistd.h>
#include <sys/resource.h>
#include <fstream>
#elif defined(Q_OS_MAC)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

PerformanceBenchmarks::PerformanceBenchmarks(QObject* parent)
    : QObject(parent)
    , m_initialMemory(0)
    , m_peakMemory(0)
    , m_cpuMonitorThread(nullptr)
    , m_cpuMonitoringActive(false)
{
    // Set default configuration
    m_config.iterations = 1;
    m_config.warmupIterations = 0;
    m_config.measureMemory = true;
    m_config.measureCPU = false;
    m_config.samplingIntervalMs = 100;
    m_config.timeoutSeconds = 60.0;
    
    // Set default output directory
    m_outputDirectory = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/performance_results";
    QDir().mkpath(m_outputDirectory);
    
    // Initialize performance counters
    initializePerformanceCounters();
}

PerformanceBenchmarks::~PerformanceBenchmarks() {
    cleanupPerformanceCounters();
}

void PerformanceBenchmarks::setBenchmarkConfig(const BenchmarkConfig& config) {
    m_config = config;
}

PerformanceBenchmarks::BenchmarkConfig PerformanceBenchmarks::getBenchmarkConfig() const {
    return m_config;
}

void PerformanceBenchmarks::setOutputDirectory(const QString& directory) {
    m_outputDirectory = directory;
    QDir().mkpath(directory);
}

QString PerformanceBenchmarks::getOutputDirectory() const {
    return m_outputDirectory;
}

PerformanceBenchmarks::PerformanceResult PerformanceBenchmarks::runBenchmark(
    const QString& name, std::function<void()> benchmarkFunction) {
    return runBenchmark(name, benchmarkFunction, m_config);
}

PerformanceBenchmarks::PerformanceResult PerformanceBenchmarks::runBenchmark(
    const QString& name, std::function<void()> benchmarkFunction, const BenchmarkConfig& config) {
    
    emit benchmarkStarted(name);
    
    PerformanceResult result = createResult(name, MetricType::ExecutionTime, 0.0, "ms");
    
    try {
        QList<qint64> executionTimes;
        MemoryStats memoryStats;
        CPUStats cpuStats;
        
        // Warmup iterations
        for (int i = 0; i < config.warmupIterations; ++i) {
            benchmarkFunction();
        }
        
        // Initialize memory tracking
        if (config.measureMemory) {
            resetMemoryTracking();
        }
        
        // Initialize CPU monitoring
        if (config.measureCPU) {
            startCPUMonitoring();
        }
        
        // Run benchmark iterations
        for (int i = 0; i < config.iterations; ++i) {
            QElapsedTimer timer;
            timer.start();
            
            benchmarkFunction();
            
            qint64 elapsed = timer.elapsed();
            executionTimes.append(elapsed);
            
            // Check for timeout
            if (elapsed > config.timeoutSeconds * 1000) {
                if (config.failOnTimeout) {
                    throw std::runtime_error("Benchmark timed out");
                }
                break;
            }
        }
        
        // Stop monitoring
        if (config.measureCPU) {
            stopCPUMonitoring();
        }
        
        // Calculate statistics
        if (!executionTimes.isEmpty()) {
            QList<double> times;
            for (qint64 time : executionTimes) {
                times.append(static_cast<double>(time));
            }
            
            result.value = calculateMean(times);
            result.metadata["min"] = *std::min_element(times.begin(), times.end());
            result.metadata["max"] = *std::max_element(times.begin(), times.end());
            result.metadata["median"] = calculateMedian(times);
            result.metadata["stddev"] = calculateStandardDeviation(times);
            result.metadata["iterations"] = config.iterations;
        }
        
        // Add memory statistics
        if (config.measureMemory) {
            result.metadata["peakMemory"] = getPeakMemoryUsage();
            result.metadata["currentMemory"] = getCurrentMemoryUsage();
        }
        
        // Add CPU statistics
        if (config.measureCPU && !m_cpuSamples.isEmpty()) {
            result.metadata["avgCPU"] = calculateMean(m_cpuSamples);
            result.metadata["peakCPU"] = *std::max_element(m_cpuSamples.begin(), m_cpuSamples.end());
        }
        
        result.timestamp = QDateTime::currentDateTime();
        result.withinThreshold = true; // Will be updated when compared with baseline
        
        emit benchmarkCompleted(name, result);
        
    } catch (const std::exception& e) {
        result.metadata["error"] = QString::fromStdString(e.what());
        result.withinThreshold = false;
        emit benchmarkFailed(name, QString::fromStdString(e.what()));
    }
    
    return result;
}

void PerformanceBenchmarks::startTimer(const QString& name) {
    m_timers[name].start();
}

qint64 PerformanceBenchmarks::stopTimer(const QString& name) {
    if (!m_timers.contains(name)) {
        qWarning() << "Timer not found:" << name;
        return -1;
    }
    
    qint64 elapsed = m_timers[name].elapsed();
    m_elapsedTimes[name] = elapsed;
    m_timers.remove(name);
    
    return elapsed;
}

qint64 PerformanceBenchmarks::getElapsedTime(const QString& name) const {
    return m_elapsedTimes.value(name, -1);
}

PerformanceBenchmarks::PerformanceResult PerformanceBenchmarks::measureExecutionTime(
    const QString& name, std::function<void()> function) {
    
    QElapsedTimer timer;
    timer.start();
    
    function();
    
    qint64 elapsed = timer.elapsed();
    
    PerformanceResult result = createResult(name, MetricType::ExecutionTime, 
                                          static_cast<double>(elapsed), "ms");
    result.timestamp = QDateTime::currentDateTime();
    
    return result;
}

PerformanceBenchmarks::MemoryStats PerformanceBenchmarks::measureMemoryUsage(
    const QString& name, std::function<void()> function) {
    
    MemoryStats stats;
    
    // Get initial memory usage
    stats.initialUsage = getCurrentMemoryUsage();
    resetMemoryTracking();
    
    // Execute function
    function();
    
    // Get final memory usage
    stats.finalUsage = getCurrentMemoryUsage();
    stats.peakUsage = getPeakMemoryUsage();
    stats.allocated = stats.finalUsage - stats.initialUsage;
    
    return stats;
}

qint64 PerformanceBenchmarks::getCurrentMemoryUsage() const {
    return getPlatformMemoryUsage();
}

qint64 PerformanceBenchmarks::getPeakMemoryUsage() const {
    QMutexLocker locker(&m_memoryMutex);
    return m_peakMemory;
}

void PerformanceBenchmarks::resetMemoryTracking() {
    QMutexLocker locker(&m_memoryMutex);
    m_initialMemory = getCurrentMemoryUsage();
    m_peakMemory = m_initialMemory;
}

PerformanceBenchmarks::PerformanceResult PerformanceBenchmarks::measureThroughput(
    const QString& name, std::function<int()> function, const QString& unit) {
    
    QElapsedTimer timer;
    timer.start();
    
    int operations = function();
    
    qint64 elapsed = timer.elapsed();
    double throughput = (static_cast<double>(operations) * 1000.0) / elapsed; // ops per second
    
    PerformanceResult result = createResult(name, MetricType::ThroughputOps, throughput, unit);
    result.metadata["operations"] = operations;
    result.metadata["duration"] = elapsed;
    result.timestamp = QDateTime::currentDateTime();
    
    return result;
}

bool PerformanceBenchmarks::createBaseline(const QString& name, const PerformanceResult& result) {
    PerformanceBaseline baseline;
    baseline.name = name;
    baseline.metricType = result.metricType;
    baseline.value = result.value;
    baseline.created = QDateTime::currentDateTime();
    baseline.lastUpdated = baseline.created;
    baseline.platform = getPlatformIdentifier();
    baseline.description = result.description;
    
    QMutexLocker locker(&m_baselinesMutex);
    m_baselines[name] = baseline;
    
    saveBaseline(name, baseline);
    emit baselineCreated(name);
    
    return true;
}

PerformanceBenchmarks::PerformanceResult PerformanceBenchmarks::compareWithBaseline(
    const QString& baselineName, const PerformanceResult& result) const {
    
    PerformanceResult comparison = result;
    
    if (!baselineExists(baselineName)) {
        comparison.metadata["error"] = "Baseline not found";
        comparison.withinThreshold = false;
        return comparison;
    }
    
    PerformanceBaseline baseline = getBaseline(baselineName);
    comparison.baseline = baseline.value;
    comparison.threshold = baseline.tolerance;
    
    double regression = calculatePerformanceRegression(result, baseline);
    comparison.metadata["regression"] = regression;
    
    // Check if within threshold
    comparison.withinThreshold = qAbs(regression) <= baseline.tolerance;
    
    if (!comparison.withinThreshold) {
        emit const_cast<PerformanceBenchmarks*>(this)->performanceRegression(baselineName, regression);
    }
    
    return comparison;
}

// Statistical analysis methods
double PerformanceBenchmarks::calculateMean(const QList<double>& values) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    
    return sum / values.size();
}

double PerformanceBenchmarks::calculateMedian(const QList<double>& values) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    QList<double> sortedValues = values;
    sortValues(sortedValues);
    
    int size = sortedValues.size();
    if (size % 2 == 0) {
        return (sortedValues[size / 2 - 1] + sortedValues[size / 2]) / 2.0;
    } else {
        return sortedValues[size / 2];
    }
}

double PerformanceBenchmarks::calculateStandardDeviation(const QList<double>& values) const {
    if (values.size() <= 1) {
        return 0.0;
    }
    
    double mean = calculateMean(values);
    double sumSquaredDiffs = 0.0;
    
    for (double value : values) {
        double diff = value - mean;
        sumSquaredDiffs += diff * diff;
    }
    
    return qSqrt(sumSquaredDiffs / (values.size() - 1));
}

double PerformanceBenchmarks::calculatePercentile(const QList<double>& values, double percentile) const {
    if (values.isEmpty()) {
        return 0.0;
    }
    
    QList<double> sortedValues = values;
    sortValues(sortedValues);
    
    double index = (percentile / 100.0) * (sortedValues.size() - 1);
    int lowerIndex = static_cast<int>(qFloor(index));
    int upperIndex = static_cast<int>(qCeil(index));
    
    if (lowerIndex == upperIndex) {
        return sortedValues[lowerIndex];
    }
    
    double weight = index - lowerIndex;
    return sortedValues[lowerIndex] * (1.0 - weight) + sortedValues[upperIndex] * weight;
}

double PerformanceBenchmarks::calculatePerformanceRegression(
    const PerformanceResult& current, const PerformanceBaseline& baseline) const {
    
    if (baseline.value == 0.0) {
        return 0.0;
    }
    
    return ((current.value - baseline.value) / baseline.value) * 100.0;
}

// Utility methods
QString PerformanceBenchmarks::formatDuration(qint64 milliseconds) {
    if (milliseconds < 1000) {
        return QString("%1ms").arg(milliseconds);
    } else if (milliseconds < 60000) {
        return QString("%1.%2s").arg(milliseconds / 1000).arg((milliseconds % 1000) / 100);
    } else {
        int minutes = milliseconds / 60000;
        int seconds = (milliseconds % 60000) / 1000;
        return QString("%1m %2s").arg(minutes).arg(seconds);
    }
}

QString PerformanceBenchmarks::formatBytes(qint64 bytes) {
    const QStringList units = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unitIndex < units.size() - 1) {
        size /= 1024.0;
        unitIndex++;
    }
    
    return QString("%1 %2").arg(size, 0, 'f', 2).arg(units[unitIndex]);
}

QString PerformanceBenchmarks::formatRate(double rate, const QString& unit) {
    if (rate >= 1000000.0) {
        return QString("%1M %2").arg(rate / 1000000.0, 0, 'f', 2).arg(unit);
    } else if (rate >= 1000.0) {
        return QString("%1K %2").arg(rate / 1000.0, 0, 'f', 2).arg(unit);
    } else {
        return QString("%1 %2").arg(rate, 0, 'f', 2).arg(unit);
    }
}

// Platform-specific implementations
qint64 PerformanceBenchmarks::getPlatformMemoryUsage() const {
#ifdef Q_OS_WIN
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<qint64>(pmc.WorkingSetSize);
    }
    return 0;
#elif defined(Q_OS_LINUX)
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label;
            long value;
            std::string unit;
            iss >> label >> value >> unit;
            return static_cast<qint64>(value * 1024); // Convert KB to bytes
        }
    }
    return 0;
#elif defined(Q_OS_MAC)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, 
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return static_cast<qint64>(info.resident_size);
    }
    return 0;
#else
    return 0; // Unsupported platform
#endif
}

double PerformanceBenchmarks::getPlatformCPUUsage() const {
    // Simplified CPU usage - would need platform-specific implementation
    return 0.0;
}

QMap<QString, QVariant> PerformanceBenchmarks::getPlatformSystemInfo() const {
    QMap<QString, QVariant> info;
    
    info["platform"] = QSysInfo::productType();
    info["version"] = QSysInfo::productVersion();
    info["architecture"] = QSysInfo::currentCpuArchitecture();
    info["kernelType"] = QSysInfo::kernelType();
    info["kernelVersion"] = QSysInfo::kernelVersion();
    info["qtVersion"] = QT_VERSION_STR;
    
    return info;
}

QString PerformanceBenchmarks::getPlatformIdentifier() const {
    return QString("%1_%2_%3")
           .arg(QSysInfo::productType())
           .arg(QSysInfo::currentCpuArchitecture())
           .arg(QSysInfo::productVersion());
}

// Private helper methods
void PerformanceBenchmarks::initializePerformanceCounters() {
    resetMemoryTracking();
}

void PerformanceBenchmarks::cleanupPerformanceCounters() {
    // Stop any ongoing monitoring
    if (m_cpuMonitoringActive) {
        stopCPUMonitoring();
    }
    
    // Clean up monitoring threads
    for (auto it = m_monitoringThreads.begin(); it != m_monitoringThreads.end(); ++it) {
        if (it.value() && it.value()->isRunning()) {
            it.value()->quit();
            it.value()->wait(1000);
        }
    }
}

PerformanceBenchmarks::PerformanceResult PerformanceBenchmarks::createResult(
    const QString& name, MetricType type, double value, const QString& unit) const {
    
    PerformanceResult result;
    result.benchmarkName = name;
    result.metricType = type;
    result.value = value;
    result.unit = unit;
    result.timestamp = QDateTime::currentDateTime();
    result.withinThreshold = true;
    
    return result;
}

bool PerformanceBenchmarks::baselineExists(const QString& name) const {
    QMutexLocker locker(&m_baselinesMutex);
    return m_baselines.contains(name) || QFile::exists(getBaselinePath(name));
}

PerformanceBenchmarks::PerformanceBaseline PerformanceBenchmarks::getBaseline(const QString& name) const {
    QMutexLocker locker(&m_baselinesMutex);
    
    if (m_baselines.contains(name)) {
        return m_baselines[name];
    }
    
    return loadBaseline(name);
}

void PerformanceBenchmarks::saveBaseline(const QString& name, const PerformanceBaseline& baseline) {
    QString filePath = getBaselinePath(name);
    
    QJsonObject json;
    json["name"] = baseline.name;
    json["metricType"] = static_cast<int>(baseline.metricType);
    json["value"] = baseline.value;
    json["tolerance"] = baseline.tolerance;
    json["created"] = baseline.created.toString(Qt::ISODate);
    json["lastUpdated"] = baseline.lastUpdated.toString(Qt::ISODate);
    json["platform"] = baseline.platform;
    json["configuration"] = baseline.configuration;
    json["description"] = baseline.description;
    
    QJsonDocument doc(json);
    
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
    }
}

PerformanceBenchmarks::PerformanceBaseline PerformanceBenchmarks::loadBaseline(const QString& name) const {
    PerformanceBaseline baseline;
    baseline.name = name;
    
    QString filePath = getBaselinePath(name);
    QFile file(filePath);
    
    if (!file.open(QIODevice::ReadOnly)) {
        return baseline;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    
    if (error.error != QJsonParseError::NoError) {
        return baseline;
    }
    
    QJsonObject json = doc.object();
    baseline.name = json["name"].toString();
    baseline.metricType = static_cast<MetricType>(json["metricType"].toInt());
    baseline.value = json["value"].toDouble();
    baseline.tolerance = json["tolerance"].toDouble();
    baseline.created = QDateTime::fromString(json["created"].toString(), Qt::ISODate);
    baseline.lastUpdated = QDateTime::fromString(json["lastUpdated"].toString(), Qt::ISODate);
    baseline.platform = json["platform"].toString();
    baseline.configuration = json["configuration"].toString();
    baseline.description = json["description"].toString();
    
    return baseline;
}

QString PerformanceBenchmarks::getBaselinePath(const QString& name) const {
    return QDir(m_outputDirectory).absoluteFilePath(QString("%1_baseline.json").arg(name));
}

void PerformanceBenchmarks::startCPUMonitoring() {
    if (m_cpuMonitoringActive) {
        return;
    }
    
    m_cpuMonitoringActive = true;
    m_cpuSamples.clear();
    
    // Start CPU monitoring in a separate thread (simplified implementation)
    // In a real implementation, this would use platform-specific APIs
}

void PerformanceBenchmarks::stopCPUMonitoring() {
    m_cpuMonitoringActive = false;
}

QList<double> PerformanceBenchmarks::extractValues(const QList<PerformanceResult>& results) const {
    QList<double> values;
    for (const PerformanceResult& result : results) {
        values.append(result.value);
    }
    return values;
}

void PerformanceBenchmarks::sortValues(QList<double>& values) const {
    std::sort(values.begin(), values.end());
}