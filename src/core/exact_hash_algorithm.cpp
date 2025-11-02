#include "exact_hash_algorithm.h"
#include "hash_calculator.h"
#include <QCryptographicHash>
#include <QFile>
#include <QDebug>

ExactHashAlgorithm::ExactHashAlgorithm()
{
    m_config = getDefaultConfiguration();
}

QString ExactHashAlgorithm::name() const
{
    return "Exact Hash";
}

QString ExactHashAlgorithm::description() const
{
    return "SHA-256 hash-based exact content matching. Provides 100% accuracy for identical files "
           "by comparing cryptographic hashes of file contents. Cannot detect similar but not identical files.";
}

QStringList ExactHashAlgorithm::supportedExtensions() const
{
    // Supports all file types
    return QStringList();
}

QByteArray ExactHashAlgorithm::computeSignature(const QString& filePath)
{
    // Use existing HashCalculator for consistency
    HashCalculator calculator;
    QString hash = calculator.calculateFileHashSync(filePath);
    
    if (hash.isEmpty()) {
        qWarning() << "Failed to calculate hash for:" << filePath;
        return QByteArray();
    }
    
    return hash.toUtf8();
}

bool ExactHashAlgorithm::compareSignatures(const QByteArray& sig1, const QByteArray& sig2)
{
    // Exact hash comparison - must be identical
    return !sig1.isEmpty() && !sig2.isEmpty() && sig1 == sig2;
}

double ExactHashAlgorithm::similarityScore(const QByteArray& sig1, const QByteArray& sig2)
{
    // Hash-based comparison is binary - either identical (1.0) or different (0.0)
    return compareSignatures(sig1, sig2) ? 1.0 : 0.0;
}

void ExactHashAlgorithm::setConfiguration(const QVariantMap& config)
{
    m_config = config;
    
    // Apply configuration to HashCalculator if needed
    // For now, ExactHashAlgorithm doesn't have configurable parameters
}

QVariantMap ExactHashAlgorithm::getConfiguration() const
{
    return m_config;
}

QVariantMap ExactHashAlgorithm::getDefaultConfiguration() const
{
    QVariantMap config;
    
    // ExactHashAlgorithm currently has no configurable parameters
    // SHA-256 is always used for maximum reliability
    config["algorithm"] = "SHA-256";
    config["description"] = "Cryptographic hash algorithm used for file comparison";
    
    return config;
}

QVariantMap ExactHashAlgorithm::getPerformanceInfo() const
{
    QVariantMap info;
    
    info["speed"] = "Medium";
    info["accuracy"] = "100%";
    info["memory_usage"] = "Low";
    info["cpu_usage"] = "Medium";
    info["typical_throughput"] = "500 MB/s";
    info["best_for"] = "All file types, guaranteed accuracy";
    info["limitations"] = "Cannot detect similar but not identical files";
    
    return info;
}