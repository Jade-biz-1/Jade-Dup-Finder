#include "quick_scan_algorithm.h"
#include <QFileInfo>
#include <QDataStream>
#include <QDebug>
#include <QRegularExpression>
#include <algorithm>

QuickScanAlgorithm::QuickScanAlgorithm()
{
    m_config = getDefaultConfiguration();
    
    // Load configuration values
    m_filenameThreshold = m_config.value("filename_threshold", 0.8).toDouble();
    m_requireSameExtension = m_config.value("require_same_extension", true).toBool();
    m_caseSensitive = m_config.value("case_sensitive", false).toBool();
}

QString QuickScanAlgorithm::name() const
{
    return "Quick Scan";
}

QString QuickScanAlgorithm::description() const
{
    return "Fast size and filename-based duplicate detection. Compares file sizes and uses "
           "fuzzy filename matching to quickly identify likely duplicates. Much faster than "
           "hash-based detection but may miss some duplicates or have false positives.";
}

QStringList QuickScanAlgorithm::supportedExtensions() const
{
    // Supports all file types
    return QStringList();
}

QByteArray QuickScanAlgorithm::computeSignature(const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    
    if (!fileInfo.exists() || !fileInfo.isFile()) {
        qWarning() << "File does not exist or is not a regular file:" << filePath;
        return QByteArray();
    }
    
    // Create signature from file size, normalized filename, and extension
    qint64 size = fileInfo.size();
    QString filename = fileInfo.completeBaseName(); // Filename without extension
    QString extension = fileInfo.suffix().toLower();
    
    QString normalizedName = normalizeFilename(filename);
    
    FileSignature sig(size, normalizedName, extension);
    return encodeSignature(sig);
}

bool QuickScanAlgorithm::compareSignatures(const QByteArray& sig1, const QByteArray& sig2)
{
    if (sig1.isEmpty() || sig2.isEmpty()) {
        return false;
    }
    
    FileSignature fs1 = parseSignature(sig1);
    FileSignature fs2 = parseSignature(sig2);
    
    // Files must have the same size
    if (fs1.size != fs2.size) {
        return false;
    }
    
    // Check extension requirement
    if (m_requireSameExtension && fs1.extension != fs2.extension) {
        return false;
    }
    
    // Calculate filename similarity
    double similarity = filenameSimilarity(fs1.normalizedName, fs2.normalizedName);
    
    return similarity >= m_filenameThreshold;
}

double QuickScanAlgorithm::similarityScore(const QByteArray& sig1, const QByteArray& sig2)
{
    if (sig1.isEmpty() || sig2.isEmpty()) {
        return 0.0;
    }
    
    FileSignature fs1 = parseSignature(sig1);
    FileSignature fs2 = parseSignature(sig2);
    
    // Different sizes = no similarity
    if (fs1.size != fs2.size) {
        return 0.0;
    }
    
    // Calculate filename similarity
    double filenameSim = filenameSimilarity(fs1.normalizedName, fs2.normalizedName);
    
    // Extension bonus/penalty
    double extensionSim = (fs1.extension == fs2.extension) ? 1.0 : 0.5;
    
    // Combine scores (filename is more important)
    return (filenameSim * 0.8) + (extensionSim * 0.2);
}

void QuickScanAlgorithm::setConfiguration(const QVariantMap& config)
{
    m_config = config;
    
    // Update internal settings
    m_filenameThreshold = config.value("filename_threshold", 0.8).toDouble();
    m_requireSameExtension = config.value("require_same_extension", true).toBool();
    m_caseSensitive = config.value("case_sensitive", false).toBool();
}

QVariantMap QuickScanAlgorithm::getConfiguration() const
{
    return m_config;
}

QVariantMap QuickScanAlgorithm::getDefaultConfiguration() const
{
    QVariantMap config;
    
    config["filename_threshold"] = 0.8;
    config["filename_threshold_description"] = "Minimum filename similarity (0.0-1.0) to consider files duplicates";
    
    config["require_same_extension"] = true;
    config["require_same_extension_description"] = "Require files to have the same extension to be considered duplicates";
    
    config["case_sensitive"] = false;
    config["case_sensitive_description"] = "Whether filename comparison is case-sensitive";
    
    return config;
}

QVariantMap QuickScanAlgorithm::getPerformanceInfo() const
{
    QVariantMap info;
    
    info["speed"] = "Very Fast";
    info["accuracy"] = "80-90%";
    info["memory_usage"] = "Very Low";
    info["cpu_usage"] = "Low";
    info["typical_throughput"] = "5000+ files/s";
    info["best_for"] = "Large datasets, quick previews, obvious duplicates";
    info["limitations"] = "May miss duplicates with different names, possible false positives";
    
    return info;
}

QuickScanAlgorithm::FileSignature QuickScanAlgorithm::parseSignature(const QByteArray& signature) const
{
    QDataStream stream(signature);
    stream.setVersion(QDataStream::Qt_6_0);
    
    FileSignature sig;
    stream >> sig.size >> sig.normalizedName >> sig.extension;
    
    return sig;
}

QByteArray QuickScanAlgorithm::encodeSignature(const FileSignature& sig) const
{
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    stream.setVersion(QDataStream::Qt_6_0);
    
    stream << sig.size << sig.normalizedName << sig.extension;
    
    return data;
}

QString QuickScanAlgorithm::normalizeFilename(const QString& filename) const
{
    QString normalized = filename;
    
    // Convert to lowercase if not case-sensitive
    if (!m_caseSensitive) {
        normalized = normalized.toLower();
    }
    
    // Remove common patterns that don't affect content
    // Remove copy indicators: " (1)", " (2)", " - Copy", etc.
    normalized.remove(QRegularExpression(R"(\s*\(\d+\)\s*)"));
    normalized.remove(QRegularExpression(R"(\s*-\s*copy\s*)", QRegularExpression::CaseInsensitiveOption));
    normalized.remove(QRegularExpression(R"(\s*copy\s*of\s*)", QRegularExpression::CaseInsensitiveOption));
    
    // Remove extra whitespace
    normalized = normalized.simplified();
    
    // Remove common prefixes/suffixes
    normalized.remove(QRegularExpression(R"(^new\s+)", QRegularExpression::CaseInsensitiveOption));
    normalized.remove(QRegularExpression(R"(\s+new$)", QRegularExpression::CaseInsensitiveOption));
    
    return normalized;
}

int QuickScanAlgorithm::levenshteinDistance(const QString& s1, const QString& s2) const
{
    const int len1 = static_cast<int>(s1.length());
    const int len2 = static_cast<int>(s2.length());
    
    // Create matrix
    QVector<QVector<int>> matrix(len1 + 1, QVector<int>(len2 + 1));
    
    // Initialize first row and column
    for (int i = 0; i <= len1; ++i) {
        matrix[i][0] = i;
    }
    for (int j = 0; j <= len2; ++j) {
        matrix[0][j] = j;
    }
    
    // Fill matrix
    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            
            matrix[i][j] = std::min({
                matrix[i-1][j] + 1,     // deletion
                matrix[i][j-1] + 1,     // insertion
                matrix[i-1][j-1] + cost // substitution
            });
        }
    }
    
    return matrix[len1][len2];
}

double QuickScanAlgorithm::filenameSimilarity(const QString& name1, const QString& name2) const
{
    if (name1.isEmpty() && name2.isEmpty()) {
        return 1.0;
    }
    
    if (name1.isEmpty() || name2.isEmpty()) {
        return 0.0;
    }
    
    // Exact match
    if (name1 == name2) {
        return 1.0;
    }
    
    // Calculate Levenshtein distance
    int distance = levenshteinDistance(name1, name2);
    int maxLength = static_cast<int>(std::max(name1.length(), name2.length()));
    
    // Convert distance to similarity score
    double similarity = 1.0 - (static_cast<double>(distance) / maxLength);
    
    return std::max(0.0, similarity);
}