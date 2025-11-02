#ifndef QUICK_SCAN_ALGORITHM_H
#define QUICK_SCAN_ALGORITHM_H

#include "detection_algorithm.h"

/**
 * @class QuickScanAlgorithm
 * @brief Fast size and filename-based duplicate detection
 * 
 * This algorithm provides rapid duplicate detection by comparing
 * file sizes and using fuzzy filename matching. It's designed for
 * speed over accuracy and is ideal for quick previews of large datasets.
 */
class QuickScanAlgorithm : public DetectionAlgorithm
{
public:
    QuickScanAlgorithm();
    virtual ~QuickScanAlgorithm() = default;

    // DetectionAlgorithm interface
    QString name() const override;
    QString description() const override;
    QStringList supportedExtensions() const override;
    QByteArray computeSignature(const QString& filePath) override;
    bool compareSignatures(const QByteArray& sig1, const QByteArray& sig2) override;
    double similarityScore(const QByteArray& sig1, const QByteArray& sig2) override;
    void setConfiguration(const QVariantMap& config) override;
    QVariantMap getConfiguration() const override;
    QVariantMap getDefaultConfiguration() const override;
    QVariantMap getPerformanceInfo() const override;

private:
    struct FileSignature {
        qint64 size;
        QString normalizedName;
        QString extension;
        
        FileSignature() : size(0) {}
        FileSignature(qint64 s, const QString& name, const QString& ext)
            : size(s), normalizedName(name), extension(ext) {}
    };

    /**
     * @brief Parse signature from byte array
     * @param signature Encoded signature
     * @return Parsed file signature
     */
    FileSignature parseSignature(const QByteArray& signature) const;

    /**
     * @brief Encode signature to byte array
     * @param sig File signature to encode
     * @return Encoded signature
     */
    QByteArray encodeSignature(const FileSignature& sig) const;

    /**
     * @brief Normalize filename for comparison
     * @param filename Original filename
     * @return Normalized filename
     */
    QString normalizeFilename(const QString& filename) const;

    /**
     * @brief Calculate Levenshtein distance between two strings
     * @param s1 First string
     * @param s2 Second string
     * @return Edit distance between strings
     */
    int levenshteinDistance(const QString& s1, const QString& s2) const;

    /**
     * @brief Calculate filename similarity score
     * @param name1 First filename
     * @param name2 Second filename
     * @return Similarity score (0.0 to 1.0)
     */
    double filenameSimilarity(const QString& name1, const QString& name2) const;

    QVariantMap m_config;
    double m_filenameThreshold;
    bool m_requireSameExtension;
    bool m_caseSensitive;
};

#endif // QUICK_SCAN_ALGORITHM_H