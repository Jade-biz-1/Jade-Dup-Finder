#ifndef EXACT_HASH_ALGORITHM_H
#define EXACT_HASH_ALGORITHM_H

#include "detection_algorithm.h"
#include "hash_calculator.h"

/**
 * @class ExactHashAlgorithm
 * @brief SHA-256 hash-based exact duplicate detection
 * 
 * This algorithm uses SHA-256 cryptographic hashing to detect
 * files with identical content. It provides 100% accuracy for
 * exact duplicates but cannot detect similar (but not identical) files.
 */
class ExactHashAlgorithm : public DetectionAlgorithm
{
public:
    ExactHashAlgorithm();
    virtual ~ExactHashAlgorithm() = default;

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
    QVariantMap m_config;
    HashCalculator m_calculator;  // Single instance shared for all hash calculations
};

#endif // EXACT_HASH_ALGORITHM_H