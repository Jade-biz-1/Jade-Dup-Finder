#ifndef PERCEPTUAL_HASH_ALGORITHM_H
#define PERCEPTUAL_HASH_ALGORITHM_H

#include "detection_algorithm.h"
#include <QImage>

/**
 * @class PerceptualHashAlgorithm
 * @brief Image similarity detection using perceptual hashing
 * 
 * This algorithm uses difference hash (dHash) to detect visually similar images.
 * It can find duplicates even when images have been resized, compressed,
 * or converted to different formats.
 */
class PerceptualHashAlgorithm : public DetectionAlgorithm
{
public:
    PerceptualHashAlgorithm();
    virtual ~PerceptualHashAlgorithm() = default;

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
    /**
     * @brief Compute difference hash (dHash) for an image
     * @param image Input image
     * @return 64-bit hash as QByteArray
     */
    QByteArray computeDHash(const QImage& image) const;

    /**
     * @brief Calculate Hamming distance between two hashes
     * @param hash1 First hash
     * @param hash2 Second hash
     * @return Number of different bits
     */
    int hammingDistance(const QByteArray& hash1, const QByteArray& hash2) const;

    /**
     * @brief Convert Hamming distance to similarity score
     * @param hammingDistance Number of different bits
     * @return Similarity score (0.0 to 1.0)
     */
    double hammingToSimilarity(int hammingDistance) const;

    /**
     * @brief Load and preprocess image for hashing
     * @param filePath Path to image file
     * @return Loaded and preprocessed image, or null image if failed
     */
    QImage loadAndPreprocessImage(const QString& filePath) const;

    QVariantMap m_config;
    double m_similarityThreshold;
    int m_hashSize;
    int m_maxHammingDistance;
};

#endif // PERCEPTUAL_HASH_ALGORITHM_H