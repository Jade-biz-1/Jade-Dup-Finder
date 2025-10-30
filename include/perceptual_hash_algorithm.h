#pragma once

#include "detection_algorithm.h"
#include <QImage>

/**
 * @brief Perceptual hash algorithm for detecting similar images
 * 
 * Implements dHash (difference hash) algorithm which:
 * 1. Resizes image to small size (e.g. 9x8)
 * 2. Converts to grayscale
 * 3. Computes differences between adjacent pixels
 * 4. Creates 64-bit hash from differences
 * 
 * Similar images will have similar hashes (measured by Hamming distance)
 */
class PerceptualHashAlgorithm : public DetectionAlgorithm {
    Q_OBJECT

public:
    explicit PerceptualHashAlgorithm(QObject* parent = nullptr);
    ~PerceptualHashAlgorithm() override = default;

    // DetectionAlgorithm interface
    Mode getMode() const override;
    QList<FileType> getSupportedFileTypes() const override;
    bool canHandle(const QString& filePath, FileType type) const override;
    QByteArray computeSignature(const QString& filePath, FileType type) override;
    SimilarityResult compareSignatures(const QByteArray& signature1,
                                      const QByteArray& signature2) override;
    double getSimilarityThreshold() const override;
    void setSimilarityThreshold(double threshold) override;
    QString getName() const override;
    QString getDescription() const override;

    /**
     * @brief Get supported image formats
     */
    static QStringList getSupportedFormats();

private:
    /**
     * @brief Compute difference hash for an image
     * @param image Input image
     * @return 64-bit hash as byte array
     */
    QByteArray computeDHash(const QImage& image);

    /**
     * @brief Compute Hamming distance between two hashes
     * @param hash1 First hash
     * @param hash2 Second hash
     * @return Number of differing bits
     */
    int hammingDistance(const QByteArray& hash1, const QByteArray& hash2);

    /**
     * @brief Convert Hamming distance to similarity score
     * @param distance Hamming distance
     * @param maxBits Maximum number of bits (usually 64)
     * @return Similarity score (0.0 to 1.0)
     */
    double distanceToSimilarity(int distance, int maxBits);

    double m_threshold = 0.90;  ///< Default threshold (90% similarity)
    int m_hashSize = 8;          ///< Hash size (8x8 = 64 bits)
};

REGISTER_DETECTION_ALGORITHM("perceptual_hash", PerceptualHashAlgorithm)
