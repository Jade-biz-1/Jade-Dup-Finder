#ifndef DETECTION_ALGORITHM_H
#define DETECTION_ALGORITHM_H

#include <QString>
#include <QByteArray>
#include <QVariantMap>
#include <memory>

/**
 * @class DetectionAlgorithm
 * @brief Base interface for duplicate detection algorithms
 * 
 * This abstract class defines the interface for all duplicate detection algorithms.
 * Implementations can provide different strategies for detecting duplicates:
 * - Exact matching (hash-based)
 * - Quick matching (size + filename)
 * - Perceptual matching (image similarity)
 * - Content matching (document similarity)
 */
class DetectionAlgorithm
{
public:
    virtual ~DetectionAlgorithm() = default;

    /**
     * @brief Get the algorithm name
     * @return Human-readable algorithm name
     */
    virtual QString name() const = 0;

    /**
     * @brief Get the algorithm description
     * @return Detailed description of what the algorithm does
     */
    virtual QString description() const = 0;

    /**
     * @brief Get supported file extensions
     * @return List of file extensions this algorithm can handle (empty = all files)
     */
    virtual QStringList supportedExtensions() const = 0;

    /**
     * @brief Compute signature for a file
     * @param filePath Path to the file to analyze
     * @return Signature/fingerprint for the file, or empty if failed
     */
    virtual QByteArray computeSignature(const QString& filePath) = 0;

    /**
     * @brief Compare two signatures for similarity
     * @param sig1 First signature
     * @param sig2 Second signature
     * @return true if signatures indicate duplicate files
     */
    virtual bool compareSignatures(const QByteArray& sig1, const QByteArray& sig2) = 0;

    /**
     * @brief Calculate similarity score between two signatures
     * @param sig1 First signature
     * @param sig2 Second signature
     * @return Similarity score (0.0 = completely different, 1.0 = identical)
     */
    virtual double similarityScore(const QByteArray& sig1, const QByteArray& sig2) = 0;

    /**
     * @brief Set algorithm configuration
     * @param config Configuration parameters specific to the algorithm
     */
    virtual void setConfiguration(const QVariantMap& config) = 0;

    /**
     * @brief Get current algorithm configuration
     * @return Current configuration parameters
     */
    virtual QVariantMap getConfiguration() const = 0;

    /**
     * @brief Get default configuration for this algorithm
     * @return Default configuration parameters with descriptions
     */
    virtual QVariantMap getDefaultConfiguration() const = 0;

    /**
     * @brief Check if algorithm can handle a specific file
     * @param filePath Path to the file to check
     * @return true if algorithm can process this file type
     */
    virtual bool canHandle(const QString& filePath) const;

    /**
     * @brief Get algorithm performance characteristics
     * @return Map with performance info (speed, accuracy, memory usage)
     */
    virtual QVariantMap getPerformanceInfo() const = 0;

protected:
    /**
     * @brief Helper function to check file extension
     * @param filePath File path to check
     * @param extensions List of supported extensions
     * @return true if file extension is supported
     */
    bool hasValidExtension(const QString& filePath, const QStringList& extensions) const;
};

#endif // DETECTION_ALGORITHM_H