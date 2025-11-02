#ifndef DETECTION_ALGORITHM_FACTORY_H
#define DETECTION_ALGORITHM_FACTORY_H

#include "detection_algorithm.h"
#include <QString>
#include <QStringList>
#include <memory>

/**
 * @class DetectionAlgorithmFactory
 * @brief Factory for creating detection algorithm instances
 * 
 * This factory provides a centralized way to create and manage
 * different types of duplicate detection algorithms.
 */
class DetectionAlgorithmFactory
{
public:
    /**
     * @brief Available detection algorithm types
     */
    enum AlgorithmType {
        ExactHash,          ///< SHA-256 hash-based exact matching (current default)
        QuickScan,          ///< Size + filename matching for speed
        PerceptualHash,     ///< Image similarity using perceptual hashing
        DocumentSimilarity  ///< Document content similarity
    };

    /**
     * @brief Create an algorithm instance
     * @param type Algorithm type to create
     * @return Unique pointer to algorithm instance, or nullptr if failed
     */
    static std::unique_ptr<DetectionAlgorithm> create(AlgorithmType type);

    /**
     * @brief Create an algorithm instance by name
     * @param algorithmName Name of the algorithm
     * @return Unique pointer to algorithm instance, or nullptr if not found
     */
    static std::unique_ptr<DetectionAlgorithm> create(const QString& algorithmName);

    /**
     * @brief Get list of available algorithm names
     * @return List of algorithm names that can be created
     */
    static QStringList availableAlgorithms();

    /**
     * @brief Get algorithm type from name
     * @param algorithmName Name of the algorithm
     * @return Algorithm type, or ExactHash if not found
     */
    static AlgorithmType algorithmTypeFromName(const QString& algorithmName);

    /**
     * @brief Get algorithm name from type
     * @param type Algorithm type
     * @return Algorithm name
     */
    static QString algorithmNameFromType(AlgorithmType type);

    /**
     * @brief Get algorithm description
     * @param type Algorithm type
     * @return Human-readable description of the algorithm
     */
    static QString algorithmDescription(AlgorithmType type);

    /**
     * @brief Get algorithm use case description
     * @param type Algorithm type
     * @return Description of when to use this algorithm
     */
    static QString algorithmUseCase(AlgorithmType type);

    /**
     * @brief Get algorithm performance characteristics
     * @param type Algorithm type
     * @return Map with speed, accuracy, and resource usage info
     */
    static QVariantMap algorithmPerformanceInfo(AlgorithmType type);

    /**
     * @brief Check if algorithm is available
     * @param type Algorithm type to check
     * @return true if algorithm can be created
     */
    static bool isAlgorithmAvailable(AlgorithmType type);

    /**
     * @brief Get recommended algorithm for file type
     * @param filePath Path to file to analyze
     * @return Recommended algorithm type for this file
     */
    static AlgorithmType recommendedAlgorithm(const QString& filePath);

private:
    DetectionAlgorithmFactory() = default; // Static class
};

#endif // DETECTION_ALGORITHM_FACTORY_H