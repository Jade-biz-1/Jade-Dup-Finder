#pragma once

#include <QObject>
#include <QString>
#include <QByteArray>
#include <QMap>
#include <QVariant>

/**
 * @brief Abstract base class for duplicate detection algorithms
 * 
 * This interface allows different detection strategies to be plugged into
 * the DuplicateDetector, enabling exact matching, fuzzy matching, perceptual
 * hashing, and other detection modes.
 */
class DetectionAlgorithm : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Detection mode type
     */
    enum class Mode {
        Exact,          ///< Exact byte-for-byte match (hash-based)
        Perceptual,     ///< Perceptual similarity (images, audio)
        Fuzzy,          ///< Fuzzy/approximate matching
        Semantic        ///< Semantic similarity (text, documents)
    };
    Q_ENUM(Mode)

    /**
     * @brief File type hint for algorithm selection
     */
    enum class FileType {
        Generic,        ///< Generic file
        Image,          ///< Image file (jpg, png, etc.)
        Audio,          ///< Audio file (mp3, wav, etc.)
        Video,          ///< Video file
        Document,       ///< Text/document file
        Archive         ///< Archive file (zip, tar, etc.)
    };
    Q_ENUM(FileType)

    /**
     * @brief Similarity result
     */
    struct SimilarityResult {
        double similarityScore = 0.0;  ///< 0.0 to 1.0, where 1.0 is identical
        bool isDuplicate = false;       ///< Whether files are considered duplicates
        QString algorithm;              ///< Algorithm used
        QMap<QString, QVariant> metadata; ///< Additional metadata
    };

    explicit DetectionAlgorithm(QObject* parent = nullptr);
    virtual ~DetectionAlgorithm() = default;

    /**
     * @brief Get the detection mode this algorithm implements
     */
    virtual Mode getMode() const = 0;

    /**
     * @brief Get supported file types
     */
    virtual QList<FileType> getSupportedFileTypes() const = 0;

    /**
     * @brief Check if algorithm can handle this file
     */
    virtual bool canHandle(const QString& filePath, FileType type) const = 0;

    /**
     * @brief Compute signature/fingerprint for a file
     * @param filePath Path to file
     * @param type File type hint
     * @return Signature data (format depends on algorithm)
     */
    virtual QByteArray computeSignature(const QString& filePath, FileType type) = 0;

    /**
     * @brief Compare two signatures
     * @param signature1 First signature
     * @param signature2 Second signature
     * @return Similarity result
     */
    virtual SimilarityResult compareSignatures(const QByteArray& signature1,
                                               const QByteArray& signature2) = 0;

    /**
     * @brief Get similarity threshold for considering files as duplicates
     * @return Threshold value (0.0 to 1.0)
     */
    virtual double getSimilarityThreshold() const = 0;

    /**
     * @brief Set similarity threshold
     */
    virtual void setSimilarityThreshold(double threshold) = 0;

    /**
     * @brief Get algorithm configuration
     */
    virtual QMap<QString, QVariant> getConfiguration() const;

    /**
     * @brief Set algorithm configuration
     */
    virtual void setConfiguration(const QMap<QString, QVariant>& config);

    /**
     * @brief Get algorithm name
     */
    virtual QString getName() const = 0;

    /**
     * @brief Get algorithm description
     */
    virtual QString getDescription() const = 0;

signals:
    /**
     * @brief Emitted during signature computation
     */
    void signatureProgress(int current, int total);

    /**
     * @brief Emitted when an error occurs
     */
    void errorOccurred(const QString& error);

protected:
    QMap<QString, QVariant> m_config;
};

/**
 * @brief Factory for creating detection algorithms
 */
class DetectionAlgorithmFactory {
public:
    /**
     * @brief Create an algorithm by name
     */
    static DetectionAlgorithm* createAlgorithm(const QString& name, QObject* parent = nullptr);

    /**
     * @brief Get list of available algorithms
     */
    static QStringList getAvailableAlgorithms();

    /**
     * @brief Register a custom algorithm
     */
    static void registerAlgorithm(const QString& name,
                                  std::function<DetectionAlgorithm*(QObject*)> factory);

private:
    static QMap<QString, std::function<DetectionAlgorithm*(QObject*)>>& getRegistry();
};

// Convenience macro for algorithm registration
#define REGISTER_DETECTION_ALGORITHM(name, className) \
    namespace { \
        struct className##Registrar { \
            className##Registrar() { \
                DetectionAlgorithmFactory::registerAlgorithm(name, \
                    [](QObject* parent) -> DetectionAlgorithm* { \
                        return new className(parent); \
                    }); \
            } \
        }; \
        static className##Registrar className##_registrar; \
    }
