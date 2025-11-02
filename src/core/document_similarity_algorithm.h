#ifndef DOCUMENT_SIMILARITY_ALGORITHM_H
#define DOCUMENT_SIMILARITY_ALGORITHM_H

#include "detection_algorithm.h"

/**
 * @class DocumentSimilarityAlgorithm
 * @brief Document content similarity detection
 * 
 * This algorithm extracts text content from documents and uses
 * text similarity algorithms to detect duplicate documents with
 * different filenames or minor content differences.
 */
class DocumentSimilarityAlgorithm : public DetectionAlgorithm
{
public:
    DocumentSimilarityAlgorithm();
    virtual ~DocumentSimilarityAlgorithm() = default;

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
     * @brief Extract text content from a file
     * @param filePath Path to the document
     * @return Extracted text content, or empty string if failed
     */
    QString extractTextContent(const QString& filePath) const;

    /**
     * @brief Extract text from plain text file
     * @param filePath Path to text file
     * @return File content as string
     */
    QString extractFromTextFile(const QString& filePath) const;

    /**
     * @brief Extract text from PDF file (basic implementation)
     * @param filePath Path to PDF file
     * @return Extracted text content
     */
    QString extractFromPDF(const QString& filePath) const;

    /**
     * @brief Normalize text for comparison
     * @param text Raw text content
     * @return Normalized text
     */
    QString normalizeText(const QString& text) const;

    /**
     * @brief Calculate Jaccard similarity between two texts
     * @param text1 First text
     * @param text2 Second text
     * @return Jaccard similarity score (0.0 to 1.0)
     */
    double calculateJaccardSimilarity(const QString& text1, const QString& text2) const;

    /**
     * @brief Calculate cosine similarity between two texts
     * @param text1 First text
     * @param text2 Second text
     * @return Cosine similarity score (0.0 to 1.0)
     */
    double calculateCosineSimilarity(const QString& text1, const QString& text2) const;

    /**
     * @brief Create word frequency map
     * @param text Input text
     * @return Map of word frequencies
     */
    QMap<QString, int> createWordFrequencyMap(const QString& text) const;

    /**
     * @brief Tokenize text into words
     * @param text Input text
     * @return List of words
     */
    QStringList tokenizeText(const QString& text) const;

    QVariantMap m_config;
    double m_similarityThreshold;
    QString m_similarityMethod;
    int m_minTextLength;
    bool m_ignoreCase;
    bool m_ignoreWhitespace;
};

#endif // DOCUMENT_SIMILARITY_ALGORITHM_H