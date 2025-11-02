#ifndef DOCUMENT_HANDLER_H
#define DOCUMENT_HANDLER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QVariantMap>

/**
 * @brief Document information for content-based duplicate detection
 */
struct DocumentInfo {
    QString filePath;           // Path to the document file
    QString fileName;           // Just the filename
    QString documentType;       // Type of document (PDF, DOC, TXT, etc.)
    QString title;              // Document title (if available)
    QString author;             // Document author (if available)
    QString subject;            // Document subject (if available)
    QString textContent;        // Extracted text content
    QVariantMap metadata;       // Additional metadata
    qint64 fileSize;           // File size in bytes
    int pageCount;             // Number of pages (for paginated documents)
    int wordCount;             // Approximate word count
    
    DocumentInfo() : fileSize(0), pageCount(0), wordCount(0) {}
};

/**
 * @brief Document scanning configuration
 */
struct DocumentScanConfig {
    bool extractPdfContent = true;          // Extract text from PDF files
    bool extractOfficeContent = true;       // Extract text from Office documents
    bool extractTextFiles = true;           // Process plain text files
    bool extractHtmlContent = true;         // Extract text from HTML files
    bool extractMarkdownContent = true;     // Process Markdown files
    bool includeMetadata = true;            // Include document metadata in comparison
    int maxDocumentSize = 50 * 1024 * 1024; // Max document size to process (50MB)
    int maxTextLength = 1024 * 1024;        // Max text content to extract (1MB)
    bool normalizeWhitespace = true;        // Normalize whitespace in text
    bool removePunctuation = false;         // Remove punctuation for comparison
    bool caseSensitive = false;             // Case-sensitive text comparison
};

/**
 * @brief Handler for document content extraction and comparison
 * 
 * This class provides functionality to extract text content from various
 * document formats and perform content-based duplicate detection.
 */
class DocumentHandler : public QObject
{
    Q_OBJECT

public:
    explicit DocumentHandler(QObject* parent = nullptr);
    ~DocumentHandler();

    /**
     * @brief Check if a file is a supported document format
     * @param filePath Path to the file to check
     * @return True if the file is a supported document format
     */
    static bool isDocumentFile(const QString& filePath);
    
    /**
     * @brief Get list of supported document extensions
     * @return List of supported extensions (e.g., "pdf", "doc", "txt")
     */
    static QStringList supportedExtensions();
    
    /**
     * @brief Set document scanning configuration
     * @param config Configuration settings for document scanning
     */
    void setConfiguration(const DocumentScanConfig& config);
    
    /**
     * @brief Get current document scanning configuration
     * @return Current configuration settings
     */
    DocumentScanConfig configuration() const;
    
    /**
     * @brief Extract document information and content
     * @param filePath Path to the document file
     * @return Document information with extracted content
     */
    DocumentInfo extractDocumentInfo(const QString& filePath);
    
    /**
     * @brief Extract text content from document
     * @param filePath Path to the document file
     * @return Extracted text content (empty if extraction failed)
     */
    QString extractTextContent(const QString& filePath);
    
    /**
     * @brief Extract metadata from document
     * @param filePath Path to the document file
     * @return Document metadata as variant map
     */
    QVariantMap extractMetadata(const QString& filePath);
    
    /**
     * @brief Compare two documents for content similarity
     * @param doc1 First document info
     * @param doc2 Second document info
     * @return Similarity score (0.0-1.0, 1.0 = identical)
     */
    double compareDocuments(const DocumentInfo& doc1, const DocumentInfo& doc2);
    
    /**
     * @brief Calculate text similarity using Jaccard index
     * @param text1 First text
     * @param text2 Second text
     * @return Jaccard similarity score (0.0-1.0)
     */
    static double calculateJaccardSimilarity(const QString& text1, const QString& text2);
    
    /**
     * @brief Calculate text similarity using cosine similarity
     * @param text1 First text
     * @param text2 Second text
     * @return Cosine similarity score (0.0-1.0)
     */
    static double calculateCosineSimilarity(const QString& text1, const QString& text2);
    
    /**
     * @brief Get document format type
     * @param filePath Path to document file
     * @return Document format name (e.g., "PDF", "DOC", "TXT")
     */
    static QString getDocumentFormat(const QString& filePath);
    
    /**
     * @brief Check if document processing is enabled for given file type
     * @param filePath Path to document file
     * @return True if processing is enabled for this document type
     */
    bool isProcessingEnabled(const QString& filePath) const;

signals:
    /**
     * @brief Emitted when document processing starts
     * @param filePath Path to document being processed
     */
    void processingStarted(const QString& filePath);
    
    /**
     * @brief Emitted when processing progress updates
     * @param filePath Path to document being processed
     * @param percentage Progress percentage (0-100)
     */
    void processingProgress(const QString& filePath, int percentage);
    
    /**
     * @brief Emitted when document processing completes
     * @param filePath Path to document that was processed
     * @param success True if processing was successful
     */
    void processingCompleted(const QString& filePath, bool success);
    
    /**
     * @brief Emitted when an error occurs during document operations
     * @param filePath Path to document that caused error
     * @param error Error message
     */
    void errorOccurred(const QString& filePath, const QString& error);

private:
    // Format-specific extraction methods
    QString extractFromPdf(const QString& filePath);
    QString extractFromOfficeDoc(const QString& filePath);
    QString extractFromTextFile(const QString& filePath);
    QString extractFromHtml(const QString& filePath);
    QString extractFromMarkdown(const QString& filePath);
    
    // Metadata extraction methods
    QVariantMap extractPdfMetadata(const QString& filePath);
    QVariantMap extractOfficeMetadata(const QString& filePath);
    
    // Text processing utilities
    QString normalizeText(const QString& text);
    QStringList tokenizeText(const QString& text);
    QSet<QString> getWordSet(const QString& text);
    QMap<QString, int> getWordFrequency(const QString& text);
    
    // Utility methods
    bool isTextFile(const QString& filePath) const;
    bool isPdfFile(const QString& filePath) const;
    bool isOfficeFile(const QString& filePath) const;
    bool isHtmlFile(const QString& filePath) const;
    bool isMarkdownFile(const QString& filePath) const;
    
    // Member variables
    DocumentScanConfig m_config;
    
    // Statistics
    int m_totalDocumentsProcessed;
    qint64 m_totalTextExtracted;
    int m_totalErrors;
};

#endif // DOCUMENT_HANDLER_H