#include "document_handler.h"
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QMimeDatabase>
#include <QMimeType>
#include <QTextStream>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>
#include <QtMath>

DocumentHandler::DocumentHandler(QObject* parent)
    : QObject(parent)
    , m_totalDocumentsProcessed(0)
    , m_totalTextExtracted(0)
    , m_totalErrors(0)
{
    qDebug() << "DocumentHandler initialized";
    
    // Set default configuration
    m_config = DocumentScanConfig();
}

DocumentHandler::~DocumentHandler()
{
    qDebug() << "DocumentHandler destroyed. Stats - Documents processed:" << m_totalDocumentsProcessed 
             << "Text extracted:" << m_totalTextExtracted << "bytes"
             << "Errors:" << m_totalErrors;
}

bool DocumentHandler::isDocumentFile(const QString& filePath)
{
    if (filePath.isEmpty() || !QFileInfo::exists(filePath)) {
        return false;
    }
    
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    // Check by extension first (fastest)
    QStringList documentExtensions = {
        "pdf",                                    // PDF documents
        "doc", "docx", "odt", "rtf",             // Word processing documents
        "xls", "xlsx", "ods",                    // Spreadsheet documents
        "ppt", "pptx", "odp",                    // Presentation documents
        "txt", "text",                           // Plain text files
        "md", "markdown",                        // Markdown files
        "html", "htm", "xhtml",                  // HTML files
        "xml",                                   // XML files
        "csv",                                   // CSV files
        "json",                                  // JSON files
        "log"                                    // Log files
    };
    
    if (documentExtensions.contains(extension)) {
        return true;
    }
    
    // Fallback to MIME type detection
    QMimeDatabase mimeDb;
    QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
    QString mimeName = mimeType.name();
    
    QStringList documentMimeTypes = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
        "text/plain",
        "text/markdown",
        "text/html",
        "application/xml",
        "text/xml",
        "text/csv",
        "application/json"
    };
    
    return documentMimeTypes.contains(mimeName);
}

QStringList DocumentHandler::supportedExtensions()
{
    return {
        "pdf", "doc", "docx", "odt", "rtf",
        "xls", "xlsx", "ods",
        "ppt", "pptx", "odp",
        "txt", "text", "md", "markdown",
        "html", "htm", "xhtml", "xml",
        "csv", "json", "log"
    };
}

void DocumentHandler::setConfiguration(const DocumentScanConfig& config)
{
    m_config = config;
    qDebug() << "Document configuration updated:"
             << "PDF:" << config.extractPdfContent
             << "Office:" << config.extractOfficeContent
             << "Text:" << config.extractTextFiles
             << "HTML:" << config.extractHtmlContent
             << "Markdown:" << config.extractMarkdownContent;
}

DocumentScanConfig DocumentHandler::configuration() const
{
    return m_config;
}

DocumentInfo DocumentHandler::extractDocumentInfo(const QString& filePath)
{
    DocumentInfo info;
    
    if (!QFileInfo::exists(filePath)) {
        emit errorOccurred(filePath, "Document file does not exist");
        return info;
    }
    
    if (!isProcessingEnabled(filePath)) {
        qDebug() << "Processing disabled for document type:" << filePath;
        return info;
    }
    
    QFileInfo fileInfo(filePath);
    if (fileInfo.size() > m_config.maxDocumentSize) {
        emit errorOccurred(filePath, QString("Document too large (%1 bytes, max: %2)")
                          .arg(fileInfo.size()).arg(m_config.maxDocumentSize));
        return info;
    }
    
    emit processingStarted(filePath);
    
    try {
        // Basic file information
        info.filePath = filePath;
        info.fileName = fileInfo.fileName();
        info.documentType = getDocumentFormat(filePath);
        info.fileSize = fileInfo.size();
        
        emit processingProgress(filePath, 25);
        
        // Extract text content
        info.textContent = extractTextContent(filePath);
        if (info.textContent.length() > m_config.maxTextLength) {
            info.textContent = info.textContent.left(m_config.maxTextLength);
        }
        
        emit processingProgress(filePath, 50);
        
        // Extract metadata
        if (m_config.includeMetadata) {
            info.metadata = extractMetadata(filePath);
            info.title = info.metadata.value("title").toString();
            info.author = info.metadata.value("author").toString();
            info.subject = info.metadata.value("subject").toString();
            info.pageCount = info.metadata.value("pageCount").toInt();
        }
        
        emit processingProgress(filePath, 75);
        
        // Calculate word count
        if (!info.textContent.isEmpty()) {
            QStringList words = info.textContent.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
            info.wordCount = words.size();
        }
        
        m_totalDocumentsProcessed++;
        m_totalTextExtracted += info.textContent.length();
        
        emit processingProgress(filePath, 100);
        emit processingCompleted(filePath, true);
        
        qDebug() << "Document processed:" << filePath 
                 << "Type:" << info.documentType
                 << "Text length:" << info.textContent.length()
                 << "Words:" << info.wordCount;
        
    } catch (const std::exception& e) {
        m_totalErrors++;
        emit errorOccurred(filePath, QString("Exception during processing: %1").arg(e.what()));
        emit processingCompleted(filePath, false);
    }
    
    return info;
}

QString DocumentHandler::extractTextContent(const QString& filePath)
{
    QString format = getDocumentFormat(filePath);
    
    if (format == "PDF") {
        return extractFromPdf(filePath);
    } else if (format == "DOC" || format == "DOCX" || format == "ODT" || format == "RTF") {
        return extractFromOfficeDoc(filePath);
    } else if (format == "TXT" || format == "TEXT" || format == "LOG" || format == "CSV" || format == "JSON") {
        return extractFromTextFile(filePath);
    } else if (format == "HTML" || format == "HTM" || format == "XHTML" || format == "XML") {
        return extractFromHtml(filePath);
    } else if (format == "MD" || format == "MARKDOWN") {
        return extractFromMarkdown(filePath);
    } else {
        qWarning() << "Unsupported document format for text extraction:" << format;
        return QString();
    }
}

QVariantMap DocumentHandler::extractMetadata(const QString& filePath)
{
    QString format = getDocumentFormat(filePath);
    
    if (format == "PDF") {
        return extractPdfMetadata(filePath);
    } else if (format == "DOC" || format == "DOCX" || format == "ODT" || format == "RTF") {
        return extractOfficeMetadata(filePath);
    } else {
        // Basic metadata for other formats
        QVariantMap metadata;
        QFileInfo fileInfo(filePath);
        metadata["fileName"] = fileInfo.fileName();
        metadata["fileSize"] = fileInfo.size();
        metadata["lastModified"] = fileInfo.lastModified();
        metadata["format"] = format;
        return metadata;
    }
}

double DocumentHandler::compareDocuments(const DocumentInfo& doc1, const DocumentInfo& doc2)
{
    if (doc1.textContent.isEmpty() || doc2.textContent.isEmpty()) {
        return 0.0;
    }
    
    // Use Jaccard similarity for document comparison
    double textSimilarity = calculateJaccardSimilarity(doc1.textContent, doc2.textContent);
    
    // Consider metadata similarity if available
    double metadataSimilarity = 0.0;
    if (m_config.includeMetadata && !doc1.title.isEmpty() && !doc2.title.isEmpty()) {
        metadataSimilarity = calculateJaccardSimilarity(doc1.title, doc2.title);
    }
    
    // Weighted combination (80% text, 20% metadata)
    return (textSimilarity * 0.8) + (metadataSimilarity * 0.2);
}

double DocumentHandler::calculateJaccardSimilarity(const QString& text1, const QString& text2)
{
    if (text1.isEmpty() || text2.isEmpty()) {
        return 0.0;
    }
    
    // Normalize texts
    QString norm1 = text1.toLower();
    QString norm2 = text2.toLower();
    
    if (!norm1.compare(norm2, Qt::CaseInsensitive)) {
        return 1.0; // Identical texts
    }
    
    // Tokenize into word sets (static helper)
    auto getWordSetStatic = [](const QString& text) -> QSet<QString> {
        QStringList words = text.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        QSet<QString> wordSet;
        for (const QString& word : words) {
            if (word.length() > 2) { // Skip very short words
                wordSet.insert(word);
            }
        }
        return wordSet;
    };
    
    QSet<QString> words1 = getWordSetStatic(norm1);
    QSet<QString> words2 = getWordSetStatic(norm2);
    
    if (words1.isEmpty() || words2.isEmpty()) {
        return 0.0;
    }
    
    // Calculate Jaccard similarity: |intersection| / |union|
    QSet<QString> intersection = words1;
    intersection.intersect(words2);
    
    QSet<QString> unionSet = words1;
    unionSet.unite(words2);
    
    return static_cast<double>(intersection.size()) / static_cast<double>(unionSet.size());
}

double DocumentHandler::calculateCosineSimilarity(const QString& text1, const QString& text2)
{
    if (text1.isEmpty() || text2.isEmpty()) {
        return 0.0;
    }
    
    // Get word frequency maps (static helper)
    auto getWordFrequencyStatic = [](const QString& text) -> QMap<QString, int> {
        QStringList words = text.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        QMap<QString, int> frequency;
        for (const QString& word : words) {
            if (word.length() > 2) { // Skip very short words
                frequency[word]++;
            }
        }
        return frequency;
    };
    
    QMap<QString, int> freq1 = getWordFrequencyStatic(text1.toLower());
    QMap<QString, int> freq2 = getWordFrequencyStatic(text2.toLower());
    
    if (freq1.isEmpty() || freq2.isEmpty()) {
        return 0.0;
    }
    
    // Calculate cosine similarity
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    // Get all unique words
    QSet<QString> allWords;
    for (auto it = freq1.begin(); it != freq1.end(); ++it) {
        allWords.insert(it.key());
    }
    for (auto it = freq2.begin(); it != freq2.end(); ++it) {
        allWords.insert(it.key());
    }
    
    // Calculate dot product and norms
    for (const QString& word : allWords) {
        int count1 = freq1.value(word, 0);
        int count2 = freq2.value(word, 0);
        
        dotProduct += count1 * count2;
        norm1 += count1 * count1;
        norm2 += count2 * count2;
    }
    
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }
    
    return dotProduct / (qSqrt(norm1) * qSqrt(norm2));
}

QString DocumentHandler::getDocumentFormat(const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    if (extension == "pdf") {
        return "PDF";
    } else if (extension == "doc") {
        return "DOC";
    } else if (extension == "docx") {
        return "DOCX";
    } else if (extension == "odt") {
        return "ODT";
    } else if (extension == "rtf") {
        return "RTF";
    } else if (extension == "txt" || extension == "text") {
        return "TXT";
    } else if (extension == "md" || extension == "markdown") {
        return "MD";
    } else if (extension == "html" || extension == "htm") {
        return "HTML";
    } else if (extension == "xhtml") {
        return "XHTML";
    } else if (extension == "xml") {
        return "XML";
    } else if (extension == "csv") {
        return "CSV";
    } else if (extension == "json") {
        return "JSON";
    } else if (extension == "log") {
        return "LOG";
    }
    
    return "UNKNOWN";
}

bool DocumentHandler::isProcessingEnabled(const QString& filePath) const
{
    QString format = getDocumentFormat(filePath);
    
    if (format == "PDF") {
        return m_config.extractPdfContent;
    } else if (format == "DOC" || format == "DOCX" || format == "ODT" || format == "RTF") {
        return m_config.extractOfficeContent;
    } else if (format == "TXT" || format == "TEXT" || format == "LOG" || format == "CSV" || format == "JSON") {
        return m_config.extractTextFiles;
    } else if (format == "HTML" || format == "HTM" || format == "XHTML" || format == "XML") {
        return m_config.extractHtmlContent;
    } else if (format == "MD" || format == "MARKDOWN") {
        return m_config.extractMarkdownContent;
    }
    
    return false;
}

// Private implementation methods

QString DocumentHandler::extractFromPdf(const QString& filePath)
{
    if (!m_config.extractPdfContent) {
        return QString();
    }
    
    // Use pdftotext command (from poppler-utils) to extract text
    QProcess pdfProcess;
    QStringList arguments;
    arguments << "-layout" << "-nopgbrk" << filePath << "-"; // Extract to stdout
    
    pdfProcess.start("pdftotext", arguments);
    if (!pdfProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start pdftotext command. Please ensure poppler-utils is installed.";
        return QString();
    }
    
    if (!pdfProcess.waitForFinished(30000)) {
        qWarning() << "PDF text extraction timed out";
        pdfProcess.kill();
        return QString();
    }
    
    if (pdfProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(pdfProcess.readAllStandardError());
        qWarning() << "PDF text extraction failed:" << error;
        return QString();
    }
    
    QString text = QString::fromUtf8(pdfProcess.readAllStandardOutput());
    return normalizeText(text);
}

QString DocumentHandler::extractFromOfficeDoc(const QString& filePath)
{
    if (!m_config.extractOfficeContent) {
        return QString();
    }
    
    QString format = getDocumentFormat(filePath);
    
    // Use appropriate tool based on format
    QProcess docProcess;
    QStringList arguments;
    
    if (format == "DOCX" || format == "ODT") {
        // Use pandoc to convert to plain text
        arguments << "-t" << "plain" << "--wrap=none" << filePath;
        docProcess.start("pandoc", arguments);
    } else if (format == "DOC") {
        // Use antiword for old DOC format
        arguments << filePath;
        docProcess.start("antiword", arguments);
    } else if (format == "RTF") {
        // Use unrtf for RTF format
        arguments << "--text" << filePath;
        docProcess.start("unrtf", arguments);
    } else {
        qWarning() << "Unsupported office document format:" << format;
        return QString();
    }
    
    if (!docProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start document extraction tool for" << format;
        return QString();
    }
    
    if (!docProcess.waitForFinished(30000)) {
        qWarning() << "Document text extraction timed out";
        docProcess.kill();
        return QString();
    }
    
    if (docProcess.exitCode() != 0) {
        QString error = QString::fromUtf8(docProcess.readAllStandardError());
        qWarning() << "Document text extraction failed:" << error;
        return QString();
    }
    
    QString text = QString::fromUtf8(docProcess.readAllStandardOutput());
    return normalizeText(text);
}

QString DocumentHandler::extractFromTextFile(const QString& filePath)
{
    if (!m_config.extractTextFiles) {
        return QString();
    }
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open text file:" << filePath;
        return QString();
    }
    
    QTextStream stream(&file);
    QString text = stream.readAll();
    file.close();
    
    return normalizeText(text);
}

QString DocumentHandler::extractFromHtml(const QString& filePath)
{
    if (!m_config.extractHtmlContent) {
        return QString();
    }
    
    // Read HTML file
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open HTML file:" << filePath;
        return QString();
    }
    
    QString html = QString::fromUtf8(file.readAll());
    file.close();
    
    // Simple HTML tag removal (basic implementation)
    QString text = html;
    text.remove(QRegularExpression("<[^>]*>"));
    text.replace("&nbsp;", " ");
    text.replace("&amp;", "&");
    text.replace("&lt;", "<");
    text.replace("&gt;", ">");
    text.replace("&quot;", "\"");
    
    return normalizeText(text);
}

QString DocumentHandler::extractFromMarkdown(const QString& filePath)
{
    if (!m_config.extractMarkdownContent) {
        return QString();
    }
    
    // Read Markdown file
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open Markdown file:" << filePath;
        return QString();
    }
    
    QString markdown = QString::fromUtf8(file.readAll());
    file.close();
    
    // Simple Markdown formatting removal
    QString text = markdown;
    text.remove(QRegularExpression("^#{1,6}\\s*", QRegularExpression::MultilineOption)); // Headers
    text.remove(QRegularExpression("\\*\\*([^*]+)\\*\\*")); // Bold
    text.remove(QRegularExpression("\\*([^*]+)\\*")); // Italic
    text.remove(QRegularExpression("`([^`]+)`")); // Inline code
    text.remove(QRegularExpression("```[\\s\\S]*?```")); // Code blocks
    text.remove(QRegularExpression("\\[([^\\]]+)\\]\\([^)]+\\)")); // Links
    
    return normalizeText(text);
}

QVariantMap DocumentHandler::extractPdfMetadata(const QString& filePath)
{
    QVariantMap metadata;
    
    // Use pdfinfo command to extract metadata
    QProcess pdfProcess;
    QStringList arguments;
    arguments << filePath;
    
    pdfProcess.start("pdfinfo", arguments);
    if (!pdfProcess.waitForStarted(5000)) {
        qWarning() << "Failed to start pdfinfo command";
        return metadata;
    }
    
    if (!pdfProcess.waitForFinished(10000)) {
        qWarning() << "PDF metadata extraction timed out";
        pdfProcess.kill();
        return metadata;
    }
    
    if (pdfProcess.exitCode() != 0) {
        return metadata;
    }
    
    QString output = QString::fromUtf8(pdfProcess.readAllStandardOutput());
    QStringList lines = output.split('\n', Qt::SkipEmptyParts);
    
    for (const QString& line : lines) {
        QStringList parts = line.split(':', Qt::SkipEmptyParts);
        if (parts.size() >= 2) {
            QString key = parts[0].trimmed().toLower();
            QString value = parts.mid(1).join(':').trimmed();
            
            if (key == "title") {
                metadata["title"] = value;
            } else if (key == "author") {
                metadata["author"] = value;
            } else if (key == "subject") {
                metadata["subject"] = value;
            } else if (key == "pages") {
                metadata["pageCount"] = value.toInt();
            } else if (key == "creator") {
                metadata["creator"] = value;
            } else if (key == "producer") {
                metadata["producer"] = value;
            }
        }
    }
    
    return metadata;
}

QVariantMap DocumentHandler::extractOfficeMetadata(const QString& filePath)
{
    QVariantMap metadata;
    
    // Basic metadata extraction for office documents
    // This would require more sophisticated tools for full metadata extraction
    QFileInfo fileInfo(filePath);
    metadata["fileName"] = fileInfo.fileName();
    metadata["fileSize"] = fileInfo.size();
    metadata["lastModified"] = fileInfo.lastModified();
    metadata["format"] = getDocumentFormat(filePath);
    
    return metadata;
}

QString DocumentHandler::normalizeText(const QString& text)
{
    QString normalized = text;
    
    if (m_config.normalizeWhitespace) {
        // Normalize whitespace
        normalized = normalized.simplified();
        normalized.replace(QRegularExpression("\\s+"), " ");
    }
    
    if (m_config.removePunctuation) {
        // Remove punctuation
        normalized.remove(QRegularExpression("[^\\w\\s]"));
    }
    
    if (!m_config.caseSensitive) {
        normalized = normalized.toLower();
    }
    
    return normalized.trimmed();
}

QStringList DocumentHandler::tokenizeText(const QString& text)
{
    return text.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
}

QSet<QString> DocumentHandler::getWordSet(const QString& text)
{
    QStringList words = tokenizeText(text);
    QSet<QString> wordSet;
    
    for (const QString& word : words) {
        if (word.length() > 2) { // Skip very short words
            wordSet.insert(word);
        }
    }
    
    return wordSet;
}

QMap<QString, int> DocumentHandler::getWordFrequency(const QString& text)
{
    QStringList words = tokenizeText(text);
    QMap<QString, int> frequency;
    
    for (const QString& word : words) {
        if (word.length() > 2) { // Skip very short words
            frequency[word]++;
        }
    }
    
    return frequency;
}

bool DocumentHandler::isTextFile(const QString& filePath) const
{
    QString format = getDocumentFormat(filePath);
    return format == "TXT" || format == "TEXT" || format == "LOG" || format == "CSV" || format == "JSON";
}

bool DocumentHandler::isPdfFile(const QString& filePath) const
{
    return getDocumentFormat(filePath) == "PDF";
}

bool DocumentHandler::isOfficeFile(const QString& filePath) const
{
    QString format = getDocumentFormat(filePath);
    return format == "DOC" || format == "DOCX" || format == "ODT" || format == "RTF";
}

bool DocumentHandler::isHtmlFile(const QString& filePath) const
{
    QString format = getDocumentFormat(filePath);
    return format == "HTML" || format == "HTM" || format == "XHTML" || format == "XML";
}

bool DocumentHandler::isMarkdownFile(const QString& filePath) const
{
    return getDocumentFormat(filePath) == "MD";
}