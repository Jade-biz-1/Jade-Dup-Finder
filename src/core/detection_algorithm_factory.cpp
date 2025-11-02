#include "detection_algorithm_factory.h"
#include "exact_hash_algorithm.h"
#include "quick_scan_algorithm.h"
#include "perceptual_hash_algorithm.h"
#include "document_similarity_algorithm.h"
#include <QFileInfo>
#include <QDebug>

std::unique_ptr<DetectionAlgorithm> DetectionAlgorithmFactory::create(AlgorithmType type)
{
    switch (type) {
        case ExactHash:
            return std::make_unique<ExactHashAlgorithm>();
            
        case QuickScan:
            return std::make_unique<QuickScanAlgorithm>();
            
        case PerceptualHash:
            return std::make_unique<PerceptualHashAlgorithm>();
            
        case DocumentSimilarity:
            return std::make_unique<DocumentSimilarityAlgorithm>();
            
        default:
            qWarning() << "Unknown algorithm type:" << type;
            return std::make_unique<ExactHashAlgorithm>(); // Fallback to default
    }
}

std::unique_ptr<DetectionAlgorithm> DetectionAlgorithmFactory::create(const QString& algorithmName)
{
    AlgorithmType type = algorithmTypeFromName(algorithmName);
    return create(type);
}

QStringList DetectionAlgorithmFactory::availableAlgorithms()
{
    return {
        algorithmNameFromType(ExactHash),
        algorithmNameFromType(QuickScan),
        algorithmNameFromType(PerceptualHash),
        algorithmNameFromType(DocumentSimilarity)
    };
}

DetectionAlgorithmFactory::AlgorithmType DetectionAlgorithmFactory::algorithmTypeFromName(const QString& algorithmName)
{
    QString name = algorithmName.toLower();
    
    if (name == "exact" || name == "exacthash" || name == "hash") {
        return ExactHash;
    } else if (name == "quick" || name == "quickscan" || name == "fast") {
        return QuickScan;
    } else if (name == "perceptual" || name == "perceptualhash" || name == "image") {
        return PerceptualHash;
    } else if (name == "document" || name == "documentsimilarity" || name == "content") {
        return DocumentSimilarity;
    }
    
    qWarning() << "Unknown algorithm name:" << algorithmName << "- using ExactHash";
    return ExactHash;
}

QString DetectionAlgorithmFactory::algorithmNameFromType(AlgorithmType type)
{
    switch (type) {
        case ExactHash:
            return "ExactHash";
        case QuickScan:
            return "QuickScan";
        case PerceptualHash:
            return "PerceptualHash";
        case DocumentSimilarity:
            return "DocumentSimilarity";
        default:
            return "ExactHash";
    }
}

QString DetectionAlgorithmFactory::algorithmDescription(AlgorithmType type)
{
    switch (type) {
        case ExactHash:
            return "SHA-256 hash-based exact content matching. 100% accurate for identical files.";
            
        case QuickScan:
            return "Fast size and filename matching. Quick results but may miss some duplicates.";
            
        case PerceptualHash:
            return "Image similarity detection using perceptual hashing. Finds visually similar images.";
            
        case DocumentSimilarity:
            return "Document content similarity using text analysis. Finds similar documents with different names.";
            
        default:
            return "Unknown algorithm";
    }
}

QString DetectionAlgorithmFactory::algorithmUseCase(AlgorithmType type)
{
    switch (type) {
        case ExactHash:
            return "Best for: All file types, guaranteed accuracy, finding exact duplicates";
            
        case QuickScan:
            return "Best for: Large datasets, quick preview, obvious duplicates";
            
        case PerceptualHash:
            return "Best for: Photo libraries, finding resized/compressed images, visual duplicates";
            
        case DocumentSimilarity:
            return "Best for: Document collections, PDFs with different names, content-based duplicates";
            
        default:
            return "Unknown use case";
    }
}

QVariantMap DetectionAlgorithmFactory::algorithmPerformanceInfo(AlgorithmType type)
{
    QVariantMap info;
    
    switch (type) {
        case ExactHash:
            info["speed"] = "Medium";
            info["accuracy"] = "100%";
            info["memory"] = "Low";
            info["cpu"] = "Medium";
            info["throughput"] = "500 MB/s";
            break;
            
        case QuickScan:
            info["speed"] = "Very Fast";
            info["accuracy"] = "80-90%";
            info["memory"] = "Very Low";
            info["cpu"] = "Low";
            info["throughput"] = "5000+ files/s";
            break;
            
        case PerceptualHash:
            info["speed"] = "Fast";
            info["accuracy"] = "95% (for images)";
            info["memory"] = "Low";
            info["cpu"] = "Medium";
            info["throughput"] = "200 images/s";
            break;
            
        case DocumentSimilarity:
            info["speed"] = "Medium";
            info["accuracy"] = "90-95%";
            info["memory"] = "Medium";
            info["cpu"] = "High";
            info["throughput"] = "100 documents/s";
            break;
            
        default:
            info["speed"] = "Unknown";
            info["accuracy"] = "Unknown";
            info["memory"] = "Unknown";
            info["cpu"] = "Unknown";
            info["throughput"] = "Unknown";
    }
    
    return info;
}

bool DetectionAlgorithmFactory::isAlgorithmAvailable(AlgorithmType type)
{
    // For now, all algorithms are available
    // In the future, this could check for dependencies (e.g., image libraries)
    Q_UNUSED(type)
    return true;
}

DetectionAlgorithmFactory::AlgorithmType DetectionAlgorithmFactory::recommendedAlgorithm(const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    // Image files - use perceptual hashing
    QStringList imageExtensions = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif", "webp", "ico"};
    if (imageExtensions.contains(extension)) {
        return PerceptualHash;
    }
    
    // Document files - use content similarity
    QStringList documentExtensions = {"pdf", "doc", "docx", "txt", "rtf", "odt"};
    if (documentExtensions.contains(extension)) {
        return DocumentSimilarity;
    }
    
    // For all other files, use exact hash matching
    return ExactHash;
}