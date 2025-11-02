#include "perceptual_hash_algorithm.h"
#include <QImageReader>
#include <QDebug>
#include <QBitArray>

PerceptualHashAlgorithm::PerceptualHashAlgorithm()
{
    m_config = getDefaultConfiguration();
    
    // Load configuration values
    m_similarityThreshold = m_config.value("similarity_threshold", 0.90).toDouble();
    m_hashSize = m_config.value("hash_size", 8).toInt();
    
    // Calculate maximum Hamming distance for threshold
    int totalBits = m_hashSize * m_hashSize;
    m_maxHammingDistance = static_cast<int>(totalBits * (1.0 - m_similarityThreshold));
}

QString PerceptualHashAlgorithm::name() const
{
    return "Perceptual Hash";
}

QString PerceptualHashAlgorithm::description() const
{
    return "Image similarity detection using perceptual hashing (dHash algorithm). "
           "Finds visually similar images even when they have been resized, compressed, "
           "or converted to different formats. Works by comparing structural differences "
           "rather than exact pixel values.";
}

QStringList PerceptualHashAlgorithm::supportedExtensions() const
{
    return {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif", "webp", "ico", "svg"};
}

QByteArray PerceptualHashAlgorithm::computeSignature(const QString& filePath)
{
    QImage image = loadAndPreprocessImage(filePath);
    
    if (image.isNull()) {
        qWarning() << "Failed to load image:" << filePath;
        return QByteArray();
    }
    
    return computeDHash(image);
}

bool PerceptualHashAlgorithm::compareSignatures(const QByteArray& sig1, const QByteArray& sig2)
{
    if (sig1.isEmpty() || sig2.isEmpty()) {
        return false;
    }
    
    int distance = hammingDistance(sig1, sig2);
    return distance <= m_maxHammingDistance;
}

double PerceptualHashAlgorithm::similarityScore(const QByteArray& sig1, const QByteArray& sig2)
{
    if (sig1.isEmpty() || sig2.isEmpty()) {
        return 0.0;
    }
    
    int distance = hammingDistance(sig1, sig2);
    return hammingToSimilarity(distance);
}

void PerceptualHashAlgorithm::setConfiguration(const QVariantMap& config)
{
    m_config = config;
    
    // Update internal settings
    m_similarityThreshold = config.value("similarity_threshold", 0.90).toDouble();
    m_hashSize = config.value("hash_size", 8).toInt();
    
    // Recalculate maximum Hamming distance
    int totalBits = m_hashSize * m_hashSize;
    m_maxHammingDistance = static_cast<int>(totalBits * (1.0 - m_similarityThreshold));
}

QVariantMap PerceptualHashAlgorithm::getConfiguration() const
{
    return m_config;
}

QVariantMap PerceptualHashAlgorithm::getDefaultConfiguration() const
{
    QVariantMap config;
    
    config["similarity_threshold"] = 0.90;
    config["similarity_threshold_description"] = "Minimum similarity (0.0-1.0) to consider images duplicates";
    config["similarity_threshold_min"] = 0.70;
    config["similarity_threshold_max"] = 0.99;
    
    config["hash_size"] = 8;
    config["hash_size_description"] = "Hash matrix size (8x8 = 64 bits, larger = more precise but slower)";
    
    return config;
}

QVariantMap PerceptualHashAlgorithm::getPerformanceInfo() const
{
    QVariantMap info;
    
    info["speed"] = "Fast";
    info["accuracy"] = "95% (for images)";
    info["memory_usage"] = "Low";
    info["cpu_usage"] = "Medium";
    info["typical_throughput"] = "200 images/s";
    info["best_for"] = "Photo libraries, finding resized/compressed images";
    info["limitations"] = "Only works with image files, may miss heavily modified images";
    
    return info;
}

QByteArray PerceptualHashAlgorithm::computeDHash(const QImage& image) const
{
    // Resize image to (hash_size + 1) x hash_size for difference calculation
    QImage resized = image.scaled(m_hashSize + 1, m_hashSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    
    // Convert to grayscale
    QImage grayscale = resized.convertToFormat(QImage::Format_Grayscale8);
    
    // Calculate difference hash
    QBitArray hash(m_hashSize * m_hashSize);
    int bitIndex = 0;
    
    for (int y = 0; y < m_hashSize; ++y) {
        for (int x = 0; x < m_hashSize; ++x) {
            // Get pixel values for comparison
            QRgb leftPixel = grayscale.pixel(x, y);
            QRgb rightPixel = grayscale.pixel(x + 1, y);
            
            // Extract grayscale values
            int leftGray = qGray(leftPixel);
            int rightGray = qGray(rightPixel);
            
            // Set bit if left pixel is brighter than right pixel
            hash.setBit(bitIndex++, leftGray > rightGray);
        }
    }
    
    // Convert bit array to byte array
    QByteArray result;
    result.resize((hash.size() + 7) / 8); // Round up to nearest byte
    result.fill(0);
    
    for (int i = 0; i < hash.size(); ++i) {
        if (hash.testBit(i)) {
            int byteIndex = i / 8;
            int bitIndex = i % 8;
            result[byteIndex] |= (1 << bitIndex);
        }
    }
    
    return result;
}

int PerceptualHashAlgorithm::hammingDistance(const QByteArray& hash1, const QByteArray& hash2) const
{
    if (hash1.size() != hash2.size()) {
        qWarning() << "Hash sizes don't match:" << hash1.size() << "vs" << hash2.size();
        return INT_MAX; // Maximum distance for mismatched sizes
    }
    
    int distance = 0;
    
    for (int i = 0; i < hash1.size(); ++i) {
        // XOR bytes and count set bits
        unsigned char xorResult = static_cast<unsigned char>(hash1[i]) ^ static_cast<unsigned char>(hash2[i]);
        
        // Count bits using Brian Kernighan's algorithm
        while (xorResult) {
            distance++;
            xorResult &= xorResult - 1; // Clear the lowest set bit
        }
    }
    
    return distance;
}

double PerceptualHashAlgorithm::hammingToSimilarity(int hammingDistance) const
{
    int totalBits = m_hashSize * m_hashSize;
    
    if (hammingDistance >= totalBits) {
        return 0.0;
    }
    
    // Convert distance to similarity score
    double similarity = 1.0 - (static_cast<double>(hammingDistance) / totalBits);
    
    return std::max(0.0, std::min(1.0, similarity));
}

QImage PerceptualHashAlgorithm::loadAndPreprocessImage(const QString& filePath) const
{
    QImageReader reader(filePath);
    
    // Set format hint for better performance
    reader.setAutoDetectImageFormat(true);
    
    // Limit image size for performance (perceptual hash doesn't need full resolution)
    QSize maxSize(1024, 1024);
    if (reader.size().isValid() && (reader.size().width() > maxSize.width() || reader.size().height() > maxSize.height())) {
        reader.setScaledSize(reader.size().scaled(maxSize, Qt::KeepAspectRatio));
    }
    
    QImage image = reader.read();
    
    if (image.isNull()) {
        qWarning() << "Failed to read image:" << filePath << "Error:" << reader.errorString();
        return QImage();
    }
    
    // Ensure image is in a format we can work with
    if (image.format() == QImage::Format_Invalid) {
        qWarning() << "Invalid image format for:" << filePath;
        return QImage();
    }
    
    return image;
}