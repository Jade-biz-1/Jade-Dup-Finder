#include "visual_testing.h"
#include <QWidget>
#include <QApplication>
#include <QScreen>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QTextStream>
#include <QDebug>
#include <QBuffer>
#include <QImageReader>
#include <QImageWriter>
#include <QPainter>
#include <QFontMetrics>
#include <QtMath>
#include <QCryptographicHash>

VisualTesting::VisualTesting(QObject* parent)
    : QObject(parent)
    , m_dpiScale(1.0)
{
    // Set default configuration
    m_config.algorithm = ComparisonAlgorithm::Perceptual;
    m_config.threshold = 0.95;
    m_config.ignoreAntialiasing = true;
    m_config.pixelTolerance = 5;
    m_config.generateDiffImage = true;
    m_config.saveFailedComparisons = true;
    
    // Set default visualization options
    m_diffVisualization.differenceColor = Qt::red;
    m_diffVisualization.highlightOpacity = 128;
    m_diffVisualization.showBoundingBoxes = true;
    m_diffVisualization.sideBySideView = true;
    
    // Create directories
    QDir().mkpath(m_config.baselineDirectory);
    QDir().mkpath(m_config.outputDirectory);
    
    // Detect platform and DPI
    m_platformSuffix = QSysInfo::productType();
    
    QScreen* screen = QApplication::primaryScreen();
    if (screen) {
        m_dpiScale = screen->devicePixelRatio();
    }
}

VisualTesting::~VisualTesting() = default;

void VisualTesting::setTestConfig(const TestConfig& config) {
    m_config = config;
    QDir().mkpath(m_config.baselineDirectory);
    QDir().mkpath(m_config.outputDirectory);
}

VisualTesting::TestConfig VisualTesting::getTestConfig() const {
    return m_config;
}

void VisualTesting::setDiffVisualization(const DiffVisualization& visualization) {
    m_diffVisualization = visualization;
}

VisualTesting::DiffVisualization VisualTesting::getDiffVisualization() const {
    return m_diffVisualization;
}

bool VisualTesting::createBaseline(const QString& name, const QPixmap& image, const QString& description) {
    if (image.isNull()) {
        qWarning() << "Cannot create baseline with null image:" << name;
        return false;
    }
    
    QString baselinePath = getBaselinePath(name);
    QFileInfo fileInfo(baselinePath);
    QDir().mkpath(fileInfo.absolutePath());
    
    // Save image
    if (!image.save(baselinePath, "PNG")) {
        qWarning() << "Failed to save baseline image:" << baselinePath;
        return false;
    }
    
    // Create baseline info
    BaselineInfo info;
    info.name = name;
    info.filePath = baselinePath;
    info.created = QDateTime::currentDateTime();
    info.lastUpdated = info.created;
    info.imageSize = image.size();
    info.platform = m_platformSuffix;
    info.qtVersion = QT_VERSION_STR;
    info.theme = m_themeSuffix;
    info.dpiScale = m_dpiScale;
    info.description = description;
    
    // Save baseline info
    if (!saveBaselineInfo(name, info)) {
        QFile::remove(baselinePath);
        return false;
    }
    
    // Update cache
    m_baselineCache[name] = info;
    m_imageCache[name] = image;
    
    emit baselineCreated(name);
    return true;
}

bool VisualTesting::createBaseline(const QString& name, QWidget* widget, const QString& description) {
    if (!widget) {
        qWarning() << "Cannot create baseline with null widget:" << name;
        return false;
    }
    
    QPixmap screenshot = captureWidget(widget);
    return createBaseline(name, screenshot, description);
}

QPixmap VisualTesting::loadBaseline(const QString& name) const {
    // Check cache first
    if (m_imageCache.contains(name)) {
        return m_imageCache[name];
    }
    
    QString baselinePath = getBaselinePath(name);
    QPixmap image(baselinePath);
    
    if (!image.isNull()) {
        m_imageCache[name] = image;
    }
    
    return image;
}

VisualTesting::ComparisonResult VisualTesting::compareImages(const QPixmap& actual, const QPixmap& expected, 
                                                           ComparisonAlgorithm algorithm, double threshold) const {
    ComparisonResult result;
    result.algorithm = algorithm;
    result.threshold = threshold;
    
    if (actual.isNull() || expected.isNull()) {
        result.errorMessage = "One or both images are null";
        return result;
    }
    
    if (actual.size() != expected.size()) {
        result.errorMessage = QString("Image size mismatch: %1x%2 vs %3x%4")
                             .arg(actual.width()).arg(actual.height())
                             .arg(expected.width()).arg(expected.height());
        return result;
    }
    
    QImage img1 = actual.toImage();
    QImage img2 = expected.toImage();
    
    // Preprocess images if needed
    if (m_config.ignoreAntialiasing || m_config.ignoreColors) {
        img1 = preprocessImage(img1);
        img2 = preprocessImage(img2);
    }
    
    // Apply comparison algorithm
    switch (algorithm) {
        case ComparisonAlgorithm::PixelPerfect:
            result = comparePixelPerfect(img1, img2, threshold);
            break;
        case ComparisonAlgorithm::Perceptual:
            result = comparePerceptual(img1, img2, threshold);
            break;
        case ComparisonAlgorithm::Structural:
            result = compareStructural(img1, img2, threshold);
            break;
        case ComparisonAlgorithm::Histogram:
            result = compareHistogram(img1, img2, threshold);
            break;
        case ComparisonAlgorithm::Fuzzy:
            result = compareFuzzy(img1, img2, threshold);
            break;
        default:
            result = comparePerceptual(img1, img2, threshold);
            break;
    }
    
    result.totalPixels = img1.width() * img1.height();
    result.matches = result.similarity >= threshold;
    
    return result;
}

VisualTesting::ComparisonResult VisualTesting::compareWithBaseline(const QString& baselineName, const QPixmap& actual) const {
    QPixmap baseline = loadBaseline(baselineName);
    if (baseline.isNull()) {
        ComparisonResult result;
        result.errorMessage = QString("Baseline not found: %1").arg(baselineName);
        return result;
    }
    
    return compareImages(actual, baseline, m_config.algorithm, m_config.threshold);
}

bool VisualTesting::verifyVisual(const QString& testName, QWidget* widget) {
    if (!widget) {
        qWarning() << "Cannot verify visual with null widget:" << testName;
        return false;
    }
    
    QPixmap screenshot = captureWidget(widget);
    return verifyVisual(testName, screenshot);
}

bool VisualTesting::verifyVisual(const QString& testName, const QPixmap& image) {
    QString baselineName = generateBaselineName(testName);
    
    if (!baselineExists(baselineName)) {
        qWarning() << "Baseline does not exist for test:" << testName;
        if (m_config.autoUpdateBaselines) {
            qDebug() << "Auto-creating baseline for:" << testName;
            return createBaseline(baselineName, image, QString("Auto-created baseline for %1").arg(testName));
        }
        return false;
    }
    
    ComparisonResult result = compareWithBaseline(baselineName, image);
    
    emit comparisonCompleted(testName, result);
    
    if (!result.matches) {
        emit visualTestFailed(testName, result);
        
        if (m_config.saveFailedComparisons) {
            QPixmap baseline = loadBaseline(baselineName);
            saveComparisonReport(testName, result, image, baseline);
        }
        
        if (m_config.autoUpdateBaselines) {
            qDebug() << "Auto-updating baseline for failed test:" << testName;
            updateBaseline(baselineName, image);
        }
    }
    
    return result.matches;
}

QPixmap VisualTesting::generateDifferenceImage(const QPixmap& actual, const QPixmap& expected, 
                                              const ComparisonResult& result) const {
    if (actual.size() != expected.size()) {
        return QPixmap();
    }
    
    QImage actualImg = actual.toImage();
    QImage expectedImg = expected.toImage();
    QImage diffImg(actualImg.size(), QImage::Format_ARGB32);
    
    int tolerance = m_config.pixelTolerance;
    
    for (int y = 0; y < actualImg.height(); ++y) {
        for (int x = 0; x < actualImg.width(); ++x) {
            QRgb actualPixel = actualImg.pixel(x, y);
            QRgb expectedPixel = expectedImg.pixel(x, y);
            
            int rDiff = qAbs(qRed(actualPixel) - qRed(expectedPixel));
            int gDiff = qAbs(qGreen(actualPixel) - qGreen(expectedPixel));
            int bDiff = qAbs(qBlue(actualPixel) - qBlue(expectedPixel));
            
            if (rDiff > tolerance || gDiff > tolerance || bDiff > tolerance) {
                // Highlight difference
                diffImg.setPixel(x, y, m_diffVisualization.differenceColor.rgba());
            } else {
                // Keep original pixel but make it semi-transparent
                QColor originalColor(actualPixel);
                originalColor.setAlpha(128);
                diffImg.setPixel(x, y, originalColor.rgba());
            }
        }
    }
    
    return QPixmap::fromImage(diffImg);
}

// Static utility methods
QPixmap VisualTesting::captureWidget(QWidget* widget, bool includeFrame) {
    if (!widget) {
        return QPixmap();
    }
    
    if (includeFrame) {
        return widget->grab();
    } else {
        return widget->grab(widget->rect());
    }
}

// Comparison algorithm implementations
VisualTesting::ComparisonResult VisualTesting::comparePixelPerfect(const QImage& img1, const QImage& img2, double threshold) const {
    ComparisonResult result;
    result.algorithm = ComparisonAlgorithm::PixelPerfect;
    result.threshold = threshold;
    
    int totalPixels = img1.width() * img1.height();
    int matchingPixels = 0;
    
    for (int y = 0; y < img1.height(); ++y) {
        for (int x = 0; x < img1.width(); ++x) {
            if (img1.pixel(x, y) == img2.pixel(x, y)) {
                matchingPixels++;
            }
        }
    }
    
    result.similarity = static_cast<double>(matchingPixels) / totalPixels;
    result.differentPixels = totalPixels - matchingPixels;
    
    return result;
}

VisualTesting::ComparisonResult VisualTesting::comparePerceptual(const QImage& img1, const QImage& img2, double threshold) const {
    ComparisonResult result;
    result.algorithm = ComparisonAlgorithm::Perceptual;
    result.threshold = threshold;
    
    int totalPixels = img1.width() * img1.height();
    int differentPixels = 0;
    double totalDifference = 0.0;
    
    int tolerance = m_config.pixelTolerance;
    
    for (int y = 0; y < img1.height(); ++y) {
        for (int x = 0; x < img1.width(); ++x) {
            QRgb pixel1 = img1.pixel(x, y);
            QRgb pixel2 = img2.pixel(x, y);
            
            int rDiff = qAbs(qRed(pixel1) - qRed(pixel2));
            int gDiff = qAbs(qGreen(pixel1) - qGreen(pixel2));
            int bDiff = qAbs(qBlue(pixel1) - qBlue(pixel2));
            
            // Calculate perceptual difference (weighted by human eye sensitivity)
            double diff = (0.299 * rDiff + 0.587 * gDiff + 0.114 * bDiff) / 255.0;
            totalDifference += diff;
            
            if (rDiff > tolerance || gDiff > tolerance || bDiff > tolerance) {
                differentPixels++;
            }
        }
    }
    
    result.similarity = 1.0 - (totalDifference / totalPixels);
    result.differentPixels = differentPixels;
    
    return result;
}

// Helper methods
QString VisualTesting::getBaselinePath(const QString& name) const {
    QString fileName = QString("%1.png").arg(name);
    return QDir(m_config.baselineDirectory).absoluteFilePath(fileName);
}

QString VisualTesting::generateBaselineName(const QString& baseName) const {
    QString name = baseName;
    
    if (!m_platformSuffix.isEmpty()) {
        name += "_" + m_platformSuffix;
    }
    
    if (!m_themeSuffix.isEmpty()) {
        name += "_" + m_themeSuffix;
    }
    
    if (m_dpiScale != 1.0) {
        name += QString("_dpi%1").arg(QString::number(m_dpiScale, 'f', 1));
    }
    
    return name;
}

bool VisualTesting::baselineExists(const QString& name) const {
    return QFile::exists(getBaselinePath(name));
}

bool VisualTesting::saveBaselineInfo(const QString& name, const BaselineInfo& info) const {
    QString infoPath = getBaselinePath(name) + ".json";
    
    QJsonObject json;
    json["name"] = info.name;
    json["created"] = info.created.toString(Qt::ISODate);
    json["lastUpdated"] = info.lastUpdated.toString(Qt::ISODate);
    json["imageWidth"] = info.imageSize.width();
    json["imageHeight"] = info.imageSize.height();
    json["platform"] = info.platform;
    json["qtVersion"] = info.qtVersion;
    json["theme"] = info.theme;
    json["dpiScale"] = info.dpiScale;
    json["description"] = info.description;
    
    QJsonArray tagsArray;
    for (const QString& tag : info.tags) {
        tagsArray.append(tag);
    }
    json["tags"] = tagsArray;
    
    QJsonDocument doc(json);
    
    QFile file(infoPath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to save baseline info:" << infoPath;
        return false;
    }
    
    file.write(doc.toJson());
    return true;
}

QImage VisualTesting::preprocessImage(const QImage& image) const {
    QImage processed = image;
    
    if (m_config.ignoreColors) {
        processed = convertToGrayscale(processed);
    }
    
    if (m_config.ignoreAntialiasing) {
        // Simple antialiasing reduction by slight blur
        processed = applyGaussianBlur(processed, 0.5);
    }
    
    return processed;
}

QImage VisualTesting::convertToGrayscale(const QImage& image) {
    QImage grayscale = image.convertToFormat(QImage::Format_Grayscale8);
    return grayscale;
}

QImage VisualTesting::applyGaussianBlur(const QImage& image, double radius) {
    // Simplified blur implementation
    Q_UNUSED(radius)
    return image; // For now, return original image
}

// Additional comparison algorithms (simplified implementations)
VisualTesting::ComparisonResult VisualTesting::compareStructural(const QImage& img1, const QImage& img2, double threshold) const {
    // Simplified structural comparison
    return comparePerceptual(img1, img2, threshold);
}

VisualTesting::ComparisonResult VisualTesting::compareHistogram(const QImage& img1, const QImage& img2, double threshold) const {
    ComparisonResult result;
    result.algorithm = ComparisonAlgorithm::Histogram;
    result.threshold = threshold;
    
    // Calculate color histograms
    QMap<QRgb, int> hist1 = calculateColorHistogram(img1);
    QMap<QRgb, int> hist2 = calculateColorHistogram(img2);
    
    // Compare histograms (simplified)
    int totalPixels = img1.width() * img1.height();
    int similarColors = 0;
    
    for (auto it = hist1.begin(); it != hist1.end(); ++it) {
        if (hist2.contains(it.key())) {
            similarColors += qMin(it.value(), hist2[it.key()]);
        }
    }
    
    result.similarity = static_cast<double>(similarColors) / totalPixels;
    return result;
}

VisualTesting::ComparisonResult VisualTesting::compareFuzzy(const QImage& img1, const QImage& img2, double threshold) const {
    // Fuzzy comparison with higher tolerance
    ComparisonResult result = comparePerceptual(img1, img2, threshold);
    result.algorithm = ComparisonAlgorithm::Fuzzy;
    
    // Apply fuzzy logic - be more lenient with small differences
    if (result.similarity > 0.85) {
        result.similarity = qMin(1.0, result.similarity + 0.1);
    }
    
    return result;
}

QMap<QRgb, int> VisualTesting::calculateColorHistogram(const QImage& image) const {
    QMap<QRgb, int> histogram;
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            QRgb pixel = image.pixel(x, y);
            histogram[pixel]++;
        }
    }
    
    return histogram;
}