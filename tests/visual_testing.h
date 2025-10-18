#pragma once

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QDir>
#include <QDateTime>
#include <QMap>
#include <QVariant>
#include <QRect>
#include <QPoint>
#include <QSize>
#include <QColor>
#include <QJsonObject>
#include <QJsonDocument>
#include <functional>

class QWidget;
class UIAutomation;

/**
 * @brief Comprehensive visual regression testing framework
 * 
 * Provides advanced visual testing capabilities including screenshot capture,
 * baseline management, image comparison algorithms, and difference visualization.
 */
class VisualTesting : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Image comparison algorithms
     */
    enum class ComparisonAlgorithm {
        PixelPerfect,       ///< Exact pixel-by-pixel comparison
        Perceptual,         ///< Human perception-based comparison
        Structural,         ///< Structural similarity (SSIM-like)
        Histogram,          ///< Color histogram comparison
        EdgeDetection,      ///< Edge-based comparison
        Fuzzy,             ///< Fuzzy matching with tolerance
        Adaptive           ///< Adaptive algorithm selection
    };

    /**
     * @brief Comparison result details
     */
    struct ComparisonResult {
        bool matches = false;               ///< Whether images match within threshold
        double similarity = 0.0;           ///< Similarity score (0.0 to 1.0)
        double threshold = 0.95;            ///< Threshold used for comparison
        ComparisonAlgorithm algorithm;      ///< Algorithm used
        QRect differenceRegion;             ///< Bounding box of differences
        int differentPixels = 0;            ///< Number of different pixels
        int totalPixels = 0;                ///< Total number of pixels
        QString errorMessage;               ///< Error message if comparison failed
        QMap<QString, QVariant> metrics;   ///< Additional comparison metrics
    };

    /**
     * @brief Baseline image metadata
     */
    struct BaselineInfo {
        QString name;                       ///< Baseline name/identifier
        QString filePath;                   ///< Path to baseline image
        QDateTime created;                  ///< Creation timestamp
        QDateTime lastUpdated;              ///< Last update timestamp
        QSize imageSize;                    ///< Image dimensions
        QString platform;                   ///< Platform (Windows, macOS, Linux)
        QString qtVersion;                  ///< Qt version used
        QString theme;                      ///< UI theme when captured
        double dpiScale = 1.0;             ///< DPI scaling factor
        QMap<QString, QVariant> metadata;  ///< Additional metadata
        QString description;                ///< Human-readable description
        QStringList tags;                   ///< Tags for organization
    };

    /**
     * @brief Visual test configuration
     */
    struct TestConfig {
        ComparisonAlgorithm algorithm = ComparisonAlgorithm::Perceptual;
        double threshold = 0.95;            ///< Similarity threshold (0.0 to 1.0)
        bool ignoreAntialiasing = true;     ///< Ignore antialiasing differences
        bool ignoreColors = false;          ///< Compare only structure, ignore colors
        bool ignoreMinorDifferences = true; ///< Ignore very small differences
        int pixelTolerance = 5;            ///< RGB tolerance per pixel (0-255)
        QRect cropRegion;                   ///< Region to compare (empty = full image)
        QList<QRect> ignoreRegions;        ///< Regions to ignore during comparison
        bool generateDiffImage = true;      ///< Generate difference visualization
        bool saveFailedComparisons = true; ///< Save failed comparison images
        QString baselineDirectory = "visual_baselines";
        QString outputDirectory = "visual_test_results";
        bool autoUpdateBaselines = false;   ///< Auto-update baselines on mismatch
        int maxDifferenceHighlights = 100; ///< Max difference regions to highlight
    };

    /**
     * @brief Difference visualization options
     */
    struct DiffVisualization {
        QColor differenceColor = Qt::red;   ///< Color for highlighting differences
        QColor addedColor = Qt::green;      ///< Color for added content
        QColor removedColor = Qt::blue;     ///< Color for removed content
        int highlightOpacity = 128;         ///< Opacity for difference overlay (0-255)
        bool showBoundingBoxes = true;      ///< Show bounding boxes around differences
        bool showPixelGrid = false;         ///< Show pixel grid for detailed view
        bool sideBySideView = true;         ///< Generate side-by-side comparison
        bool overlayView = true;            ///< Generate overlay comparison
        bool animatedGif = false;           ///< Generate animated GIF comparison
    };

    explicit VisualTesting(QObject* parent = nullptr);
    ~VisualTesting();

    // Configuration
    void setTestConfig(const TestConfig& config);
    TestConfig getTestConfig() const;
    void setDiffVisualization(const DiffVisualization& visualization);
    DiffVisualization getDiffVisualization() const;

    // Baseline management
    bool createBaseline(const QString& name, const QPixmap& image, const QString& description = "");
    bool createBaseline(const QString& name, QWidget* widget, const QString& description = "");
    bool updateBaseline(const QString& name, const QPixmap& image);
    bool deleteBaseline(const QString& name);
    bool baselineExists(const QString& name) const;
    BaselineInfo getBaselineInfo(const QString& name) const;
    QStringList getAvailableBaselines() const;
    QPixmap loadBaseline(const QString& name) const;

    // Image comparison
    ComparisonResult compareImages(const QPixmap& actual, const QPixmap& expected, 
                                 ComparisonAlgorithm algorithm = ComparisonAlgorithm::Perceptual,
                                 double threshold = 0.95) const;
    ComparisonResult compareWithBaseline(const QString& baselineName, const QPixmap& actual) const;
    ComparisonResult compareWithBaseline(const QString& baselineName, QWidget* widget) const;

    // Visual testing methods
    bool verifyVisual(const QString& testName, QWidget* widget);
    bool verifyVisual(const QString& testName, const QPixmap& image);
    bool verifyRegion(const QString& testName, QWidget* widget, const QRect& region);
    bool verifyMultipleWidgets(const QString& testName, const QList<QWidget*>& widgets);

    // Difference visualization
    QPixmap generateDifferenceImage(const QPixmap& actual, const QPixmap& expected, 
                                   const ComparisonResult& result) const;
    QPixmap generateSideBySideComparison(const QPixmap& actual, const QPixmap& expected,
                                        const ComparisonResult& result) const;
    QPixmap generateOverlayComparison(const QPixmap& actual, const QPixmap& expected,
                                     const ComparisonResult& result) const;
    bool saveComparisonReport(const QString& testName, const ComparisonResult& result,
                             const QPixmap& actual, const QPixmap& expected) const;

    // Batch operations
    QMap<QString, ComparisonResult> runVisualTestSuite(const QStringList& testNames, 
                                                      const QList<QWidget*>& widgets);
    bool updateAllBaselines(const QList<QWidget*>& widgets, const QStringList& testNames);
    QMap<QString, ComparisonResult> compareAllBaselines(const QList<QWidget*>& widgets,
                                                       const QStringList& testNames);

    // Platform and environment handling
    void setPlatformSuffix(const QString& platform);
    void setThemeSuffix(const QString& theme);
    void setDpiScale(double scale);
    QString generateBaselineName(const QString& baseName) const;

    // Utility methods
    static QPixmap captureWidget(QWidget* widget, bool includeFrame = false);
    static QPixmap captureRegion(QWidget* widget, const QRect& region);
    static QPixmap cropImage(const QPixmap& image, const QRect& region);
    static QPixmap scaleImage(const QPixmap& image, const QSize& targetSize, Qt::AspectRatioMode mode = Qt::KeepAspectRatio);
    static QImage convertToGrayscale(const QImage& image);
    static QImage applyGaussianBlur(const QImage& image, double radius);

    // Analysis and metrics
    QMap<QString, double> analyzeImage(const QPixmap& image) const;
    QList<QRect> findDifferenceRegions(const QPixmap& actual, const QPixmap& expected, 
                                      int tolerance = 5) const;
    double calculateStructuralSimilarity(const QImage& img1, const QImage& img2) const;
    double calculatePerceptualHash(const QImage& image) const;
    QMap<QRgb, int> calculateColorHistogram(const QImage& image) const;

    // Reporting and export
    bool generateHtmlReport(const QString& outputPath, 
                           const QMap<QString, ComparisonResult>& results) const;
    bool exportBaselines(const QString& exportPath, const QStringList& baselineNames = {}) const;
    bool importBaselines(const QString& importPath);
    QJsonObject generateTestReport(const QMap<QString, ComparisonResult>& results) const;

signals:
    void baselineCreated(const QString& name);
    void baselineUpdated(const QString& name);
    void baselineDeleted(const QString& name);
    void comparisonCompleted(const QString& testName, const ComparisonResult& result);
    void visualTestFailed(const QString& testName, const ComparisonResult& result);
    void batchTestCompleted(int totalTests, int passedTests, int failedTests);

private:
    TestConfig m_config;
    DiffVisualization m_diffVisualization;
    QString m_platformSuffix;
    QString m_themeSuffix;
    double m_dpiScale;
    
    mutable QMap<QString, BaselineInfo> m_baselineCache;
    mutable QMap<QString, QPixmap> m_imageCache;

    // Internal helper methods
    QString getBaselinePath(const QString& name) const;
    QString getOutputPath(const QString& testName) const;
    bool saveBaselineInfo(const QString& name, const BaselineInfo& info) const;
    BaselineInfo loadBaselineInfo(const QString& name) const;
    void clearCache();
    
    // Comparison algorithms
    ComparisonResult comparePixelPerfect(const QImage& img1, const QImage& img2, double threshold) const;
    ComparisonResult comparePerceptual(const QImage& img1, const QImage& img2, double threshold) const;
    ComparisonResult compareStructural(const QImage& img1, const QImage& img2, double threshold) const;
    ComparisonResult compareHistogram(const QImage& img1, const QImage& img2, double threshold) const;
    ComparisonResult compareEdgeDetection(const QImage& img1, const QImage& img2, double threshold) const;
    ComparisonResult compareFuzzy(const QImage& img1, const QImage& img2, double threshold) const;
    
    // Image processing utilities
    QImage preprocessImage(const QImage& image) const;
    QImage detectEdges(const QImage& image) const;
    QList<QPoint> findKeyPoints(const QImage& image) const;
    QImage normalizeImage(const QImage& image) const;
    
    // Difference detection
    QImage createDifferenceMap(const QImage& img1, const QImage& img2) const;
    QList<QRect> clusterDifferences(const QList<QPoint>& differencePoints) const;
    QRect calculateBoundingBox(const QList<QPoint>& points) const;
    
    // Visualization helpers
    QPixmap highlightDifferences(const QPixmap& image, const QList<QRect>& differences) const;
    QPixmap addAnnotations(const QPixmap& image, const ComparisonResult& result) const;
    QString formatComparisonMetrics(const ComparisonResult& result) const;
};

/**
 * @brief Visual test baseline manager for organizing and maintaining baselines
 */
class VisualBaselineManager : public QObject {
    Q_OBJECT

public:
    explicit VisualBaselineManager(const QString& baselineDirectory, QObject* parent = nullptr);

    // Baseline organization
    bool createBaselineSet(const QString& setName, const QString& description = "");
    bool deleteBaselineSet(const QString& setName);
    QStringList getBaselineSets() const;
    bool addToSet(const QString& setName, const QString& baselineName);
    bool removeFromSet(const QString& setName, const QString& baselineName);
    QStringList getBaselinesInSet(const QString& setName) const;

    // Baseline maintenance
    bool validateBaselines(const QString& setName = "");
    QStringList findOrphanedBaselines() const;
    bool cleanupOrphanedBaselines();
    bool archiveOldBaselines(int daysOld = 30);

    // Version control integration
    bool exportForVersionControl(const QString& exportPath, const QString& setName = "") const;
    bool importFromVersionControl(const QString& importPath);
    QStringList getBaselineHistory(const QString& baselineName) const;

    // Statistics and reporting
    QMap<QString, QVariant> getBaselineStatistics() const;
    bool generateMaintenanceReport(const QString& outputPath) const;

signals:
    void baselineSetCreated(const QString& setName);
    void baselineSetDeleted(const QString& setName);
    void baselinesValidated(int totalBaselines, int validBaselines, int invalidBaselines);
    void maintenanceCompleted(const QString& operation, bool success);

private:
    QString m_baselineDirectory;
    QMap<QString, QStringList> m_baselineSets;
    
    void loadBaselineSets();
    void saveBaselineSets();
    bool isBaselineValid(const QString& baselineName) const;
};

/**
 * @brief Convenience macros for visual testing
 */
#define VISUAL_VERIFY(testName, widget) \
    do { \
        if (!visualTesting.verifyVisual(testName, widget)) { \
            QFAIL(QString("Visual verification failed: %1").arg(testName).toUtf8().constData()); \
        } \
    } while(0)

#define VISUAL_CREATE_BASELINE(name, widget, description) \
    do { \
        if (!visualTesting.createBaseline(name, widget, description)) { \
            QFAIL(QString("Failed to create baseline: %1").arg(name).toUtf8().constData()); \
        } \
    } while(0)

#define VISUAL_COMPARE_WITH_BASELINE(baselineName, widget) \
    do { \
        auto result = visualTesting.compareWithBaseline(baselineName, widget); \
        if (!result.matches) { \
            QFAIL(QString("Visual comparison failed: %1 (similarity: %2)") \
                  .arg(baselineName).arg(result.similarity).toUtf8().constData()); \
        } \
    } while(0)