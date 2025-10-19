#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QDateTime>
#include <QJsonObject>
#include <QJsonArray>
#include <QUrl>
#include <memory>

// Forward declarations
class TestHarness;
class TestResults;
class TestMaintenanceCoordinator;

/**
 * @brief Test trend analysis data point
 */
struct TestTrendDataPoint {
    QDateTime timestamp;
    QString testName;
    QString suiteName;
    bool passed = false;
    qint64 executionTime = 0;
    double memoryUsage = 0.0;
    double cpuUsage = 0.0;
    QMap<QString, QVariant> customMetrics;
};

/**
 * @brief Test effectiveness metrics
 */
struct TestEffectivenessMetrics {
    QString testName;
    QString suiteName;
    int totalRuns = 0;
    int defectsFound = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double effectivenessScore = 0.0; // 0.0-1.0
    double maintenanceCost = 0.0; // Hours per month
    double valueScore = 0.0; // Effectiveness / Cost ratio
    QStringList improvementSuggestions;
    QString recommendation; // "keep", "improve", "remove"
};

/**
 * @brief Test suite analytics summary
 */
struct TestSuiteAnalytics {
    QString suiteName;
    int totalTests = 0;
    int activeTests = 0;
    int flakyTests = 0;
    int slowTests = 0;
    double averageExecutionTime = 0.0;
    double codeCoverage = 0.0;
    double successRate = 0.0;
    double maintenanceScore = 0.0; // 0.0-1.0, higher is better
    QMap<QString, int> testCategories;
    QList<QString> topIssues;
};

/**
 * @brief Performance trend analysis
 */
struct PerformanceTrendAnalysis {
    QString metricName;
    QList<TestTrendDataPoint> dataPoints;
    double currentValue = 0.0;
    double trendSlope = 0.0; // Positive = improving, Negative = degrading
    double volatility = 0.0; // Standard deviation of changes
    QString trendDirection; // "improving", "stable", "degrading"
    QString prediction; // Short-term prediction
    QStringList alerts;
};

/**
 * @brief Interactive dashboard configuration
 */
struct DashboardConfig {
    QString title = "Test Analytics Dashboard";
    QStringList enabledWidgets;
    QMap<QString, QVariant> widgetSettings;
    QString refreshInterval = "5m"; // 5 minutes
    bool autoRefresh = true;
    QString theme = "light"; // "light", "dark", "auto"
    QStringList userRoles; // For access control
};

/**
 * @brief Test trend analyzer for historical data analysis
 */
class TestTrendAnalyzer : public QObject {
    Q_OBJECT

public:
    explicit TestTrendAnalyzer(QObject* parent = nullptr);

    // Data management
    void addDataPoint(const TestTrendDataPoint& dataPoint);
    void addDataPoints(const QList<TestTrendDataPoint>& dataPoints);
    void loadHistoricalData(const QString& dataFilePath);
    void saveHistoricalData(const QString& dataFilePath) const;

    // Trend analysis
    PerformanceTrendAnalysis analyzePerformanceTrend(const QString& testName, 
                                                    const QString& metricName,
                                                    int daysPeriod = 30) const;
    QList<PerformanceTrendAnalysis> analyzeAllTrends(int daysPeriod = 30) const;
    
    // Success rate trends
    double calculateSuccessRateTrend(const QString& testName, int daysPeriod = 30) const;
    QMap<QString, double> getSuccessRateTrends(int daysPeriod = 30) const;
    
    // Performance predictions
    double predictExecutionTime(const QString& testName, int daysAhead = 7) const;
    QMap<QString, double> predictAllExecutionTimes(int daysAhead = 7) const;
    
    // Anomaly detection
    QStringList detectAnomalies(const QString& testName, double threshold = 2.0) const;
    QMap<QString, QStringList> detectAllAnomalies(double threshold = 2.0) const;
    
    // Reporting
    QJsonObject exportTrendData() const;
    void generateTrendReport(const QString& outputPath) const;

signals:
    void trendAnalysisCompleted(const QString& testName);
    void anomalyDetected(const QString& testName, const QString& anomaly);
    void performanceRegressionPredicted(const QString& testName, double predictedIncrease);

private:
    QList<TestTrendDataPoint> getDataPoints(const QString& testName, 
                                           const QDateTime& startDate,
                                           const QDateTime& endDate) const;
    double calculateTrendSlope(const QList<double>& values) const;
    double calculateVolatility(const QList<double>& values) const;
    QString determineTrendDirection(double slope, double volatility) const;
    QStringList generateTrendAlerts(const PerformanceTrendAnalysis& analysis) const;

    QList<TestTrendDataPoint> m_historicalData;
    QMap<QString, QList<TestTrendDataPoint>> m_testDataCache;
};

/**
 * @brief Test effectiveness analyzer for measuring test value
 */
class TestEffectivenessAnalyzer : public QObject {
    Q_OBJECT

public:
    explicit TestEffectivenessAnalyzer(QObject* parent = nullptr);

    // Configuration
    void setDefectTrackingEnabled(bool enabled);
    void setMaintenanceCostTracking(bool enabled);
    void setCostPerHour(double costPerHour);

    // Analysis
    void analyzeTestEffectiveness(const QString& testName);
    void analyzeAllTestEffectiveness();
    TestEffectivenessMetrics getTestEffectiveness(const QString& testName) const;
    
    // Defect correlation
    void recordDefectFound(const QString& testName, const QString& defectId);
    void recordFalsePositive(const QString& testName, const QString& reason);
    void recordFalseNegative(const QString& testName, const QString& missedDefect);
    
    // Maintenance cost tracking
    void recordMaintenanceTime(const QString& testName, double hours, const QString& reason);
    double getMaintenanceCost(const QString& testName) const;
    
    // Value analysis
    QList<TestEffectivenessMetrics> getHighValueTests(int topN = 10) const;
    QList<TestEffectivenessMetrics> getLowValueTests(int bottomN = 10) const;
    QStringList getTestsToRemove() const;
    QStringList getTestsToImprove() const;
    
    // Reporting
    void generateEffectivenessReport(const QString& outputPath) const;
    QJsonObject exportEffectivenessData() const;

signals:
    void lowValueTestDetected(const QString& testName, double valueScore);
    void highMaintenanceCostDetected(const QString& testName, double cost);
    void effectivenessAnalysisCompleted(int totalTests, int lowValueTests);

private:
    double calculateEffectivenessScore(const TestEffectivenessMetrics& metrics) const;
    double calculateValueScore(const TestEffectivenessMetrics& metrics) const;
    QString generateRecommendation(const TestEffectivenessMetrics& metrics) const;
    QStringList generateImprovementSuggestions(const TestEffectivenessMetrics& metrics) const;

    bool m_defectTrackingEnabled = true;
    bool m_maintenanceCostTracking = true;
    double m_costPerHour = 100.0; // Default cost per hour
    QMap<QString, TestEffectivenessMetrics> m_effectivenessData;
    QMap<QString, QStringList> m_defectHistory;
    QMap<QString, double> m_maintenanceCosts;
};

/**
 * @brief Comprehensive HTML report generator
 */
class HtmlReportGenerator : public QObject {
    Q_OBJECT

public:
    explicit HtmlReportGenerator(QObject* parent = nullptr);

    // Configuration
    void setTemplate(const QString& templatePath);
    void setOutputDirectory(const QString& directory);
    void setIncludeCharts(bool includeCharts);
    void setIncludeInteractivity(bool interactive);

    // Report generation
    void generateComprehensiveReport(const TestResults& results, 
                                   const QString& outputPath);
    void generateTrendReport(const QList<PerformanceTrendAnalysis>& trends,
                           const QString& outputPath);
    void generateEffectivenessReport(const QList<TestEffectivenessMetrics>& metrics,
                                   const QString& outputPath);
    void generateMaintenanceReport(const QJsonObject& maintenanceData,
                                 const QString& outputPath);

    // Chart generation
    void generatePerformanceChart(const QList<TestTrendDataPoint>& data,
                                const QString& outputPath);
    void generateSuccessRateChart(const QMap<QString, double>& successRates,
                                const QString& outputPath);
    void generateCoverageChart(const QMap<QString, double>& coverage,
                             const QString& outputPath);

    // Interactive elements
    void addFilterControls(const QStringList& filterOptions);
    void addSortingControls(const QStringList& sortOptions);
    void addSearchFunctionality();
    void addExportFunctionality();

signals:
    void reportGenerationStarted(const QString& reportType);
    void reportGenerationCompleted(const QString& outputPath);
    void chartGenerationCompleted(const QString& chartPath);

private:
    QString loadTemplate(const QString& templateName) const;
    QString generateHtmlTable(const QJsonArray& data, const QStringList& columns) const;
    QString generateChartScript(const QString& chartType, const QJsonObject& data) const;
    QString generateCssStyles() const;
    QString generateJavaScript() const;
    void copyStaticAssets(const QString& outputDir) const;

    QString m_templatePath;
    QString m_outputDirectory;
    bool m_includeCharts = true;
    bool m_includeInteractivity = true;
    QMap<QString, QString> m_templates;
};

/**
 * @brief Interactive test dashboard generator
 */
class TestDashboard : public QObject {
    Q_OBJECT

public:
    explicit TestDashboard(QObject* parent = nullptr);

    // Configuration
    void setConfiguration(const DashboardConfig& config);
    DashboardConfig getConfiguration() const { return m_config; }
    
    // Widget management
    void addWidget(const QString& widgetId, const QString& widgetType, 
                  const QJsonObject& settings);
    void removeWidget(const QString& widgetId);
    void updateWidget(const QString& widgetId, const QJsonObject& data);
    
    // Data sources
    void setTestHarness(TestHarness* harness);
    void setTrendAnalyzer(TestTrendAnalyzer* analyzer);
    void setEffectivenessAnalyzer(TestEffectivenessAnalyzer* analyzer);
    void setMaintenanceCoordinator(TestMaintenanceCoordinator* coordinator);
    
    // Dashboard generation
    void generateDashboard(const QString& outputPath);
    void updateDashboard(); // Refresh with latest data
    void startAutoRefresh();
    void stopAutoRefresh();
    
    // Real-time updates
    void enableWebSocketUpdates(int port = 8080);
    void broadcastUpdate(const QString& widgetId, const QJsonObject& data);
    
    // Export functionality
    void exportDashboardData(const QString& format, const QString& outputPath);
    QJsonObject getDashboardSnapshot() const;

signals:
    void dashboardGenerated(const QString& outputPath);
    void dashboardUpdated();
    void widgetDataUpdated(const QString& widgetId);
    void realTimeUpdateSent(const QString& widgetId, const QJsonObject& data);

private slots:
    void onAutoRefreshTimer();
    void onTestCompleted(const QString& suiteName, const QString& testName, bool passed);
    void onTrendAnalysisCompleted(const QString& testName);

private:
    struct DashboardWidget {
        QString id;
        QString type;
        QString title;
        QJsonObject settings;
        QJsonObject data;
        QDateTime lastUpdated;
    };

    void initializeWidgets();
    void updateWidgetData(const QString& widgetId);
    QString generateWidgetHtml(const DashboardWidget& widget) const;
    QString generateDashboardHtml() const;
    QJsonObject collectWidgetData(const QString& widgetType) const;
    void setupWebSocketServer();

    DashboardConfig m_config;
    QMap<QString, DashboardWidget> m_widgets;
    
    // Data sources
    TestHarness* m_testHarness = nullptr;
    TestTrendAnalyzer* m_trendAnalyzer = nullptr;
    TestEffectivenessAnalyzer* m_effectivenessAnalyzer = nullptr;
    TestMaintenanceCoordinator* m_maintenanceCoordinator = nullptr;
    
    // Auto-refresh
    QTimer* m_refreshTimer = nullptr;
    bool m_autoRefreshEnabled = false;
    
    // WebSocket server for real-time updates
    QObject* m_webSocketServer = nullptr; // Would be QWebSocketServer in real implementation
    QList<QObject*> m_connectedClients;
};

/**
 * @brief Advanced analytics coordinator
 */
class AdvancedAnalyticsCoordinator : public QObject {
    Q_OBJECT

public:
    explicit AdvancedAnalyticsCoordinator(QObject* parent = nullptr);

    // Component management
    void setTrendAnalyzer(std::shared_ptr<TestTrendAnalyzer> analyzer);
    void setEffectivenessAnalyzer(std::shared_ptr<TestEffectivenessAnalyzer> analyzer);
    void setHtmlReportGenerator(std::shared_ptr<HtmlReportGenerator> generator);
    void setTestDashboard(std::shared_ptr<TestDashboard> dashboard);

    // Comprehensive analytics
    void performFullAnalytics(TestHarness* harness);
    void performIncrementalAnalytics(const QStringList& changedTests);
    void schedulePeriodicAnalytics(int intervalMinutes = 60);
    
    // Report generation
    void generateAllReports(const QString& outputDirectory);
    void generateExecutiveSummary(const QString& outputPath);
    void generateTechnicalReport(const QString& outputPath);
    void generateMaintenanceReport(const QString& outputPath);
    
    // Dashboard management
    void setupDashboard(const DashboardConfig& config);
    void updateDashboard();
    void publishDashboard(const QString& url);
    
    // Data export
    void exportAnalyticsData(const QString& format, const QString& outputPath);
    void importAnalyticsData(const QString& filePath);
    
    // Configuration
    void loadConfiguration(const QString& configFile);
    void saveConfiguration(const QString& configFile) const;

signals:
    void analyticsStarted();
    void analyticsCompleted(const QString& summary);
    void reportGenerated(const QString& reportType, const QString& outputPath);
    void dashboardUpdated(const QUrl& dashboardUrl);
    void alertGenerated(const QString& alertType, const QString& message);

private slots:
    void onPeriodicAnalyticsTimer();
    void onTrendAnalysisCompleted(const QString& testName);
    void onEffectivenessAnalysisCompleted(int totalTests, int lowValueTests);

private:
    void aggregateAnalyticsResults();
    void generateAlerts();
    QString generateExecutiveSummary() const;
    QJsonObject collectAllAnalyticsData() const;

    // Components
    std::shared_ptr<TestTrendAnalyzer> m_trendAnalyzer;
    std::shared_ptr<TestEffectivenessAnalyzer> m_effectivenessAnalyzer;
    std::shared_ptr<HtmlReportGenerator> m_htmlGenerator;
    std::shared_ptr<TestDashboard> m_dashboard;

    // Configuration
    QString m_outputDirectory;
    bool m_periodicAnalyticsEnabled = false;
    int m_analyticsIntervalMinutes = 60;
    
    // State
    QTimer* m_periodicTimer = nullptr;
    QDateTime m_lastAnalyticsRun;
    QJsonObject m_lastAnalyticsResults;
};