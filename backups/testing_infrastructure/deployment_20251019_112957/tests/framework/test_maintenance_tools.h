#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <memory>

// Forward declarations
class TestHarness;
class TestSuite;

/**
 * @brief Test flakiness detection and analysis
 */
struct TestFlakinessInfo {
    QString testName;
    QString suiteName;
    int totalRuns = 0;
    int failures = 0;
    int successes = 0;
    double flakinessRate = 0.0; // Percentage of runs that are inconsistent
    QStringList failureReasons;
    QList<qint64> executionTimes;
    QDateTime firstSeen;
    QDateTime lastSeen;
    bool isFlaky = false;
    QString recommendation;
};

/**
 * @brief Test execution performance metrics
 */
struct TestPerformanceMetrics {
    QString testName;
    QString suiteName;
    qint64 averageExecutionTime = 0;
    qint64 minExecutionTime = 0;
    qint64 maxExecutionTime = 0;
    qint64 standardDeviation = 0;
    QList<qint64> recentExecutionTimes;
    double performanceTrend = 0.0; // Positive = getting slower, Negative = getting faster
    bool isPerformanceRegression = false;
    QString optimizationSuggestion;
};

/**
 * @brief Test coverage analysis results
 */
struct TestCoverageGap {
    QString moduleName;
    QString className;
    QString methodName;
    int lineNumber = 0;
    QString filePath;
    QString gapType; // "uncovered_line", "uncovered_branch", "missing_test"
    int priority = 0; // 1-5, 5 being highest priority
    QString suggestedTestName;
    QString suggestedTestCode;
};

/**
 * @brief Baseline update recommendation
 */
struct BaselineUpdateRecommendation {
    QString testName;
    QString baselineType; // "visual", "performance", "data"
    QString currentBaseline;
    QString suggestedBaseline;
    QString reason;
    double confidence = 0.0; // 0.0-1.0
    QDateTime lastUpdated;
    bool autoUpdateRecommended = false;
    QString reviewNotes;
};

/**
 * @brief Test maintenance recommendation
 */
struct TestMaintenanceRecommendation {
    enum Type {
        RemoveObsoleteTest,
        UpdateTestData,
        RefactorDuplicateCode,
        OptimizeSlowTest,
        FixFlakyTest,
        AddMissingTest,
        UpdateBaseline,
        ImproveAssertion
    };
    
    Type type;
    QString testName;
    QString suiteName;
    QString description;
    QString suggestedAction;
    int priority = 0; // 1-5
    double estimatedEffort = 0.0; // Hours
    QStringList affectedTests;
    QString automationPossible; // "yes", "partial", "no"
};

/**
 * @brief Automated test flakiness detector
 */
class TestFlakinessDetector : public QObject {
    Q_OBJECT

public:
    explicit TestFlakinessDetector(QObject* parent = nullptr);

    // Configuration
    void setFlakinessThreshold(double threshold); // 0.0-1.0, default 0.05 (5%)
    void setMinimumRuns(int minRuns); // Minimum runs before considering flakiness
    void setAnalysisWindow(int days); // Days of history to analyze

    // Analysis
    void analyzeTestHistory(const QString& testName, const QList<bool>& results, 
                           const QList<qint64>& executionTimes);
    void analyzeAllTests(TestHarness* harness);
    
    // Results
    QList<TestFlakinessInfo> getFlakyTests() const;
    TestFlakinessInfo getTestFlakinessInfo(const QString& testName) const;
    bool isTestFlaky(const QString& testName) const;
    
    // Recommendations
    QStringList generateFlakinessRecommendations(const QString& testName) const;
    QString suggestFlakinessFixStrategy(const TestFlakinessInfo& info) const;

    // Reporting
    void generateFlakinessReport(const QString& outputPath) const;
    QJsonObject exportFlakinessData() const;
    void importFlakinessData(const QJsonObject& data);

signals:
    void flakyTestDetected(const QString& testName, double flakinessRate);
    void flakinessAnalysisCompleted(int totalTests, int flakyTests);

private:
    double calculateFlakinessRate(const QList<bool>& results) const;
    QString analyzeFlakinessPattern(const QList<bool>& results) const;
    QString generateRecommendation(const TestFlakinessInfo& info) const;

    double m_flakinessThreshold = 0.05;
    int m_minimumRuns = 10;
    int m_analysisWindowDays = 30;
    QMap<QString, TestFlakinessInfo> m_flakinessData;
};

/**
 * @brief Test execution time optimizer and analyzer
 */
class TestExecutionOptimizer : public QObject {
    Q_OBJECT

public:
    explicit TestExecutionOptimizer(QObject* parent = nullptr);

    // Configuration
    void setPerformanceRegressionThreshold(double threshold); // e.g., 1.5 = 50% slower
    void setOptimizationTargets(const QStringList& targets); // "slow_tests", "memory_usage", etc.

    // Analysis
    void analyzeTestPerformance(const QString& testName, const QList<qint64>& executionTimes);
    void analyzeAllTestPerformance(TestHarness* harness);
    
    // Optimization suggestions
    QList<TestPerformanceMetrics> getSlowTests(int topN = 10) const;
    QList<TestPerformanceMetrics> getPerformanceRegressions() const;
    QStringList generateOptimizationSuggestions(const QString& testName) const;
    
    // Execution time prediction
    qint64 predictExecutionTime(const QString& testName) const;
    qint64 estimateTotalExecutionTime(const QStringList& testNames) const;
    
    // Reporting
    void generatePerformanceReport(const QString& outputPath) const;
    QJsonObject exportPerformanceData() const;

signals:
    void performanceRegressionDetected(const QString& testName, qint64 oldTime, qint64 newTime);
    void optimizationOpportunityFound(const QString& testName, const QString& suggestion);

private:
    double calculatePerformanceTrend(const QList<qint64>& times) const;
    QString generateOptimizationSuggestion(const TestPerformanceMetrics& metrics) const;
    bool isPerformanceRegression(const QList<qint64>& times) const;

    double m_regressionThreshold = 1.5;
    QStringList m_optimizationTargets;
    QMap<QString, TestPerformanceMetrics> m_performanceData;
};

/**
 * @brief Test coverage gap analyzer
 */
class TestCoverageAnalyzer : public QObject {
    Q_OBJECT

public:
    explicit TestCoverageAnalyzer(QObject* parent = nullptr);

    // Configuration
    void setSourceDirectories(const QStringList& directories);
    void setTestDirectories(const QStringList& directories);
    void setCoverageThreshold(double threshold); // Minimum acceptable coverage

    // Analysis
    void analyzeCoverage(const QString& coverageFilePath);
    void analyzeCodeStructure();
    void identifyTestGaps();
    
    // Results
    QList<TestCoverageGap> getCoverageGaps() const;
    QList<TestCoverageGap> getHighPriorityGaps() const;
    double getOverallCoverage() const;
    QMap<QString, double> getModuleCoverage() const;
    
    // Recommendations
    QStringList generateTestSuggestions(const QString& className) const;
    QString generateTestCode(const TestCoverageGap& gap) const;
    
    // Reporting
    void generateCoverageReport(const QString& outputPath) const;

signals:
    void coverageGapFound(const TestCoverageGap& gap);
    void coverageAnalysisCompleted(double overallCoverage, int gaps);

private:
    void parseSourceFiles();
    void parseTestFiles();
    void identifyUncoveredCode();
    int calculateGapPriority(const TestCoverageGap& gap) const;

    QStringList m_sourceDirectories;
    QStringList m_testDirectories;
    double m_coverageThreshold = 0.85;
    QList<TestCoverageGap> m_coverageGaps;
    QMap<QString, double> m_moduleCoverage;
    double m_overallCoverage = 0.0;
};

/**
 * @brief Automated baseline management system
 */
class BaselineManager : public QObject {
    Q_OBJECT

public:
    explicit BaselineManager(QObject* parent = nullptr);

    // Configuration
    void setBaselineDirectory(const QString& directory);
    void setAutoUpdateThreshold(double threshold); // Confidence threshold for auto-updates
    void setReviewPeriod(int days); // Days between baseline reviews

    // Baseline analysis
    void analyzeBaselines();
    void checkBaselineAge();
    void detectBaselineChanges();
    
    // Update recommendations
    QList<BaselineUpdateRecommendation> getUpdateRecommendations() const;
    QList<BaselineUpdateRecommendation> getAutoUpdateCandidates() const;
    
    // Baseline operations
    bool updateBaseline(const QString& testName, const QString& baselineType, 
                       const QString& newBaseline, const QString& reason);
    bool revertBaseline(const QString& testName, const QString& baselineType);
    void backupBaselines();
    void restoreBaselines(const QDateTime& backupDate);
    
    // Reporting
    void generateBaselineReport(const QString& outputPath) const;
    QJsonObject exportBaselineData() const;

signals:
    void baselineUpdateRecommended(const BaselineUpdateRecommendation& recommendation);
    void baselineOutdated(const QString& testName, int daysSinceUpdate);
    void baselineUpdated(const QString& testName, const QString& baselineType);

private:
    void scanBaselineFiles();
    double calculateUpdateConfidence(const QString& testName, const QString& baselineType) const;
    bool shouldAutoUpdate(const BaselineUpdateRecommendation& recommendation) const;

    QString m_baselineDirectory;
    double m_autoUpdateThreshold = 0.9;
    int m_reviewPeriodDays = 30;
    QList<BaselineUpdateRecommendation> m_updateRecommendations;
    QMap<QString, QDateTime> m_baselineAges;
};

/**
 * @brief Comprehensive test maintenance and optimization coordinator
 */
class TestMaintenanceCoordinator : public QObject {
    Q_OBJECT

public:
    explicit TestMaintenanceCoordinator(QObject* parent = nullptr);

    // Component management
    void setFlakinessDetector(std::shared_ptr<TestFlakinessDetector> detector);
    void setExecutionOptimizer(std::shared_ptr<TestExecutionOptimizer> optimizer);
    void setCoverageAnalyzer(std::shared_ptr<TestCoverageAnalyzer> analyzer);
    void setBaselineManager(std::shared_ptr<BaselineManager> manager);

    // Comprehensive analysis
    void performFullAnalysis(TestHarness* harness);
    void performIncrementalAnalysis(const QStringList& changedTests);
    
    // Maintenance recommendations
    QList<TestMaintenanceRecommendation> getAllRecommendations() const;
    QList<TestMaintenanceRecommendation> getHighPriorityRecommendations() const;
    QList<TestMaintenanceRecommendation> getAutomatableRecommendations() const;
    
    // Automated maintenance
    void enableAutomaticMaintenance(bool enabled);
    void performAutomaticMaintenance();
    void scheduleMaintenanceTasks();
    
    // Reporting and dashboards
    void generateMaintenanceReport(const QString& outputPath) const;
    void generateMaintenanceDashboard(const QString& outputPath) const;
    QJsonObject exportMaintenanceData() const;
    
    // Configuration
    void loadConfiguration(const QString& configFile);
    void saveConfiguration(const QString& configFile) const;

signals:
    void analysisStarted();
    void analysisCompleted(int totalRecommendations, int highPriority);
    void maintenanceTaskCompleted(const TestMaintenanceRecommendation& task);
    void automaticMaintenanceCompleted(int tasksCompleted);

private slots:
    void onFlakyTestDetected(const QString& testName, double flakinessRate);
    void onPerformanceRegressionDetected(const QString& testName, qint64 oldTime, qint64 newTime);
    void onCoverageGapFound(const TestCoverageGap& gap);
    void onBaselineUpdateRecommended(const BaselineUpdateRecommendation& recommendation);

private:
    void aggregateRecommendations();
    void prioritizeRecommendations();
    bool canAutomate(const TestMaintenanceRecommendation& recommendation) const;
    void executeAutomaticTask(const TestMaintenanceRecommendation& task);

    // Components
    std::shared_ptr<TestFlakinessDetector> m_flakinessDetector;
    std::shared_ptr<TestExecutionOptimizer> m_executionOptimizer;
    std::shared_ptr<TestCoverageAnalyzer> m_coverageAnalyzer;
    std::shared_ptr<BaselineManager> m_baselineManager;

    // Configuration
    bool m_automaticMaintenanceEnabled = false;
    QStringList m_automationWhitelist;
    
    // State
    QList<TestMaintenanceRecommendation> m_allRecommendations;
    QElapsedTimer m_lastAnalysisTime;
};