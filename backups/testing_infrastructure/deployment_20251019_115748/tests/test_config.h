#pragma once

#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QJsonObject>
#include <QJsonDocument>

/**
 * @brief Test configuration management system
 * 
 * Provides centralized configuration for all test categories and execution parameters.
 * Supports test categorization, tagging, and runtime configuration management.
 */
class TestConfig {
public:
    /**
     * @brief Test categories for organization and filtering
     */
    enum class Category {
        Unit,           ///< Unit tests for individual components
        Integration,    ///< Integration tests for component interactions
        Performance,    ///< Performance and benchmark tests
        UI,            ///< User interface and visual tests
        EndToEnd,      ///< Complete workflow tests
        Security,      ///< Security and safety tests
        Regression,    ///< Regression tests for bug fixes
        Smoke          ///< Basic functionality smoke tests
    };

    /**
     * @brief Test execution modes
     */
    enum class ExecutionMode {
        Sequential,    ///< Run tests one after another
        Parallel,      ///< Run tests in parallel where possible
        Isolated       ///< Run each test in isolated environment
    };

    /**
     * @brief Test priority levels
     */
    enum class Priority {
        Critical,      ///< Must pass for release
        High,          ///< Important functionality
        Medium,        ///< Standard functionality
        Low            ///< Nice to have
    };

    /**
     * @brief Test configuration structure
     */
    struct TestSuiteConfig {
        QString name;                    ///< Test suite name
        Category category;               ///< Test category
        Priority priority;               ///< Test priority
        QStringList tags;               ///< Custom tags for filtering
        int timeoutSeconds;             ///< Maximum execution time
        bool enabledByDefault;          ///< Whether test runs by default
        ExecutionMode executionMode;    ///< How to execute this test
        QMap<QString, QVariant> customProperties; ///< Custom configuration
    };

    /**
     * @brief Global test execution configuration
     */
    struct GlobalConfig {
        ExecutionMode defaultExecutionMode = ExecutionMode::Sequential;
        int defaultTimeoutSeconds = 300;
        bool verboseOutput = false;
        bool generateReports = true;
        QString reportOutputDirectory = "test_reports";
        QStringList enabledCategories;
        QStringList enabledTags;
        QStringList disabledTests;
        int maxParallelTests = 4;
        bool stopOnFirstFailure = false;
        bool enableCodeCoverage = false;
        QString coverageOutputDirectory = "coverage_reports";
    };

    // Static configuration access
    static TestConfig& instance();
    
    // Configuration management
    void loadConfiguration(const QString& configFile = "");
    void saveConfiguration(const QString& configFile = "");
    void resetToDefaults();
    
    // Test suite registration and management
    void registerTestSuite(const QString& suiteName, const TestSuiteConfig& config);
    TestSuiteConfig getTestSuiteConfig(const QString& suiteName) const;
    QStringList getRegisteredTestSuites() const;
    
    // Filtering and selection
    QStringList getTestSuitesByCategory(Category category) const;
    QStringList getTestSuitesByTag(const QString& tag) const;
    QStringList getTestSuitesByPriority(Priority priority) const;
    QStringList getEnabledTestSuites() const;
    
    // Global configuration access
    GlobalConfig& globalConfig() { return m_globalConfig; }
    const GlobalConfig& globalConfig() const { return m_globalConfig; }
    
    // Utility functions
    static QString categoryToString(Category category);
    static Category categoryFromString(const QString& categoryStr);
    static QString priorityToString(Priority priority);
    static Priority priorityFromString(const QString& priorityStr);
    static QString executionModeToString(ExecutionMode mode);
    static ExecutionMode executionModeFromString(const QString& modeStr);
    
    // Environment detection
    bool isRunningInCI() const;
    QString getPlatformName() const;
    bool shouldRunTest(const QString& suiteName) const;

private:
    TestConfig() = default;
    
    GlobalConfig m_globalConfig;
    QMap<QString, TestSuiteConfig> m_testSuites;
    
    void setupDefaultConfiguration();
    QJsonObject configToJson() const;
    void configFromJson(const QJsonObject& json);
};

/**
 * @brief Macro for registering test suites with standardized configuration
 */
#define REGISTER_TEST_SUITE(className, category, priority, tags...) \
    do { \
        TestConfig::TestSuiteConfig config; \
        config.name = #className; \
        config.category = TestConfig::Category::category; \
        config.priority = TestConfig::Priority::priority; \
        config.tags = QStringList{tags}; \
        config.timeoutSeconds = TestConfig::instance().globalConfig().defaultTimeoutSeconds; \
        config.enabledByDefault = true; \
        config.executionMode = TestConfig::instance().globalConfig().defaultExecutionMode; \
        TestConfig::instance().registerTestSuite(#className, config); \
    } while(0)

/**
 * @brief Macro for conditional test execution based on configuration
 */
#define SKIP_TEST_IF_DISABLED(testName) \
    do { \
        if (!TestConfig::instance().shouldRunTest(testName)) { \
            QSKIP("Test disabled by configuration"); \
        } \
    } while(0)

/**
 * @brief Macro for test categorization in test methods
 */
#define TEST_CATEGORY(category) \
    Q_CLASSINFO("TestCategory", TestConfig::categoryToString(TestConfig::Category::category).toUtf8().constData())

/**
 * @brief Macro for test tagging
 */
#define TEST_TAGS(...) \
    Q_CLASSINFO("TestTags", QStringList{__VA_ARGS__}.join(",").toUtf8().constData())

/**
 * @brief Macro for test priority
 */
#define TEST_PRIORITY(priority) \
    Q_CLASSINFO("TestPriority", TestConfig::priorityToString(TestConfig::Priority::priority).toUtf8().constData())