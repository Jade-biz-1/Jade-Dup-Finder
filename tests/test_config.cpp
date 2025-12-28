#include "test_config.h"
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QJsonArray>
#include <QCoreApplication>
#include <QDebug>
#include <QProcessEnvironment>

TestConfig& TestConfig::instance() {
    static TestConfig instance;
    return instance;
}

void TestConfig::loadConfiguration(const QString& configFile) {
    QString filePath = configFile;
    if (filePath.isEmpty()) {
        // Try multiple locations for config file
        QStringList searchPaths = {
            QDir::currentPath() + "/test_config.json",
            QStandardPaths::writableLocation(QStandardPaths::ConfigLocation) + "/CloneClean/test_config.json",
            QCoreApplication::applicationDirPath() + "/test_config.json"
        };
        
        for (const QString& path : searchPaths) {
            if (QFile::exists(path)) {
                filePath = path;
                break;
            }
        }
    }
    
    if (filePath.isEmpty() || !QFile::exists(filePath)) {
        qDebug() << "No test configuration file found, using defaults";
        setupDefaultConfiguration();
        return;
    }
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open test configuration file:" << filePath;
        setupDefaultConfiguration();
        return;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError) {
        qWarning() << "Failed to parse test configuration:" << error.errorString();
        setupDefaultConfiguration();
        return;
    }
    
    configFromJson(doc.object());
    qDebug() << "Loaded test configuration from:" << filePath;
}

void TestConfig::saveConfiguration(const QString& configFile) {
    QString filePath = configFile;
    if (filePath.isEmpty()) {
        QString configDir = QStandardPaths::writableLocation(QStandardPaths::ConfigLocation) + "/CloneClean";
        QDir().mkpath(configDir);
        filePath = configDir + "/test_config.json";
    }
    
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to open test configuration file for writing:" << filePath;
        return;
    }
    
    QJsonDocument doc(configToJson());
    file.write(doc.toJson());
    qDebug() << "Saved test configuration to:" << filePath;
}

void TestConfig::resetToDefaults() {
    m_testSuites.clear();
    setupDefaultConfiguration();
}

void TestConfig::registerTestSuite(const QString& suiteName, const TestSuiteConfig& config) {
    m_testSuites[suiteName] = config;
}

TestConfig::TestSuiteConfig TestConfig::getTestSuiteConfig(const QString& suiteName) const {
    return m_testSuites.value(suiteName);
}

QStringList TestConfig::getRegisteredTestSuites() const {
    return m_testSuites.keys();
}

QStringList TestConfig::getTestSuitesByCategory(Category category) const {
    QStringList result;
    for (auto it = m_testSuites.begin(); it != m_testSuites.end(); ++it) {
        if (it.value().category == category) {
            result.append(it.key());
        }
    }
    return result;
}

QStringList TestConfig::getTestSuitesByTag(const QString& tag) const {
    QStringList result;
    for (auto it = m_testSuites.begin(); it != m_testSuites.end(); ++it) {
        if (it.value().tags.contains(tag)) {
            result.append(it.key());
        }
    }
    return result;
}

QStringList TestConfig::getTestSuitesByPriority(Priority priority) const {
    QStringList result;
    for (auto it = m_testSuites.begin(); it != m_testSuites.end(); ++it) {
        if (it.value().priority == priority) {
            result.append(it.key());
        }
    }
    return result;
}

QStringList TestConfig::getEnabledTestSuites() const {
    QStringList result;
    for (auto it = m_testSuites.begin(); it != m_testSuites.end(); ++it) {
        if (shouldRunTest(it.key())) {
            result.append(it.key());
        }
    }
    return result;
}

QString TestConfig::categoryToString(Category category) {
    switch (category) {
        case Category::Unit: return "Unit";
        case Category::Integration: return "Integration";
        case Category::Performance: return "Performance";
        case Category::UI: return "UI";
        case Category::EndToEnd: return "EndToEnd";
        case Category::Security: return "Security";
        case Category::Regression: return "Regression";
        case Category::Smoke: return "Smoke";
    }
    return "Unknown";
}

TestConfig::Category TestConfig::categoryFromString(const QString& categoryStr) {
    if (categoryStr == "Unit") return Category::Unit;
    if (categoryStr == "Integration") return Category::Integration;
    if (categoryStr == "Performance") return Category::Performance;
    if (categoryStr == "UI") return Category::UI;
    if (categoryStr == "EndToEnd") return Category::EndToEnd;
    if (categoryStr == "Security") return Category::Security;
    if (categoryStr == "Regression") return Category::Regression;
    if (categoryStr == "Smoke") return Category::Smoke;
    return Category::Unit; // Default
}

QString TestConfig::priorityToString(Priority priority) {
    switch (priority) {
        case Priority::Critical: return "Critical";
        case Priority::High: return "High";
        case Priority::Medium: return "Medium";
        case Priority::Low: return "Low";
    }
    return "Medium";
}

TestConfig::Priority TestConfig::priorityFromString(const QString& priorityStr) {
    if (priorityStr == "Critical") return Priority::Critical;
    if (priorityStr == "High") return Priority::High;
    if (priorityStr == "Medium") return Priority::Medium;
    if (priorityStr == "Low") return Priority::Low;
    return Priority::Medium; // Default
}

QString TestConfig::executionModeToString(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential: return "Sequential";
        case ExecutionMode::Parallel: return "Parallel";
        case ExecutionMode::Isolated: return "Isolated";
    }
    return "Sequential";
}

TestConfig::ExecutionMode TestConfig::executionModeFromString(const QString& modeStr) {
    if (modeStr == "Sequential") return ExecutionMode::Sequential;
    if (modeStr == "Parallel") return ExecutionMode::Parallel;
    if (modeStr == "Isolated") return ExecutionMode::Isolated;
    return ExecutionMode::Sequential; // Default
}

bool TestConfig::isRunningInCI() const {
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    return env.contains("CI") || 
           env.contains("GITHUB_ACTIONS") || 
           env.contains("JENKINS_URL") ||
           env.contains("TRAVIS") ||
           env.contains("APPVEYOR");
}

QString TestConfig::getPlatformName() const {
#ifdef Q_OS_WIN
    return "Windows";
#elif defined(Q_OS_MAC)
    return "macOS";
#elif defined(Q_OS_LINUX)
    return "Linux";
#else
    return "Unknown";
#endif
}

bool TestConfig::shouldRunTest(const QString& suiteName) const {
    // Check if test is explicitly disabled
    if (m_globalConfig.disabledTests.contains(suiteName)) {
        return false;
    }
    
    // Get test configuration
    TestSuiteConfig config = getTestSuiteConfig(suiteName);
    if (!config.enabledByDefault) {
        return false;
    }
    
    // Check category filtering
    if (!m_globalConfig.enabledCategories.isEmpty()) {
        QString categoryStr = categoryToString(config.category);
        if (!m_globalConfig.enabledCategories.contains(categoryStr)) {
            return false;
        }
    }
    
    // Check tag filtering
    if (!m_globalConfig.enabledTags.isEmpty()) {
        bool hasMatchingTag = false;
        for (const QString& tag : config.tags) {
            if (m_globalConfig.enabledTags.contains(tag)) {
                hasMatchingTag = true;
                break;
            }
        }
        if (!hasMatchingTag) {
            return false;
        }
    }
    
    return true;
}

void TestConfig::setupDefaultConfiguration() {
    // Setup default global configuration
    m_globalConfig.defaultExecutionMode = ExecutionMode::Sequential;
    m_globalConfig.defaultTimeoutSeconds = 300;
    m_globalConfig.verboseOutput = false;
    m_globalConfig.generateReports = true;
    m_globalConfig.reportOutputDirectory = "test_reports";
    m_globalConfig.maxParallelTests = 4;
    m_globalConfig.stopOnFirstFailure = false;
    m_globalConfig.enableCodeCoverage = false;
    m_globalConfig.coverageOutputDirectory = "coverage_reports";
    
    // Enable all categories by default
    m_globalConfig.enabledCategories = {
        "Unit", "Integration", "Performance", "UI", "EndToEnd", "Security", "Regression", "Smoke"
    };
    
    // Register default test suites based on existing tests
    registerDefaultTestSuites();
}

void TestConfig::registerDefaultTestSuites() {
    // Unit test suites
    TestSuiteConfig unitConfig;
    unitConfig.category = Category::Unit;
    unitConfig.priority = Priority::High;
    unitConfig.timeoutSeconds = 60;
    unitConfig.enabledByDefault = true;
    unitConfig.executionMode = ExecutionMode::Parallel;
    
    QStringList unitTests = {
        "HashCalculatorTest", "FileManagerTest", "SafetyManagerTest",
        "ThumbnailCacheTest", "ThumbnailDelegateTest", "ExcludePatternWidgetTest",
        "PresetManagerTest", "ScanConfigurationValidationTest", "ScanScopePreviewWidgetTest",
        "ScanProgressTrackingTest", "ScanProgressDialogTest", "FileScannerCoverageTest"
    };
    
    for (const QString& testName : unitTests) {
        unitConfig.name = testName;
        unitConfig.tags = {"unit", "core"};
        if (testName.contains("UI") || testName.contains("Widget") || testName.contains("Dialog")) {
            unitConfig.tags.append("ui");
        }
        registerTestSuite(testName, unitConfig);
    }
    
    // Integration test suites
    TestSuiteConfig integrationConfig;
    integrationConfig.category = Category::Integration;
    integrationConfig.priority = Priority::High;
    integrationConfig.timeoutSeconds = 300;
    integrationConfig.enabledByDefault = true;
    integrationConfig.executionMode = ExecutionMode::Sequential;
    
    QStringList integrationTests = {
        "IntegrationWorkflowTest", "FileScannerHashCalculatorTest", "FileScannerDuplicateDetectorTest",
        "EndToEndWorkflowTest", "ScanToDeleteWorkflowTest", "RestoreFunctionalityTest",
        "ErrorScenariosTest"
    };
    
    for (const QString& testName : integrationTests) {
        integrationConfig.name = testName;
        integrationConfig.tags = {"integration", "workflow"};
        if (testName.contains("Error")) {
            integrationConfig.tags.append("error-handling");
        }
        if (testName.contains("EndToEnd")) {
            integrationConfig.category = Category::EndToEnd;
            integrationConfig.tags.append("end-to-end");
        }
        registerTestSuite(testName, integrationConfig);
    }
    
    // Performance test suites
    TestSuiteConfig performanceConfig;
    performanceConfig.category = Category::Performance;
    performanceConfig.priority = Priority::Medium;
    performanceConfig.timeoutSeconds = 600;
    performanceConfig.enabledByDefault = !isRunningInCI(); // Disable in CI by default
    performanceConfig.executionMode = ExecutionMode::Isolated;
    
    QStringList performanceTests = {
        "FileScannerPerformanceTest", "BatchProcessingTest", "IOOptimizationTest"
    };
    
    for (const QString& testName : performanceTests) {
        performanceConfig.name = testName;
        performanceConfig.tags = {"performance", "benchmark"};
        registerTestSuite(testName, performanceConfig);
    }
}

QJsonObject TestConfig::configToJson() const {
    QJsonObject root;
    
    // Global configuration
    QJsonObject globalObj;
    globalObj["defaultExecutionMode"] = executionModeToString(m_globalConfig.defaultExecutionMode);
    globalObj["defaultTimeoutSeconds"] = m_globalConfig.defaultTimeoutSeconds;
    globalObj["verboseOutput"] = m_globalConfig.verboseOutput;
    globalObj["generateReports"] = m_globalConfig.generateReports;
    globalObj["reportOutputDirectory"] = m_globalConfig.reportOutputDirectory;
    globalObj["maxParallelTests"] = m_globalConfig.maxParallelTests;
    globalObj["stopOnFirstFailure"] = m_globalConfig.stopOnFirstFailure;
    globalObj["enableCodeCoverage"] = m_globalConfig.enableCodeCoverage;
    globalObj["coverageOutputDirectory"] = m_globalConfig.coverageOutputDirectory;
    
    QJsonArray enabledCategories;
    for (const QString& category : m_globalConfig.enabledCategories) {
        enabledCategories.append(category);
    }
    globalObj["enabledCategories"] = enabledCategories;
    
    QJsonArray enabledTags;
    for (const QString& tag : m_globalConfig.enabledTags) {
        enabledTags.append(tag);
    }
    globalObj["enabledTags"] = enabledTags;
    
    QJsonArray disabledTests;
    for (const QString& test : m_globalConfig.disabledTests) {
        disabledTests.append(test);
    }
    globalObj["disabledTests"] = disabledTests;
    
    root["global"] = globalObj;
    
    // Test suites configuration
    QJsonObject testSuitesObj;
    for (auto it = m_testSuites.begin(); it != m_testSuites.end(); ++it) {
        const TestSuiteConfig& config = it.value();
        QJsonObject suiteObj;
        suiteObj["name"] = config.name;
        suiteObj["category"] = categoryToString(config.category);
        suiteObj["priority"] = priorityToString(config.priority);
        suiteObj["timeoutSeconds"] = config.timeoutSeconds;
        suiteObj["enabledByDefault"] = config.enabledByDefault;
        suiteObj["executionMode"] = executionModeToString(config.executionMode);
        
        QJsonArray tags;
        for (const QString& tag : config.tags) {
            tags.append(tag);
        }
        suiteObj["tags"] = tags;
        
        QJsonObject customProps;
        for (auto propIt = config.customProperties.begin(); propIt != config.customProperties.end(); ++propIt) {
            customProps[propIt.key()] = QJsonValue::fromVariant(propIt.value());
        }
        suiteObj["customProperties"] = customProps;
        
        testSuitesObj[it.key()] = suiteObj;
    }
    root["testSuites"] = testSuitesObj;
    
    return root;
}

void TestConfig::configFromJson(const QJsonObject& json) {
    // Load global configuration
    if (json.contains("global")) {
        QJsonObject globalObj = json["global"].toObject();
        
        if (globalObj.contains("defaultExecutionMode")) {
            m_globalConfig.defaultExecutionMode = executionModeFromString(globalObj["defaultExecutionMode"].toString());
        }
        if (globalObj.contains("defaultTimeoutSeconds")) {
            m_globalConfig.defaultTimeoutSeconds = globalObj["defaultTimeoutSeconds"].toInt();
        }
        if (globalObj.contains("verboseOutput")) {
            m_globalConfig.verboseOutput = globalObj["verboseOutput"].toBool();
        }
        if (globalObj.contains("generateReports")) {
            m_globalConfig.generateReports = globalObj["generateReports"].toBool();
        }
        if (globalObj.contains("reportOutputDirectory")) {
            m_globalConfig.reportOutputDirectory = globalObj["reportOutputDirectory"].toString();
        }
        if (globalObj.contains("maxParallelTests")) {
            m_globalConfig.maxParallelTests = globalObj["maxParallelTests"].toInt();
        }
        if (globalObj.contains("stopOnFirstFailure")) {
            m_globalConfig.stopOnFirstFailure = globalObj["stopOnFirstFailure"].toBool();
        }
        if (globalObj.contains("enableCodeCoverage")) {
            m_globalConfig.enableCodeCoverage = globalObj["enableCodeCoverage"].toBool();
        }
        if (globalObj.contains("coverageOutputDirectory")) {
            m_globalConfig.coverageOutputDirectory = globalObj["coverageOutputDirectory"].toString();
        }
        
        if (globalObj.contains("enabledCategories")) {
            m_globalConfig.enabledCategories.clear();
            QJsonArray categories = globalObj["enabledCategories"].toArray();
            for (const QJsonValue& value : categories) {
                m_globalConfig.enabledCategories.append(value.toString());
            }
        }
        
        if (globalObj.contains("enabledTags")) {
            m_globalConfig.enabledTags.clear();
            QJsonArray tags = globalObj["enabledTags"].toArray();
            for (const QJsonValue& value : tags) {
                m_globalConfig.enabledTags.append(value.toString());
            }
        }
        
        if (globalObj.contains("disabledTests")) {
            m_globalConfig.disabledTests.clear();
            QJsonArray tests = globalObj["disabledTests"].toArray();
            for (const QJsonValue& value : tests) {
                m_globalConfig.disabledTests.append(value.toString());
            }
        }
    }
    
    // Load test suites configuration
    if (json.contains("testSuites")) {
        QJsonObject testSuitesObj = json["testSuites"].toObject();
        
        for (auto it = testSuitesObj.begin(); it != testSuitesObj.end(); ++it) {
            QJsonObject suiteObj = it.value().toObject();
            TestSuiteConfig config;
            
            config.name = suiteObj["name"].toString();
            config.category = categoryFromString(suiteObj["category"].toString());
            config.priority = priorityFromString(suiteObj["priority"].toString());
            config.timeoutSeconds = suiteObj["timeoutSeconds"].toInt();
            config.enabledByDefault = suiteObj["enabledByDefault"].toBool();
            config.executionMode = executionModeFromString(suiteObj["executionMode"].toString());
            
            if (suiteObj.contains("tags")) {
                QJsonArray tags = suiteObj["tags"].toArray();
                for (const QJsonValue& value : tags) {
                    config.tags.append(value.toString());
                }
            }
            
            if (suiteObj.contains("customProperties")) {
                QJsonObject customProps = suiteObj["customProperties"].toObject();
                for (auto propIt = customProps.begin(); propIt != customProps.end(); ++propIt) {
                    config.customProperties[propIt.key()] = propIt.value().toVariant();
                }
            }
            
            m_testSuites[it.key()] = config;
        }
    }
}