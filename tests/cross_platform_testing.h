#pragma once

#include "workflow_testing.h"
#include <QObject>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QElapsedTimer>
#include <QOperatingSystemVersion>
#include <QScreen>
#include <QApplication>
#include <functional>
#include <memory>

/**
 * @brief Supported operating system platforms
 */
enum class Platform {
    Windows,        ///< Microsoft Windows
    macOS,          ///< Apple macOS
    Linux,          ///< Linux distributions
    Unknown         ///< Unknown or unsupported platform
};

/**
 * @brief Display configuration for testing different screen setups
 */
struct DisplayConfiguration {
    QSize resolution;                                   ///< Screen resolution
    qreal dpiScale = 1.0;                              ///< DPI scaling factor
    int refreshRate = 60;                              ///< Refresh rate in Hz
    QString colorProfile = "sRGB";                      ///< Color profile
    bool isHighDPI = false;                            ///< High DPI display flag
    bool isMultiMonitor = false;                       ///< Multi-monitor setup
    QList<QSize> additionalScreens;                    ///< Additional screen resolutions
};

/**
 * @brief File system configuration for testing different file systems
 */
struct FileSystemConfiguration {
    QString type;                                       ///< File system type (NTFS, HFS+, ext4, etc.)
    bool caseSensitive = true;                         ///< Case sensitivity
    bool supportsSymlinks = true;                      ///< Symbolic link support
    bool supportsHardlinks = true;                     ///< Hard link support
    bool supportsExtendedAttributes = true;            ///< Extended attributes support
    qint64 maxFileSize = 0;                            ///< Maximum file size (0 = unlimited)
    int maxPathLength = 260;                           ///< Maximum path length
    QStringList reservedNames;                         ///< Reserved file names
    QStringList invalidCharacters;                     ///< Invalid filename characters
    QString pathSeparator = "/";                       ///< Path separator character
};

/**
 * @brief Platform-specific behavior configuration
 */
struct PlatformBehavior {
    Platform platform;                                  ///< Target platform
    QString version;                                    ///< OS version
    FileSystemConfiguration fileSystem;                ///< File system configuration
    DisplayConfiguration display;                      ///< Display configuration
    QMap<QString, QString> environmentVariables;       ///< Environment variables
    QStringList availableApplications;                 ///< Available system applications
    QMap<QString, QVariant> systemSettings;           ///< System-specific settings
    QStringList supportedFileTypes;                    ///< Supported file types
    QMap<QString, QString> defaultApplications;        ///< Default applications for file types
    bool supportsTrash = true;                         ///< Trash/Recycle bin support
    bool supportsFilePermissions = true;               ///< File permission support
    QString trashLocation;                             ///< Trash/Recycle bin location
};

/**
 * @brief Cross-platform test specification
 */
struct CrossPlatformTest {
    QString id;                                         ///< Test identifier
    QString name;                                       ///< Test name
    QString description;                                ///< Test description
    QList<Platform> targetPlatforms;                   ///< Platforms to test on
    UserWorkflow baseWorkflow;                          ///< Base workflow to adapt
    QMap<Platform, QMap<QString, QVariant>> platformAdaptations; ///< Platform-specific adaptations
    QMap<Platform, QStringList> expectedDifferences;   ///< Expected platform differences
    QStringList commonValidation;                      ///< Common validation across platforms
    QMap<Platform, QStringList> platformValidation;    ///< Platform-specific validation
    bool allowPlatformDifferences = true;              ///< Allow differences between platforms
    QString category;                                   ///< Test category
    QStringList tags;                                   ///< Test tags
};

/**
 * @brief Cross-platform test result
 */
struct CrossPlatformResult {
    QString testId;                                     ///< Test identifier
    Platform currentPlatform;                          ///< Platform where test was executed
    bool success = false;                               ///< Overall success
    QMap<Platform, WorkflowResult> platformResults;    ///< Results per platform
    QMap<Platform, QStringList> platformDifferences;   ///< Detected differences per platform
    QStringList unexpectedDifferences;                 ///< Unexpected differences found
    QStringList missingFeatures;                       ///< Features not available on current platform
    QMap<QString, QVariant> performanceComparison;     ///< Performance comparison across platforms
    QStringList compatibilityIssues;                   ///< Compatibility issues found
    QString platformSpecificFeedback;                  ///< Platform-specific feedback
};

/**
 * @brief OS integration test specification
 */
struct OSIntegrationTest {
    QString id;                                         ///< Test identifier
    QString name;                                       ///< Test name
    QString description;                                ///< Test description
    QString integrationType;                           ///< Type of integration (file_manager, system_dialogs, etc.)
    QMap<QString, QVariant> integrationParameters;     ///< Integration-specific parameters
    QStringList requiredSystemFeatures;               ///< Required system features
    QStringList expectedBehaviors;                     ///< Expected integration behaviors
    QMap<Platform, QStringList> platformSpecificBehaviors; ///< Platform-specific behaviors
    bool requiresSystemPermissions = false;            ///< Requires special system permissions
    QString fallbackBehavior;                          ///< Fallback behavior if integration unavailable
};

/**
 * @brief Comprehensive cross-platform workflow validation framework
 * 
 * Provides testing capabilities for validating workflows across different
 * operating systems, file systems, display configurations, and OS integrations.
 */
class CrossPlatformTesting : public QObject {
    Q_OBJECT

public:
    explicit CrossPlatformTesting(QObject* parent = nullptr);
    ~CrossPlatformTesting();

    // Workflow testing integration
    void setWorkflowTesting(std::shared_ptr<WorkflowTesting> workflowTesting);
    std::shared_ptr<WorkflowTesting> getWorkflowTesting() const { return m_workflowTesting; }

    // Platform detection and configuration
    Platform getCurrentPlatform() const;
    QString getPlatformVersion() const;
    FileSystemConfiguration detectFileSystemConfiguration(const QString& path) const;
    DisplayConfiguration detectDisplayConfiguration() const;
    PlatformBehavior capturePlatformBehavior() const;

    // Cross-platform test management
    void registerCrossPlatformTest(const CrossPlatformTest& test);
    void unregisterCrossPlatformTest(const QString& testId);
    QStringList getRegisteredTests() const;
    CrossPlatformTest getCrossPlatformTest(const QString& testId) const;
    QStringList getTestsForPlatform(Platform platform) const;

    // Predefined cross-platform test creation
    CrossPlatformTest createFileOperationTest();
    CrossPlatformTest createPathHandlingTest();
    CrossPlatformTest createPermissionTest();
    CrossPlatformTest createSymlinkTest();
    CrossPlatformTest createLargeFileTest();
    CrossPlatformTest createUnicodeFilenameTest();
    CrossPlatformTest createDisplayScalingTest();
    CrossPlatformTest createMultiMonitorTest();

    // Cross-platform test execution
    CrossPlatformResult executeCrossPlatformTest(const QString& testId);
    CrossPlatformResult executeCrossPlatformTest(const CrossPlatformTest& test);
    QList<CrossPlatformResult> executeAllCrossPlatformTests();
    QList<CrossPlatformResult> executePlatformTestSuite(Platform platform);

    // Platform-specific behavior testing
    bool testPlatformSpecificBehavior(const QString& behaviorId, const QMap<QString, QVariant>& parameters);
    QStringList validatePlatformCompatibility(const UserWorkflow& workflow);
    QMap<Platform, QStringList> identifyPlatformDifferences(const UserWorkflow& workflow);

    // File system compatibility testing
    bool testFileSystemCompatibility(const QString& path);
    QStringList validateFileOperations(const QString& basePath);
    bool testCaseSensitivity(const QString& path);
    bool testSymlinkSupport(const QString& path);
    bool testHardlinkSupport(const QString& path);
    bool testExtendedAttributes(const QString& path);
    bool testLongPathSupport(const QString& basePath);
    bool testSpecialCharacters(const QString& basePath);

    // Display configuration testing
    bool testDisplayScaling(qreal targetScale);
    bool testMultiMonitorSetup();
    bool testHighDPISupport();
    bool testColorProfileSupport();
    QStringList validateUIScaling(qreal dpiScale);

    // OS integration testing
    void registerOSIntegrationTest(const OSIntegrationTest& test);
    bool executeOSIntegrationTest(const QString& testId);
    bool testFileManagerIntegration();
    bool testSystemDialogIntegration();
    bool testTrashIntegration();
    bool testFileAssociationIntegration();
    bool testNotificationIntegration();
    bool testSystemTrayIntegration();

    // Platform adaptation
    UserWorkflow adaptWorkflowForPlatform(const UserWorkflow& workflow, Platform platform);
    QMap<QString, QVariant> getPlatformAdaptations(Platform platform) const;
    void setPlatformAdaptations(Platform platform, const QMap<QString, QVariant>& adaptations);

    // Validation and comparison
    bool validateCrossPlatformConsistency(const QList<CrossPlatformResult>& results);
    QStringList comparePlatformResults(const CrossPlatformResult& result1, const CrossPlatformResult& result2);
    QMap<QString, QVariant> analyzePlatformPerformance(const QList<CrossPlatformResult>& results);

    // Reporting and analysis
    bool generateCrossPlatformReport(const CrossPlatformResult& result, const QString& outputPath);
    bool generatePlatformComparisonReport(const QList<CrossPlatformResult>& results, const QString& outputPath);
    QStringList identifyPortabilityIssues(const CrossPlatformResult& result);

    // Configuration and settings
    void enablePlatformEmulation(bool enable);
    void setEmulatedPlatform(Platform platform);
    void enableFileSystemEmulation(bool enable);
    void setEmulatedFileSystem(const FileSystemConfiguration& config);
    void enableDisplayEmulation(bool enable);
    void setEmulatedDisplay(const DisplayConfiguration& config);

    // Utility functions
    static QString platformToString(Platform platform);
    static Platform stringToPlatform(const QString& platformStr);
    static bool isPlatformSupported(Platform platform);
    static QStringList getSupportedPlatforms();
    static QString getFileSystemType(const QString& path);
    static bool isPathValid(const QString& path, Platform platform);
    static QString normalizePath(const QString& path, Platform platform);

signals:
    void crossPlatformTestStarted(const QString& testId, Platform platform);
    void crossPlatformTestCompleted(const QString& testId, const CrossPlatformResult& result);
    void platformDifferenceDetected(const QString& testId, Platform platform, const QString& difference);
    void compatibilityIssueFound(const QString& testId, const QString& issue);
    void osIntegrationTested(const QString& integrationType, bool success);
    void fileSystemFeatureDetected(const QString& feature, bool supported);
    void displayConfigurationChanged(const DisplayConfiguration& config);

private:
    // Platform detection helpers
    Platform detectCurrentPlatform() const;
    QString detectPlatformVersion() const;
    FileSystemConfiguration analyzeFileSystem(const QString& path) const;
    DisplayConfiguration analyzeDisplaySetup() const;

    // Test execution helpers
    bool prepareCrossPlatformTest(const CrossPlatformTest& test);
    bool finalizeCrossPlatformTest(const CrossPlatformTest& test, CrossPlatformResult& result);
    UserWorkflow adaptWorkflowSteps(const UserWorkflow& workflow, Platform platform);
    bool validatePlatformRequirements(const CrossPlatformTest& test, Platform platform);

    // Platform-specific helpers
    QStringList getWindowsPlatformFeatures() const;
    QStringList getMacOSPlatformFeatures() const;
    QStringList getLinuxPlatformFeatures() const;
    QMap<QString, QVariant> getWindowsDefaults() const;
    QMap<QString, QVariant> getMacOSDefaults() const;
    QMap<QString, QVariant> getLinuxDefaults() const;

    // File system testing helpers
    bool createTestFile(const QString& path, const QByteArray& content);
    bool createTestSymlink(const QString& linkPath, const QString& targetPath);
    bool createTestHardlink(const QString& linkPath, const QString& targetPath);
    bool setTestFileAttributes(const QString& path, const QMap<QString, QVariant>& attributes);
    bool testFileOperation(const QString& operation, const QStringList& parameters);

    // Display testing helpers
    bool simulateDisplayScale(qreal scale);
    bool simulateMultiMonitor(const QList<QSize>& screenSizes);
    bool testUIElementScaling(qreal scale);
    bool validateFontRendering(qreal scale);
    bool testColorAccuracy();

    // OS integration helpers
    bool testFileManagerOpen(const QString& path);
    bool testSystemDialogOpen(const QString& dialogType, const QMap<QString, QVariant>& parameters);
    bool testTrashOperation(const QString& filePath);
    bool testFileAssociation(const QString& fileType);
    bool testSystemNotification(const QString& message);
    bool testSystemTrayIcon();

    // Validation helpers
    bool compareWorkflowResults(const WorkflowResult& result1, const WorkflowResult& result2, qreal tolerance = 0.1);
    QStringList extractPlatformDifferences(const WorkflowResult& result, Platform platform);
    bool isExpectedDifference(const QString& difference, Platform platform);
    QStringList validatePlatformSpecificBehavior(const WorkflowResult& result, Platform platform);

    // Emulation helpers
    bool enablePlatformEmulationMode(Platform platform);
    bool disablePlatformEmulationMode();
    bool emulateFileSystemBehavior(const FileSystemConfiguration& config);
    bool emulateDisplayBehavior(const DisplayConfiguration& config);

private:
    // Core components
    std::shared_ptr<WorkflowTesting> m_workflowTesting;

    // Platform information
    Platform m_currentPlatform;
    QString m_platformVersion;
    FileSystemConfiguration m_fileSystemConfig;
    DisplayConfiguration m_displayConfig;
    PlatformBehavior m_platformBehavior;

    // Test management
    QMap<QString, CrossPlatformTest> m_registeredTests;
    QMap<QString, OSIntegrationTest> m_osIntegrationTests;

    // Platform adaptations
    QMap<Platform, QMap<QString, QVariant>> m_platformAdaptations;

    // Emulation settings
    bool m_platformEmulationEnabled;
    Platform m_emulatedPlatform;
    bool m_fileSystemEmulationEnabled;
    FileSystemConfiguration m_emulatedFileSystem;
    bool m_displayEmulationEnabled;
    DisplayConfiguration m_emulatedDisplay;

    // Execution state
    QString m_currentTestId;
    Platform m_currentTestPlatform;
    QElapsedTimer m_testTimer;

    // Configuration
    bool m_strictValidation;
    qreal m_performanceTolerance;
    bool m_allowPlatformSpecificBehavior;
};

/**
 * @brief Convenience macros for cross-platform testing
 */
#define PLATFORM_SPECIFIC_TEST(platform, test) \
    do { \
        if (getCurrentPlatform() == platform) { \
            test; \
        } \
    } while(0)

#define SKIP_ON_PLATFORM(platform, reason) \
    do { \
        if (getCurrentPlatform() == platform) { \
            qDebug() << "Skipping test on" << platformToString(platform) << ":" << reason; \
            return true; \
        } \
    } while(0)

#define EXPECT_PLATFORM_DIFFERENCE(platform, difference) \
    do { \
        if (getCurrentPlatform() == platform) { \
            emit platformDifferenceDetected(m_currentTestId, platform, difference); \
        } \
    } while(0)

#define VALIDATE_FILE_SYSTEM_FEATURE(feature, path) \
    do { \
        bool supported = testFileSystemFeature(feature, path); \
        emit fileSystemFeatureDetected(feature, supported); \
        if (!supported && isFeatureRequired(feature)) { \
            return false; \
        } \
    } while(0)

#define CROSS_PLATFORM_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            emit compatibilityIssueFound(m_currentTestId, QString("Cross-platform assertion failed: %1").arg(message)); \
            return false; \
        } \
    } while(0)