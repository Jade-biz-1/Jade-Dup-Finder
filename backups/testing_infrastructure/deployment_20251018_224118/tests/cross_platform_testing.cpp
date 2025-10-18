#include "cross_platform_testing.h"
#include <QApplication>
#include <QWidget>
#include <QMainWindow>
#include <QDialog>
#include <QTimer>
#include <QEventLoop>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QStorageInfo>
#include <QScreen>
#include <QDesktopServices>
#include <QUrl>
#include <QProcess>
#include <QOperatingSystemVersion>
#include <QSysInfo>
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QRandomGenerator>
#include <stdexcept>

#ifdef Q_OS_WIN
#include <windows.h>
#include <shellapi.h>
#endif

#ifdef Q_OS_MACOS
#include <CoreFoundation/CoreFoundation.h>
#endif

#ifdef Q_OS_LINUX
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#endif

CrossPlatformTesting::CrossPlatformTesting(QObject* parent)
    : QObject(parent)
    , m_currentPlatform(detectCurrentPlatform())
    , m_platformVersion(detectPlatformVersion())
    , m_platformEmulationEnabled(false)
    , m_emulatedPlatform(Platform::Unknown)
    , m_fileSystemEmulationEnabled(false)
    , m_displayEmulationEnabled(false)
    , m_strictValidation(true)
    , m_performanceTolerance(0.2) // 20% tolerance
    , m_allowPlatformSpecificBehavior(true)
{
    // Initialize platform-specific configurations
    m_fileSystemConfig = detectFileSystemConfiguration(".");
    m_displayConfig = detectDisplayConfiguration();
    m_platformBehavior = capturePlatformBehavior();
    
    // Setup platform adaptations
    setupPlatformAdaptations();
}

CrossPlatformTesting::~CrossPlatformTesting() = default;

void CrossPlatformTesting::setWorkflowTesting(std::shared_ptr<WorkflowTesting> workflowTesting) {
    m_workflowTesting = workflowTesting;
}

Platform CrossPlatformTesting::getCurrentPlatform() const {
    if (m_platformEmulationEnabled) {
        return m_emulatedPlatform;
    }
    return m_currentPlatform;
}

QString CrossPlatformTesting::getPlatformVersion() const {
    return m_platformVersion;
}

FileSystemConfiguration CrossPlatformTesting::detectFileSystemConfiguration(const QString& path) const {
    FileSystemConfiguration config;
    
    QStorageInfo storage(path);
    config.type = storage.fileSystemType();
    
    // Platform-specific file system detection
    switch (m_currentPlatform) {
        case Platform::Windows:
            config.caseSensitive = false;
            config.supportsSymlinks = true; // Windows 10+ with developer mode
            config.supportsHardlinks = true;
            config.supportsExtendedAttributes = false;
            config.maxFileSize = 4294967295LL; // 4GB for FAT32, larger for NTFS
            config.maxPathLength = 260; // Traditional limit, 32767 with long path support
            config.reservedNames = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", 
                                   "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", 
                                   "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};
            config.invalidCharacters = {"<", ">", ":", "\"", "|", "?", "*"};
            config.pathSeparator = "\\";
            break;
            
        case Platform::macOS:
            config.caseSensitive = false; // HFS+ default, APFS can be case-sensitive
            config.supportsSymlinks = true;
            config.supportsHardlinks = true;
            config.supportsExtendedAttributes = true;
            config.maxFileSize = 8589934591LL; // 8GB for HFS+
            config.maxPathLength = 1024;
            config.reservedNames = {};
            config.invalidCharacters = {":"};
            config.pathSeparator = "/";
            break;
            
        case Platform::Linux:
            config.caseSensitive = true;
            config.supportsSymlinks = true;
            config.supportsHardlinks = true;
            config.supportsExtendedAttributes = true;
            config.maxFileSize = 0; // Depends on file system
            config.maxPathLength = 4096;
            config.reservedNames = {};
            config.invalidCharacters = {"\0"};
            config.pathSeparator = "/";
            break;
            
        case Platform::Unknown:
            // Use conservative defaults
            config.caseSensitive = true;
            config.supportsSymlinks = false;
            config.supportsHardlinks = false;
            config.supportsExtendedAttributes = false;
            config.maxFileSize = 2147483647LL; // 2GB
            config.maxPathLength = 260;
            config.pathSeparator = "/";
            break;
    }
    
    return config;
}

DisplayConfiguration CrossPlatformTesting::detectDisplayConfiguration() const {
    DisplayConfiguration config;
    
    QScreen* primaryScreen = QApplication::primaryScreen();
    if (primaryScreen) {
        config.resolution = primaryScreen->size();
        config.dpiScale = primaryScreen->devicePixelRatio();
        config.refreshRate = primaryScreen->refreshRate();
        config.isHighDPI = config.dpiScale > 1.0;
        
        // Check for multiple monitors
        QList<QScreen*> screens = QApplication::screens();
        config.isMultiMonitor = screens.size() > 1;
        
        for (QScreen* screen : screens) {
            if (screen != primaryScreen) {
                config.additionalScreens.append(screen->size());
            }
        }
    }
    
    return config;
}

PlatformBehavior CrossPlatformTesting::capturePlatformBehavior() const {
    PlatformBehavior behavior;
    behavior.platform = m_currentPlatform;
    behavior.version = m_platformVersion;
    behavior.fileSystem = m_fileSystemConfig;
    behavior.display = m_displayConfig;
    
    // Capture environment variables
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QStringList keys = env.keys();
    for (const QString& key : keys) {
        behavior.environmentVariables[key] = env.value(key);
    }
    
    // Platform-specific behavior detection
    switch (m_currentPlatform) {
        case Platform::Windows:
            behavior.supportsTrash = true;
            behavior.supportsFilePermissions = true;
            behavior.trashLocation = "C:\\$Recycle.Bin";
            behavior.supportedFileTypes = {"exe", "dll", "bat", "cmd", "msi", "lnk"};
            behavior.defaultApplications["txt"] = "notepad.exe";
            behavior.defaultApplications["jpg"] = "Photos";
            break;
            
        case Platform::macOS:
            behavior.supportsTrash = true;
            behavior.supportsFilePermissions = true;
            behavior.trashLocation = QDir::homePath() + "/.Trash";
            behavior.supportedFileTypes = {"app", "dmg", "pkg", "plist"};
            behavior.defaultApplications["txt"] = "TextEdit";
            behavior.defaultApplications["jpg"] = "Preview";
            break;
            
        case Platform::Linux:
            behavior.supportsTrash = true;
            behavior.supportsFilePermissions = true;
            behavior.trashLocation = QDir::homePath() + "/.local/share/Trash";
            behavior.supportedFileTypes = {"deb", "rpm", "appimage", "desktop"};
            behavior.defaultApplications["txt"] = "gedit";
            behavior.defaultApplications["jpg"] = "eog";
            break;
            
        case Platform::Unknown:
            behavior.supportsTrash = false;
            behavior.supportsFilePermissions = false;
            break;
    }
    
    return behavior;
}

void CrossPlatformTesting::registerCrossPlatformTest(const CrossPlatformTest& test) {
    m_registeredTests[test.id] = test;
}

void CrossPlatformTesting::unregisterCrossPlatformTest(const QString& testId) {
    m_registeredTests.remove(testId);
}

QStringList CrossPlatformTesting::getRegisteredTests() const {
    return m_registeredTests.keys();
}

CrossPlatformTest CrossPlatformTesting::getCrossPlatformTest(const QString& testId) const {
    return m_registeredTests.value(testId);
}

QStringList CrossPlatformTesting::getTestsForPlatform(Platform platform) const {
    QStringList tests;
    for (auto it = m_registeredTests.begin(); it != m_registeredTests.end(); ++it) {
        if (it.value().targetPlatforms.contains(platform)) {
            tests.append(it.key());
        }
    }
    return tests;
}

CrossPlatformTest CrossPlatformTesting::createFileOperationTest() {
    CrossPlatformTest test;
    test.id = "file_operation_cross_platform";
    test.name = "Cross-Platform File Operations";
    test.description = "Test file operations across different platforms and file systems";
    test.targetPlatforms = {Platform::Windows, Platform::macOS, Platform::Linux};
    test.category = "file_operations";
    test.tags = {"files", "cross_platform", "compatibility"};
    
    // Create base workflow for file operations
    test.baseWorkflow.id = "file_operations_workflow";
    test.baseWorkflow.name = "File Operations Workflow";
    test.baseWorkflow.description = "Test various file operations";
    
    // Platform-specific adaptations
    QMap<QString, QVariant> windowsAdaptations;
    windowsAdaptations["path_separator"] = "\\";
    windowsAdaptations["case_sensitive"] = false;
    windowsAdaptations["reserved_names"] = QStringList{"CON", "PRN", "AUX"};
    test.platformAdaptations[Platform::Windows] = windowsAdaptations;
    
    QMap<QString, QVariant> macosAdaptations;
    macosAdaptations["path_separator"] = "/";
    macosAdaptations["case_sensitive"] = false;
    macosAdaptations["resource_forks"] = true;
    test.platformAdaptations[Platform::macOS] = macosAdaptations;
    
    QMap<QString, QVariant> linuxAdaptations;
    linuxAdaptations["path_separator"] = "/";
    linuxAdaptations["case_sensitive"] = true;
    linuxAdaptations["permissions"] = true;
    test.platformAdaptations[Platform::Linux] = linuxAdaptations;
    
    // Expected differences
    test.expectedDifferences[Platform::Windows] = {"case_insensitive_paths", "backslash_separators"};
    test.expectedDifferences[Platform::macOS] = {"resource_forks", "case_insensitive_default"};
    test.expectedDifferences[Platform::Linux] = {"case_sensitive_paths", "unix_permissions"};
    
    // Common validation
    test.commonValidation = {"files_created_successfully", "files_deleted_successfully", "directories_created"};
    
    // Platform-specific validation
    test.platformValidation[Platform::Windows] = {"recycle_bin_integration", "ntfs_features"};
    test.platformValidation[Platform::macOS] = {"trash_integration", "finder_integration"};
    test.platformValidation[Platform::Linux] = {"trash_integration", "file_manager_integration"};
    
    return test;
}

CrossPlatformTest CrossPlatformTesting::createPathHandlingTest() {
    CrossPlatformTest test;
    test.id = "path_handling_cross_platform";
    test.name = "Cross-Platform Path Handling";
    test.description = "Test path handling and normalization across platforms";
    test.targetPlatforms = {Platform::Windows, Platform::macOS, Platform::Linux};
    test.category = "path_handling";
    test.tags = {"paths", "normalization", "compatibility"};
    
    // Create workflow for path handling tests
    test.baseWorkflow.id = "path_handling_workflow";
    test.baseWorkflow.name = "Path Handling Workflow";
    test.baseWorkflow.description = "Test path handling across platforms";
    
    // Platform adaptations for path handling
    test.platformAdaptations[Platform::Windows]["max_path_length"] = 260;
    test.platformAdaptations[Platform::Windows]["path_separator"] = "\\";
    test.platformAdaptations[Platform::Windows]["drive_letters"] = true;
    
    test.platformAdaptations[Platform::macOS]["max_path_length"] = 1024;
    test.platformAdaptations[Platform::macOS]["path_separator"] = "/";
    test.platformAdaptations[Platform::macOS]["case_preservation"] = true;
    
    test.platformAdaptations[Platform::Linux]["max_path_length"] = 4096;
    test.platformAdaptations[Platform::Linux]["path_separator"] = "/";
    test.platformAdaptations[Platform::Linux]["case_sensitive"] = true;
    
    return test;
}

CrossPlatformTest CrossPlatformTesting::createDisplayScalingTest() {
    CrossPlatformTest test;
    test.id = "display_scaling_cross_platform";
    test.name = "Cross-Platform Display Scaling";
    test.description = "Test UI scaling across different DPI settings and platforms";
    test.targetPlatforms = {Platform::Windows, Platform::macOS, Platform::Linux};
    test.category = "display_scaling";
    test.tags = {"ui", "scaling", "dpi", "high_dpi"};
    
    // Create workflow for display scaling tests
    test.baseWorkflow.id = "display_scaling_workflow";
    test.baseWorkflow.name = "Display Scaling Workflow";
    test.baseWorkflow.description = "Test UI scaling at different DPI settings";
    
    // Platform-specific scaling behaviors
    test.platformAdaptations[Platform::Windows]["scaling_method"] = "system_dpi";
    test.platformAdaptations[Platform::Windows]["per_monitor_dpi"] = true;
    
    test.platformAdaptations[Platform::macOS]["scaling_method"] = "retina";
    test.platformAdaptations[Platform::macOS]["automatic_scaling"] = true;
    
    test.platformAdaptations[Platform::Linux]["scaling_method"] = "manual";
    test.platformAdaptations[Platform::Linux]["fractional_scaling"] = true;
    
    return test;
}

CrossPlatformResult CrossPlatformTesting::executeCrossPlatformTest(const QString& testId) {
    if (!m_registeredTests.contains(testId)) {
        CrossPlatformResult result;
        result.testId = testId;
        result.currentPlatform = getCurrentPlatform();
        result.success = false;
        result.compatibilityIssues.append(QString("Test not found: %1").arg(testId));
        return result;
    }
    
    return executeCrossPlatformTest(m_registeredTests[testId]);
}

CrossPlatformResult CrossPlatformTesting::executeCrossPlatformTest(const CrossPlatformTest& test) {
    CrossPlatformResult result;
    result.testId = test.id;
    result.currentPlatform = getCurrentPlatform();
    
    m_currentTestId = test.id;
    m_currentTestPlatform = getCurrentPlatform();
    m_testTimer.start();
    
    emit crossPlatformTestStarted(test.id, getCurrentPlatform());
    
    if (!prepareCrossPlatformTest(test)) {
        result.success = false;
        result.compatibilityIssues.append("Failed to prepare cross-platform test");
        return result;
    }
    
    // Check if current platform is supported by this test
    if (!test.targetPlatforms.contains(getCurrentPlatform())) {
        result.success = false;
        result.compatibilityIssues.append(QString("Test not supported on platform: %1")
                                         .arg(platformToString(getCurrentPlatform())));
        return result;
    }
    
    // Validate platform requirements
    if (!validatePlatformRequirements(test, getCurrentPlatform())) {
        result.success = false;
        result.compatibilityIssues.append("Platform requirements not met");
        return result;
    }
    
    // Adapt workflow for current platform
    UserWorkflow adaptedWorkflow = adaptWorkflowForPlatform(test.baseWorkflow, getCurrentPlatform());
    
    // Execute the adapted workflow
    if (m_workflowTesting) {
        WorkflowResult workflowResult = m_workflowTesting->executeWorkflow(adaptedWorkflow);
        result.platformResults[getCurrentPlatform()] = workflowResult;
        result.success = workflowResult.success;
    } else {
        result.success = false;
        result.compatibilityIssues.append("Workflow testing not available");
        return result;
    }
    
    // Detect platform-specific differences
    QStringList differences = extractPlatformDifferences(result.platformResults[getCurrentPlatform()], getCurrentPlatform());
    result.platformDifferences[getCurrentPlatform()] = differences;
    
    // Check for unexpected differences
    QStringList expectedDiffs = test.expectedDifferences.value(getCurrentPlatform());
    for (const QString& diff : differences) {
        if (!expectedDiffs.contains(diff)) {
            result.unexpectedDifferences.append(diff);
        }
    }
    
    // Validate platform-specific behavior
    QStringList validationIssues = validatePlatformSpecificBehavior(result.platformResults[getCurrentPlatform()], getCurrentPlatform());
    result.compatibilityIssues.append(validationIssues);
    
    // Check for missing features
    result.missingFeatures = identifyMissingFeatures(test, getCurrentPlatform());
    
    // Generate platform-specific feedback
    result.platformSpecificFeedback = generatePlatformFeedback(result, getCurrentPlatform());
    
    finalizeCrossPlatformTest(test, result);
    
    emit crossPlatformTestCompleted(test.id, result);
    
    return result;
}

bool CrossPlatformTesting::testFileSystemCompatibility(const QString& path) {
    QDir testDir(path);
    if (!testDir.exists()) {
        return false;
    }
    
    bool allTestsPassed = true;
    
    // Test case sensitivity
    if (!testCaseSensitivity(path)) {
        emit fileSystemFeatureDetected("case_sensitivity", false);
        allTestsPassed = false;
    }
    
    // Test symlink support
    if (!testSymlinkSupport(path)) {
        emit fileSystemFeatureDetected("symlink_support", false);
    }
    
    // Test hardlink support
    if (!testHardlinkSupport(path)) {
        emit fileSystemFeatureDetected("hardlink_support", false);
    }
    
    // Test extended attributes
    if (!testExtendedAttributes(path)) {
        emit fileSystemFeatureDetected("extended_attributes", false);
    }
    
    // Test long path support
    if (!testLongPathSupport(path)) {
        emit fileSystemFeatureDetected("long_path_support", false);
    }
    
    // Test special characters
    if (!testSpecialCharacters(path)) {
        emit fileSystemFeatureDetected("special_characters", false);
    }
    
    return allTestsPassed;
}

bool CrossPlatformTesting::testCaseSensitivity(const QString& path) {
    QString testFile1 = QDir(path).absoluteFilePath("CaseSensitivityTest.txt");
    QString testFile2 = QDir(path).absoluteFilePath("casesensitivitytest.txt");
    
    // Create first file
    QFile file1(testFile1);
    if (!file1.open(QIODevice::WriteOnly)) {
        return false;
    }
    file1.write("test content 1");
    file1.close();
    
    // Try to create second file with different case
    QFile file2(testFile2);
    bool canCreateBoth = file2.open(QIODevice::WriteOnly);
    if (canCreateBoth) {
        file2.write("test content 2");
        file2.close();
    }
    
    // Cleanup
    QFile::remove(testFile1);
    if (canCreateBoth) {
        QFile::remove(testFile2);
    }
    
    return canCreateBoth; // Case sensitive if we can create both files
}

bool CrossPlatformTesting::testSymlinkSupport(const QString& path) {
    QString targetFile = QDir(path).absoluteFilePath("symlink_target.txt");
    QString linkFile = QDir(path).absoluteFilePath("symlink_test.lnk");
    
    // Create target file
    QFile target(targetFile);
    if (!target.open(QIODevice::WriteOnly)) {
        return false;
    }
    target.write("symlink target content");
    target.close();
    
    // Try to create symlink
    bool success = QFile::link(targetFile, linkFile);
    
    // Cleanup
    QFile::remove(targetFile);
    if (success) {
        QFile::remove(linkFile);
    }
    
    return success;
}

bool CrossPlatformTesting::testHardlinkSupport(const QString& path) {
    QString originalFile = QDir(path).absoluteFilePath("hardlink_original.txt");
    QString linkFile = QDir(path).absoluteFilePath("hardlink_test.txt");
    
    // Create original file
    QFile original(originalFile);
    if (!original.open(QIODevice::WriteOnly)) {
        return false;
    }
    original.write("hardlink test content");
    original.close();
    
    // Try to create hard link (platform-specific implementation needed)
    bool success = false;
    
#ifdef Q_OS_WIN
    success = CreateHardLinkA(linkFile.toLocal8Bit().constData(), 
                             originalFile.toLocal8Bit().constData(), nullptr);
#elif defined(Q_OS_UNIX)
    success = (link(originalFile.toLocal8Bit().constData(), 
                   linkFile.toLocal8Bit().constData()) == 0);
#endif
    
    // Cleanup
    QFile::remove(originalFile);
    if (success) {
        QFile::remove(linkFile);
    }
    
    return success;
}

bool CrossPlatformTesting::testExtendedAttributes(const QString& path) {
    QString testFile = QDir(path).absoluteFilePath("extended_attr_test.txt");
    
    // Create test file
    QFile file(testFile);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    file.write("extended attributes test");
    file.close();
    
    bool success = false;
    
    // Try to set extended attribute (platform-specific)
#ifdef Q_OS_MACOS
    // macOS extended attributes
    success = (setxattr(testFile.toLocal8Bit().constData(), "user.test", "value", 5, 0, 0) == 0);
#elif defined(Q_OS_LINUX)
    // Linux extended attributes
    success = (setxattr(testFile.toLocal8Bit().constData(), "user.test", "value", 5, 0) == 0);
#endif
    
    // Cleanup
    QFile::remove(testFile);
    
    return success;
}

bool CrossPlatformTesting::testLongPathSupport(const QString& basePath) {
    // Create a very long path
    QString longPath = basePath;
    QString longDirName = "very_long_directory_name_that_exceeds_normal_limits_and_tests_the_maximum_path_length_supported_by_the_file_system";
    
    // Build path until we approach the limit
    while (longPath.length() < m_fileSystemConfig.maxPathLength - longDirName.length() - 10) {
        longPath = QDir(longPath).absoluteFilePath(longDirName);
    }
    
    // Try to create the directory structure
    QDir dir;
    bool success = dir.mkpath(longPath);
    
    if (success) {
        // Try to create a file in the long path
        QString testFile = QDir(longPath).absoluteFilePath("test.txt");
        QFile file(testFile);
        success = file.open(QIODevice::WriteOnly);
        if (success) {
            file.write("long path test");
            file.close();
        }
        
        // Cleanup
        QDir(basePath).removeRecursively();
    }
    
    return success;
}

bool CrossPlatformTesting::testSpecialCharacters(const QString& basePath) {
    // Test various special characters in filenames
    QStringList specialChars;
    
    switch (getCurrentPlatform()) {
        case Platform::Windows:
            // Test characters that should be invalid on Windows
            specialChars = {"test<file.txt", "test>file.txt", "test:file.txt", 
                           "test\"file.txt", "test|file.txt", "test?file.txt", "test*file.txt"};
            break;
            
        case Platform::macOS:
            // Test characters that should be invalid on macOS
            specialChars = {"test:file.txt"};
            break;
            
        case Platform::Linux:
            // Test characters that should be valid on Linux
            specialChars = {"test<file.txt", "test>file.txt", "test file.txt", 
                           "test'file.txt", "test[file].txt"};
            break;
            
        case Platform::Unknown:
            return false;
    }
    
    bool allTestsPassed = true;
    
    for (const QString& filename : specialChars) {
        QString testFile = QDir(basePath).absoluteFilePath(filename);
        QFile file(testFile);
        bool canCreate = file.open(QIODevice::WriteOnly);
        
        if (canCreate) {
            file.write("special character test");
            file.close();
            QFile::remove(testFile);
        }
        
        // On Windows, these should fail; on Linux, most should succeed
        bool expectedResult = (getCurrentPlatform() == Platform::Linux);
        if (canCreate != expectedResult && getCurrentPlatform() != Platform::macOS) {
            allTestsPassed = false;
        }
    }
    
    return allTestsPassed;
}

bool CrossPlatformTesting::testDisplayScaling(qreal targetScale) {
    if (m_displayEmulationEnabled) {
        return simulateDisplayScale(targetScale);
    }
    
    // Test if UI elements scale properly at the target scale
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return false;
    }
    
    qreal currentScale = screen->devicePixelRatio();
    
    // If we can't change the scale, just validate current scaling
    return testUIElementScaling(currentScale);
}

bool CrossPlatformTesting::testFileManagerIntegration() {
    QString testPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QString testFile = QDir(testPath).absoluteFilePath("file_manager_test.txt");
    
    // Create test file
    QFile file(testFile);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    file.write("File manager integration test");
    file.close();
    
    // Try to open file manager at the test location
    bool success = testFileManagerOpen(testPath);
    
    // Cleanup
    QFile::remove(testFile);
    
    return success;
}

bool CrossPlatformTesting::testTrashIntegration() {
    QString testPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QString testFile = QDir(testPath).absoluteFilePath("trash_test.txt");
    
    // Create test file
    QFile file(testFile);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    file.write("Trash integration test");
    file.close();
    
    // Try to move file to trash
    bool success = testTrashOperation(testFile);
    
    // If trash operation failed, cleanup manually
    if (!success) {
        QFile::remove(testFile);
    }
    
    return success;
}

UserWorkflow CrossPlatformTesting::adaptWorkflowForPlatform(const UserWorkflow& workflow, Platform platform) {
    UserWorkflow adaptedWorkflow = workflow;
    adaptedWorkflow.id = workflow.id + "_" + platformToString(platform).toLower();
    
    // Get platform-specific adaptations
    QMap<QString, QVariant> adaptations = getPlatformAdaptations(platform);
    
    // Apply adaptations to workflow steps
    for (WorkflowStep& step : adaptedWorkflow.steps) {
        // Adapt file paths
        if (step.parameters.contains("file_path")) {
            QString path = step.parameters["file_path"].toString();
            step.parameters["file_path"] = normalizePath(path, platform);
        }
        
        // Adapt directory paths
        if (step.parameters.contains("directory_path")) {
            QString path = step.parameters["directory_path"].toString();
            step.parameters["directory_path"] = normalizePath(path, platform);
        }
        
        // Apply platform-specific parameters
        for (auto it = adaptations.begin(); it != adaptations.end(); ++it) {
            step.parameters[it.key()] = it.value();
        }
        
        // Adjust timeouts for platform performance characteristics
        switch (platform) {
            case Platform::Windows:
                step.timeoutMs = qRound(step.timeoutMs * 1.2); // 20% longer on Windows
                break;
            case Platform::macOS:
                step.timeoutMs = qRound(step.timeoutMs * 1.0); // No adjustment
                break;
            case Platform::Linux:
                step.timeoutMs = qRound(step.timeoutMs * 0.9); // 10% shorter on Linux
                break;
            case Platform::Unknown:
                step.timeoutMs = qRound(step.timeoutMs * 1.5); // 50% longer for unknown platforms
                break;
        }
    }
    
    return adaptedWorkflow;
}

// Helper method implementations
Platform CrossPlatformTesting::detectCurrentPlatform() const {
#ifdef Q_OS_WIN
    return Platform::Windows;
#elif defined(Q_OS_MACOS)
    return Platform::macOS;
#elif defined(Q_OS_LINUX)
    return Platform::Linux;
#else
    return Platform::Unknown;
#endif
}

QString CrossPlatformTesting::detectPlatformVersion() const {
    QOperatingSystemVersion version = QOperatingSystemVersion::current();
    return QString("%1.%2.%3").arg(version.majorVersion())
                              .arg(version.minorVersion())
                              .arg(version.microVersion());
}

bool CrossPlatformTesting::prepareCrossPlatformTest(const CrossPlatformTest& test) {
    // Setup platform-specific environment
    return configureForPlatform(getCurrentPlatform());
}

bool CrossPlatformTesting::finalizeCrossPlatformTest(const CrossPlatformTest& test, CrossPlatformResult& result) {
    // Cleanup platform-specific setup
    return true;
}

bool CrossPlatformTesting::validatePlatformRequirements(const CrossPlatformTest& test, Platform platform) {
    // Check if platform has required features
    // This is a simplified check - full implementation would verify specific requirements
    return test.targetPlatforms.contains(platform);
}

QStringList CrossPlatformTesting::extractPlatformDifferences(const WorkflowResult& result, Platform platform) {
    QStringList differences;
    
    // Analyze workflow result for platform-specific behaviors
    // This is a placeholder - full implementation would analyze actual differences
    
    switch (platform) {
        case Platform::Windows:
            differences.append("windows_specific_behavior");
            break;
        case Platform::macOS:
            differences.append("macos_specific_behavior");
            break;
        case Platform::Linux:
            differences.append("linux_specific_behavior");
            break;
        case Platform::Unknown:
            break;
    }
    
    return differences;
}

QStringList CrossPlatformTesting::validatePlatformSpecificBehavior(const WorkflowResult& result, Platform platform) {
    QStringList issues;
    
    // Validate that platform-specific behaviors are correct
    // This is a placeholder for actual validation logic
    
    return issues;
}

QStringList CrossPlatformTesting::identifyMissingFeatures(const CrossPlatformTest& test, Platform platform) {
    QStringList missingFeatures;
    
    // Check for features that are not available on the current platform
    switch (platform) {
        case Platform::Windows:
            if (!m_fileSystemConfig.supportsExtendedAttributes) {
                missingFeatures.append("extended_attributes");
            }
            break;
        case Platform::macOS:
            // macOS generally has good feature support
            break;
        case Platform::Linux:
            // Linux generally has good feature support
            break;
        case Platform::Unknown:
            missingFeatures.append("unknown_platform_limitations");
            break;
    }
    
    return missingFeatures;
}

QString CrossPlatformTesting::generatePlatformFeedback(const CrossPlatformResult& result, Platform platform) {
    QString feedback;
    
    if (result.success) {
        feedback = QString("Test completed successfully on %1").arg(platformToString(platform));
    } else {
        feedback = QString("Test failed on %1").arg(platformToString(platform));
        if (!result.compatibilityIssues.isEmpty()) {
            feedback += ". Issues: " + result.compatibilityIssues.join(", ");
        }
    }
    
    return feedback;
}

bool CrossPlatformTesting::configureForPlatform(Platform platform) {
    // Configure testing environment for specific platform
    // This would set up platform-specific settings, paths, etc.
    return true;
}

// Platform-specific helper implementations
bool CrossPlatformTesting::testFileManagerOpen(const QString& path) {
    return QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

bool CrossPlatformTesting::testTrashOperation(const QString& filePath) {
    // Platform-specific trash operations
#ifdef Q_OS_WIN
    // Windows: Move to Recycle Bin
    SHFILEOPSTRUCTA fileOp;
    fileOp.hwnd = nullptr;
    fileOp.wFunc = FO_DELETE;
    fileOp.pFrom = filePath.toLocal8Bit().constData();
    fileOp.pTo = nullptr;
    fileOp.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT;
    return (SHFileOperationA(&fileOp) == 0);
#elif defined(Q_OS_MACOS)
    // macOS: Move to Trash
    QProcess process;
    process.start("osascript", QStringList() << "-e" 
                  << QString("tell application \"Finder\" to delete POSIX file \"%1\"").arg(filePath));
    process.waitForFinished();
    return (process.exitCode() == 0);
#elif defined(Q_OS_LINUX)
    // Linux: Move to Trash (using gio)
    QProcess process;
    process.start("gio", QStringList() << "trash" << filePath);
    process.waitForFinished();
    return (process.exitCode() == 0);
#else
    return false;
#endif
}

bool CrossPlatformTesting::simulateDisplayScale(qreal scale) {
    // This would simulate display scaling for testing purposes
    // Implementation depends on platform and testing framework capabilities
    return true;
}

bool CrossPlatformTesting::testUIElementScaling(qreal scale) {
    // Test that UI elements scale properly
    // This would check font sizes, widget dimensions, etc.
    return true;
}

void CrossPlatformTesting::setupPlatformAdaptations() {
    // Setup default adaptations for each platform
    QMap<QString, QVariant> windowsAdaptations;
    windowsAdaptations["path_separator"] = "\\";
    windowsAdaptations["case_sensitive"] = false;
    windowsAdaptations["line_ending"] = "\r\n";
    m_platformAdaptations[Platform::Windows] = windowsAdaptations;
    
    QMap<QString, QVariant> macosAdaptations;
    macosAdaptations["path_separator"] = "/";
    macosAdaptations["case_sensitive"] = false;
    macosAdaptations["line_ending"] = "\n";
    m_platformAdaptations[Platform::macOS] = macosAdaptations;
    
    QMap<QString, QVariant> linuxAdaptations;
    linuxAdaptations["path_separator"] = "/";
    linuxAdaptations["case_sensitive"] = true;
    linuxAdaptations["line_ending"] = "\n";
    m_platformAdaptations[Platform::Linux] = linuxAdaptations;
}

QMap<QString, QVariant> CrossPlatformTesting::getPlatformAdaptations(Platform platform) const {
    return m_platformAdaptations.value(platform);
}

void CrossPlatformTesting::setPlatformAdaptations(Platform platform, const QMap<QString, QVariant>& adaptations) {
    m_platformAdaptations[platform] = adaptations;
}

// Static utility methods
QString CrossPlatformTesting::platformToString(Platform platform) {
    switch (platform) {
        case Platform::Windows: return "Windows";
        case Platform::macOS: return "macOS";
        case Platform::Linux: return "Linux";
        case Platform::Unknown: return "Unknown";
    }
    return "Unknown";
}

Platform CrossPlatformTesting::stringToPlatform(const QString& platformStr) {
    if (platformStr == "Windows") return Platform::Windows;
    if (platformStr == "macOS") return Platform::macOS;
    if (platformStr == "Linux") return Platform::Linux;
    return Platform::Unknown;
}

bool CrossPlatformTesting::isPlatformSupported(Platform platform) {
    return platform != Platform::Unknown;
}

QStringList CrossPlatformTesting::getSupportedPlatforms() {
    return {"Windows", "macOS", "Linux"};
}

QString CrossPlatformTesting::getFileSystemType(const QString& path) {
    QStorageInfo storage(path);
    return storage.fileSystemType();
}

bool CrossPlatformTesting::isPathValid(const QString& path, Platform platform) {
    FileSystemConfiguration config;
    
    // Get platform-specific configuration
    switch (platform) {
        case Platform::Windows:
            config.maxPathLength = 260;
            config.invalidCharacters = {"<", ">", ":", "\"", "|", "?", "*"};
            break;
        case Platform::macOS:
            config.maxPathLength = 1024;
            config.invalidCharacters = {":"};
            break;
        case Platform::Linux:
            config.maxPathLength = 4096;
            config.invalidCharacters = {"\0"};
            break;
        case Platform::Unknown:
            return false;
    }
    
    // Check path length
    if (path.length() > config.maxPathLength) {
        return false;
    }
    
    // Check for invalid characters
    for (const QString& invalidChar : config.invalidCharacters) {
        if (path.contains(invalidChar)) {
            return false;
        }
    }
    
    return true;
}

QString CrossPlatformTesting::normalizePath(const QString& path, Platform platform) {
    QString normalized = path;
    
    switch (platform) {
        case Platform::Windows:
            normalized.replace("/", "\\");
            break;
        case Platform::macOS:
        case Platform::Linux:
            normalized.replace("\\", "/");
            break;
        case Platform::Unknown:
            break;
    }
    
    return normalized;
}

// Configuration methods
void CrossPlatformTesting::enablePlatformEmulation(bool enable) {
    m_platformEmulationEnabled = enable;
}

void CrossPlatformTesting::setEmulatedPlatform(Platform platform) {
    m_emulatedPlatform = platform;
}

void CrossPlatformTesting::enableFileSystemEmulation(bool enable) {
    m_fileSystemEmulationEnabled = enable;
}

void CrossPlatformTesting::setEmulatedFileSystem(const FileSystemConfiguration& config) {
    m_emulatedFileSystem = config;
}

void CrossPlatformTesting::enableDisplayEmulation(bool enable) {
    m_displayEmulationEnabled = enable;
}

void CrossPlatformTesting::setEmulatedDisplay(const DisplayConfiguration& config) {
    m_emulatedDisplay = config;
}

#include "cross_platform_testing.moc"