#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDir>
#include <QTemporaryDir>
#include <QProcess>
#include <QMap>
#include <QVariant>
#include <memory>

/**
 * @brief Specification for creating test files
 */
struct TestFileSpec {
    QString fileName;
    QByteArray content;
    qint64 size = -1;  // If size is specified and > content.size(), file will be padded
    QDateTime lastModified = QDateTime::currentDateTime();
    bool isHidden = false;
    QFile::Permissions permissions = QFile::ReadOwner | QFile::WriteOwner | QFile::ReadGroup | QFile::ReadOther;
};

/**
 * @brief Specification for creating test directory structures
 */
struct TestDirectorySpec {
    QString name;
    QList<TestFileSpec> files;
    QList<TestDirectorySpec> subdirectories;
    bool isHidden = false;
    QFile::Permissions permissions = QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner | 
                                   QFile::ReadGroup | QFile::ExeGroup | QFile::ReadOther | QFile::ExeOther;
};

/**
 * @brief Application launch configuration
 */
struct AppLaunchConfig {
    QString executablePath;
    QStringList arguments;
    QMap<QString, QString> environment;
    QString workingDirectory;
    int timeoutMs = 30000;
    bool waitForStartup = true;
};

/**
 * @brief System simulation parameters
 */
struct SystemSimulation {
    qreal dpiScale = 1.0;
    int cpuDelayMs = 0;
    int memoryLimitMB = 0;
    int diskSpeedKBps = 0;
    int networkLatencyMs = 0;
    int networkBandwidthKbps = 0;
    bool simulateSlowSystem = false;
};

/**
 * @brief Test environment management and setup
 * 
 * This class provides comprehensive test environment management including:
 * - Test data creation and cleanup
 * - Application lifecycle management
 * - System condition simulation
 * - Environment isolation
 */
class TestEnvironment : public QObject {
    Q_OBJECT

public:
    explicit TestEnvironment(QObject* parent = nullptr);
    ~TestEnvironment();

    // Environment lifecycle
    bool setupTestEnvironment();
    bool cleanupTestEnvironment();
    bool resetToDefaults();
    bool isSetup() const { return m_isSetup; }

    // Test directory management
    QString getTestDataDirectory() const;
    QString createTestDirectory(const QString& name = QString());
    bool removeTestDirectory(const QString& path);
    void setTestDataRoot(const QString& rootPath);

    // Test file creation
    bool createTestFile(const QString& filePath, const TestFileSpec& spec);
    bool createTestFiles(const QString& basePath, const QList<TestFileSpec>& specs);
    bool createTestDirectoryStructure(const QString& basePath, const TestDirectorySpec& spec);
    
    // Predefined test datasets
    bool createPhotoLibraryDataset(const QString& basePath, int numPhotos, int duplicatePercentage);
    bool createDocumentDataset(const QString& basePath, int numDocs, const QStringList& formats);
    bool createMixedMediaDataset(const QString& basePath, int totalFiles, const QMap<QString, int>& typeDistribution);
    bool createLargeFileDataset(const QString& basePath, int numFiles, qint64 minSizeMB, qint64 maxSizeMB);
    bool createDuplicateGroupDataset(const QString& basePath, int groups, int filesPerGroup);

    // Application management
    bool launchApplication(const AppLaunchConfig& config = AppLaunchConfig());
    bool closeApplication();
    bool restartApplication();
    bool isApplicationRunning() const;
    QProcess* getApplicationProcess() const { return m_appProcess.get(); }

    // Application state management
    bool resetApplicationState();
    bool clearApplicationSettings();
    bool loadApplicationSettings(const QString& settingsFile);
    bool saveApplicationSettings(const QString& settingsFile);

    // System simulation
    void setSystemSimulation(const SystemSimulation& simulation);
    SystemSimulation getSystemSimulation() const { return m_systemSimulation; }
    bool simulateHighDPI(qreal scale);
    bool simulateSlowSystem(bool enabled = true);
    bool simulateNetworkConditions(int latencyMs, int bandwidthKbps);
    bool simulateLowMemory(int limitMB);

    // Database management
    bool setupTestDatabase(const QString& dbPath = QString());
    bool clearTestDatabase();
    bool loadTestData(const QString& dataFile);

    // Configuration management
    QString getConfigDirectory() const;
    bool createTestConfiguration(const QMap<QString, QVariant>& config);
    bool loadTestConfiguration(const QString& configFile);
    QVariant getConfigValue(const QString& key, const QVariant& defaultValue = QVariant()) const;
    void setConfigValue(const QString& key, const QVariant& value);

    // Utility methods
    QString generateRandomContent(int sizeBytes);
    QByteArray generateBinaryContent(int sizeBytes);
    QString generateUniqueFileName(const QString& baseName, const QString& extension);
    bool copyFile(const QString& source, const QString& destination);
    bool compareFiles(const QString& file1, const QString& file2);
    qint64 getDirectorySize(const QString& path);
    QStringList findFiles(const QString& path, const QStringList& patterns = QStringList());

    // Environment validation
    bool validateEnvironment();
    QStringList getEnvironmentIssues();
    bool hasRequiredPermissions(const QString& path);

signals:
    void environmentSetup();
    void environmentCleanup();
    void applicationLaunched();
    void applicationClosed();
    void testDataCreated(const QString& path);
    void testDataRemoved(const QString& path);

private slots:
    void onApplicationFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onApplicationError(QProcess::ProcessError error);

private:
    // Helper methods
    bool createDirectory(const QString& path, QFile::Permissions permissions = QFile::Permissions());
    bool setFilePermissions(const QString& filePath, QFile::Permissions permissions);
    bool setFileTimestamp(const QString& filePath, const QDateTime& timestamp);
    QString generateTestContent(const QString& contentType, int size);
    bool padFileToSize(const QString& filePath, qint64 targetSize);
    void cleanupTemporaryFiles();
    bool killApplicationProcess();

    // State management
    bool m_isSetup = false;
    QString m_testDataRoot;
    std::unique_ptr<QTemporaryDir> m_tempDir;
    QStringList m_createdDirectories;
    QStringList m_createdFiles;
    
    // Application management
    std::unique_ptr<QProcess> m_appProcess;
    AppLaunchConfig m_appConfig;
    QString m_originalWorkingDir;
    
    // System simulation
    SystemSimulation m_systemSimulation;
    qreal m_originalDpiScale = 1.0;
    
    // Configuration
    QMap<QString, QVariant> m_testConfig;
    QString m_configDirectory;
    QString m_originalConfigDir;
    
    // Cleanup tracking
    QStringList m_filesToCleanup;
    QStringList m_directoriesToCleanup;
    QStringList m_processesToKill;
};

/**
 * @brief RAII helper for test environment management
 */
class TestEnvironmentGuard {
public:
    explicit TestEnvironmentGuard(TestEnvironment* env) : m_env(env) {
        if (m_env) {
            m_env->setupTestEnvironment();
        }
    }
    
    ~TestEnvironmentGuard() {
        if (m_env) {
            m_env->cleanupTestEnvironment();
        }
    }
    
    TestEnvironment* operator->() { return m_env; }
    TestEnvironment* get() { return m_env; }

private:
    TestEnvironment* m_env;
};

/**
 * @brief Utility macros for test environment usage
 */
#define TEST_ENV_GUARD(env) TestEnvironmentGuard guard(env)

#define CREATE_TEST_FILE(env, path, content) \
    do { \
        TestFileSpec spec; \
        spec.fileName = QFileInfo(path).fileName(); \
        spec.content = content; \
        TEST_VERIFY(env->createTestFile(path, spec)); \
    } while(0)

#define CREATE_TEST_DIR(env, path) \
    TEST_VERIFY(!env->createTestDirectory(path).isEmpty())

#define LAUNCH_APP(env) \
    TEST_VERIFY(env->launchApplication())

#define CLEANUP_APP(env) \
    do { \
        if (env->isApplicationRunning()) { \
            env->closeApplication(); \
        } \
    } while(0)