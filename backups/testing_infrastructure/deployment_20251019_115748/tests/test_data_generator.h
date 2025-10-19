#pragma once

#include <QString>
#include <QStringList>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDateTime>
#include <QRandomGenerator>
#include <QCryptographicHash>
#include <QMap>
#include <QVariant>
#include <QJsonObject>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QMimeDatabase>
#include <QPixmap>
#include <QImage>

/**
 * @brief Comprehensive test data generation system for DupFinder testing
 * 
 * Provides utilities for creating realistic test datasets including:
 * - File hierarchies with various types and sizes
 * - Duplicate file scenarios for testing detection algorithms
 * - Database test data for configuration and metadata testing
 * - Performance test datasets with controlled characteristics
 */
class TestDataGenerator {
public:
    /**
     * @brief File generation specifications
     */
    struct FileSpec {
        QString fileName;
        QString extension;
        qint64 sizeBytes = 0;
        QString content;
        QDateTime lastModified;
        bool isDuplicate = false;
        QString duplicateOf; // Original file path for duplicates
        QMap<QString, QVariant> metadata;
    };

    /**
     * @brief Directory structure specifications
     */
    struct DirectorySpec {
        QString name;
        int depth = 1;
        int filesPerDirectory = 10;
        int subdirectories = 3;
        QStringList fileTypes = {"txt", "jpg", "png", "pdf", "doc"};
        qint64 minFileSize = 1024;      // 1KB
        qint64 maxFileSize = 1048576;   // 1MB
        double duplicateRatio = 0.2;    // 20% duplicates
        bool createSymlinks = false;
        bool createHiddenFiles = false;
    };

    /**
     * @brief Test scenario specifications
     */
    enum class TestScenario {
        EmptyDirectory,           ///< Empty directory for edge case testing
        SingleFile,              ///< Single file scenarios
        SmallDataset,            ///< Small dataset (< 100 files)
        MediumDataset,           ///< Medium dataset (100-1000 files)
        LargeDataset,            ///< Large dataset (1000+ files)
        DeepHierarchy,           ///< Deep directory structure
        WideHierarchy,           ///< Wide directory structure
        MixedFileTypes,          ///< Various file types and sizes
        DuplicateHeavy,          ///< High percentage of duplicates
        PerformanceStress,       ///< Large files for performance testing
        EdgeCases,               ///< Special characters, long names, etc.
        CrossPlatform,           ///< Platform-specific file scenarios
        CorruptedFiles,          ///< Files with corruption for error testing
        SymlinkScenario,         ///< Symbolic links and junctions
        HiddenFiles              ///< Hidden and system files
    };

    /**
     * @brief Database test data specifications
     */
    struct DatabaseSpec {
        QString databaseType = "SQLite";
        QString connectionString;
        QStringList tables;
        QMap<QString, QVariantList> testData;
        bool populateWithSampleData = true;
        int recordCount = 100;
    };

    TestDataGenerator();
    ~TestDataGenerator();

    // File and directory generation
    QString generateTestDirectory(const DirectorySpec& spec, const QString& basePath = "");
    QString generateTestFile(const FileSpec& spec, const QString& directory = "");
    QStringList generateFileHierarchy(TestScenario scenario, const QString& basePath = "");
    
    // Duplicate file generation
    QString createDuplicateFile(const QString& originalFile, const QString& targetDirectory = "");
    QStringList createDuplicateSet(const QString& originalFile, int duplicateCount, const QString& baseDirectory = "");
    void generateDuplicateScenario(const QString& directory, double duplicateRatio = 0.3);
    
    // Content generation
    QByteArray generateFileContent(const QString& fileType, qint64 sizeBytes);
    QByteArray generateRandomBinaryData(qint64 sizeBytes);
    QString generateTextContent(qint64 sizeBytes, const QString& pattern = "");
    QPixmap generateImageContent(int width, int height, const QString& format = "PNG");
    
    // Realistic file scenarios
    QString generateDocumentFile(const QString& directory, const QString& type = "txt");
    QString generateImageFile(const QString& directory, const QString& format = "jpg");
    QString generateMediaFile(const QString& directory, const QString& format = "mp4");
    QString generateArchiveFile(const QString& directory, const QStringList& filesToArchive = {});
    
    // Special scenarios
    void generateCorruptedFiles(const QString& directory, int count = 5);
    void generateLongPathFiles(const QString& directory, int pathLength = 250);
    void generateSpecialCharacterFiles(const QString& directory);
    void generateSymlinks(const QString& directory, const QStringList& targets);
    void generateHiddenFiles(const QString& directory, int count = 3);
    
    // Database test data
    bool setupTestDatabase(const DatabaseSpec& spec);
    bool populateTestDatabase(const QString& connectionString, const QMap<QString, QVariantList>& data);
    void cleanupTestDatabase(const QString& connectionString);
    
    // Performance test data
    QString generatePerformanceDataset(qint64 totalSizeBytes, const QString& basePath = "");
    QString generateLargeFileSet(int fileCount, qint64 fileSizeBytes, const QString& basePath = "");
    QString generateDeepDirectoryStructure(int depth, int width, const QString& basePath = "");
    
    // Cleanup and management
    void registerGeneratedPath(const QString& path);
    void cleanupGeneratedData();
    void cleanupPath(const QString& path);
    QStringList getGeneratedPaths() const;
    
    // Validation and verification
    bool verifyGeneratedStructure(const QString& path, const DirectorySpec& expectedSpec);
    QMap<QString, qint64> analyzeGeneratedData(const QString& path);
    QString generateDatasetReport(const QString& path);
    
    // Configuration and customization
    void setRandomSeed(quint32 seed);
    void setTemporaryDirectory(const QString& tempDir);
    void setCleanupOnDestruction(bool cleanup);
    
    // Utility functions
    static QString generateUniqueFileName(const QString& baseName, const QString& extension = "");
    static QString generateRandomString(int length, bool alphaNumericOnly = true);
    static QDateTime generateRandomDateTime(const QDateTime& start = QDateTime::currentDateTime().addDays(-365),
                                          const QDateTime& end = QDateTime::currentDateTime());
    static QString calculateFileHash(const QString& filePath, QCryptographicHash::Algorithm algorithm = QCryptographicHash::Md5);
    
    // Predefined scenarios
    static DirectorySpec getScenarioSpec(TestScenario scenario);
    static QStringList getCommonFileExtensions();
    static QStringList getImageExtensions();
    static QStringList getDocumentExtensions();
    static QStringList getMediaExtensions();

private:
    QTemporaryDir* m_tempDir;
    QStringList m_generatedPaths;
    QRandomGenerator* m_randomGenerator;
    QMimeDatabase m_mimeDatabase;
    bool m_cleanupOnDestruction;
    QString m_customTempDirectory;
    
    // Internal helper methods
    QString createDirectoryStructure(const QString& basePath, const DirectorySpec& spec, int currentDepth = 0);
    void populateDirectoryWithFiles(const QString& directory, const DirectorySpec& spec);
    FileSpec generateRandomFileSpec(const DirectorySpec& dirSpec);
    QString selectRandomFileType(const QStringList& types);
    qint64 generateRandomFileSize(qint64 minSize, qint64 maxSize);
    void createFileWithContent(const QString& filePath, const QByteArray& content, const QDateTime& lastModified = QDateTime());
    
    // Content generators for specific file types
    QByteArray generateTextFileContent(qint64 sizeBytes);
    QByteArray generateBinaryFileContent(qint64 sizeBytes);
    QByteArray generateImageFileContent(const QString& format, int width = 800, int height = 600);
    QByteArray generateDocumentContent(const QString& type, qint64 sizeBytes);
    
    // Platform-specific helpers
    void setPlatformSpecificAttributes(const QString& filePath, const QMap<QString, QVariant>& attributes);
    bool createSymbolicLink(const QString& linkPath, const QString& targetPath);
    void setFileHidden(const QString& filePath, bool hidden = true);
    
    // Validation helpers
    bool validateDirectorySpec(const DirectorySpec& spec);
    bool validateFileSpec(const FileSpec& spec);
    void logGenerationProgress(const QString& operation, int current, int total);
};

/**
 * @brief Test environment isolation manager
 * 
 * Provides isolated test environments to prevent test interference
 */
class TestEnvironmentIsolator {
public:
    TestEnvironmentIsolator();
    ~TestEnvironmentIsolator();
    
    // Environment isolation
    QString createIsolatedEnvironment(const QString& testName);
    void destroyIsolatedEnvironment(const QString& environmentId);
    QString getEnvironmentPath(const QString& environmentId);
    
    // Process isolation
    bool runInIsolatedProcess(const QString& command, const QStringList& arguments, const QString& workingDirectory = "");
    
    // Resource monitoring
    void startResourceMonitoring(const QString& environmentId);
    void stopResourceMonitoring(const QString& environmentId);
    QMap<QString, QVariant> getResourceUsage(const QString& environmentId);
    
    // Cleanup management
    void scheduleCleanup(const QString& environmentId, int delaySeconds = 0);
    void cleanupAllEnvironments();

private:
    QMap<QString, QString> m_environments;
    QMap<QString, QVariant> m_resourceMonitors;
    QString m_baseIsolationPath;
    
    QString generateEnvironmentId();
    void setupEnvironmentIsolation(const QString& environmentPath);
    void cleanupEnvironment(const QString& environmentPath);
};

/**
 * @brief Macros for convenient test data generation
 */
#define GENERATE_TEST_DIRECTORY(scenario, basePath) \
    TestDataGenerator::getScenarioSpec(TestDataGenerator::TestScenario::scenario), basePath

#define CREATE_TEMP_FILE(name, content) \
    do { \
        TestDataGenerator generator; \
        TestDataGenerator::FileSpec spec; \
        spec.fileName = name; \
        spec.content = content; \
        generator.generateTestFile(spec); \
    } while(0)

#define CLEANUP_TEST_DATA(generator) \
    do { \
        generator.cleanupGeneratedData(); \
    } while(0)